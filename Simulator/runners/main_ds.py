import os
import numpy as np
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  #设置环境变量KMP_DUPLICATE_LIB_OK为TRUE，这是为了避免在使用某些PyTorch版本时出现的OpenMP库重复加载问题。os.environ是一个字典对象，它代表了当前进程的环境变量。通过修改它，可以改变当前进程的环境变量设置。
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer  #从Simulator.Approximator模块导入预训练网络、偏置网络、全网络、损失函数计算和训练器类。
from Simulator.Plotter import ErrorVisualizer  # 导入误差可视化器
import time  # 导入时间模块用于测量训练时间
import torch  #导入PyTorch深度学习库。
from Simulator import PROJECT_ROOT  #从Simulator模块导入PROJECT_ROOT，即项目根目录。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #检查是否有可用的GPU，如果有则使用GPU，否则使用CPU。



def compute_errors_random_dtheta_safe(error_calculator, nn_model, ppc, device, init_params_dict, n_dtheta=5, n_samples_per_dtheta=10, zero_dtheta=False):
    """
        随机生成 dtheta，计算可行性和最优性误差
        参数:
            error_calculator: ErrorCalculator 实例
            nn_model: 当前训练的神经网络模型 (FullNet, BiasNet, PreTrainNet)
            ppc: 电力系统案例数据
            device: 计算设备 (cpu/cuda)
            init_params_dict: 参数字典，包含 'Pd_meta' 和 'Qd_meta' 的初始值
            n_dtheta: 随机生成的 dtheta 数量
            n_samples_per_dtheta: 每个 dtheta 的误差计算样本数
            zero_dtheta: 若为True，dtheta固定为0（用于pretrainnet）
        """
    """
    安全版本：使用error_calculator的副本进行计算，不修改原始对象
    支持单相（1D参数）和三相（2D参数）
    """
    # 创建error_calculator的副本，避免修改原始对象
    error_calc_copy = error_calculator.copy()

    # 从init_params_dict获取初始值，自动适配1D/2D（单相/三相）
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    total_params = Pd_init.size + Qd_init.size

    # 配置参数
    dtheta_range = (-0.5, 0.5)
    np.random.seed(42)  # 固定种子，但仅影响误差计算，不影响训练

    # 生成dtheta值
    if zero_dtheta:
        dtheta_values = [np.zeros(total_params)]
    else:
        dtheta_values = [np.random.uniform(dtheta_range[0], dtheta_range[1], total_params) for _ in range(n_dtheta)]

    # 初始化存储误差的列表
    feas_errors = []
    opt_errors = []

    for dtheta in dtheta_values:
        # dtheta拆分为Pd和Qd部分，加到初始值上
        Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
        Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)

        # 通过update_parameters更新，自动处理1D/2D索引映射
        error_calc_copy.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})

        # 使用当前神经网络模型预测A和b（不修改模型状态）
        with torch.no_grad():
            dtheta_tensor = torch.tensor(dtheta, dtype=torch.float32).to(device)
            if isinstance(nn_model, FullNet):
                A_pred, b_pred = nn_model(dtheta_tensor)
            elif isinstance(nn_model, PreTrainNet):
                A_pred, b_pred = nn_model()
                A_pred = A_pred.repeat(1, 1, 1)
                b_pred = b_pred.repeat(1, 1)
            elif isinstance(nn_model, BiasNet):
                b_pred = nn_model(dtheta_tensor)
                A_pred = nn_model.A_pretrained.repeat(1, 1, 1)
            else:
                raise ValueError(f"不支持的模型类型: {type(nn_model)}")

            A_pred_np = A_pred[0].detach().cpu().numpy()
            b_pred_np = b_pred[0].detach().cpu().numpy()

        # 在副本上更新近似多面体
        error_calc_copy.update_polytope(A_hat=A_pred_np, b_hat=b_pred_np)

        # 在副本上计算误差
        feas_results, opt_results = error_calc_copy.calculate(
            n_cal=n_samples_per_dtheta, cal_feas=True, cal_opt=True)

        # 提取误差值
        feas_errors.extend([r['error'] for r in feas_results])
        opt_errors.extend([r['error'] for r in opt_results])

    # 转换为numpy数组
    return np.array(feas_errors), np.array(opt_errors)



model_type = 'fullnet'  #设置模型类型为全网络（fullnet）。其他选项包括预训练网络（pretrainnet）和偏置网络（biasnet）。
parallel= False  #设置是否使用并行训练。启用并行计算加速训练cplex并行训练。因true代码会卡住，所以改为false
record_errors = False  # 是否在训练过程中记录误差数据（设为False可加速训练，不记录误差也不保存误差文件）
dscases = {
    # 'case10ba_ds': TD_case.case10ba_ds(),
    # 'case17me_ds': TD_case.case17me_ds(),
     'case33bw_ds': TD_case.case33bw_ds(),
    # 'case51ga_ds': TD_case.case51ga_ds(),
    # 'case74_ds': TD_case.case74_ds(),
    # 'case118zh_ds': TD_case.case118zh_ds(),
    # 'case136ma_ds': TD_case.case136ma_ds(),
    # 'case533mt_hi_ds': TD_case.case533mt_hi_ds(),
    # 'case36real_3phase_ds': DS_case_3phase.case36real_3phase_ds(),
}

for casename, ppc in dscases.items():
    P_rated = sum(ppc['bus'][:,2])/ppc['baseMVA']
    #lr = 1e-1/P_rated
    lr = 3e-5/P_rated  #fullnet
    rate_opt_feas = 0.6

    # 记录总开始时间
    total_start_time = time.time()
    is_3phase = '3phase' in casename
    if is_3phase:
        case = DS_case_3phase.DScase_3phase_train(casedata=ppc, model_type=model_type, device=device)
    else:
        case = TD_case.DScase_train(casedata=ppc, model_type=model_type, device=device, plot_flag=True)


    # 根据开关决定是否记录训练过程中的误差数据
    if record_errors:
        visualizer = ErrorVisualizer()

        if model_type == 'pretrainnet':
            # pretrainnet 不依赖参数，固定 dtheta=0
            if 'training_callback' in case['trainer_configure']:
                original_callback = case['trainer_configure']['training_callback']

                def enhanced_callback(error_calculator, epoch):
                    original_callback(error_calculator, epoch)
                    if epoch % 200 == 0:
                        feas_errors, opt_errors = compute_errors_random_dtheta_safe(error_calculator, model, ppc, device, case['params']['params_dict'], n_dtheta=1, n_samples_per_dtheta=50, zero_dtheta=True)
                        visualizer.error_history['iterations'].append(epoch)
                        visualizer.error_history['error_feas'].append(feas_errors)
                        visualizer.error_history['error_opt'].append(opt_errors)
            else:
                def enhanced_callback(error_calculator, epoch):
                    if epoch % 200 == 0:
                        feas_errors, opt_errors = compute_errors_random_dtheta_safe(error_calculator, model, ppc, device, case['params']['params_dict'], n_dtheta=1, n_samples_per_dtheta=50, zero_dtheta=True)
                        visualizer.error_history['iterations'].append(epoch)
                        visualizer.error_history['error_feas'].append(feas_errors)
                        visualizer.error_history['error_opt'].append(opt_errors)
        else:
            # fullnet / biasnet 使用随机 dtheta，用 offset 拼接两阶段横坐标
            if 'training_callback' in case['trainer_configure']:
                original_callback = case['trainer_configure']['training_callback']

                def enhanced_callback(error_calculator, epoch):
                    # 调用原始回调函数
                    original_callback(error_calculator, epoch)
                    # 每隔200次迭代才记录误差分布，减少不必要的计算
                    if epoch % 200 == 0:
                        feas_errors, opt_errors = compute_errors_random_dtheta_safe(error_calculator, model, ppc, device, case['params']['params_dict'], n_dtheta=5, n_samples_per_dtheta=10)
                        visualizer.error_history['iterations'].append(enhanced_callback.offset + epoch)
                        visualizer.error_history['error_feas'].append(feas_errors)
                        visualizer.error_history['error_opt'].append(opt_errors)
            else:
                def enhanced_callback(error_calculator, epoch):
                    if epoch % 200 == 0:
                        feas_errors, opt_errors = compute_errors_random_dtheta_safe(error_calculator, model, ppc, device, case['params']['params_dict'], n_dtheta=5, n_samples_per_dtheta=10)
                        visualizer.error_history['iterations'].append(enhanced_callback.offset + epoch)
                        visualizer.error_history['error_feas'].append(feas_errors)
                        visualizer.error_history['error_opt'].append(opt_errors)
            enhanced_callback.offset = 0

        case['trainer_configure']['training_callback'] = enhanced_callback


    if model_type=='pretrainnet':
        n_train = 500
        model   = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=False, device = device)
    else:
        n_train = 20
        # 加载预训练权重
        model = PreTrainNet(case['A_hat'], case['b_hat'],is_epigraph=False,device=device)
        result_dir = os.path.dirname(case['result_path'])
        model.load_state_dict(torch.load(os.path.join(result_dir, 'pretrainnet_weights.pth'),map_location=device))
        # 提取预训练参数
        A_pretrained, b_pretrained = model()
        b_pretrained = b_pretrained[0].detach().cpu().numpy()

        # 偏置网络
        if model_type == 'biasnet':
            A_pretrained = A_pretrained[0].detach().to(device)

            case['trainer_configure'].update(A_pretrained = A_pretrained)  # 新增或更新这个键值对
            model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,n_hidden=128,device = device).to(device)

        # 全网络
        elif model_type == 'fullnet':
            A_pretrained = A_pretrained[0].detach().cpu().numpy()
            model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,n_hidden=128,device = device).to(device)

    trainer = Trainer(  #初始化训练器，传入模型、误差计算器和损失函数。
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )
                    # ** 操作符，字典解包：用于将字典中的键值对作为关键字参数传递给函数或方法。 #个数可变的关键字参数  #*个数可变的位置参数，可以解包列表
    trainer.configure(**case['trainer_configure'])  #配置训练器的参数，包括学习率和rate_opt_feas。
    trainer.configure(lr = lr)
    trainer.configure(rate_opt_feas = rate_opt_feas)
    trainer.initialize()  #初始化训练器。

    if model_type == 'pretrainnet':
        # pretrainnet 只有一个训练阶段
        phase1_start = time.time()
        trainer.train(n_train=n_train*4, params_data=case['params'], parallel=parallel)
        phase1_end = time.time()
        torch.save(model.state_dict(), case['result_path'])

        if not record_errors:
            phase1_duration = phase1_end - phase1_start
            total_end_time = time.time()
            total_overall_duration = total_end_time - total_start_time
            print(f"训练时间统计 - {casename}:")
            print(f"  训练时间: {phase1_duration:.2f} 秒 ({phase1_duration/60:.2f} 分钟)")
            print(f"  总耗时（端到端）: {total_overall_duration:.2f} 秒 ({total_overall_duration/60:.2f} 分钟)")
            # 保存训练时间
            result_dir = os.path.dirname(case['result_path'])
            os.makedirs(result_dir, exist_ok=True)
            time_data_path = os.path.join(result_dir, f'{model_type}_training_time.npz')
            np.savez(time_data_path,
                     phase1_time=phase1_duration,
                     total_time=total_overall_duration)
            print(f"训练时间已保存至: {time_data_path}")

    else:
        # fullnet / biasnet 有两个训练阶段
        # 第一阶段
        phase1_start = time.time()
        trainer.train(n_train=n_train * 4-1, params_data=case['params'], parallel=parallel)
        phase1_end = time.time()
        torch.save(model.state_dict(), case['result_path'])

        # 第二阶段：最小化可行性误差
        if record_errors:
            # 更新 offset，使 Phase 2 的横坐标从 Phase 1 末尾接续
            enhanced_callback.offset = (n_train * 4) * len(case['params']['dataloader'])
        trainer.configure(lr=1e-5/P_rated)
        trainer.configure(rate_opt_feas=1e-4)
        trainer.initialize()
        phase2_start = time.time()
        trainer.train(n_train=n_train * 2 , params_data=case['params'], parallel=parallel)
        phase2_end = time.time()
        torch.save(model.state_dict(), f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\A(8,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)\\{model_type}_weights_feasible.pth')

        if not record_errors:
            phase1_duration = phase1_end - phase1_start
            phase2_duration = phase2_end - phase2_start
            total_duration = phase1_duration + phase2_duration
            total_end_time = time.time()
            total_overall_duration = total_end_time - total_start_time
            print(f"训练时间统计 - {casename}:")
            print(f"  第一阶段训练: {phase1_duration:.2f} 秒 ({phase1_duration/60:.2f} 分钟)")
            print(f"  第二阶段训练: {phase2_duration:.2f} 秒 ({phase2_duration/60:.2f} 分钟)")
            print(f"  总训练时间: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
            print(f"  总耗时（端到端）: {total_overall_duration:.2f} 秒 ({total_overall_duration/60:.2f} 分钟)")
            # 保存训练时间
            result_dir = os.path.dirname(case['result_path'])
            os.makedirs(result_dir, exist_ok=True)
            time_data_path = os.path.join(result_dir, f'{model_type}_training_time.npz')
            np.savez(time_data_path,
                     phase1_time=phase1_duration,
                     phase2_time=phase2_duration,
                     total_train_time=total_duration,
                     total_time=total_overall_duration)
            print(f"训练时间已保存至: {time_data_path}")

    # 保存训练过程误差数据（绘图请运行 main_ds_plot.py）
    if record_errors:
        result_dir = os.path.dirname(case['result_path'])
        os.makedirs(result_dir, exist_ok=True)
        error_filename = f'{model_type}_error_data.npz'
        error_data_path = os.path.join(result_dir, error_filename)

        error_history = visualizer.error_history
        original_iterations = error_history['iterations']

        # 修复迭代次数为200的倍数
        if len(original_iterations) > 0:
            start_rounded = round(original_iterations[0] / 200) * 200
            iterations = np.array([start_rounded + i * 200 for i in range(len(original_iterations))])
        else:
            iterations = np.array([])

        save_dict = {'iterations': iterations}
        if len(error_history['error_feas']) > 0:
            save_dict['error_feas'] = np.array(error_history['error_feas'])
            save_dict['error_opt'] = np.array(error_history['error_opt'])

        np.savez(error_data_path, **save_dict)
        print(f"训练误差数据已保存至: {error_data_path}")
