import numpy as np
from sympy import Reals

from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
import torch
from torch.utils.data import Dataset, DataLoader
from pyomo.environ import *
from Simulator.Plotter import ShapeDrawer_2D
import matplotlib.pyplot as plt
from Simulator import PROJECT_ROOT
import os
import pickle
from Simulator.Plotter import ErrorVisualizer

def case_polygon(total_samples = 200, noise_scale=0.15, batch_size=1, model_type = 'pretrainnet',device = 'cpu'):  # total_samples：生成的数据样本数量。noise_scale：添加到数据的噪声大小。batch_size：训练时的批量大小（批量梯度下降、随机梯度下降、小批量梯度下降）
    """固定参数的原始案例实现"""
    # 固定初始化参数
    A_init = np.array([
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, -1], [1, -1], [-1, 1]
    ])
    b_init = np.array([1, 1, 0, 0, 1.5, -0.5, 0.7, 0.7])  # 定义八边形原始范围

    # 自动推导维度
    dim = A_init.shape[1]  #获取数组的列数
    ncons = A_init.shape[0]  #获取数组的行数
    dim_theta = 2  #含义：可调参数 theta 的维度。解释：在这个问题中，有2个约束的参数是可以学习调整的
    num_b = ncons - dim_theta  #含义：固定参数 b 的数量。解释：在8个约束中，有2个是可调的，剩下6个是固定的

    # 构建模型
    model = ConcreteModel()

    # 参数定义
    model.A = Param(range(ncons), range(dim),
                    initialize={(i, j): A_init[i, j] for i, j in np.ndindex(A_init.shape)},
                    mutable=True)  #A_init.shape = (8, 2) → 8行2列。np.ndindex((8, 2)) 生成一个迭代器，产生所有可能的索引组合
    model.b = Param(range(num_b),
                    initialize={i: b_init[i] for i in range(num_b)},
                    mutable=True)  #固定参数 b，从 b_init 的前6个元素取值。固定约束：物理限制、安全边界等不可改变的条件，保持问题的基本结构（通过固定约束）
    model.theta = Param(range(dim_theta),
                        initialize={i: b_init[num_b + i] for i in range(2)},
                        mutable=True)  #可调参数 theta，从 b_init 的第6、7个位置取值。可调约束：可以通过学习优化的参数

    # 变量定义
    model.var_proj = Var(range(dim), domain=Reals)

    # 约束定义
    def constraint_rule(model, i, is_adjustable):  #i：约束的索引（在固定约束中为0-5，在可调约束中为0-1）。is_adjustable：布尔值，标识这是固定约束还是可调约束
        idx = i + num_b if is_adjustable else i    #如果 is_adjustable=True：idx = i + num_b = i + 6（可调约束）；如果 is_adjustable=False：idx = i（固定约束）
        param = model.theta[i] if is_adjustable else model.b[i]
        return sum(model.A[idx, j] * model.var_proj[j] for j in range(dim)) <= param  #return 的意思是"返回"或"输出"。它告诉函数："计算到这里结束，把这个结果送出去"。

    model.con_fixed = Constraint(range(num_b), rule=lambda m, i: constraint_rule(m, i, False))  #model.constraint_name = Constraint(索引集合, rule=规则函数)。规则函数必须是 function(model, index) 形式
    model.con_adj = Constraint(range(dim_theta), rule=lambda m, i: constraint_rule(m, i, True))  #lambda 语法：lambda 参数: 表达式。m → 传递给 constraint_rule 的 model 参数；i → 传递给 constraint_rule 的 i 参数；Ture → 可调约束的标志

    original_model = {  #创建了一个字典 original_model，用于封装原始优化模型的信息。
        'model': model,
        # 'baseline': baseline,
    }

    # 数据集配置
    class CaseData(Dataset):  #定义了一个名为CaseData的类，它继承自PyTorch的Dataset类。这意味着我们可以用它来创建数据加载器（DataLoader），从而方便地进行批量训练。
        def __init__(self, size=total_samples):  #初始化方法 __init__:
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):  #返回数据集的大小200
            return self.size

        def __getitem__(self, idx):  #根据索引idx（数据集的大小）返回一个样本。每个样本是一个字典，包含一个键'theta'，其值是一个从均值为0、标准差为noise_scale的正态分布中抽取的随机向量，向量的维度是dim_theta（固定为2），并且指定了设备（CPU或GPU）。生成均值为0，标准差为0.15的2维随机向量
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    A_hat = np.vstack([  #将两个矩阵垂直堆叠在一起
        np.eye(dim),  # 上界  #单位矩阵
        -np.eye(dim),  # 下界  #负单位矩阵
    ])  #作为初始近似
    errorcalculator = ErrorCalculator(  #初始化了一个 ErrorCalculator 对象
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',     #将cplex求解器改为gurobi
    )
    A_list = [errorcalculator.A_hat]  # 存储初始的约束矩阵
    b_list = [errorcalculator.b_hat]  # 存储初始的约束右端项

    visualizer = ErrorVisualizer()  # 创建可视化器实例。ErrorVisualizer 是一个专门用于分析和可视化误差的工具类
    num_sample = 50  # 这个参数指定了用于误差评估的采样点数量
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)  # 计算误差

    case_name = 'polygon'  #定义案例名称
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'  # 定义存放图片的文件夹路径。
    os.makedirs(figure_folder, exist_ok=True)  #创建上述文件夹。如果文件夹已经存在，不会抛出错误（因为exist_ok=True）。
    plt.figure(figsize=(8, 6))  #创建一个新的图形窗口，设置大小为8英寸×6英寸。这是使用matplotlib库来创建图形。
    plotter = ShapeDrawer_2D()  #创建一个二维形状绘制器实例
    def callback(errorcalculator, epoch):  #回调函数：在训练过程中定期调用的函数。epoch：当前的训练迭代次数
        len_his = len(errorcalculator.training_history['feas'])  #获取可行性误差历史的长度。training_history 是存储训练过程中各种指标的字典。'feas' 键存储可行性误差的历史值。
        print(f"Iter {epoch}: FeasErr={np.mean(errorcalculator.training_history['feas'][-min(10,len_his):]):.2e}, "
              f"OptErr={np.mean(errorcalculator.training_history['opt'][-min(10,len_his):]):.2e}")
        if model_type.lower() == 'pretrainnet':  #.lower() 确保大小写不敏感
            xlim = [-0.5, 1.5]
            ylim = [-0.5, 1.5]  # 坐标轴范围设置。固定显示范围，确保所有图形有一致的视角
            if not epoch:  # 等价于 if epoch == 0:
                plotter.plot_polygon(A_init, b_init,  #绘制原始多边形
                                     facecolor='blue', xlim=xlim, ylim=ylim,  #蓝色：原始多边形
                                     label=f'Original Region',  #标签："Original Region"
                                     title=f'Training step = {0}',  #标题：显示训练步数
                                     )
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,  #绘制初始近似
                                     facecolor='green', xlim=xlim, ylim=ylim,  #绿色：近似多边形
                                     label=f'Approximation',  #标签："Approximation"
                                     title=f'Training step = {epoch}'  #与原始多边形在同一坐标系中
                                     )
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])  #移除旧的近似。plotter.shapes[-1]：获取最后一个形状（最近绘制的近似多边形）。['id']：获取该形状的ID。remove_shape()：从图形中移除这个形状
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                     facecolor='green', xlim=xlim, ylim=ylim,
                                     label=f'Approximation',
                                     title=f'Training step = {epoch}'
                                     )
            # 保存图片（自动创建目录）
            plotter.save(figure_folder + f'/pretrain_process/step{epoch}')
            if (epoch + 20) % 100 == 0:
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)  #将当前的近似约束矩阵A_hat和约束向量b_hat分别添加到A_list和b_list中，以记录训练过程中近似多边形的演变。
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)  #调用visualizer.compute_errors计算当前近似模型的误差，使用50个样本（由num_sample=50指定）进行误差评估。
            if epoch>=980:
                with open(f'{PROJECT_ROOT}/results/{case_name}/A_list.pkl', "wb") as f:  #A_list.pkl: 保存整个训练过程中收集的A_hat列表。
                    pickle.dump(A_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/b_list.pkl', "wb") as f:  #b_list.pkl: 保存整个训练过程中收集的b_hat列表。
                    pickle.dump(b_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/error_history.pkl', "wb") as f:  #error_history.pkl: 保存可视化器计算得到的误差历史。
                    pickle.dump(visualizer.error_history, f)  #pickle 格式。with open('file.pkl', 'wb') as f:  # 'wb' 表示二进制写模式

    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
                'call_interval':20,  #调用回调函数的间隔，即每多少次迭代调用一次callback函数。
                'training_callback':callback,  #回调函数
                "optimizer": "SGD",  #优化器的类型.随机梯度下降Stochastic Gradient Descent
                "lr": 0.5,
                "batch_size": 1,  #批量大小。训练过程中 θ 样本的批次大小。
                "scheduler": {  #学习率调度器的配置，包括类型、步长和衰减因子。
                    "type": "StepLR",  #调度器类型，这里都是StepLR，即按步长调整学习率。
                    "step_size": 100,  #调整步长，每多少步调整一次。
                    "gamma": 0.98  #调整的倍数，这里每step_size步将学习率乘以0.98。
                },
                "n_cal": 2,  #探索次数：每次迭代中用于方向误差计算的采样方向数量。。每次迭代进行2次完整的梯度计算流程，取平均获得更稳定的更新方向
                "cal_feas": True,  #是否计算可行性误差。
                "cal_opt": True,  #是否计算最优性误差。
                "rate_opt_feas": 1  #权重，用于平衡最优性误差和可行性误差在损失函数中的比例。
            }#这些配置被存储在trainer_configure字典中，供后续训练使用。
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",  #自适应矩估计（Adaptive Moment Estimation）
            "lr": 0.015,
            "batch_size": batch_size,
            "scheduler": {
                "type": "StepLR",
                "step_size": 100,
                "gamma": 0.98
            },
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }
    params_dict, param_count = pyomo_params_to_numpy(model)  #将 Pyomo 模型中的参数转换为 NumPy 格式，便于神经网络处理。
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,  #所有参数的名称及其初始值构成的字典
        'dataloader': DataLoader(  #用于参数采样的数据集加载器
            CaseData(),  #CaseData()：前面定义的数据集类（生成带噪声的theta）
            batch_size=batch_size,  #根据模型类型设置
            shuffle=True  #打乱数据顺序，提高训练效果
        ),
        'count':dim_theta,  #可调参数的总数
    }
    return {
        'casename':'polygon',
        'A_hat': A_hat,  #初始矩阵 A
        'b_hat': errorcalculator.b_hat,  #初始向量 b
        'errorcalculator': errorcalculator,
        'params':params,
        'trainer_configure': trainer_configure,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
        'metadata':{'A_init':A_init,'b_init':b_init}
    }

def case_ellipse(total_samples=100, noise_scale=0.2, batch_size=2, model_type='pretrainnet',device = 'cpu'):
    """椭圆约束案例实现"""
    # 固定初始化参数 (二次型矩阵)
    Sigma_init = np.array([
        [5 / 2, -3 / 2],       # [ [a, b],
        [-3 / 2, 5 / 2]        #   [b, a] ]
    ])  #椭圆[x1, x2] * Sigma * [x1; x2]
    dim = 2  # 固定为二维问题
    a_init = 5 / 2
    b_init = -3 / 2

    # 构建Pyomo模型
    model = ConcreteModel()

    # 可调参数定义 (Sigma矩阵的上三角元素)
    model.a = Param(initialize=a_init, mutable=True)
    model.b = Param(initialize=b_init, mutable=True)

    # 决策变量
    model.var_proj = Var(range(dim), domain=Reals)

    # 椭圆约束 (二次型)
    model.constraints = Constraint(
        expr=(model.a * model.var_proj[0] ** 2
        + 2 * model.b * model.var_proj[0] * model.var_proj[1]
        + model.a * model.var_proj[1] ** 2) <= 1
    )  #定义椭圆约束：a*x₁² + 2b*x₁x₂ + a*x₂² ≤ 1

    original_model = {'model': model}

    # 数据集配置
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):  #成训练样本，为椭圆约束的参数添加随机噪声。每个样本代表一个略微不同的椭圆形状，模型学习适应这个椭圆族的变化，最终得到的近似器对参数扰动具有鲁棒性。
            return {"a":torch.normal(0, self.noise_scale, (1,),device=device), #0: 均值 - 噪声围绕0值波动。self.noise_scale: 标准差 - 控制噪声的幅度（默认0.2）。(1,): 输出形状 - 生成标量张量
                    "b":torch.normal(0, self.noise_scale, (1,),device=device),
            }  # theta维度固定为2
    # 近似器参数
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
        np.array([[1, 1], [-1, -1]])  # 对角线约束
    ])  #六边形
    # 误差计算器
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi'   #将cplex求解器改为gurobi
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    visualizer = ErrorVisualizer()
    num_sample = 50
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)

    case_name = 'ellipse'
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
    os.makedirs(figure_folder, exist_ok=True)

    # 可视化回调函数
    plt.figure(figsize=(8, 6))
    plotter = ShapeDrawer_2D()
    xlim = [-2, 2]
    ylim = [-2, 2]

    # marking_epoches = [1,6,12,25,50,100,195]
    marking_epoches = list(range(1,195,20))+[195]  #设置一个标记epoch的列表：标记点包括从1到195（不含195）每隔20个epoch取一个，再加上195这个epoch。共 11个
    def callback(error_calculator, epoch):  #这是一个回调函数，在训练过程中定期被调用，接收两个参数：error_calculator: 误差计算器对象；epoch: 当前训练轮次
        if not hasattr(callback, "idx_mark"):  #检查回调函数是否有一个名为idx_mark的属性。如果没有，则初始化这个属性为0。这个属性用于记录当前已经处理到的marking_epoches列表的索引。因为marking_epoches列表定义了需要执行额外记录操作的epoch。
            callback.idx_mark = 0  # 初始化计数器
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "  #滑动平均计算
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")

        if model_type.lower() == 'pretrainnet':
            if epoch == 0:
                plotter.plot_ellipse(Sigma_init, xlim=xlim, ylim=ylim,
                                     facecolor='blue', label='Original region')
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation'
                                        , title=f'Training step {epoch}')
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            plotter.save(f"{figure_folder}/step{epoch}")
            if callback.idx_mark>=len(marking_epoches):  # 当 callback.idx_mark = 11 (超过列表长度10) 时触发
                with open(f'{PROJECT_ROOT}/results/{case_name}/A_list.pkl', "wb") as f:
                    pickle.dump(A_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/b_list.pkl', "wb") as f:
                    pickle.dump(b_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/error_history.pkl', "wb") as f:
                    pickle.dump(visualizer.error_history, f)
            elif epoch >= marking_epoches[callback.idx_mark]:  # 到达标记点记录数据
                callback.idx_mark+=1
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)  ## 计算并记录误差




    # 训练参数配置
    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 5,  #回调函数频率
            "training_callback": callback,
            "optimizer": "SGD",  #随机梯度下降
            "lr": 0.2,
            "batch_size": 1,  #可变参数
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},  # ！学习率逐渐衰减
            "n_cal": 2,  #探索次数
            "cal_feas": True,
            "cal_opt": True,  #需不需要
            "rate_opt_feas": 1  #最优和可行误差权重比
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            # "optimizer": "sgd",
            "lr": 0.0003,   #由于训练效果不好，调整学习率
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }


    params_dict, param_count = pyomo_params_to_numpy(model)  #它将Pyomo模型中的参数转换为NumPy数组的形式，并返回一个参数字典和参数个数。
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,  # 参数字典，包含从Pyomo模型转换而来的参数
        'dataloader': DataLoader(  #是一个PyTorch的DataLoader，它使用之前定义的CaseData数据集，设置批大小和打乱数据。
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,  #需要学习的参数总数（这里是2个：a和b）
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,   # 初始多边形近似矩阵
        'b_hat': errorcalculator.b_hat,  # 初始多边形近似偏移
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,  # 训练配置
        'params':params,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
        'metadata': {   # 案例元数据
            'Sigma_init': Sigma_init,  # 初始椭圆矩阵（用于参考和可视化）
            'dim': dim    # 问题维度（固定为2）
        }
    }

def case_epigraph(total_samples=100, noise_scale=0.15, batch_size=5, model_type='pretrainnet',device = 'cpu'):
    """Epigraph问题案例实现"""
    # 初始参数设置
    theta_init = np.array([1.0, 1.0])  # theta[0]=x上界, theta[1]=x²系数
    dim = 1  # 变量维度 (x, f)
    dim_theta = 2

    # 构建Pyomo模型
    model = ConcreteModel()

    # 可调参数定义
    model.theta = Param(range(dim_theta),
                        initialize={i: theta_init[i] for i in range(dim_theta)},
                        mutable=True)

    # 决策变量

    model.var_proj = Var(range(dim+1), domain=Reals) # var_proj[0]=x, var_proj[1]=f

    model.x = model.var_proj[0]  # 前n-1个元素的表达式别名
    model.f = model.var_proj[dim]

    model.obj = Expression(expr=model.theta[1] * model.x ** 2)
    # model.f = Var(domain=Reals)

    # 约束定义
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 0)  # x下界固定
    model.constraints.add(model.x <= model.theta[0])  # 上界由theta控制

    model.constraints.add(model.f >= model.obj)  # 下界约束
    # model.constraints.add(model.f <= model.theta[1] * model.theta[0]**2)  # Epigraph上界

    original_model = {'model': model}

    # 数据集配置
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    # 近似器参数 (线性化约束矩阵)

    A_hat = np.vstack([
        [1, 0],  # x <= theta0
        [-1, 0],  # x >= 0
        [0, -1],  # f >= theta1*x² (需要后续处理)
        [2,-1],
        [1, -1],
    ],dtype=float) #所有的A矩阵写在这里
    A_hat = np.vstack([A_hat,[0,1.]]) #这一行不能变，这表示目标函数的上界
    # [0, 1],  # f <= theta0*theta1

    case_name = 'epigraph'
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
    os.makedirs(figure_folder, exist_ok=True)

    # 可视化工具
    plotter = ShapeDrawer_2D()
    xlim = [-0.5, 1.5]
    ylim = [-0.5, 1.5]

    def callback(error_calculator, epoch):
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")

        if model_type.lower() == 'pretrainnet':
            if epoch == 0:
                plotter.plot_epigraph(
                    x_range=(0, 1),
                    f_min_func=lambda x: x**2,
                    # facecolor='rgba(135,206,250,0.3)',
                    label="Original epigraph",
                    xlim = xlim,
                    ylim = ylim
                )
                plotter.plot_polygon(np.vstack([error_calculator.A_hat,[0,1]]), np.hstack([error_calculator.b_hat,[error_calculator.fmax]]),
                                     xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(np.vstack([error_calculator.A_hat,[0,1]]), np.hstack([error_calculator.b_hat,[error_calculator.fmax]]),
                                     xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            plotter.save(f"{figure_folder}/step{epoch}.png")

    # 误差计算器
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        is_epigraph=True,
        solver='ipopt'
    )

    # 训练参数配置
    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 10,
            "training_callback": callback,
            "optimizer": "SGD",
            "lr": 0.15,
            "batch_size": 3,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.97},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.01,
            # "optimizer": "SGD",
            # "lr": 0.05,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.99},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
            # "theta_init": theta_init
        }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'fmax':errorcalculator.fmax,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
    }

def case_nonconvex(total_samples=200, noise_scale=0.3, batch_size=5, model_type='pretrainnet',device = 'cpu'):
    """非凸优化问题案例实现"""
    # 初始参数设置
    theta_init = np.array([1, 1, 1])  # theta[0]: 圆心x偏移,theta[1]: 圆心y偏移, theta[2]: 最小半径
    dim = 2  # 变量维度
    dim_theta = len(theta_init)

    # 构建Pyomo模型
    model = ConcreteModel()

    # 定义可变参数
    model.theta = Param(range(dim_theta),
                        initialize={i: theta_init[i] for i in range(dim_theta)},
                        mutable=True)
    # 定义变量及非对称边界
    def variable_bounds(m, i):
        return (-1, 1)  # var_proj[0] ∈ [-1,2], var_proj[1] ∈ [-2,4]

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # 非凸约束定义
    model.constraints = ConstraintList()
    # 单位圆约束 (凸)
    model.constraints.add(model.var_proj[0] ** 2 + model.var_proj[1] ** 2 <= 1)
    # 动态环形约束 (非凸)
    model.constraints.add(
        (model.var_proj[0] - model.theta[0]) ** 2 + (model.var_proj[1] - model.theta[1]) ** 2 >= model.theta[2] ** 2
    )

    original_model = {'model': model}

    # 数据集配置（生成theta扰动）
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    # 近似器矩阵（包含边界约束）
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
        [[1, 1], [-1, -1]]  # 对角线约束
    ])
    # 误差计算器（支持非凸约束评估）
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='ipopt',  # 使用支持非凸的求解器
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    visualizer = ErrorVisualizer()
    num_sample = 50
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)

    case_name = 'nonconvex'
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
    os.makedirs(figure_folder, exist_ok=True)
    plt.figure(figsize=(6, 6))
    # 可视化回调函数
    plotter = ShapeDrawer_2D()
    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    marking_epoches = list(range(1,300,30))+[280]

    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # 初始化计数器
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
        if model_type.lower() == 'pretrainnet':
            # 绘制原始区域
            if epoch == 0:
                # 单位圆
                plotter.plot_circle_regions(
                    theta=theta_init,
                    xlim=xlim,  # 包含两个圆的可视范围
                    ylim=ylim,
                    edgecolor='skyblue',
                    facecolor='skyblue',  # 区域填充色
                    alpha=0.3,  # 透明度
                    label = 'Nonconvex region'
                )
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation'
                                     , title=f'Training step {epoch}')
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            plotter.save(f"{figure_folder}/step{epoch}.png")

            if callback.idx_mark>=len(marking_epoches):
                with open(f'{PROJECT_ROOT}/results/{case_name}/A_list.pkl', "wb") as f:
                    pickle.dump(A_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/b_list.pkl', "wb") as f:
                    pickle.dump(b_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/error_history.pkl', "wb") as f:
                    pickle.dump(visualizer.error_history, f)
            elif epoch >= marking_epoches[callback.idx_mark]:
                callback.idx_mark+=1
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)




    if model_type.lower() == 'pretrainnet':
        # 训练参数配置
        trainer_configure = {
            "call_interval": 20,
            "training_callback": callback,
            # "optimizer": "Adam",
            "optimizer": "sgd",
            "lr": 0.2,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.99},
            "n_cal": 2,  # 减少校准次数提高稳定性
            "cal_feas": True,
            "cal_opt": True,  # 非凸问题暂不优化目标
            "rate_opt_feas": 1
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.006,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
    }


def case_ball(dim = 2, total_samples=200, noise_scale=0.15, batch_size=5, model_type='pretrainnet',device = 'cpu'):
    """非凸优化问题案例实现"""
    # 初始参数设置
    R_init = 1.0  # theta 圆半径
    dim_theta = 1

    # 构建Pyomo模型
    model = ConcreteModel()

    # 定义可变参数
    model.R = Param(initialize= R_init, mutable=True)
    # 定义变量及非对称边界
    def variable_bounds(m, i):
        return (-5, 5)

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # 非凸约束定义
    model.constraints = ConstraintList()
    # 单位圆约束 (凸)
    model.constraints.add(expr = sum(model.var_proj[i] ** 2 for i in range(dim))<= model.R)

    original_model = {'model': model}

    # 数据集配置（生成theta扰动）
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    # 近似器矩阵（包含边界约束）
    # A_hat = np.vstack([
    #     np.eye(dim),  # 上界
    #     -np.eye(dim),  # 下界
    # ])
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
    ])
    A_hat += np.random.normal(loc=0, scale=0.5/np.sqrt(dim), size=A_hat.shape)


    case_name = 'ball'
    results_folder = f'{PROJECT_ROOT}\\results\\{case_name}'
    os.makedirs(results_folder, exist_ok=True)

    def callback(error_calculator, epoch):
        len_his = len(error_calculator.training_history['feas'])
        e_feas = np.mean(error_calculator.training_history['feas'][-min(10, len_his):])
        e_opt = np.mean(error_calculator.training_history['opt'][-min(10, len_his):])
        print(f"Iter {epoch}: FeasErr={e_feas:.2e}, "
              f"OptErr={e_opt:.2e}")
    # 误差计算器（支持非凸约束评估）
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',  # 使用支持非凸的求解器
    )


    if model_type.lower() == 'pretrainnet':
        # 训练参数配置
        trainer_configure = {
            "call_interval": 5,
            "training_callback": callback,
            # "optimizer": "Adam",
            "optimizer": "sgd",
            "lr": 0.3,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 1.00},
            "n_cal": 2,  # 减少校准次数提高稳定性
            "cal_feas": True,
            "cal_opt": True,  # 非凸问题暂不优化目标
            "rate_opt_feas": 1
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.006/dim,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 20, "gamma": 1.0},
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1.0,
        }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': errorcalculator.A_hat,
        'b_hat': errorcalculator.b_hat,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}',
    }

def case_cube(dim = 2, model_type='pretrainnet',device = 'cpu'):
    """非凸优化问题案例实现"""
    # 初始参数设置
    d_init = 1.0

    # 构建Pyomo模型
    model = ConcreteModel()
    def variable_bounds(m, i):
        return (-1, 1)

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # 非凸约束定义
    model.constraints = ConstraintList()

    original_model = {'model': model}

    # 近似器矩阵（包含边界约束）
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
    ])

    A_hat += np.random.normal(loc=0, scale=0.5/sqrt(dim), size=A_hat.shape)
    # 误差计算器（支持非凸约束评估）
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',  # 使用支持非凸的求解器
    )

    case_name = 'cube'
    results_folder = f'{PROJECT_ROOT}\\results\\{case_name}'
    os.makedirs(results_folder, exist_ok=True)

    def callback(error_calculator, epoch):
        # 初始化end_flag（如果尚未存在）
        if not hasattr(callback, 'end_flag'):
            callback.end_flag = False
            callback.start_flag = True

        # 打开文件用于追加写入
        len_his = len(error_calculator.training_history['feas'])
        e_feas = np.mean(error_calculator.training_history['feas'][-min(10, len_his):])
        e_opt = np.mean(error_calculator.training_history['opt'][-min(10, len_his):])
        print(f"Iter {epoch}: FeasErr={e_feas:.2e}, "
              f"OptErr={e_opt:.2e}")
        with open(f'{results_folder}\\results_dim{dim}.txt', 'a') as f:
            if callback.start_flag:
                f.write(f"Initial: FeasErr={e_feas:.4e}, OptErr={e_opt:.4e}\n")
                callback.start_flag = False
            if (e_feas + e_opt) / 2 < 1e-6 and (not callback.end_flag):
                print(epoch)
                callback.end_flag = True
                # 写入epoch
                f.write(f"Converged at epoch: {epoch}\n")


    trainer_configure = {
        "call_interval": 10,
        "training_callback": callback,
        # "optimizer": "Adam",
        "optimizer": "sgd",
        "lr": 0.3,
        "batch_size": 1,
        "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 1.05},
        "n_cal": 3,  # 减少校准次数提高稳定性
        "cal_feas": True,
        "cal_opt": True,  # 非凸问题暂不优化目标
        "rate_opt_feas": 1
    }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': [None],
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}',
    }