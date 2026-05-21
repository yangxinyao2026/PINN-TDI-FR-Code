import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.basic_cases import case_polygon  #从Simulator.cases.basic_cases导入case_polygon，这是一个生成案例的函数。
from Simulator import PROJECT_ROOT  #从Simulator导入PROJECT_ROOT，这是项目根目录的路径。
from Simulator.Plotter import ErrorVisualizer  # 导入误差可视化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel = True  #parallel设置为True，表示使用并行训练（但后面训练器中并没有直接使用这个变量，可能是在训练器内部使用）。
model_type = 'fullnet'  #model_type设置为'pretrainnet'，表示使用预训练网络。
case = case_polygon(model_type = model_type,device=device)  #根据model_type和device创建案例，返回一个字典，包含案例的相关参数。调用case_polygon函数创建测试案例，传入模型类型和设备参数，返回包含所有案例配置的字典

# 创建误差可视化器并增强回调函数以记录误差分布
visualizer = ErrorVisualizer()
original_callback = case['trainer_configure']['training_callback']

def enhanced_callback(error_calculator, epoch):
    # 调用原始回调函数
    original_callback(error_calculator, epoch)
    # 记录误差分布（减少采样次数以提高性能）
    visualizer.compute_errors(error_calculator, num_sample=10)

# 替换训练配置中的回调函数
case['trainer_configure']['training_callback'] = enhanced_callback
if model_type=='pretrainnet':
    n_train = 1000  #设置训练次数为1000次
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)  #创建PreTrainNet模型，传入A_hat和b_hat矩阵，移动到指定设备
else:
    n_train = 6  #设置训练次数为20次（微调阶段）
    model = PreTrainNet(case['A_hat'], case['b_hat'])  #创建空的PreTrainNet模型
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth', map_location=device))  #从文件加载预训练好的权重
    A_pretrained, b_pretrained = model()  #运行预训练模型，获取A_pretrained和b_pretrained。注意，这里模型已经被加载了预训练权重，所以输出的是预训练的结果。
    b_pretrained = b_pretrained[0].detach().cpu().numpy()  #提取b参数，分离计算图，转移到CPU并转为numpy数组。#将b_pretrained取第一个元素（可能是一个batch中的第一个）。detach(): 断开计算图，避免梯度传播。移动到CPU并转换为numpy数组。
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)  #提取A参数并转移到设备
        case['trainer_configure'].update(A_pretrained = A_pretrained)  # 更新训练器配置，添加预训练的A矩阵
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,device = device).to(device)  #创建BiasNet模型，初始化偏置参数
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()  #提取A参数并转为numpy数组
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,device = device).to(device)  #创建FullNet模型，初始化A和b参数

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)  #创建Trainer训练器实例，传入模型、误差计算器和损失函数

trainer.configure(**case['trainer_configure'])  #使用案例中的训练配置来配置训练器（解包字典参数）
# trainer.configure(lr = 0.9)  #手动学习率配置
trainer.initialize()  #初始化训练器
trainer.train(n_train = n_train , params_data=case["params"], parallel = parallel)  #开始训练，传入训练次数、参数数据和并行标志
torch.save(model.state_dict(), case['result_path'])  #训练完成后保存模型权重到指定路径

# 绘制并保存误差箱线图（间隔采样）
import os
result_dir = os.path.dirname(case['result_path'])
os.makedirs(result_dir, exist_ok=True)
boxplot_path = os.path.join(result_dir, 'error_boxplot_interval.png')

interval = 100  # 默认值
#print(f"未找到误差历史数据，使用默认间隔: {interval}")

visualizer.plot_dual_boxplot_interval(save_path=boxplot_path, interval=interval)
print(f"误差箱线图（间隔显示）已保存至: {boxplot_path}")

# 绘制并保存误差对比分布图（KDE和带抖动的箱型图）
comparison_path = os.path.join(result_dir, 'error_comparison_distributions.png')
# 使用误差计算器进行误差分布分析
if 'errorcalculator' in case:
    error_calculator = case['errorcalculator']
    visualizer.plot_comparison_distributions(model=error_calculator, num_sample=1000, save_path=comparison_path)
    print(f"误差对比分布图已保存至: {comparison_path}")
else:
    print("警告：未找到误差计算器，无法绘制误差分布图")
