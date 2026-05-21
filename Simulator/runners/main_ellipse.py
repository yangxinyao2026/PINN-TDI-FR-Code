import os  #这行代码导入了Python的os模块，该模块提供了与操作系统交互的功能，包括环境变量的访问和修改。
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  #设置环境变量KMP_DUPLICATE_LIB_OK为TRUE，这是为了避免在使用某些PyTorch版本时出现的OpenMP库重复加载问题。os.environ是一个字典对象，它代表了当前进程的环境变量。通过修改它，可以改变当前进程的环境变量设置。
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
from Simulator import  PROJECT_ROOT  #从Simulator模块导入项目根目录常量
import torch
from Simulator.cases.basic_cases import case_ellipse  #从Simulator.cases.basic_cases中导入case_ellipse函数，用于创建椭圆案例。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #设置设备，如果有GPU则使用GPU，否则使用CPU。
parallel= False  #设置并行标志为False，表示不使用并行训练。
model_type = 'fullnet'  #设置模型类型为'fullnet'全网络，其他可选值为'pretrainnet'预训练网络和'biasnet'偏置网络。
case = case_ellipse(model_type = model_type,device = device)  #调用case_ellipse函数，传入模型类型和设备，获取一个案例配置字典。

if model_type=='pretrainnet':  #如果模型类型是预训练网络，则设置训练次数为500，并创建PreTrainNet模型，传入案例中的A_hat和b_hat矩阵，并将模型移动到设备上。
    n_train = 500
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
else:  #如果不是预训练网络（即biasnet或fullnet），则设置训练次数为15，先创建PreTrainNet模型，然后加载预训练好的权重。权重文件路径由项目根目录、案例名称和固定文件名拼接而成。
    n_train = 15
    model = PreTrainNet(case['A_hat'], case['b_hat'])  #创建PreTrainNet实例（用于加载预训练权重）
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth",map_location=device))  #加载预训练好的权重文件。map_location=device: 确保权重加载到正确的设备上
    A_pretrained, b_pretrained = model()  #运行预训练模型，获取A_pretrained和b_pretrained。注意，这里模型已经被加载了预训练权重，所以输出的是预训练的结果。
    b_pretrained = b_pretrained[0].detach().cpu().numpy()  #将b_pretrained取第一个元素（可能是一个batch中的第一个）。detach(): 断开计算图，避免梯度传播。移动到CPU并转换为numpy数组。
    if model_type == 'biasnet':  #如果是biasnet，则处理A_pretrained（同样取第一个元素，断开计算图并确保在设备上），然后更新训练器配置，加入A_pretrained。接着创建BiasNet模型，传入参数维度（从case['params']['count']获取），用预训练的b_pretrained初始化偏置，只训练偏置项。并移动到设备。
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,device = device).to(device)
    elif model_type == 'fullnet':  #如果是fullnet，则将A_pretrained转换为numpy数组，然后创建FullNet模型，用预训练的A和b进行初始化，训练所有参数。
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,device = device).to(device)

trainer = Trainer(  #创建Trainer对象，传入模型、案例中的误差计算器和损失计算函数。
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)
trainer.configure(**case['trainer_configure'])  #使用案例中的训练器配置来配置训练器（学习率、优化器等）。
trainer.initialize()  #初始化训练器可能包括优化器、损失记录等。
trainer.train(n_train = n_train , params_data = case['params'], parallel= parallel)  #开始训练，传入训练次数、参数数据以及是否并行。
torch.save(model.state_dict(), case['result_path'])  #用于保存模型权重


'''
总结
这段代码实现了一个渐进式的神经网络训练流程：
   预训练阶段：大量迭代学习基础线性关系
   精调阶段：少量迭代微调特定组件
   BiasNet：固定线性变换，只调整偏置
   FullNet：全面调整所有参数
这种设计常用于物理模拟或数值逼近问题，其中预训练提供了良好的初始点，精调阶段针对特定任务进行优化。
'''