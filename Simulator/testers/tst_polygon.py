import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet, ErrorCalculator
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D, ErrorVisualizer
import torch
from Simulator.cases.basic_cases import case_polygon
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = 'pretrainnet'
case = case_polygon(model_type = model_type)
dim_theta = 2
pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'], device=device)
pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth', map_location=device))
pretrainnet = pretrainnet.to(device)
A_pretrained,b_pretrained = pretrainnet()
A_pretrained = A_pretrained[0].detach().cpu().numpy()
b_pretrained = b_pretrained[0].detach().cpu().numpy()
biasnet = BiasNet(dim_theta = dim_theta, b_init = b_pretrained, device=device)
biasnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\biasnet_weights.pth', map_location=device))
biasnet = biasnet.to(device)
fullnet = FullNet(dim_theta = dim_theta, A_init=A_pretrained,b_init = b_pretrained, device=device)
fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\fullnet_weights.pth', map_location=device))
fullnet = fullnet.to(device)
plt.figure(figsize=(8, 6))
plotter = ShapeDrawer_2D()

dtheta = 0.2
xlim = [-0.5, 1.5]
ylim = [-0.5, 1.5]
b = np.array([1, 1, 0, 0, 1.5, -0.5, 0.7+dtheta , 0.7+dtheta ])

figure_folder_name = f'{PROJECT_ROOT}\\results\\{case['casename']}\\figures\\comparison\\'
os.makedirs(figure_folder_name, exist_ok=True)

plotter.plot_polygon(case['metadata']['A_init'], b,
                     facecolor='blue', xlim=xlim, ylim=ylim,
                     label=f'Original region',
                     # title=f'Training step = {0}',
                     )

plotter.plot_polygon(A_pretrained, b_pretrained,
                     facecolor='green', xlim=xlim, ylim=ylim,
                     label=f'pretrain',
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'pretrain dtheta = {dtheta:.2e}.png')
plotter.remove_shape(plotter.shapes[-1]['id'])

b_pred = biasnet(torch.tensor([dtheta,dtheta], dtype=torch.float32).to(device))
b_pred = b_pred.detach().cpu().numpy()
plotter.plot_polygon(A_pretrained, b_pred,
                     facecolor='yellow', xlim=xlim, ylim=ylim,
                     label=f'biasnet',
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'biasnet dtheta = {dtheta:.2e}.png')
plotter.remove_shape(plotter.shapes[-1]['id'])
A_pred, b_pred = fullnet(torch.tensor([dtheta,dtheta], dtype=torch.float32).to(device))
A_pred = A_pred[0].detach().cpu().numpy()
b_pred = b_pred[0].detach().cpu().numpy()

plotter.plot_polygon(A_pred, b_pred,
                     facecolor='red', xlim=xlim, ylim=ylim,
                     label=f'fullnet',
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'fullnet dtheta = {dtheta:.2e}.png')


# ============================================================================
# 随机生成 dtheta，测试可行性与最优性误差，绘制分布图
# ============================================================================

# 检查是否存在误差计算器
if 'errorcalculator' not in case:
    print("警告：未找到误差计算器，无法进行误差分析")
else:
    error_calculator = case['errorcalculator']

    # 配置误差计算器的阈值参数
    error_calculator.configure(feas_tol=1e-8, opt_tol=1e-8)

    # 配置参数
    n_dtheta = 100  # 随机 dtheta 的数量
    n_samples_per_dtheta = 50  # 每个 dtheta 的误差计算样本数
    dtheta_range = (-0.5, 0.5)  # dtheta 的采样范围

    # 生成随机 dtheta 值
    np.random.seed(42)  # 固定随机种子以便复现
    dtheta_values = np.random.uniform(dtheta_range[0], dtheta_range[1], n_dtheta)

    # 初始化存储误差的列表
    feas_errors_biasnet = []
    opt_errors_biasnet = []
    feas_errors_fullnet = []
    opt_errors_fullnet = []

    print(f"开始测试 {n_dtheta} 个随机 dtheta 值...")

    for idx, dtheta in enumerate(dtheta_values):
        if (idx + 1) % 10 == 0:
            print(f"  处理第 {idx + 1}/{n_dtheta} 个 dtheta = {dtheta:.3f}")

        # 更新原始模型的 theta 参数 (可调参数)
        # 在 polygon 案例中，theta 参数对应 b_init 的最后两个元素
        # 根据 basic_cases.py，theta 参数名为 'theta'，是二维的
        theta_new = np.array([0.7 + dtheta, 0.7 + dtheta])  # 原始值为 0.7, 0.7
        error_calculator.update_parameters({'theta': theta_new})

        # 计算 biasnet 预测
        b_pred_bias = biasnet(torch.tensor([dtheta, dtheta], dtype=torch.float32).to(device))
        b_pred_bias = b_pred_bias.detach().cpu().numpy()

        # 计算 fullnet 预测
        A_pred_full, b_pred_full = fullnet(torch.tensor([dtheta, dtheta], dtype=torch.float32).to(device))
        A_pred_full = A_pred_full[0].detach().cpu().numpy()
        b_pred_full = b_pred_full[0].detach().cpu().numpy()

        # 计算 biasnet 误差
        error_calculator.update_polytope(A_hat=A_pretrained, b_hat=b_pred_bias)
        feas_results, opt_results = error_calculator.calculate(
            n_cal=n_samples_per_dtheta, cal_feas=True, cal_opt=True)

        # 提取误差值
        feas_errors_biasnet.extend([r['error'] for r in feas_results])
        opt_errors_biasnet.extend([r['error'] for r in opt_results])

        # 计算 fullnet 误差
        error_calculator.update_polytope(A_hat=A_pred_full, b_hat=b_pred_full)
        feas_results, opt_results = error_calculator.calculate(
            n_cal=n_samples_per_dtheta, cal_feas=True, cal_opt=True)

        # 提取误差值
        feas_errors_fullnet.extend([r['error'] for r in feas_results])
        opt_errors_fullnet.extend([r['error'] for r in opt_results])

    print("误差计算完成，开始绘制分布图...")

    # 转换为 numpy 数组
    feas_errors_biasnet = np.array(feas_errors_biasnet)
    opt_errors_biasnet = np.array(opt_errors_biasnet)
    feas_errors_fullnet = np.array(feas_errors_fullnet)
    opt_errors_fullnet = np.array(opt_errors_fullnet)

    # 创建输出目录
    error_dist_dir = os.path.join(os.path.dirname(figure_folder_name), 'error_distributions')
    os.makedirs(error_dist_dir, exist_ok=True)

    # 导入绘图所需库
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # ============================================================================
    # 绘制可行性误差分布图 (KDE 和箱型图)
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左侧：KDE 密度图
    ax_kde = axes[0]

    # 计算 KDE (只计算 FullNet)
    kde_fullnet = gaussian_kde(feas_errors_fullnet)

    # 创建 x 轴范围 (基于 FullNet)
    x_min = feas_errors_fullnet.min()
    x_max = feas_errors_fullnet.max()
    x_range = np.linspace(x_min, x_max, 1000)

    # 绘制 KDE (只绘制 FullNet)
    ax_kde.plot(x_range, kde_fullnet(x_range), label='FullNet', color='red', linewidth=2)
    ax_kde.fill_between(x_range, kde_fullnet(x_range), alpha=0.3, color='red')

    ax_kde.set_xlabel('Feasibility Error', fontsize=12)
    ax_kde.set_ylabel('Density', fontsize=12)
    ax_kde.set_title('KDE Distribution of Feasibility Errors (FullNet Only)', fontsize=14)
    ax_kde.legend()
    ax_kde.grid(True, alpha=0.3)

    # 右侧：带抖动的箱型图
    ax_box = axes[1]

    # 准备数据 (只包含 FullNet)
    data = [feas_errors_fullnet]
    labels = ['FullNet']

    # 绘制箱型图
    bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True)

    # 设置箱型图颜色
    colors = ['lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # 添加抖动散点
    for i, d in enumerate(data):
        # 添加随机抖动
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax_box.scatter(x, d, alpha=0.5, s=10, color=colors[i], edgecolors='black', linewidths=0.5)

    ax_box.set_ylabel('Feasibility Error', fontsize=12)
    ax_box.set_title('Boxplot with Jitter of Feasibility Errors (FullNet Only)', fontsize=14)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()
    feas_save_path = os.path.join(error_dist_dir, 'feasibility_error_distribution.png')
    plt.savefig(feas_save_path, dpi=300, bbox_inches='tight')
    print(f"可行性误差分布图已保存至: {feas_save_path}")
    plt.close()

    # ============================================================================
    # 绘制最优性误差分布图 (KDE 和箱型图)
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左侧：KDE 密度图
    ax_kde = axes[0]

    # 计算 KDE (只计算 FullNet)
    kde_fullnet = gaussian_kde(opt_errors_fullnet)

    # 创建 x 轴范围 (基于 FullNet)
    x_min = opt_errors_fullnet.min()
    x_max = opt_errors_fullnet.max()
    x_range = np.linspace(x_min, x_max, 1000)

    # 绘制 KDE (只绘制 FullNet)
    ax_kde.plot(x_range, kde_fullnet(x_range), label='FullNet', color='red', linewidth=2)
    ax_kde.fill_between(x_range, kde_fullnet(x_range), alpha=0.3, color='red')

    ax_kde.set_xlabel('Optimality Error', fontsize=12)
    ax_kde.set_ylabel('Density', fontsize=12)
    ax_kde.set_title('KDE Distribution of Optimality Errors (FullNet Only)', fontsize=14)
    ax_kde.legend()
    ax_kde.grid(True, alpha=0.3)

    # 右侧：带抖动的箱型图
    ax_box = axes[1]

    # 准备数据 (只包含 FullNet)
    data = [opt_errors_fullnet]
    labels = ['FullNet']

    # 绘制箱型图
    bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True)

    # 设置箱型图颜色
    colors = ['lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # 添加抖动散点
    for i, d in enumerate(data):
        # 添加随机抖动
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax_box.scatter(x, d, alpha=0.5, s=10, color=colors[i], edgecolors='black', linewidths=0.5)

    ax_box.set_ylabel('Optimality Error', fontsize=12)
    ax_box.set_title('Boxplot with Jitter of Optimality Errors (FullNet Only)', fontsize=14)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()
    opt_save_path = os.path.join(error_dist_dir, 'optimality_error_distribution.png')
    plt.savefig(opt_save_path, dpi=300, bbox_inches='tight')
    print(f"最优性误差分布图已保存至: {opt_save_path}")
    plt.close()

    # ============================================================================
    # 输出统计摘要
    # ============================================================================
    print("\n" + "="*60)
    print("误差统计摘要:")
    print("="*60)

    def print_stats(name, errors):
        print(f"{name}:")
        print(f"  样本数: {len(errors)}")
        print(f"  均值: {errors.mean():.2e}")
        print(f"  标准差: {errors.std():.2e}")
        print(f"  最小值: {errors.min():.2e}")
        print(f"  25%分位数: {np.percentile(errors, 25):.2e}")
        print(f"  中位数: {np.median(errors):.2e}")
        print(f"  75%分位数: {np.percentile(errors, 75):.2e}")
        print(f"  最大值: {errors.max():.2e}")
        print()

    print_stats("BiasNet 可行性误差", feas_errors_biasnet)
    print_stats("BiasNet 最优性误差", opt_errors_biasnet)
    print_stats("FullNet 可行性误差", feas_errors_fullnet)
    print_stats("FullNet 最优性误差", opt_errors_fullnet)

    print("="*60)
    print("误差分布分析完成！")


