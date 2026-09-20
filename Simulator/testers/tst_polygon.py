import os
import sys

# Add the project root directory to Python path
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
# randomly generated dtheta, Test feasibility and optimality errors and draw distribution diagrams
# ============================================================================

# Check if error calculator exists
if 'errorcalculator' not in case:
    print("Warning: Error calculator not found, error analysis cannot be performed")
else:
    error_calculator = case['errorcalculator']

    # Configuring the threshold parameters of the error calculator
    error_calculator.configure(feas_tol=1e-8, opt_tol=1e-8)

    # Configuration parameters
    n_dtheta = 100  # random dtheta quantity
    n_samples_per_dtheta = 50  # each dtheta Error calculation sample number
    dtheta_range = (-0.5, 0.5)  # dtheta The sampling range

    # Generate random dtheta value
    np.random.seed(42)  # Fixed random seed for easy reproducibility
    dtheta_values = np.random.uniform(dtheta_range[0], dtheta_range[1], n_dtheta)

    # Initialize a list to store errors
    feas_errors_biasnet = []
    opt_errors_biasnet = []
    feas_errors_fullnet = []
    opt_errors_fullnet = []

    print(f"Start testing {n_dtheta} random dtheta value...")

    for idx, dtheta in enumerate(dtheta_values):
        if (idx + 1) % 10 == 0:
            print(f"  processing section {idx + 1}/{n_dtheta} a dtheta = {dtheta:.3f}")

        # Update the original model theta parameters (Adjustable parameters)
        # in polygon in case, theta Parameter correspondence b_init The last two elements of
        # basic_cases.py defines the two-dimensional parameter as 'theta'.
        theta_new = np.array([0.7 + dtheta, 0.7 + dtheta])  # The original value is 0.7, 0.7
        error_calculator.update_parameters({'theta': theta_new})

        # Calculate biasnet Forecast
        b_pred_bias = biasnet(torch.tensor([dtheta, dtheta], dtype=torch.float32).to(device))
        b_pred_bias = b_pred_bias.detach().cpu().numpy()

        # Calculate fullnet Forecast
        A_pred_full, b_pred_full = fullnet(torch.tensor([dtheta, dtheta], dtype=torch.float32).to(device))
        A_pred_full = A_pred_full[0].detach().cpu().numpy()
        b_pred_full = b_pred_full[0].detach().cpu().numpy()

        # Calculate biasnet Error
        error_calculator.update_polytope(A_hat=A_pretrained, b_hat=b_pred_bias)
        feas_results, opt_results = error_calculator.calculate(
            n_cal=n_samples_per_dtheta, cal_feas=True, cal_opt=True)

        # Extract error value
        feas_errors_biasnet.extend([r['error'] for r in feas_results])
        opt_errors_biasnet.extend([r['error'] for r in opt_results])

        # Calculate fullnet Error
        error_calculator.update_polytope(A_hat=A_pred_full, b_hat=b_pred_full)
        feas_results, opt_results = error_calculator.calculate(
            n_cal=n_samples_per_dtheta, cal_feas=True, cal_opt=True)

        # Extract error value
        feas_errors_fullnet.extend([r['error'] for r in feas_results])
        opt_errors_fullnet.extend([r['error'] for r in opt_results])

    print("The error calculation is completed and the distribution chart begins to be drawn....")

    # Convert to numpy array
    feas_errors_biasnet = np.array(feas_errors_biasnet)
    opt_errors_biasnet = np.array(opt_errors_biasnet)
    feas_errors_fullnet = np.array(feas_errors_fullnet)
    opt_errors_fullnet = np.array(opt_errors_fullnet)

    # Create output directory
    error_dist_dir = os.path.join(os.path.dirname(figure_folder_name), 'error_distributions')
    os.makedirs(error_dist_dir, exist_ok=True)

    # Import the libraries required for drawing
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # ============================================================================
    # Draw feasibility error distribution diagram (KDE and boxplot)
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left side: KDE Density plot
    ax_kde = axes[0]

    # Calculate KDE (Calculate only FullNet)
    kde_fullnet = gaussian_kde(feas_errors_fullnet)

    # create x Axis range (Based on FullNet)
    x_min = feas_errors_fullnet.min()
    x_max = feas_errors_fullnet.max()
    x_range = np.linspace(x_min, x_max, 1000)

    # draw KDE (draw only FullNet)
    ax_kde.plot(x_range, kde_fullnet(x_range), label='FullNet', color='red', linewidth=2)
    ax_kde.fill_between(x_range, kde_fullnet(x_range), alpha=0.3, color='red')

    ax_kde.set_xlabel('Feasibility Error', fontsize=12)
    ax_kde.set_ylabel('Density', fontsize=12)
    ax_kde.set_title('KDE Distribution of Feasibility Errors (FullNet Only)', fontsize=14)
    ax_kde.legend()
    ax_kde.grid(True, alpha=0.3)

    # Right: Boxplot with jitter
    ax_box = axes[1]

    # Prepare data (Contains only FullNet)
    data = [feas_errors_fullnet]
    labels = ['FullNet']

    # Draw a boxplot
    bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True)

    # Set boxplot color
    colors = ['lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add jitter scatter
    for i, d in enumerate(data):
        # Add random jitter
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax_box.scatter(x, d, alpha=0.5, s=10, color=colors[i], edgecolors='black', linewidths=0.5)

    ax_box.set_ylabel('Feasibility Error', fontsize=12)
    ax_box.set_title('Boxplot with Jitter of Feasibility Errors (FullNet Only)', fontsize=14)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()
    feas_save_path = os.path.join(error_dist_dir, 'feasibility_error_distribution.png')
    plt.savefig(feas_save_path, dpi=300, bbox_inches='tight')
    print(f"The feasibility error distribution plot has been saved to: {feas_save_path}")
    plt.close()

    # ============================================================================
    # Plot optimality error distribution graph (KDE and boxplot)
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left side: KDE Density plot
    ax_kde = axes[0]

    # Calculate KDE (Calculate only FullNet)
    kde_fullnet = gaussian_kde(opt_errors_fullnet)

    # create x Axis range (Based on FullNet)
    x_min = opt_errors_fullnet.min()
    x_max = opt_errors_fullnet.max()
    x_range = np.linspace(x_min, x_max, 1000)

    # draw KDE (draw only FullNet)
    ax_kde.plot(x_range, kde_fullnet(x_range), label='FullNet', color='red', linewidth=2)
    ax_kde.fill_between(x_range, kde_fullnet(x_range), alpha=0.3, color='red')

    ax_kde.set_xlabel('Optimality Error', fontsize=12)
    ax_kde.set_ylabel('Density', fontsize=12)
    ax_kde.set_title('KDE Distribution of Optimality Errors (FullNet Only)', fontsize=14)
    ax_kde.legend()
    ax_kde.grid(True, alpha=0.3)

    # Right: Boxplot with jitter
    ax_box = axes[1]

    # Prepare data (Contains only FullNet)
    data = [opt_errors_fullnet]
    labels = ['FullNet']

    # Draw a boxplot
    bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True)

    # Set boxplot color
    colors = ['lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add jitter scatter
    for i, d in enumerate(data):
        # Add random jitter
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax_box.scatter(x, d, alpha=0.5, s=10, color=colors[i], edgecolors='black', linewidths=0.5)

    ax_box.set_ylabel('Optimality Error', fontsize=12)
    ax_box.set_title('Boxplot with Jitter of Optimality Errors (FullNet Only)', fontsize=14)
    ax_box.grid(True, alpha=0.3)

    plt.tight_layout()
    opt_save_path = os.path.join(error_dist_dir, 'optimality_error_distribution.png')
    plt.savefig(opt_save_path, dpi=300, bbox_inches='tight')
    print(f"The optimality error distribution plot has been saved to: {opt_save_path}")
    plt.close()

    # ============================================================================
    # Output statistical summary
    # ============================================================================
    print("\n" + "="*60)
    print("Summary of error statistics:")
    print("="*60)

    def print_stats(name, errors):
        print(f"{name}:")
        print(f"  Number of samples: {len(errors)}")
        print(f"  mean: {errors.mean():.2e}")
        print(f"  standard deviation: {errors.std():.2e}")
        print(f"  minimum value: {errors.min():.2e}")
        print(f"  25%Quantile: {np.percentile(errors, 25):.2e}")
        print(f"  median: {np.median(errors):.2e}")
        print(f"  75%Quantile: {np.percentile(errors, 75):.2e}")
        print(f"  maximum value: {errors.max():.2e}")
        print()

    print_stats("BiasNet feasibility error", feas_errors_biasnet)
    print_stats("BiasNet optimality error", opt_errors_biasnet)
    print_stats("FullNet feasibility error", feas_errors_fullnet)
    print_stats("FullNet optimality error", opt_errors_fullnet)

    print("="*60)
    print("Error distribution analysis completed!")

