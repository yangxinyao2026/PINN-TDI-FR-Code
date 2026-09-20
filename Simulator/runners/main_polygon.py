import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.basic_cases import case_polygon
from Simulator import PROJECT_ROOT
from Simulator.Plotter import ErrorVisualizer  # Import error visualizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel = True
model_type = 'fullnet'  #model_typeset to'pretrainnet', Indicates using a pre-trained network.
case = case_polygon(model_type = model_type,device=device)

# Record the error distribution through the training callback.
visualizer = ErrorVisualizer()
original_callback = case['trainer_configure']['training_callback']

def enhanced_callback(error_calculator, epoch):
    # Call the original callback function
    original_callback(error_calculator, epoch)
    # Record error distribution (reduce the number of samples to improve performance)
    visualizer.compute_errors(error_calculator, num_sample=10)

# Replace the callback function in the training configuration
case['trainer_configure']['training_callback'] = enhanced_callback
if model_type=='pretrainnet':
    n_train = 1000  #Set the number of training times to1000times
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
else:
    n_train = 6  #Set the number of training times to20times (fine-tuning stage)
    model = PreTrainNet(case['A_hat'], case['b_hat'])  #create emptyPreTrainNetmodel
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth', map_location=device))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

trainer.configure(**case['trainer_configure'])
# trainer.configure(lr = 0.9)  #Manual learning rate configuration
trainer.initialize()
trainer.train(n_train = n_train , params_data=case["params"], parallel = parallel)
torch.save(model.state_dict(), case['result_path'])

# Draw and save error boxplots (interval sampling)
import os
result_dir = os.path.dirname(case['result_path'])
os.makedirs(result_dir, exist_ok=True)
boxplot_path = os.path.join(result_dir, 'error_boxplot_interval.png')

interval = 100  # Default plotting interval.
#print(f"Error history data not found, using default interval: {interval}")

visualizer.plot_dual_boxplot_interval(save_path=boxplot_path, interval=interval)
print(f"Error boxplot (interval display) saved to: {boxplot_path}")

# Draw and save error comparison distribution graph (KDEand boxplots with jitter)
comparison_path = os.path.join(result_dir, 'error_comparison_distributions.png')
# Error distribution analysis using the error calculator
if 'errorcalculator' in case:
    error_calculator = case['errorcalculator']
    visualizer.plot_comparison_distributions(model=error_calculator, num_sample=1000, save_path=comparison_path)
    print(f"The error comparison distribution chart has been saved to: {comparison_path}")
else:
    print("Warning: Error calculator not found, unable to plot error distribution")
