import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Permit duplicate OpenMP runtimes when required by the local stack.
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
from Simulator import  PROJECT_ROOT
import torch
from Simulator.cases.basic_cases import case_ellipse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #Set up the device, if anyGPUthen useGPU, Otherwise useCPU.
parallel= False  #Set the parallel flag toFalse, Indicates that parallel training is not used.
model_type = 'fullnet'  # Options: pretrainnet, biasnet, or fullnet.
case = case_ellipse(model_type = model_type,device = device)

if model_type=='pretrainnet':
    n_train = 500
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
else:
    n_train = 15
    model = PreTrainNet(case['A_hat'], case['b_hat'])  #createPreTrainNetInstance (for loading pre-trained weights)
    model.load_state_dict(torch.load(f"{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth",map_location=device))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,device = device).to(device)
    elif model_type == 'fullnet':  # Initialize the full model from pretrained A and b.
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)
trainer.configure(**case['trainer_configure'])
trainer.initialize()
trainer.train(n_train = n_train , params_data = case['params'], parallel= parallel)
torch.save(model.state_dict(), case['result_path'])


'''
Summary
This code implements a progressive neural network training process:
   Pre-training phase: a large number of iterations to learn basic linear relationships
   Fine-tuning phase: fine-tuning specific components in small iterations
   BiasNet: Fixed linear transformation, only adjusts the offset
   FullNet: Comprehensive adjustment of all parameters
This design is often used for physical simulation or numerical approximation problems, where pre-training provides a good starting point and the fine-tuning phase is optimized for specific tasks..
'''
