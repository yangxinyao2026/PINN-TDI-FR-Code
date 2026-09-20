import os
import numpy as np
from Simulator.cases import TD_case
import Simulator.cases.DS_case_3phase as DS_case_3phase
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Permit duplicate OpenMP runtimes when required by the local stack.
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
from Simulator.Plotter import ErrorVisualizer  # Import error visualizer
import time  # Import time module for measuring training time
import torch  #importPyTorchDeep learning library.
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def compute_errors_random_dtheta_safe(error_calculator, nn_model, ppc, device, init_params_dict, n_dtheta=5, n_samples_per_dtheta=10, zero_dtheta=False):
    """
        randomly generated dtheta, Calculate feasibility and optimality errors
        parameters:
            error_calculator: ErrorCalculator Example
            nn_model: Currently trained neural network model (FullNet, BiasNet, PreTrainNet)
            ppc: Power system case data
            device: computing equipment (cpu/cuda)
            init_params_dict: parameter dictionary, containing 'Pd_meta' and 'Qd_meta' initial value
            n_dtheta: randomly generated dtheta Quantity
            n_samples_per_dtheta: each dtheta Error calculation sample number
            zero_dtheta: IfTrue, dthetafixed to0 (used forpretrainnet)
        """
    """
    Safe version: Useerror_calculatorComputes a copy of the object without modifying the original object
    Support single phase (1Dparameters) and three-phase (2Dparameters)
    """
    # createerror_calculatora copy to avoid modifying the original object
    error_calc_copy = error_calculator.copy()

    # frominit_params_dictGet the initial value and automatically adapt1D/2D (single phase/Three phases)
    Pd_init = init_params_dict['Pd_meta']['initial_value']
    Qd_init = init_params_dict['Qd_meta']['initial_value']
    total_params = Pd_init.size + Qd_init.size

    # Configuration parameters
    dtheta_range = (-0.5, 0.5)
    np.random.seed(42)  # Fixed seed, but only affects error calculation, not training

    # generatedthetavalue
    if zero_dtheta:
        dtheta_values = [np.zeros(total_params)]
    else:
        dtheta_values = [np.random.uniform(dtheta_range[0], dtheta_range[1], total_params) for _ in range(n_dtheta)]

    # Initialize a list to store errors
    feas_errors = []
    opt_errors = []

    for dtheta in dtheta_values:
        # dthetaSplit intoPdandQdpart, added to the initial value
        Pd_meta_new = Pd_init + dtheta[:Pd_init.size].reshape(Pd_init.shape)
        Qd_meta_new = Qd_init + dtheta[Pd_init.size:].reshape(Qd_init.shape)

        # Passupdate_parametersUpdate, automatically processed1D/2Dindex mapping
        error_calc_copy.update_parameters({'Pd_meta': Pd_meta_new, 'Qd_meta': Qd_meta_new})

        # Prediction using current neural network modelAandb (Do not modify model state)
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
                raise ValueError(f"Unsupported model type: {type(nn_model)}")

            A_pred_np = A_pred[0].detach().cpu().numpy()
            b_pred_np = b_pred[0].detach().cpu().numpy()

        # Update approximate polyhedron on copy
        error_calc_copy.update_polytope(A_hat=A_pred_np, b_hat=b_pred_np)

        # Calculate error on copy
        feas_results, opt_results = error_calc_copy.calculate(
            n_cal=n_samples_per_dtheta, cal_feas=True, cal_opt=True)

        # Extract error value
        feas_errors.extend([r['error'] for r in feas_results])
        opt_errors.extend([r['error'] for r in opt_results])

    # Convert tonumpyarray
    return np.array(feas_errors), np.array(opt_errors)



model_type = 'fullnet'  # Options: pretrainnet, biasnet, or fullnet.
parallel = False
record_errors = True  # Disable to skip error evaluation and output files during training.
dscases = {
     'case10ba_ds': TD_case.case10ba_ds(),
    # 'case17me_ds': TD_case.case17me_ds(),
    # 'case33bw_ds': TD_case.case33bw_ds(),
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
    lr = 1e-4/P_rated  #fullnet
    rate_opt_feas = 0.6

    # Record total start time
    total_start_time = time.time()
    is_3phase = '3phase' in casename
    if is_3phase:
        case = DS_case_3phase.DScase_3phase_train(
            casedata=ppc, model_type=model_type, device=device,
            plot_flag=False)
    else:
        case = TD_case.DScase_train(
            casedata=ppc, model_type=model_type, device=device,
            plot_flag=False)


    # Determine whether to record error data during training according to the switch
    if record_errors:
        visualizer = ErrorVisualizer()

        if model_type == 'pretrainnet':
            # pretrainnet Does not depend on parameters, fixed dtheta=0
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
            # fullnet / biasnet Use random dtheta, use offset Splicing two stages of abscissa
            if 'training_callback' in case['trainer_configure']:
                original_callback = case['trainer_configure']['training_callback']

                def enhanced_callback(error_calculator, epoch):
                    # Call the original callback function
                    original_callback(error_calculator, epoch)
                    # every200Record the error distribution only after iterations to reduce unnecessary calculations
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
        # Load pretrained weights
        model = PreTrainNet(case['A_hat'], case['b_hat'],is_epigraph=False,device=device)
        result_dir = os.path.dirname(case['result_path'])
        model.load_state_dict(torch.load(os.path.join(result_dir, 'pretrainnet_weights.pth'),map_location=device))
        # Extract pre-training parameters
        A_pretrained, b_pretrained = model()
        b_pretrained = b_pretrained[0].detach().cpu().numpy()

        # bias network
        if model_type == 'biasnet':
            A_pretrained = A_pretrained[0].detach().to(device)

            case['trainer_configure'].update(A_pretrained = A_pretrained)  # Add or update this key-value pair
            model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,n_hidden=128,device = device).to(device)

        # Whole network
        elif model_type == 'fullnet':
            A_pretrained = A_pretrained[0].detach().cpu().numpy()
            model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,n_hidden=128,device = device).to(device)

    trainer = Trainer(
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )
    trainer.configure(**case['trainer_configure'])  #Configure the parameters of the trainer, including the learning rate andrate_opt_feas.
    trainer.configure(lr = lr)
    trainer.configure(rate_opt_feas = rate_opt_feas)
    trainer.initialize()

    if model_type == 'pretrainnet':
        # pretrainnet There is only one training phase
        phase1_start = time.time()
        trainer.train(n_train=n_train*4, params_data=case['params'], parallel=parallel)
        phase1_end = time.time()
        torch.save(model.state_dict(), case['result_path'])

        if not record_errors:
            phase1_duration = phase1_end - phase1_start
            total_end_time = time.time()
            total_overall_duration = total_end_time - total_start_time
            print(f"Training time statistics - {casename}:")
            print(f"  training time: {phase1_duration:.2f} seconds ({phase1_duration/60:.2f} minutes)")
            print(f"  Total time taken (end-to-end) : {total_overall_duration:.2f} seconds ({total_overall_duration/60:.2f} minutes)")
            # Save training time
            result_dir = os.path.dirname(case['result_path'])
            os.makedirs(result_dir, exist_ok=True)
            time_data_path = os.path.join(result_dir, f'{model_type}_training_time.npz')
            np.savez(time_data_path,
                     phase1_time=phase1_duration,
                     total_time=total_overall_duration)
            print(f"Training time has been saved to: {time_data_path}")

    else:
        # fullnet / biasnet There are two training phases
        # first stage
        phase1_start = time.time()
        trainer.train(n_train=n_train * 4-1, params_data=case['params'], parallel=parallel)
        phase1_end = time.time()
        torch.save(model.state_dict(), case['result_path'])

        # Stage 2: Minimize Feasibility Error
        if record_errors:
            # update offset, make Phase 2 The abscissa of Phase 1 Continue at the end
            enhanced_callback.offset = (n_train * 4) * len(case['params']['dataloader'])
        trainer.configure(lr=1e-5/P_rated)
        trainer.configure(rate_opt_feas=1e-4)
        trainer.initialize()
        phase2_start = time.time()
        trainer.train(n_train=n_train * 2 , params_data=case['params'], parallel=parallel)
        phase2_end = time.time()
        torch.save(model.state_dict(), f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{casename}\\A(36,2)_type3(1)_lr1(1e-4)_lr2(1e-5)_rate(1e-4)\\{model_type}_weights_feasible.pth')

        if not record_errors:
            phase1_duration = phase1_end - phase1_start
            phase2_duration = phase2_end - phase2_start
            total_duration = phase1_duration + phase2_duration
            total_end_time = time.time()
            total_overall_duration = total_end_time - total_start_time
            print(f"Training time statistics - {casename}:")
            print(f"  first phase training: {phase1_duration:.2f} seconds ({phase1_duration/60:.2f} minutes)")
            print(f"  Second phase training: {phase2_duration:.2f} seconds ({phase2_duration/60:.2f} minutes)")
            print(f"  total training time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            print(f"  Total time taken (end-to-end) : {total_overall_duration:.2f} seconds ({total_overall_duration/60:.2f} minutes)")
            # Save training time
            result_dir = os.path.dirname(case['result_path'])
            os.makedirs(result_dir, exist_ok=True)
            time_data_path = os.path.join(result_dir, f'{model_type}_training_time.npz')
            np.savez(time_data_path,
                     phase1_time=phase1_duration,
                     phase2_time=phase2_duration,
                     total_train_time=total_duration,
                     total_time=total_overall_duration)
            print(f"Training time has been saved to: {time_data_path}")

    # Save the error data of the training process (please run the drawing main_ds_plot.py)
    if record_errors:
        result_dir = os.path.dirname(case['result_path'])
        os.makedirs(result_dir, exist_ok=True)
        error_filename = f'{model_type}_error_data.npz'
        error_data_path = os.path.join(result_dir, error_filename)

        error_history = visualizer.error_history
        original_iterations = error_history['iterations']

        # The number of repair iterations is200multiples of
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
        print(f"Training error data has been saved to: {error_data_path}")
