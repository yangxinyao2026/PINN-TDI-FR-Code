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

def case_polygon(total_samples=200, noise_scale=0.15, batch_size=1,
                 model_type='pretrainnet', device='cpu',
                 save_training_frames=False):
    """Original case implementation of fixed parameters"""
    # Fixed initialization parameters
    A_init = np.array([
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, -1], [1, -1], [-1, 1]
    ])
    b_init = np.array([1, 1, 0, 0, 1.5, -0.5, 0.7, 0.7])  # Define the original extent of the octagon

    # Automatically derive dimensions
    dim = A_init.shape[1]  #Get the number of columns of an array
    ncons = A_init.shape[0]  #Get the number of rows in an array
    dim_theta = 2  # Number of adjustable input parameters.
    num_b = ncons - dim_theta  #Meaning: fixed parameters b quantity. Explanation: at8Among the constraints, there are2One is adjustable, the rest6one is fixed

    # Build model
    model = ConcreteModel()

    # Parameter definition
    model.A = Param(range(ncons), range(dim),
                    initialize={(i, j): A_init[i, j] for i, j in np.ndindex(A_init.shape)},
                    mutable=True)  # Initialize every entry of A as a mutable parameter.
    model.b = Param(range(num_b),
                    initialize={i: b_init[i] for i in range(num_b)},
                    mutable=True)  # Fixed physical and safety constraints.
    model.theta = Param(range(dim_theta),
                        initialize={i: b_init[num_b + i] for i in range(2)},
                        mutable=True)  # The final two offsets depend on theta.

    # variable definition
    model.var_proj = Var(range(dim), domain=Reals)

    # Constraint definition
    def constraint_rule(model, i, is_adjustable):
        idx = i + num_b if is_adjustable else i    #if is_adjustable=True: idx = i + num_b = i + 6 (adjustable constraints); if is_adjustable=False: idx = i (fixed constraints)
        param = model.theta[i] if is_adjustable else model.b[i]
        return sum(model.A[idx, j] * model.var_proj[j] for j in range(dim)) <= param

    model.con_fixed = Constraint(range(num_b), rule=lambda m, i: constraint_rule(m, i, False))
    model.con_adj = Constraint(range(dim_theta), rule=lambda m, i: constraint_rule(m, i, True))

    original_model = {
        'model': model,
        # 'baseline': baseline,
    }

    # Dataset configuration
    class CaseData(Dataset):
        def __init__(self, size=total_samples):  #Initialization method __init__:
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):  #Returns the size of the dataset200
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # thetaThe dimensions are fixed to2

    A_hat = np.vstack([  #Stack two matrices vertically together
        np.eye(dim),  # upper bound  #Identity matrix
        -np.eye(dim),  # Nether  #negative identity matrix
    ])  #as an initial approximation
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',     #willcplexThe solver is changed togurobi
    )
    A_list = [errorcalculator.A_hat]  # Store the initial constraint matrix
    b_list = [errorcalculator.b_hat]  # Store the initial constraint right-hand term

    visualizer = ErrorVisualizer()
    num_sample = 50  # Number of samples used for error evaluation.
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)  # Calculation error

    case_name = 'polygon'  #Define case name
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'  # Define the folder path where images are stored.
    os.makedirs(figure_folder, exist_ok=True)  #Create the above folder. If the folder already exists, no error will be thrown (becauseexist_ok=True) .
    plt.figure(figsize=(8, 6))
    plotter = ShapeDrawer_2D()
    def callback(errorcalculator, epoch):  # Record polygon evolution at selected epochs.
        len_his = len(errorcalculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(errorcalculator.training_history['feas'][-min(10,len_his):]):.2e}, "
              f"OptErr={np.mean(errorcalculator.training_history['opt'][-min(10,len_his):]):.2e}")
        if model_type.lower() == 'pretrainnet':  #.lower() Make sure it is not case sensitive
            xlim = [-0.5, 1.5]
            ylim = [-0.5, 1.5]  # Axis range settings. Fixed display range to ensure consistent viewing angle for all graphics
            if not epoch:  # Equivalent to if epoch == 0:
                plotter.plot_polygon(A_init, b_init,  #Draw original polygon
                                     facecolor='blue', xlim=xlim, ylim=ylim,  #Blue: original polygon
                                     label=f'Original Region',  #label: "Original Region"
                                     title=f'Training step = {0}',  #Title: Display number of training steps
                                     )
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,  #Draw initial approximation
                                     facecolor='green', xlim=xlim, ylim=ylim,  #Green: Approximate polygon
                                     label=f'Approximation',  #label: "Approximation"
                                     title=f'Training step = {epoch}'  #In the same coordinate system as the original polygon
                                     )
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])  # Remove the previous approximation.
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                     facecolor='green', xlim=xlim, ylim=ylim,
                                     label=f'Approximation',
                                     title=f'Training step = {epoch}'
                                     )
            # Save pictures (automatically create directory)
            if save_training_frames:
                plotter.save(figure_folder + f'/pretrain_process/step{epoch}')
            if (epoch + 20) % 100 == 0:
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)  # Store the current learned polygon.
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)
            if epoch>=980:
                with open(f'{PROJECT_ROOT}/results/{case_name}/A_list.pkl', "wb") as f:  #A_list.pkl: Save the data collected throughout the training processA_hatlist.
                    pickle.dump(A_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/b_list.pkl', "wb") as f:  #b_list.pkl: Save the data collected throughout the training processb_hatlist.
                    pickle.dump(b_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/error_history.pkl', "wb") as f:  #error_history.pkl: Save history of errors calculated by the visualizer.
                    pickle.dump(visualizer.error_history, f)

    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
                'call_interval':20,  # Invoke the callback every 20 iterations.
                'training_callback':callback,  #callback function
                "optimizer": "SGD",  #Type of optimizer.stochastic gradient descentStochastic Gradient Descent
                "lr": 0.5,
                "batch_size": 1,  #Batch size. During training θ Sample batch size.
                "scheduler": {  #Configuration of the learning rate scheduler, including type, step size, and decay factor.
                    "type": "StepLR",  #Scheduler type, here are allStepLR, That is, adjust the learning rate according to the step size.
                    "step_size": 100,  #Adjust the step size every few steps.
                    "gamma": 0.98  #Adjustment multiple, here eachstep_sizestep multiplies the learning rate by0.98.
                },
                "n_cal": 2,  # Sampled directions per iteration.
                "cal_feas": True,  #Whether to calculate feasibility error.
                "cal_opt": True,  #Whether to calculate optimality error.
                "rate_opt_feas": 1  # Relative weight of optimality and feasibility errors.
            }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",  #adaptive moment estimation (Adaptive Moment Estimation)
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
    params_dict, param_count = pyomo_params_to_numpy(model)  #will Pyomo The parameters in the model are converted to NumPy format to facilitate neural network processing.
    params = { #Name, initial value, error data set
        'params_dict':params_dict,  #A dictionary of the names of all parameters and their initial values
        'dataloader': DataLoader(  #Dataset loader for parameter sampling
            CaseData(),  #CaseData(): The dataset class defined earlier (generates noisytheta)
            batch_size=batch_size,  #Set according to model type
            shuffle=True  #Disorganize the order of data and improve training effect
        ),
        'count':dim_theta,  #Total number of tunable parameters
    }
    return {
        'casename':'polygon',
        'A_hat': A_hat,  #initial matrix A
        'b_hat': errorcalculator.b_hat,  #initial vector b
        'errorcalculator': errorcalculator,
        'params':params,
        'trainer_configure': trainer_configure,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
        'metadata':{'A_init':A_init,'b_init':b_init}
    }

def case_ellipse(total_samples=100, noise_scale=0.2, batch_size=2, model_type='pretrainnet',device = 'cpu'):
    """Elliptical constraint case implementation"""
    # Fixed initialization parameters (quadratic matrix)
    Sigma_init = np.array([
        [5 / 2, -3 / 2],       # [ [a, b],
        [-3 / 2, 5 / 2]        #   [b, a] ]
    ])  #ellipse[x1, x2] * Sigma * [x1; x2]
    dim = 2  # Fixed as a two-dimensional problem
    a_init = 5 / 2
    b_init = -3 / 2

    # buildPyomomodel
    model = ConcreteModel()

    # Adjustable parameter definition (Sigmaupper triangular element of matrix)
    model.a = Param(initialize=a_init, mutable=True)
    model.b = Param(initialize=b_init, mutable=True)

    # decision variables
    model.var_proj = Var(range(dim), domain=Reals)

    # Ellipse constraint (Quadratic type)
    model.constraints = Constraint(
        expr=(model.a * model.var_proj[0] ** 2
        + 2 * model.b * model.var_proj[0] * model.var_proj[1]
        + model.a * model.var_proj[1] ** 2) <= 1
    )  #Define elliptical constraints: a*x₁² + 2b*x₁x₂ + a*x₂² ≤ 1

    original_model = {'model': model}

    # Dataset configuration
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"a":torch.normal(0, self.noise_scale, (1,),device=device),
                    "b":torch.normal(0, self.noise_scale, (1,),device=device),
            }  # thetaThe dimensions are fixed to2
    # Approximator parameters
    A_hat = np.vstack([
        np.eye(dim),  # upper bound
        -np.eye(dim),  # Nether
        np.array([[1, 1], [-1, -1]])  # diagonal constraints
    ])  #hexagon
    # Error calculator
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi'   #willcplexThe solver is changed togurobi
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    visualizer = ErrorVisualizer()
    num_sample = 50
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)

    case_name = 'ellipse'
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
    os.makedirs(figure_folder, exist_ok=True)

    # Visual callback function
    plt.figure(figsize=(8, 6))
    plotter = ShapeDrawer_2D()
    xlim = [-2, 2]
    ylim = [-2, 2]

    # marking_epoches = [1,6,12,25,50,100,195]
    marking_epoches = list(range(1,195,20))+[195]  #set a markepochList of: marked points include from1Arrive195 (Does not contain195) every20aepochTake one and add195thisepoch.total 11a
    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # Initialize counter
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "  #Moving average calculation
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
            if callback.idx_mark>=len(marking_epoches):  # when callback.idx_mark = 11 (List length exceeded10) triggered when
                with open(f'{PROJECT_ROOT}/results/{case_name}/A_list.pkl', "wb") as f:
                    pickle.dump(A_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/b_list.pkl', "wb") as f:
                    pickle.dump(b_list, f)
                with open(f'{PROJECT_ROOT}/results/{case_name}/error_history.pkl', "wb") as f:
                    pickle.dump(visualizer.error_history, f)
            elif epoch >= marking_epoches[callback.idx_mark]:  # Arrive at the mark point and record data
                callback.idx_mark+=1
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)  ## Calculate and record errors




    # Training parameter configuration
    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 5,  #callback function frequency
            "training_callback": callback,
            "optimizer": "SGD",  #stochastic gradient descent
            "lr": 0.2,
            "batch_size": 1,  #Variable parameters
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},  # !The learning rate gradually decays
            "n_cal": 2,  #Number of explorations
            "cal_feas": True,
            "cal_opt": True,  # Evaluate projection optimality during training.
            "rate_opt_feas": 1  #Optimal and feasible error weight ratio
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            # "optimizer": "sgd",
            "lr": 0.0003,   #Due to poor training results, adjust the learning rate
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }


    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #Name, initial value, error data set
        'params_dict':params_dict,  # parameter dictionary, containing fromPyomoParameters converted from the model
        'dataloader': DataLoader(  #is aPyTorchofDataLoader, It uses the previously definedCaseDataDatasets, setting batch sizes and shuffling data.
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,  #The total number of parameters that need to be learned (here is2a: aandb)
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,   # Initial polygon approximation matrix
        'b_hat': errorcalculator.b_hat,  # Initial polygon approximate offset
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,  # training configuration
        'params':params,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
        'metadata': {   # Case metadata
            'Sigma_init': Sigma_init,  # Initial ellipse matrix (used for reference and visualization)
            'dim': dim    # Problem dimensions (fixed to2)
        }
    }

def case_epigraph(total_samples=100, noise_scale=0.15, batch_size=5, model_type='pretrainnet',device = 'cpu'):
    """EpigraphProblem case implementation"""
    # Initial parameter settings
    theta_init = np.array([1.0, 1.0])  # theta[0]=xupper bound, theta[1]=x²coefficient
    dim = 1  # variable dimension (x, f)
    dim_theta = 2

    # buildPyomomodel
    model = ConcreteModel()

    # Adjustable parameter definition
    model.theta = Param(range(dim_theta),
                        initialize={i: theta_init[i] for i in range(dim_theta)},
                        mutable=True)

    # decision variables

    model.var_proj = Var(range(dim+1), domain=Reals) # var_proj[0]=x, var_proj[1]=f

    model.x = model.var_proj[0]  # beforen-1expression alias for elements
    model.f = model.var_proj[dim]

    model.obj = Expression(expr=model.theta[1] * model.x ** 2)
    # model.f = Var(domain=Reals)

    # Constraint definition
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 0)  # xFixed lower bound
    model.constraints.add(model.x <= model.theta[0])  # The upper bound is given bythetacontrol

    model.constraints.add(model.f >= model.obj)  # lower bound constraint
    # model.constraints.add(model.f <= model.theta[1] * model.theta[0]**2)  # Epigraphupper bound

    original_model = {'model': model}

    # Dataset configuration
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # thetaThe dimensions are fixed to2

    # Approximator parameters (linearized constraint matrix)

    A_hat = np.vstack([
        [1, 0],  # x <= theta0
        [-1, 0],  # x >= 0
        [0, -1],  # f >= theta1*x² (Need follow-up processing)
        [2,-1],
        [1, -1],
    ],dtype=float) #allAThe matrix is written here
    A_hat = np.vstack([A_hat,[0,1.]])  # Append the objective upper-bound row.
    # [0, 1],  # f <= theta0*theta1

    case_name = 'epigraph'
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
    os.makedirs(figure_folder, exist_ok=True)

    # Visualization tools
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

    # Error calculator
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        is_epigraph=True,
        solver='ipopt'
    )

    # Training parameter configuration
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
    params = { #Name, initial value, error data set
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
    """Non-convex optimization problem case implementation"""
    # Initial parameter settings
    theta_init = np.array([1, 1, 1])  # theta[0]: center of circlexoffset,theta[1]: center of circleyoffset, theta[2]: minimum radius
    dim = 2  # variable dimension
    dim_theta = len(theta_init)

    # buildPyomomodel
    model = ConcreteModel()

    # Define variable parameters
    model.theta = Param(range(dim_theta),
                        initialize={i: theta_init[i] for i in range(dim_theta)},
                        mutable=True)
    # Define variables and asymmetric boundaries
    def variable_bounds(m, i):
        return (-1, 1)  # var_proj[0] ∈ [-1,2], var_proj[1] ∈ [-2,4]

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # Non-convex constraint definition
    model.constraints = ConstraintList()
    # unit circle constraint (convex)
    model.constraints.add(model.var_proj[0] ** 2 + model.var_proj[1] ** 2 <= 1)
    # Dynamic circular constraints (non-convex)
    model.constraints.add(
        (model.var_proj[0] - model.theta[0]) ** 2 + (model.var_proj[1] - model.theta[1]) ** 2 >= model.theta[2] ** 2
    )

    original_model = {'model': model}

    # Dataset configuration (generatedthetadisturbance)
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # thetaThe dimensions are fixed to2

    # approximator matrix (contains boundary constraints)
    A_hat = np.vstack([
        np.eye(dim),  # upper bound
        -np.eye(dim),  # Nether
        [[1, 1], [-1, -1]]  # diagonal constraints
    ])
    # Error calculator (supports non-convex constraint evaluation)
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='ipopt',  # Use a solver that supports non-convexity
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
    # Visual callback function
    plotter = ShapeDrawer_2D()
    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    marking_epoches = list(range(1,300,30))+[280]

    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # Initialize counter
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
        if model_type.lower() == 'pretrainnet':
            # draw original area
            if epoch == 0:
                # unit circle
                plotter.plot_circle_regions(
                    theta=theta_init,
                    xlim=xlim,  # Contains the visual range of the two circles
                    ylim=ylim,
                    edgecolor='skyblue',
                    facecolor='skyblue',  # area fill color
                    alpha=0.3,  # Transparency
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
        # Training parameter configuration
        trainer_configure = {
            "call_interval": 20,
            "training_callback": callback,
            # "optimizer": "Adam",
            "optimizer": "sgd",
            "lr": 0.2,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.99},
            "n_cal": 2,  # Reduce calibration times and improve stability
            "cal_feas": True,
            "cal_opt": True,  # Non-convex problems do not optimize the target for the time being.
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
    params = { #Name, initial value, error data set
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
    """Non-convex optimization problem case implementation"""
    # Initial parameter settings
    R_init = 1.0  # theta circle radius
    dim_theta = 1

    # buildPyomomodel
    model = ConcreteModel()

    # Define variable parameters
    model.R = Param(initialize= R_init, mutable=True)
    # Define variables and asymmetric boundaries
    def variable_bounds(m, i):
        return (-5, 5)

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # Non-convex constraint definition
    model.constraints = ConstraintList()
    # unit circle constraint (convex)
    model.constraints.add(expr = sum(model.var_proj[i] ** 2 for i in range(dim))<= model.R)

    original_model = {'model': model}

    # Dataset configuration (generatedthetadisturbance)
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # thetaThe dimensions are fixed to2

    # approximator matrix (contains boundary constraints)
    # A_hat = np.vstack([
    #     np.eye(dim),  # upper bound
    #     -np.eye(dim),  # Nether
    # ])
    A_hat = np.vstack([
        np.eye(dim),  # upper bound
        -np.eye(dim),  # Nether
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
    # Error calculator (supports non-convex constraint evaluation)
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',  # Use a solver that supports non-convexity
    )


    if model_type.lower() == 'pretrainnet':
        # Training parameter configuration
        trainer_configure = {
            "call_interval": 5,
            "training_callback": callback,
            # "optimizer": "Adam",
            "optimizer": "sgd",
            "lr": 0.3,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 1.00},
            "n_cal": 2,  # Reduce calibration times and improve stability
            "cal_feas": True,
            "cal_opt": True,  # Non-convex problems do not optimize the target for the time being.
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
    params = { #Name, initial value, error data set
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
    """Non-convex optimization problem case implementation"""
    # Initial parameter settings
    d_init = 1.0

    # buildPyomomodel
    model = ConcreteModel()
    def variable_bounds(m, i):
        return (-1, 1)

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # Non-convex constraint definition
    model.constraints = ConstraintList()

    original_model = {'model': model}

    # approximator matrix (contains boundary constraints)
    A_hat = np.vstack([
        np.eye(dim),  # upper bound
        -np.eye(dim),  # Nether
    ])

    A_hat += np.random.normal(loc=0, scale=0.5/sqrt(dim), size=A_hat.shape)
    # Error calculator (supports non-convex constraint evaluation)
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',  # Use a solver that supports non-convexity
    )

    case_name = 'cube'
    results_folder = f'{PROJECT_ROOT}\\results\\{case_name}'
    os.makedirs(results_folder, exist_ok=True)

    def callback(error_calculator, epoch):
        # initializationend_flag (if it does not exist yet)
        if not hasattr(callback, 'end_flag'):
            callback.end_flag = False
            callback.start_flag = True

        # Open file for append writing
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
                # writeepoch
                f.write(f"Converged at epoch: {epoch}\n")


    trainer_configure = {
        "call_interval": 10,
        "training_callback": callback,
        # "optimizer": "Adam",
        "optimizer": "sgd",
        "lr": 0.3,
        "batch_size": 1,
        "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 1.05},
        "n_cal": 3,  # Reduce calibration times and improve stability
        "cal_feas": True,
        "cal_opt": True,  # Non-convex problems do not optimize the target for the time being.
        "rate_opt_feas": 1
    }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #Name, initial value, error data set
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
