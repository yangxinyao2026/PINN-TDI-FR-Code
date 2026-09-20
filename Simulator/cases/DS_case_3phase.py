import numpy as np
import torch

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
from Simulator.Plotter import ShapeDrawer_2D
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pyomo.environ as pyo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from scipy.io import loadmat

def define_nodal_flex_P(ppc, percent_3 = 0.05, percent_2 = 0.45, rate = 0.3):
    # Create list of (node, load) tuples
    node_loads = [(int(bus_data[0]), bus_data[2]) for bus_data in ppc["bus"]]
    # Sort nodes by load in descending order
    sorted_nodes = sorted(node_loads, key=lambda x: x[1], reverse=True)
    max_P=sorted_nodes[0][1]
    # Determine the top 50% nodes with the highest load
    num_nodes = len(sorted_nodes)
    type3_percent_num = round(num_nodes * percent_3)  # Determine the number of high-load nodes to select: Total number of nodes × Percentage (rounded)
    type2_percent_num = round(num_nodes * percent_2)
    type3_nodes = {node for node, _ in sorted_nodes[9:9 + type3_percent_num]}
    type2_nodes = {node for node, _ in sorted_nodes[:9] + sorted_nodes[9 + type3_percent_num:type2_percent_num + type3_percent_num]}
    type0_nodes = {node for node, _ in sorted_nodes[type2_percent_num + type3_percent_num:]}

    # Create node_flex_dict
    node_flex_dict = {}
    for node, _ in node_loads:  # The load value is not needed in this loop.
        if node in type2_nodes:  # For high load nodes (node in top_nodes):
            node_flex_dict[node] = {"type": 2, "rate": (rate,rate)}
        elif node in type3_nodes:
            node_flex_dict[node] = {"type": 3, "rate": rate}  # Let the active power of the medium load node be adjustable, andP^2+Q^2<=set value
        else:  # For ordinary nodes:
            node_flex_dict[node] = {"type": 0, "rate": None}  # ''type'': 0: Indicates that the node does not have flexible load capacity.rate: None: No ability to adjust
    ppc["node_flex_dict"] = node_flex_dict  # Add the created flexible load dictionary to the power system case data.
    ppc["type3_nodes"] = type3_nodes
    ppc["type2_nodes"] = type2_nodes
    ppc["max_P"] = max_P
    ppc["sorted_nodes"] = sorted_nodes
    ppc["type0_nodes"] = type0_nodes



def case36real_3phase_ds(flex_percent_3=0.05, flex_percent_2=0.45, flex_rate=0.3):
    """Load the publication-ready 36-bus three-phase case."""
    data_path = f'{PROJECT_ROOT}/Simulator/data/TD_OPF/ds_data/case36real_3phase_ds.mat'
    mpc = loadmat(data_path)["mpc"][0, 0]

    ppc = {
        "version": "2",
        "casename": "case36real_3phase_ds",
        "baseMVA": float(np.asarray(mpc["baseMVA"]).squeeze()),
        "bus": np.asarray(mpc["bus"], dtype=float).copy(),
        "branch": np.asarray(mpc["branch"], dtype=float).copy(),
        "gen": np.asarray(mpc["gen"], dtype=float).copy(),
        "gencost": np.asarray(mpc["gencost"], dtype=float).copy(),
        "branch_R_mat": np.asarray(mpc["branch_R_mat"], dtype=float).copy(),
        "branch_X_mat": np.asarray(mpc["branch_X_mat"], dtype=float).copy(),
        "bus_P": np.asarray(mpc["bus_P"], dtype=float).copy(),
        "bus_Q": np.asarray(mpc["bus_Q"], dtype=float).copy(),
    }

    define_nodal_flex_P(
        ppc,
        percent_3=flex_percent_3,
        percent_2=flex_percent_2,
        rate=flex_rate,
    )
    return ppc


def DScase_3phase_train(casedata, model_type='pretrainnet', plot_flag=False,
                       total_samples=100, batch_size=5, device='cpu'):
    baseMVA = casedata['baseMVA']
    bus_data = casedata['bus']  # per line: [bus_i, type, Pd, Qd, ...]
    branch_data = casedata['branch']  # per line: [from, to, ..., rateA, angle_min, angle_max]
    # List of precalculated branch impedance matrices
    # case36_3phase Zhongying has provided: branch_R_mat, branch_X_mat
    branch_R_mat = casedata.get('branch_R_mat')
    branch_X_mat = casedata.get('branch_X_mat')
    node_flex_dict = casedata.get('node_flex_dict')
    bus_P = casedata.get('bus_P')
    bus_Q = casedata.get('bus_Q')
    # phase index mapping
    phase_list = ['a', 'b', 'c']
    phase_dict = {'a': 0, 'b': 1, 'c': 2}
    ph_idx = {ph: i for i, ph in enumerate(phase_list)}

    # Number of nodes and branches
    bus_ids = [int(row[0]) for row in bus_data]
    line_ids = list(range(len(branch_data)))
    dim_n = len(bus_ids)

    # Pyomo model
    model = pyo.ConcreteModel()
    model.BUS = pyo.Set(initialize=bus_ids)
    model.LINE = pyo.Set(initialize=line_ids)
    model.PH = pyo.Set(initialize=phase_list)

    # Branch start and end index mapping
    model.from_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 0]) for l in model.LINE}, within=model.BUS, mutable=False)
    model.to_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 1]) for l in model.LINE}, within=model.BUS, mutable=False)

    # Branch impedance parameters: R[(l,phi,psi)], X[(l,phi,psi)]
    def R_init(model, l, ph, ps):
        return branch_R_mat[l][ph_idx[ph], ph_idx[ps]]

    def X_init(model, l, ph, ps):
        return branch_X_mat[l][ph_idx[ph], ph_idx[ps]]

    model.R = pyo.Param(model.LINE, model.PH, model.PH, initialize=R_init, mutable=False)
    model.X = pyo.Param(model.LINE, model.PH, model.PH, initialize=X_init, mutable=False)

    # Normalized load parameters
    Pd_i = bus_P / baseMVA  # shape: (n_bus, 3)
    Qd_i = bus_Q / baseMVA

    Pd_min = 0.8 * Pd_i
    Pd_max = 1.2 * Pd_i
    Qd_min = 0.8 * Qd_i
    Qd_max = 1.2 * Qd_i

    def _normalize_2d(x, xmin, xmax):
        result = []
        for i in range(len(x)):
            row = []
            for j in range(x.shape[1]):
                if xmax[i, j] == xmin[i, j]:
                    row.append(0.5)
                else:
                    row.append((x[i, j] - xmin[i, j]) / (xmax[i, j] - xmin[i, j]))
            result.append(row)
        return result

    Pd_meta_init = _normalize_2d(Pd_i, Pd_min, Pd_max)
    Qd_meta_init = _normalize_2d(Qd_i, Qd_min, Qd_max)

    # Pyomo Perturbable parameters (normalized values)
    def Pd_meta_init_bus_ph(model, i, ph):
        idx = bus_ids.index(i)
        return Pd_meta_init[idx][ph_idx[ph]]

    def Qd_meta_init_bus_ph(model, i, ph):
        idx = bus_ids.index(i)
        return Qd_meta_init[idx][ph_idx[ph]]

    model.Pd_meta = pyo.Param(model.BUS, model.PH, initialize=Pd_meta_init_bus_ph, mutable=True, domain=pyo.Reals)
    model.Qd_meta = pyo.Param(model.BUS, model.PH, initialize=Qd_meta_init_bus_ph, mutable=True, domain=pyo.Reals)

    # denormalization expression: Pd = Pd_meta * (Pd_max - Pd_min) + Pd_min
    def Pd_denormalize_expr(model, i, ph):
        idx = bus_ids.index(i)
        p = ph_idx[ph]
        return model.Pd_meta[i, ph] * (Pd_max[idx, p] - Pd_min[idx, p]) + Pd_min[idx, p]

    def Qd_denormalize_expr(model, i, ph):
        idx = bus_ids.index(i)
        p = ph_idx[ph]
        return model.Qd_meta[i, ph] * (Qd_max[idx, p] - Qd_min[idx, p]) + Qd_min[idx, p]

    model.Pd = pyo.Expression(model.BUS, model.PH, rule=Pd_denormalize_expr)
    model.Qd = pyo.Expression(model.BUS, model.PH, rule=Qd_denormalize_expr)

    model.V_root = pyo.Param(initialize=1.0, mutable=False)
    # variable definition
    model.V2 = pyo.Var(model.BUS, model.PH, within=pyo.NonNegativeReals)  # Node voltage squared
    model.Pf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # Branch active power flow
    model.Qf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # Branch reactive power flow
    model.I2 = pyo.Var(model.LINE, model.PH, within=pyo.NonNegativeReals)  # branch current square

    # model.P_total = pyo.Var(model.PH, within=pyo.Reals)  # Root node active power flow
    # model.Q_total = pyo.Var(model.PH, within=pyo.Reals)  # Root node reactive power flow
    model.Pn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # Node active power injection
    model.Qn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # Node reactive power injection

    model.var_proj = pyo.Var(range(2), within=pyo.Reals)  # aggregate power variable

    Vmin = 0.90 ** 2
    Vmax = 1.10 ** 2

    # constraint
    model.constraints = pyo.ConstraintList()

    # voltage boundary constraints
    for i in model.BUS:
        for ph in model.PH:
            model.constraints.add(expr=model.V2[i, ph] >= Vmin)
            model.constraints.add(expr=model.V2[i, ph] <= Vmax)

    # Voltage drop equation (simplified form)
    for l in model.LINE:
        i = model.from_bus[l]
        j = model.to_bus[l]
        for ph in model.PH:
            # 2*sum_phi' (R_ij^ph,ph' * P_ij^ph' + X_ij^ph,ph' * Q_ij^ph') + sum_phi' |Z_ij^ph,ph'|^2 * I2
            lin_loss = sum(2 * (model.R[l, ph, ps] * model.Pf[l, ps] + model.X[l, ph, ps] * model.Qf[l, ps])
                           for ps in model.PH)
            quad_loss = sum((model.R[l, ph, ps] ** 2 + model.X[l, ph, ps] ** 2) * model.I2[l, ps] for ps in model.PH)
            model.constraints.add(
                expr=model.V2[j, ph] == model.V2[i, ph] - lin_loss + quad_loss
                # expr = (model.V2[j, ph] == model.V2[i, ph] - lin_loss)
            )

    # current-The exact relationship between power
    for l in model.LINE:
        i = model.from_bus[l]
        for ph in model.PH:
            # There are numerical problems in converting equations into inequalities
            # model.constraints.add(
            #     expr=model.I2[l, ph] * model.V2[i, ph] >= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            # model.constraints.add(
            #     expr=model.I2[l, ph] * model.V2[i, ph] <= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            model.constraints.add(
                expr=model.I2[l, ph] * model.V2[i, ph] == model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            )

    # Power balance constraints
    for n in model.BUS:
        if not n == 1:
            for ph in model.PH:
                inflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_P = sum(sum(model.R[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr=(inflow_P - loss_P - outflow_P + model.Pn[n, ph] == 0.0))

                inflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_Q = sum(sum(model.X[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr=(inflow_Q - loss_Q - outflow_Q + model.Qn[n, ph] == 0.0))

                node_flex_info = node_flex_dict.get(n, 0)
                if not node_flex_info['type']:
                    model.constraints.add(expr=(model.Pn[n, ph] == -model.Pd[n, ph]))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 1:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 2:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] <= -model.Qd[n, ph] + node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] >= -model.Qd[n, ph] - node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
                elif node_flex_info['type'] == 3:
                    model.constraints.add((model.Pn[n, ph]+model.Pd[n, ph])**2 + (model.Qn[n, ph]+model.Qd[n, ph])**2 <= 0.3*(model.Pd[n, ph]**2+model.Qd[n, ph]**2))




    # Balanced node (bus) power and voltage settings, assuming bus 1 for reference
    for ph in model.PH:
        outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        model.constraints.add(expr=(model.Pn[1, ph] - outflow_P == model.Pd[1, ph]))
        model.constraints.add(expr=(model.Qn[1, ph] - outflow_Q == model.Qd[1, ph]))
        model.constraints.add(expr=model.V2[bus_ids[0], ph] == model.V_root ** 2)

    model.constraints.add(expr=(model.var_proj[0] == sum(model.Pn[1, ph] for ph in model.PH)))
    model.constraints.add(expr=(model.var_proj[1] == sum(model.Qn[1, ph] for ph in model.PH)))

    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            Pd_meta_tensor = torch.tensor(Pd_meta_init, device=device, dtype=torch.float32)  # (n_bus, 3)
            Qd_meta_tensor = torch.tensor(Qd_meta_init, device=device, dtype=torch.float32)
            return {
                'Pd_meta': torch.rand(dim_n, 3, device=device) - Pd_meta_tensor,  # normalized value ∈ [-0.5,0.5]
                'Qd_meta': torch.rand(dim_n, 3, device=device) - Qd_meta_tensor,
            }

    dim = 2
    num = 36
    theta_num = np.linspace(0, 2 * np.pi, num, endpoint=False)
    A_hat = np.column_stack((np.cos(theta_num), np.sin(theta_num)))
    errorcalculator = ErrorCalculator(
        original_model={'model': model},
        A_hat=A_hat,
        solver='ipopt',
    )

    case_name = casedata['casename']
    figure_folder = f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{case_name}\\A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)\\figures'
    os.makedirs(figure_folder, exist_ok=True)


    n_train = 501

    if plot_flag:
        plt.figure(figsize=(8, 6))
        total_P = sum(sum(bus_P))
        total_Q = sum(sum(bus_Q))
        xlim = np.array([total_P - 0.5*abs(total_P),total_P + 0.5*abs(total_P)])/ baseMVA
        ylim = np.array([total_Q - 0.5*abs(total_Q),total_Q + 0.5*abs(total_Q)])/ baseMVA
        plotter = ShapeDrawer_2D()
        plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                             facecolor='green', xlim=xlim, ylim=ylim,
                             label=f'Approximation',
                             title=f'Training step = {0}'
                             )

        os.makedirs(figure_folder + f'/pretrain_process', exist_ok=True)
        plotter.save(figure_folder + f'/pretrain_process/step0{0}.png')
    def training_callback(errorcalculator, epoch):
        len_his = len(errorcalculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(errorcalculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(errorcalculator.training_history['opt'][-min(10, len_his):]):.2e}")
        # print(errorcalculator.b_hat)
        if model_type.lower() == 'pretrainnet' and plot_flag:
            plotter.remove_shape(plotter.shapes[-1]['id'])
            plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                 facecolor='green', xlim=xlim, ylim=ylim,
                                 label=f'Approximation',
                                 title=f'Training step = {epoch}'
                                 )
            plotter.save(figure_folder + f'/pretrain_process/step{epoch}.png')

    # Training parameter configuration
    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 5,
            "training_callback": training_callback,
            "optimizer": 'sgd',
            "lr": 2e-1,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            'feas_tol': 1e-10,
            'opt_tol': 1e-10,
            "rate_opt_feas": 0.6
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": training_callback,
            "optimizer": "adam",
            # "optimizer": "sgd",
            "lr": 4e-4,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            'feas_tol': 1e-10,
            'opt_tol': 1e-10,
            "rate_opt_feas": 0.6,
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
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'params': params,
        'result_path': f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{case_name}\\A(36,2)_type3(8, 11)_lr1(3e-4)_lr2(1e-4)_rate(1e-4)\\{model_type}_weights.pth',
        'n_train': n_train,
        'metadata': {
            'dscasedata': casedata,
        }
    }

def DScase_3phase(model, casedata):
    baseMVA = casedata['baseMVA']
    bus_data = casedata['bus']  # per line: [bus_i, type, Pd, Qd, ...]
    branch_data = casedata['branch']  # per line: [from, to, ..., rateA, angle_min, angle_max]
    # List of precalculated branch impedance matrices
    # case36_3phase Zhongying has provided: branch_R_mat, branch_X_mat
    branch_R_mat = casedata.get('branch_R_mat')
    branch_X_mat = casedata.get('branch_X_mat')
    node_flex_dict = casedata.get('node_flex_dict')
    bus_P = casedata.get('bus_P')
    bus_Q = casedata.get('bus_Q')
    # phase index mapping
    phase_list = ['a', 'b', 'c']
    phase_dict = {'a':0, 'b':1, 'c':2}
    ph_idx = {ph: i for i, ph in enumerate(phase_list)}

    # Number of nodes and branches
    bus_ids = [int(row[0]) for row in bus_data]
    line_ids = list(range(len(branch_data)))

    # Pyomo model
    # model = pyo.ConcreteModel() #The model is input
    model.BUS = pyo.Set(initialize=bus_ids)
    model.LINE = pyo.Set(initialize=line_ids)
    model.PH = pyo.Set(initialize=phase_list)

    # Branch start and end index mapping
    model.from_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 0]) for l in model.LINE}, within=model.BUS)
    model.to_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 1]) for l in model.LINE}, within=model.BUS)


    # Branch impedance parameters: R[(l,phi,psi)], X[(l,phi,psi)]
    def R_init(model, l, ph, ps):
        return branch_R_mat[l][ph_idx[ph], ph_idx[ps]]

    def X_init(model, l, ph, ps):
        return branch_X_mat[l][ph_idx[ph], ph_idx[ps]]


    model.R = pyo.Param(model.LINE, model.PH, model.PH, initialize=R_init, mutable=False)
    model.X = pyo.Param(model.LINE, model.PH, model.PH, initialize=X_init, mutable=False)



    # Load injection (positive injection into the network, Pd,Qd Take the negative of the load)
    def Pd_init(model, i, ph):
        idx = bus_ids.index(i)
        # print(bus_P[idx, phase_dict[ph]])
        return bus_P[idx, ph_idx[ph]] / baseMVA


    def Qd_init(model, i, ph):
        idx = bus_ids.index(i)
        return bus_Q[idx, ph_idx[ph]] / baseMVA


    model.Pd = pyo.Param(model.BUS, model.PH, initialize=Pd_init, mutable=False)
    model.Qd = pyo.Param(model.BUS, model.PH, initialize=Qd_init, mutable=False)

    # variable definition
    model.V2 = pyo.Var(model.BUS, model.PH, within=pyo.NonNegativeReals)  # Node voltage squared
    model.Pf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # Branch active power flow
    model.Qf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # Branch reactive power flow
    model.I2 = pyo.Var(model.LINE, model.PH, within=pyo.NonNegativeReals)  # branch current square

    # model.P_total = pyo.Var(model.PH, within=pyo.Reals)  # Root node active power flow
    # model.Q_total = pyo.Var(model.PH, within=pyo.Reals)  # Root node reactive power flow
    model.Pn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # Node active power injection
    model.Qn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # Node reactive power injection

    model.var_proj = pyo.Var(range(2),within = pyo.Reals)  # aggregate power variable

    # Voltage upper and lower limits (pu^2)
    Vmin = 0.90 ** 2
    Vmax = 1.10 ** 2

    # constraint
    model.constraints = pyo.ConstraintList()

    # voltage boundary constraints
    for i in model.BUS:
        for ph in model.PH:
            model.constraints.add(expr = model.V2[i, ph] >= Vmin)
            model.constraints.add(expr = model.V2[i, ph] <= Vmax)

    # Voltage drop equation (simplified form)
    for l in model.LINE:
        i = model.from_bus[l]
        j = model.to_bus[l]
        for ph in model.PH:
            # 2*sum_phi' (R_ij^ph,ph' * P_ij^ph' + X_ij^ph,ph' * Q_ij^ph') + sum_phi' |Z_ij^ph,ph'|^2 * I2
            lin_loss = sum(2 * (model.R[l, ph, ps] * model.Pf[l, ps] + model.X[l, ph, ps] * model.Qf[l, ps])
                           for ps in model.PH)
            quad_loss = sum((model.R[l, ph, ps] ** 2 + model.X[l, ph, ps] ** 2) * model.I2[l, ps] for ps in model.PH)
            model.constraints.add(
                expr = model.V2[j, ph] == model.V2[i, ph] - lin_loss + quad_loss
                # expr = (model.V2[j, ph] == model.V2[i, ph] - lin_loss)
            )

    # current-The exact relationship between power
    for l in model.LINE:
        i = model.from_bus[l]
        for ph in model.PH:
            # model.constraints.add(
            #     expr = model.I2[l, ph] * model.V2[i, ph] >= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            # model.constraints.add(
            #     expr = model.I2[l, ph] * model.V2[i, ph] <= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            model.constraints.add(
                expr = model.I2[l, ph] * model.V2[i, ph] == model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            )
    # Power balance constraints
    for n in model.BUS:
        if not n == 1:
            for ph in model.PH:
                inflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_P = sum(sum(model.R[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr=(inflow_P - loss_P - outflow_P + model.Pn[n,ph] == 0.0))

                inflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_Q = sum(sum(model.X[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr = (inflow_Q - loss_Q - outflow_Q + model.Qn[n,ph] == 0.0))

                node_flex_info = node_flex_dict.get(n, 0)
                if not node_flex_info['type']:
                    model.constraints.add(expr=(model.Pn[n, ph] == -model.Pd[n, ph]))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 1:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 2:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] <= -model.Qd[n, ph] + node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] >= -model.Qd[n, ph] - node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
                elif node_flex_info['type'] == 3:
                    model.constraints.add((model.Pn[n, ph]+model.Pd[n, ph])**2 + (model.Qn[n, ph]+model.Qd[n, ph])**2 <= 0.3*(model.Pd[n, ph]**2+model.Qd[n, ph]**2))
    # Balanced node (bus) power and voltage settings, assuming bus 1 for reference
    for ph in model.PH:
        outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        model.constraints.add(expr=( model.Pn[1, ph] - outflow_P == model.Pd[1, ph]))
        model.constraints.add(expr=( model.Qn[1, ph] - outflow_Q == model.Qd[1, ph]))

    model.constraints.add(expr=(model.var_proj[0] == sum(model.Pn[1, ph] for ph in model.PH)))
    model.constraints.add(expr=(model.var_proj[1] == sum(model.Qn[1, ph] for ph in model.PH)))

def disagg_DS_3phase(P_target, Q_target, v_target, dscasedata):
    model = pyo.ConcreteModel()
    DScase_3phase(model, dscasedata)
    for ph in model.PH:
        model.constraints.add(v_target ** 2 == model.V2[dscasedata['bus'][0,0], ph]) # 0 Node is the root node
    model.obj = pyo.Objective(
        expr=(P_target-model.var_proj[0]) ** 2 + (Q_target-model.var_proj[1]) ** 2,
        sense=pyo.minimize
    )
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model, tee=True)

    return model.obj()

# Usage example
if __name__ == "__main__":
    #
    # # Optimal power flow model of three-phase unbalanced distribution network (nonlinear Branch Flow)
    # # Use Pyomo Modeling + Ipopt solver
    #
    # # Load pypower format data
    ppc = case36real_3phase_ds()
    model = pyo.ConcreteModel()
    DScase_3phase_train(casedata=ppc, )




    # Power flow calculation testcase
    from pypower.api import ppoption, runpf

    # Let’s start testing the one-way power flow calculation
    ppopt = ppoption()  # Use default options
    ppopt['VERBOSE'] = 2 # Control output verbosity, 0 means outputting less information
    ppopt['OUT_ALL'] = 0   # Does not output any results (except error messages)
    # Or you can set it to output more information, for example:
    # ppopt = ppoption(VERBOSE=2, OUT_ALL=1)  # Output detailed results

    # Run a power flow calculation
    results, success = runpf(ppc, ppopt)

    # Check for convergence
    if success:
        print("Power flow calculation converges!")
        # Output the voltage amplitude and phase angle (angle)
        print("\nNode voltage results: ")
        print("nodeID  voltage amplitude(pu)  Voltage phase angle(degree)")
        for i in range(len(results["bus"])):
            bus_id = int(results["bus"][i][0])
            voltage_mag = results["bus"][i][7]#*results["bus"][i][9]*1e3
            voltage_angle = results["bus"][i][8]  # angle, not rad
            print(f"{bus_id:4d}    {voltage_mag:.6f}      {voltage_angle:.6f}")
        # print(ppc['loadvolt'])
    else:
        print("Power flow calculation has not converged!")
    print(ppc["type3_nodes"])
    print(ppc["type2_nodes"])
    print(ppc["type0_nodes"])
    print(ppc["node_flex_dict"])
    print(ppc["max_P"])
    print(ppc["sorted_nodes"])
    # print(ppc["branch"])
