import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
from Simulator.Plotter import ShapeDrawer_2D
import matplotlib.pyplot as plt
import os
import pyomo.environ as pyo

#from Simulator.testers.tst_TD import tsppc  #注意

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from scipy.io import loadmat
def define_nodal_flex_P(ppc, percent_3 = 0.05, percent_2 = 0.45, rate = 0.3):  #这个函数用于 为电力系统中的节点 定义 柔性负荷能力，即哪些节点可以参与需求响应或负荷调节。ppc: 电力系统案例数据。percent: 柔性负荷节点比例，默认50%。rate: 柔性负荷调节速率，默认30%
    # Create list of (node, load) tuples
    node_loads = [(int(bus_data[0]), bus_data[2]) for bus_data in ppc["bus"]]  #遍历所有母线数据，提取节点编号和负荷值：生成元组列表 (节点编号, 负荷值)；bus_data[0] 是节点编号，bus_data[2] 通常是节点有功负荷
    # Sort nodes by load in descending order
    sorted_nodes = sorted(node_loads, key=lambda x: x[1], reverse=True)  #使用sorted函数按负荷值降序排列：key=lambda x: x[1] 表示按元组的第二个元素（负荷值）排序：lambda: 创建匿名函数的关键字，x: 函数的输入参数，代表node_loads列表中的每个元素，x[1]: 函数的返回值，取每个元素的索引1位置的值。reverse=True 表示从大到小排列
    max_P=sorted_nodes[0][1]
    # Determine the type3 and type2 nodes with the highest load
    num_nodes = len(sorted_nodes)  #计算节点总数
    type3_percent_num = round(num_nodes*percent_3)  #确定要选择的高负荷节点数量：节点总数 × 百分比（四舍五入）
    type2_percent_num = round(num_nodes*percent_2)
    type3_nodes = {node for node, _ in sorted_nodes[9:9+type3_percent_num]}  #创建集合top_nodes，包含前top_percent个高负荷节点的编号
    type2_nodes = {node for node, _ in sorted_nodes[:9]+sorted_nodes[9+type3_percent_num:type2_percent_num+type3_percent_num]}
    type0_nodes = {node for node, _ in sorted_nodes[type2_percent_num + type3_percent_num:]}


    # Create node_flex_dict
    node_flex_dict = {}  #创建一个空字典，用于存储所有节点的柔性负荷配置。
    for node, _ in node_loads:  # 遍历所有节点：node_loads 是之前创建的 (节点编号, 负荷值) 元组列表。node 接收节点编号。_ 是Python惯例，表示忽略负荷值（因为在这个循环中不需要使用）
        if node in type2_nodes:  #对于高负荷节点 (node in top_nodes)：
            node_flex_dict[node] = {"type": 2, "rate": (rate,rate)}  #''type'': 2: 表示该节点具有柔性负荷能力。rate: (rate, rate): 调节速率元组，通常表示：第一个值：向下调节能力（负荷减少）；第二个值：向上调节能力（负荷增加）；(0.3, 0.3) 表示可以向上和向下各调节30%
        elif node in type3_nodes:
            node_flex_dict[node] = {"type": 3, "rate": rate}  #令中负荷节点的有功可调，且P^2+Q^2<=设定值
        else:  #对于普通节点：
            node_flex_dict[node] = {"type": 0, "rate": None}  #''type'': 0: 表示该节点没有柔性负荷能力。rate: None: 没有调节能力
    ppc["node_flex_dict"] = node_flex_dict  #将创建的柔性负荷字典添加到电力系统案例数据中。
    ppc["type3_nodes"]= type3_nodes
    ppc["type2_nodes"]= type2_nodes
    ppc["max_P"] = max_P
    ppc["sorted_nodes"]=sorted_nodes
    ppc["type0_nodes"]= type0_nodes

def case533mt_hi_ds(root_voltage = 1.0, radial = True, flex_percent_3 = 0.05,flex_percent_2 = 0.45, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case533mt_hi_ds'
    # ppc["basekV"] = 12/np.sqrt(3)
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case533mt_hi_ds.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    #ppc["branch"][:, 2:4] /= 2

    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]

    ppc["bus"][:,-2] = 1.1 #1.05
    ppc["bus"][:,-1] = 0.9 #0.95

    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent_3 = flex_percent_3,percent_2 = flex_percent_2, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

def case136ma_ds(root_voltage = 1.0, radial = True, flex_percent = 0.5, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case136ma_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case136ma_ds.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["branch"][:, 2:4] /= 2

    ppc["bus"][:,-2] = 1.15
    ppc["bus"][:,-1] = 0.85
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent = flex_percent, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

def case17me_ds(root_voltage = 1.0, radial = True, flex_percent = 0.5, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case17me_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case17me_ds.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    ppc["branch"][:, 2:4] /= 2
    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["bus"][:,-2] = 1.15
    ppc["bus"][:,-1] = 0.85
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent = flex_percent, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

def case118zh_ds(root_voltage = 1.0, radial = True, flex_percent_3 = 0.05,flex_percent_2 = 0.45, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case118zh_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case118zh_ds.mat')  #系统找不到该文件，故改之，如174行
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    #ppc["branch"][:, 2:4] /= 2
    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["bus"][:,-2] = 1.15  #1.15
    ppc["bus"][:,-1] = 0.85  #0.85
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent_3 = flex_percent_3,percent_2 = flex_percent_2, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

def case74_ds(root_voltage = 1.0, radial = True, flex_percent = 0.5, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case74_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case74_ds.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    ppc["branch"][:, 2:4] /= 2
    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["bus"][:,-2] = 1.15
    ppc["bus"][:,-1] = 0.85
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent = flex_percent, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

def case51ga_ds(root_voltage = 1.0, radial = True, flex_percent = 0.5, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case51ga_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case51ga_ds.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    ppc["branch"][:, 2:4] /= 2
    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["bus"][:,-2] = 1.15
    ppc["bus"][:,-1] = 0.85
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent = flex_percent, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

def case33bw_ds(root_voltage = 1.0, radial = True, flex_percent_3 = 0.05,flex_percent_2 = 0.45, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case33bw_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case33bw_ds.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    #ppc["branch"][:, 2:4] /= 2
    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["bus"][:,-2] = 1.1
    ppc["bus"][:,-1] = 0.9
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent_3 = flex_percent_3, percent_2 = flex_percent_2, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

def case10ba_ds(root_voltage = 1.0, radial = True, flex_percent_3 = 0.1,flex_percent_2 = 0.4, flex_rate = 0.3):  # 根节点电压，默认1.0 p.u.。radial: 是否为辐射状网络，默认True。flex_percent: 柔性负荷百分比，默认50%。flex_rate: 柔性负荷调节速率，默认30%。
    ppc = {"version": '2'}  #创建字典: 初始化一个Python字典来存储所有电力系统参数。"version": '2': 设置数据格式版本号为2。
    ppc['casename'] = 'case10ba_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    #data = loadmat(f'{PROJECT_ROOT}\\data\\TD_OPF\\ds_data\\case10ba_ds.mat')  #系统找不到数据文件 case10ba_ds.mat，因为代码中指定的路径与文件实际位置不匹配,所以进行了修改，将D:\A毕业设计\毕设相关代码\share\share\data\TD_OPF\ds_data\case10ba_ds.mat改为D:\A毕业设计\毕设相关代码\share\share\Simulator\data\TD_OPF\ds_data\case10ba_ds.mat
    data = loadmat(f'{PROJECT_ROOT}\\Simulator\\data\\TD_OPF\\ds_data\\case10ba_ds.mat')  #加载数据文件:从MATLAB文件加载10节点配电网数据。
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]  #设置基准功率值。data["mpc"]["baseMVA"][0, 0]: 从MATLAB的cell数组中提取第一个元素。[0, 0]: 再从NumPy数组中提取标量值
    ppc["branch"] = data["mpc"]["branch"][0, 0]  # 提取支路（线路/变压器）数据
    ppc["branch"][:, 2:4] /= 2  #作用: 将所有支路的电阻和电抗值减半。“：” 所有行（所有支路）。“2:4”: 第3列和第4列（Python从0开始索引）
    if radial:  #条件判断: 如果radial参数为True（默认值）
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial  #ppc["branch"][:,10] 获取所有支路的第11列（索引10）。==1 筛选出该列值为1的支路。这通常表示"是否在辐射状网络中启用"的标志位。效果: 只保留标记为辐射状的支路，形成树状网络结构
    ppc["bus"] = data["mpc"]["bus"][0, 0]  #取数据: 从MATLAB数据中获取母线信息
    ppc["bus"][:,-2] = 1.1  # 设置电压上限
    ppc["bus"][:,-1] = 0.9  # 设置电压下限
    ppc["gen"] = data["mpc"]["gen"][0, 0]  #从MATLAB数据中获取发电机参数，包括：发电功率限制，电压调节能力，成本系数等
    define_nodal_flex_P(ppc,percent_3 = flex_percent_3, percent_2 = flex_percent_2, rate = flex_rate)  #功能: 为每个节点定义可调节的负荷范围
    ppc["root_voltage"] = root_voltage
    return ppc




def case118_ts():
    ppc = {"version": '2'}
    ppc['casename'] = 'case118_ts'
    data = loadmat(f'{PROJECT_ROOT}\\data\\TD_OPF\\case118_ts.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:]
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    # ppc["bus"][:,-2] = 1.2
    # ppc["bus"][:,-1] = 0.8
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    ppc["gen"][:,3:5] *=2

    ppc["gencost"] = data["mpc"]["gencost"][0, 0]
    # define_nodal_flex_P(ppc)
    return ppc

def case300_ts(): #潮流一直不可行
    ppc = {"version": '2'}
    ppc['casename'] = 'case300_ts'
    data = loadmat(f'{PROJECT_ROOT}\\data\\TD_OPF\\case300_ts.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    # ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:]
    ppc["branch"][:,2:4]/=5
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["bus"][:,-2] = 1.1
    ppc["bus"][:,-1] = 0.9
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    Qgap = (ppc["gen"][:,3]-ppc["gen"][:,4])
    ppc["gen"][:,3] +=Qgap*2
    ppc["gen"][:, 4] -= Qgap * 2
    ppc["gen"][:, 8] *= 1.5

    ppc["gencost"] = data["mpc"]["gencost"][0, 0]
    # define_nodal_flex_P(ppc)
    return ppc

def case4gs_ts():
    ppc = {}
    ppc['casename'] = 'case4gs_ts'
    ppc["version"] = '2'
    ppc["baseMVA"] = 100
    ppc["bus"] = np.array([
        [1, 3, 50, 30.99, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [2, 1, 170, 105.35, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [3, 1, 200, 123.94, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [4, 2, 80, 49.58, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]
    ])
    ppc["gen"] = np.array([
        [4, 418, 0, 500, -500, 1.02, 100, 1, 718, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 500, 0, 500, -500, 1, 100, 1, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    ppc["branch"] = np.array([
        [1, 2, 0.01008, 0.0504, 0.1025, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 3, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 4, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 4, 0.01272, 0.0636, 0.1275, 250, 250, 250, 0, 0, 1, -360, 360]
    ])
    # gencost: model a quadratic cost function for each generator: a*P^2 + b*P + c
    # [model, startup, shutdown, n, c(n-1), ..., c0]
    ppc["gencost"] = np.array([
        [2, 0, 0, 3, 0.01, 40, 0],   # Gen at bus 4: 0.01*P^2 + 40*P
        [2, 0, 0, 3, 0.02, 20, 0]    # Gen at bus 1: 0.02*P^2 + 20*P
    ])
    return ppc

def TDcase(tscasedata, dscasedata_dict, is_apx = False, is_3phase = False):  # 使用单相算例
    model = TScase(tscasedata, dscasedata_dict = dscasedata_dict)
    baseMVA = tscasedata['baseMVA']
    bus = tscasedata['bus']
    Pd_value = {int(i): bus[i, 2] / baseMVA for i in model.N}
    Qd_value = {int(i): bus[i, 3] / baseMVA for i in model.N}

    model.DS = pyo.Block(dscasedata_dict.keys())

    for tsnode, dscasedata in dscasedata_dict.items():
        block = model.DS[tsnode]
        if 'vertex' in dscasedata:
            # 使用基于顶点的凸包约束
            DScase_vtx(block, dscasedata)
        elif (not is_apx) and is_3phase:
            import Simulator.cases.DS_case_3phase as DS_case_3phase
            DS_case_3phase.DScase_3phase(block, dscasedata)
            block.constraints.add(model.Pd[tsnode] == Pd_value[tsnode] + block.var_proj[0] * dscasedata['baseMVA'] / baseMVA)
            block.constraints.add(model.Qd[tsnode] == Qd_value[tsnode] + block.var_proj[1] * dscasedata['baseMVA'] / baseMVA)
            for ph in block.PH:
                block.constraints.add(model.V[tsnode]**2 == block.V2[dscasedata['bus'][0, 0], ph])  # 0 号节点为根节点
            continue
        elif (not is_apx) and (not is_3phase):
            # 调用原配电网模型
            DScase(block, dscasedata)
        else:
            DScase_apx(block, dscasedata)

        # 将变量和约束注册进 Block
        # for v in submodel.component_objects(ctype=pyo.Var, descend_into=True):
        #     setattr(block, v.name, v)
        # for c in submodel.constraints:
        #     block.add_component(c.local_name, c)
        # block = submodel
        # 与输电网母线耦合：第0配电网挂接在母线1、第1个配电网挂接在母线2（注意编号从0）
        # substation_bus = tsnode  # 假设第i个配网接在母线 i
        # def substation_active_power_rule(m):
        #     return m.Pd[substation_bus] == Pd_value[substation_bus] + block.Pn[1]
        block.constraints.add(model.Pd[tsnode] == Pd_value[tsnode] + block.Pn[1]*dscasedata['baseMVA']/baseMVA)
        # def substation_reactive_power_rule(m):
        #     return m.Qd[substation_bus] == Qd_value[substation_bus] + block.Qn[1]
        block.constraints.add(model.Qd[tsnode] == Qd_value[tsnode] + block.Qn[1]*dscasedata['baseMVA']/baseMVA)
        # def substation_voltage_rule(m):
        #     return m.V[substation_bus]**2 == block.V2[1]

        block.constraints.add(model.V[tsnode]**2 == block.V2[1])
    return model

# def TDcase(tscasedata, dscasedata_dict, is_apx = False, is_3phase = False):  # 使用单相算例
#     model = TScase(tscasedata, dscasedata_dict = dscasedata_dict)
#     baseMVA = tscasedata['baseMVA']
#     bus = tscasedata['bus']
#     Pd_value = {int(i): bus[i, 2] / baseMVA for i in model.N}
#     Qd_value = {int(i): bus[i, 3] / baseMVA for i in model.N}
#
#     model.DS = pyo.Block(dscasedata_dict.keys())
#
#     for tsnode, dscasedata in dscasedata_dict.items():
#         block = model.DS[tsnode]
#
#         if (not is_apx) and is_3phase:
#             import Simulator.cases.DS_case_3phase as DS_case_3phase
#             DS_case_3phase.DScase_3phase(block, dscasedata)
#             block.constraints.add(model.Pd[tsnode] == Pd_value[tsnode] + block.var_proj[0] * dscasedata['baseMVA'] / baseMVA)
#             block.constraints.add(model.Qd[tsnode] == Qd_value[tsnode] + block.var_proj[1] * dscasedata['baseMVA'] / baseMVA)
#             for ph in block.PH:
#                 block.constraints.add(model.V[tsnode]**2 == block.V2[dscasedata['bus'][0, 0], ph])  # 0 号节点为根节点
#
#             continue
#         if (not is_apx) and (not is_3phase):
#             # 调用原配电网模型
#             DScase(block, dscasedata)
#         else:
#             DScase_apx(block, dscasedata)
#
#         # 将变量和约束注册进 Block
#         # for v in submodel.component_objects(ctype=pyo.Var, descend_into=True):
#         #     setattr(block, v.name, v)
#         # for c in submodel.constraints:
#         #     block.add_component(c.local_name, c)
#         # block = submodel
#         # 与输电网母线耦合：第0配电网挂接在母线1、第1个配电网挂接在母线2（注意编号从0）
#         # substation_bus = tsnode  # 假设第i个配网接在母线 i
#         # def substation_active_power_rule(m):
#         #     return m.Pd[substation_bus] == Pd_value[substation_bus] + block.Pn[1]
#         block.constraints.add(model.Pd[tsnode] == Pd_value[tsnode] + block.Pn[1]*dscasedata['baseMVA']/baseMVA)
#         # def substation_reactive_power_rule(m):
#         #     return m.Qd[substation_bus] == Qd_value[substation_bus] + block.Qn[1]
#         block.constraints.add(model.Qd[tsnode] == Qd_value[tsnode] + block.Qn[1]*dscasedata['baseMVA']/baseMVA)
#         # def substation_voltage_rule(m):
#         #     return m.V[substation_bus]**2 == block.V2[1]
#
#         block.constraints.add(model.V[tsnode]**2 == block.V2[1])
#     return model

def TScase(tscasedata ,dscasedata_dict ={}, is_base = False):
    baseMVA = tscasedata['baseMVA']
    bus = tscasedata['bus']
    gen = tscasedata['gen']
    branch = tscasedata['branch']
    gencost = tscasedata['gencost']

    bus_number2idx = dict(zip(bus[:, 0], range(len(bus))))
    bus_idx2number = bus[:, 0]
    nb = bus.shape[0]  # number of buses
    nl = branch.shape[0]  # number of lines
    ng = gen.shape[0]  # number of generators

    model = pyo.ConcreteModel()

    model.N = pyo.RangeSet(0, nb - 1)
    model.L = pyo.RangeSet(0, nl - 1)
    model.G = pyo.RangeSet(0, ng - 1)

    # Parameters
    Pd_value = {int(i): bus[i, 2] / baseMVA for i in model.N}
    Qd_value = {int(i): bus[i, 3] / baseMVA for i in model.N}

    bus_type = {int(i): int(bus[i, 1]) for i in model.N}  # 1 = PQ, 2 = PV, 3 = Slack

    # Generator location mapping: gen_id -> bus_id
    gen_bus_map = {int(i): int(gen[i, 0]) - 1 for i in model.G}

    # Bounds
    Vmax = {int(i): bus[i, 11] for i in model.N}
    Vmin = {int(i): bus[i, 12] for i in model.N}

    Vgen = {int(i): gen[i, 5] for i in model.G}
    Pg_value = {int(i_gen): gen[i_gen, 1] / baseMVA for i_gen in model.G}

    Pmax = {int(i): gen[i, 8] / baseMVA for i in model.G}
    Pmin = {int(i): gen[i, 9] / baseMVA for i in model.G}
    Qmax = {int(i): gen[i, 3] / baseMVA for i in model.G}
    Qmin = {int(i): gen[i, 4] / baseMVA for i in model.G}

    # Cost coefficients
    cost_a = {int(i): gencost[i, 4] * baseMVA ** 2 for i in model.G}  # quadratic
    cost_b = {int(i): gencost[i, 5] * baseMVA for i in model.G}  # linear
    cost_c = {int(i): gencost[i, 6] for i in model.G}  # constant

    # Admittance matrix elements
    from scipy.sparse import lil_matrix
    Ybus = lil_matrix((nb, nb), dtype=complex)
    for k in range(nl):
        i_number, j_number = int(branch[k, 0]), int(branch[k, 1])
        i, j = bus_number2idx[i_number], bus_number2idx[j_number]
        r, x, b = branch[k, 2], branch[k, 3], branch[k, 4]
        z = complex(r, x)
        y = 1 / z
        Ybus[i, j] -= y
        Ybus[j, i] -= y
        Ybus[i, i] += y + complex(0, b / 2)
        Ybus[j, j] += y + complex(0, b / 2)

    G = Ybus.real.toarray()
    B = Ybus.imag.toarray()

    # Variables
    model.V = pyo.Var(model.N, within=pyo.NonNegativeReals, bounds=lambda m, i: (Vmin[i], Vmax[i]), initialize=1.0)
    model.theta = pyo.Var(model.N, bounds=(-2 * np.pi, 2 * np.pi), initialize=0.0)

    model.Pg = pyo.Var(model.G, bounds=lambda m, i: (Pmin[i], Pmax[i]),
                       initialize=lambda m, i: (Pmin[i] + Pmax[i]) / 2)  # , initialize=lambda m, i:gen[i,1]/baseMVA
    model.Qg = pyo.Var(model.G, bounds=lambda m, i: (Qmin[i], Qmax[i]),
                       initialize=lambda m, i: (Qmin[i] + Qmax[i]) / 2 if Qmax[i] < 1e8 else 0.0)

    model.Pd = pyo.Var(model.N, initialize=lambda m, i: Pd_value[i])
    model.Qd = pyo.Var(model.N, initialize=lambda m, i: Qd_value[i])

    # Reference bus (Slack): fix angle and voltage
    for i in model.N:
        if is_base or (not i in dscasedata_dict.keys()):
            model.Pd[i].fix(Pd_value[i])
            model.Qd[i].fix(Qd_value[i])
        if bus_type[i] == 3:
            i_gen = np.where(gen[:, 0] == bus_idx2number[i])[0][0]
            # model.V[i].fix(Vgen[i_gen])
            model.V[i].fix(1.0)
            model.theta[i].fix(0.0)
        # elif bus_type[i] == 2: #最优潮流，PV节点不需要定义P和V
        #     i_gen = np.where(gen[:,0]==bus_idx2number[i])[0][0]
        #     model.V[i].fix(Vgen[i_gen])
        # model.Pg[i_gen].fix(Pg_value[i_gen])
    # Objective: minimize generation cost
    model.obj = pyo.Objective(
        expr=sum(cost_a[g] * model.Pg[g] ** 2 + cost_b[g] * model.Pg[g] + cost_c[g] for g in model.G),
        sense=pyo.minimize
    )

    # Power balance constraints
    def real_power_balance_rule(m, i):
        return sum(
            m.V[i] * m.V[j] * (G[i, j] * pyo.cos(m.theta[i] - m.theta[j]) + B[i, j] * pyo.sin(m.theta[i] - m.theta[j]))
            for j in model.N
        ) + m.Pd[i] == sum(m.Pg[g] for g in model.G if gen_bus_map[g] == i)

    def reactive_power_balance_rule(m, i):
        return sum(
            m.V[i] * m.V[j] * (G[i, j] * pyo.sin(m.theta[i] - m.theta[j]) - B[i, j] * pyo.cos(m.theta[i] - m.theta[j]))
            for j in model.N
        ) + m.Qd[i] == sum(m.Qg[g] for g in model.G if gen_bus_map[g] == i)

    model.real_power_balance = pyo.Constraint(model.N, rule=real_power_balance_rule)
    model.reactive_power_balance = pyo.Constraint(model.N, rule=reactive_power_balance_rule)
    return model


def DScase(model, dscasedata):  # 使用单相算例，如case33
    baseMVA = dscasedata['baseMVA']
    bus_data = dscasedata['bus']
    branch_data = dscasedata['branch']
    # 单相阻抗改为标量值列表
    branch_R = branch_data[:,2]
    branch_X = branch_data[:,3]

    node_flex_dict = dscasedata.get('node_flex_dict')
    bus_P = bus_data[:,2]
    bus_Q = bus_data[:,3]

    # 节点和支路
    bus_ids = [int(row[0]) for row in bus_data]
    line_ids = list(range(len(branch_data)))

    model.BUS = pyo.Set(initialize=bus_ids)
    model.LINE = pyo.Set(initialize=line_ids)

    # 支路起止映射
    model.from_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 0]) for l in model.LINE}, within=model.BUS)
    model.to_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 1]) for l in model.LINE}, within=model.BUS)

    # 阻抗参数改为标量 (单相无相间耦合)
    model.R = pyo.Param(model.LINE, initialize=lambda model, l: branch_R[l], mutable=False)
    model.X = pyo.Param(model.LINE, initialize=lambda model, l: branch_X[l], mutable=False)

    # 负荷注入 (单相)
    def Pd_init(model, i):
        idx = bus_ids.index(i)
        return bus_P[idx] / baseMVA  # 单相负荷

    def Qd_init(model, i):
        idx = bus_ids.index(i)
        return bus_Q[idx] / baseMVA  # 单相负荷

    model.Pd = pyo.Param(model.BUS, initialize=Pd_init, mutable=False)
    model.Qd = pyo.Param(model.BUS, initialize=Qd_init, mutable=False)

    # 变量定义
    model.V2 = pyo.Var(model.BUS, within=pyo.NonNegativeReals)  # 移除了相维度
    model.Pf = pyo.Var(model.LINE, within=pyo.Reals)  # 单相潮流
    model.Qf = pyo.Var(model.LINE, within=pyo.Reals)
    model.I2 = pyo.Var(model.LINE, within=pyo.NonNegativeReals)  # 单相电流
    model.Pn = pyo.Var(model.BUS, within=pyo.Reals)  # 节点注入功率
    model.Qn = pyo.Var(model.BUS, within=pyo.Reals)
    model.var_proj = pyo.Var(range(2), within=pyo.Reals)


    model.constraints = pyo.ConstraintList()

    V2max = {i: bus_data[bus_ids.index(i), 11]**2 for i in model.BUS}
    V2min = {i: bus_data[bus_ids.index(i), 12]**2 for i in model.BUS}
    # 电压边界
    for i in model.BUS:
        model.constraints.add(model.V2[i] >= V2min[i])
        model.constraints.add(model.V2[i] <= V2max[i])

    for l in model.LINE:
        i = model.from_bus[l]
        j = model.to_bus[l]
        # 单相版本去除了相循环和相间耦合项
        lin_loss = 2 * (model.R[l] * model.Pf[l] + model.X[l] * model.Qf[l])
        quad_loss = (model.R[l] ** 2 + model.X[l] ** 2) * model.I2[l]
        model.constraints.add(model.V2[j] == model.V2[i] - lin_loss + quad_loss)

    for l in model.LINE:
        i = model.from_bus[l]
        # model.constraints.add(model.I2[l] * model.V2[i] >= model.Pf[l] ** 2 + model.Qf[l] ** 2)
        # model.constraints.add(model.I2[l] * model.V2[i] <= model.Pf[l] ** 2 + model.Qf[l] ** 2)
        model.constraints.add(model.I2[l] * model.V2[i] == model.Pf[l] ** 2 + model.Qf[l] ** 2)

    for n in model.BUS:
        if n != 1:  # 非根节点
            inflow_P = sum(model.Pf[l] for l in model.LINE if model.to_bus[l] == n)
            loss_P = sum(model.R[l] * model.I2[l] for l in model.LINE if model.to_bus[l] == n)
            outflow_P = sum(model.Pf[l] for l in model.LINE if model.from_bus[l] == n)
            model.constraints.add(inflow_P - loss_P - outflow_P + model.Pn[n] == 0.0)

            inflow_Q = sum(model.Qf[l] for l in model.LINE if model.to_bus[l] == n)
            loss_Q = sum(model.X[l] * model.I2[l] for l in model.LINE if model.to_bus[l] == n)
            outflow_Q = sum(model.Qf[l] for l in model.LINE if model.from_bus[l] == n)
            model.constraints.add(inflow_Q - loss_Q - outflow_Q + model.Qn[n] == 0.0)

            # 灵活负荷处理
            node_flex_info = node_flex_dict.get(n, 0)
            if not node_flex_info['type']:
                model.constraints.add(model.Pn[n] == -model.Pd[n])
                model.constraints.add(model.Qn[n] == -model.Qd[n])
            elif node_flex_info['type'] == 1:
                model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'] * abs(model.Pd[n]))
                model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'] * abs(model.Pd[n]))
                model.constraints.add(model.Qn[n] == -model.Qd[n])
            elif node_flex_info['type'] == 2:
                model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'][0] * abs(model.Pd[n]))
                model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'][0] * abs(model.Pd[n]))
                model.constraints.add(model.Qn[n] <= -model.Qd[n] + node_flex_info['rate'][1] * abs(model.Qd[n]))
                model.constraints.add(model.Qn[n] >= -model.Qd[n] - node_flex_info['rate'][1] * abs(model.Qd[n]))

    # 平衡节点约束
    outflow_P = sum(model.Pf[l] for l in model.LINE if model.from_bus[l] == 1)
    outflow_Q = sum(model.Qf[l] for l in model.LINE if model.from_bus[l] == 1)
    model.constraints.add(model.Pn[1] - outflow_P == model.Pd[1])
    model.constraints.add(model.Qn[1] - outflow_Q == model.Qd[1])

    # model.constraints.add(model.var_proj[0] == model.Pn[1])
    # model.constraints.add(model.var_proj[1] == model.Qn[1])
    # return model


def generate_vertices(dscasedata,  #改成加扰动
                       A_directions =  np.vstack([np.eye(2),
                                                -np.eye(2),
                                                [1,1],
                                                [-1,-1],
                                                [1, -1],
                                                [-1, 1],
                                                ]),
                       solver_name='ipopt'):
    model = pyo.ConcreteModel()
    DScase(model, dscasedata)
    n_directions = A_directions.shape[0]
    vertices = np.zeros((n_directions, 2))
    feasible_flags = np.zeros(n_directions, dtype=bool)

    # 初始化求解器
    solver = pyo.SolverFactory(solver_name)

    # 在模型中添加方向参数（可变）
    model.direction_P = pyo.Param(initialize=0.0, mutable=True)
    model.direction_Q = pyo.Param(initialize=0.0, mutable=True)

    # 定义目标函数：最大化 A_i * [P1, Q1]（只定义一次）
    def objective_rule(model):
        return model.direction_P * model.Pn[1] + model.direction_Q * model.Qn[1]

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    for i in range(n_directions):
        # 更新方向参数值
        model.direction_P.set_value(A_directions[i, 0])
        model.direction_Q.set_value(A_directions[i, 1])

        try:
            # 求解优化问题
            results = solver.solve(model, tee=False)

            # 检查求解状态
            if (results.solver.status == pyo.SolverStatus.ok and
                    results.solver.termination_condition == pyo.TerminationCondition.optimal):

                # 提取最优解
                P1_opt = pyo.value(model.Pn[1])
                Q1_opt = pyo.value(model.Qn[1])

                vertices[i, 0] = P1_opt
                vertices[i, 1] = Q1_opt
                feasible_flags[i] = True

            else:
                print(f"  方向 {i + 1} 求解失败: {results.solver.termination_condition}")
                feasible_flags[i] = False

        except Exception as e:
            print(f"  方向 {i + 1} 求解出错: {str(e)}")
            feasible_flags[i] = False

    # 过滤出可行的顶点
    feasible_vertices = vertices[feasible_flags]

    return feasible_vertices, feasible_flags

def DScase_vtx(model, dscasedata_vtx):  # 使用单相算例，基于顶点凸包
    vertices, _   = dscasedata_vtx['vertex']  # n*2维数组，存储P和Q的n个顶点
    n_vertices= vertices.shape[0]

    # 定义节点集合和变量
    model.BUS = pyo.Set(initialize=range(1, 2))
    model.Pn = pyo.Var(model.BUS, within=pyo.Reals)  # 节点注入功率
    model.Qn = pyo.Var(model.BUS, within=pyo.Reals)
    model.V2 = pyo.Var(model.BUS, within=pyo.Reals)

    # 定义凸组合权重变量
    model.VERTICES = pyo.Set(initialize=range(n_vertices))
    model.lambda_weights = pyo.Var(model.VERTICES, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.constraints = pyo.ConstraintList()

    # 凸组合约束：权重之和等于1
    model.constraints.add(sum(model.lambda_weights[i] for i in model.VERTICES) == 1)

    # Pn约束：Pn[1]必须是所有顶点P坐标的凸组合
    model.constraints.add(model.Pn[1] == sum(model.lambda_weights[i] * vertices[i, 0] for i in model.VERTICES))

    # Qn约束：Qn[1]必须是所有顶点Q坐标的凸组合
    model.constraints.add(model.Qn[1] == sum(model.lambda_weights[i] * vertices[i, 1] for i in model.VERTICES))


def DScase_apx(model, dscasedata_apx):  # 使用单相算例，如case33
    A_hat = dscasedata_apx['A_hat']
    b_hat = dscasedata_apx['b_hat']
    n_cons = A_hat.shape[0]
    model.BUS = pyo.Set(initialize=range(1,2))
    model.Pn = pyo.Var(model.BUS, within=pyo.Reals)  # 节点注入功率
    model.Qn = pyo.Var(model.BUS, within=pyo.Reals)
    model.V2 = pyo.Var(model.BUS, within=pyo.Reals)
    # def matrix_constraint_rule(model, i):
    #     return A_hat[i, 0] * model.Pn[1] +  A_hat[i, 1]*model.Qn[1] <= b_hat[i]
    # model.constraints = pyo.Constraint(range(n_cons), rule=matrix_constraint_rule)

    model.constraints = pyo.ConstraintList()
    for i in range(n_cons):
        model.constraints.add( A_hat[i, 0] * model.Pn[1] +  A_hat[i, 1]*model.Qn[1] <= b_hat[i])

def DScase_train(casedata, model_type='pretrainnet', plot_flag = True, total_samples=100, batch_size=5,device = 'cpu'):  # 使用单相算例，如case33  #这是一个配电系统优化训练程序，用于生成电力系统运行边界的数据集来训练神经网络
    baseMVA = casedata['baseMVA']
    bus_data = casedata['bus']
    branch_data = casedata['branch']

    # 单相阻抗改为标量值列表
    branch_R = branch_data[:,2]
    branch_X = branch_data[:,3]

    node_flex_dict = casedata.get('node_flex_dict')
    bus_P = bus_data[:,2]  #有功负荷
    bus_Q = bus_data[:,3]  #无功负荷

    # 节点和支路
    bus_ids = [int(row[0]) for row in bus_data]   # 所有节点编号的列表  #[int(row[0]) for row in bus_data] 遍历所有行，提取每个节点的编号。int(row[0]) 确保节点编号是整数类型。# eg.执行后得到：bus_ids = [1, 2, 3, ...]
    line_ids = list(range(len(branch_data)))  # 支路索引编号  #range(len(branch_data)) 生成从0到(支路数-1)的连续整数。list(...) 将range对象转换为列表。# eg.执行后得到：line_ids = [0, 1, 2, 3, 4]
    dim_n=len(bus_ids)
    #total_Pd=sum(bus_P)/baseMVA
    #total_Qd=sum(bus_Q)/baseMVA

    # Pyomo 模型
    model = pyo.ConcreteModel(name='ds')
    model.BUS = pyo.Set(initialize=bus_ids)  # 创建名为BUS的集合，包含所有节点编号：[1, 2, 3, ...]
    model.LINE = pyo.Set(initialize=line_ids)  # 创建名为LINE的集合，包含支路索引：[0, 1, 2, ...]

    # 支路起止映射
    model.from_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 0]) for l in model.LINE}, within=model.BUS, mutable=False)  #within=model.BUS：约束起止节点必须在BUS集合内
    model.to_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 1]) for l in model.LINE}, within=model.BUS, mutable=False)

    # 阻抗参数改为标量 (单相无相间耦合)
    model.R = pyo.Param(model.LINE, initialize=lambda model, l: branch_R[l], mutable=False)  #lambda model, l: branch_R[l]：匿名函数，对于支路l返回branch_R[l].等价于：initialize={l: branch_R[l] for l in model.LINE}
    model.X = pyo.Param(model.LINE, initialize=lambda model, l: branch_X[l], mutable=False)

    #修改后
    Pd_i = bus_P / baseMVA
    Qd_i = bus_Q / baseMVA
    #V_root_1 = 1
    Pd_min = 0.8 * Pd_i
    Pd_max = 1.2 * Pd_i
    Qd_min = 0.8 * Qd_i
    Qd_max = 1.2 * Qd_i

    def _normalize(x, xmin, xmax):
        result = []  #创建一个空列表
        for i in range(len(x)):
            if xmax[i] == xmin[i]:
                result.append(0.5)
            else:
                result.append((x[i] - xmin[i]) / (xmax[i] - xmin[i]))
        return result


    #归一化
    Pd_meta_init = _normalize(Pd_i, Pd_min, Pd_max)
    Qd_meta_init = _normalize(Qd_i, Qd_min, Qd_max)

    #与model.BUS相对应
    def Pd_meta_init_bus(model, i):
        idx = bus_ids.index(i)
        return Pd_meta_init[idx]
    def Qd_meta_init_bus(model, i):
        idx = bus_ids.index(i)
        return Qd_meta_init[idx]

    model.Pd_meta = pyo.Param(model.BUS,initialize=Pd_meta_init_bus, mutable=True, domain=pyo.Reals)
    model.Qd_meta = pyo.Param(model.BUS,initialize=Qd_meta_init_bus, mutable=True, domain=pyo.Reals)

    #反归一化表达式
    def Pd_denormalize_expr(model, i):
        idx = bus_ids.index(i)
        return model.Pd_meta[i] * (Pd_max[idx] - Pd_min[idx]) + Pd_min[idx]

    def Qd_denormalize_expr(model, i):
        idx = bus_ids.index(i)
        return model.Qd_meta[i] * (Qd_max[idx] - Qd_min[idx]) + Qd_min[idx]


    model.Pd = pyo.Expression(model.BUS, rule=Pd_denormalize_expr)
    model.Qd = pyo.Expression(model.BUS, rule=Qd_denormalize_expr)
    model.V_root = pyo.Param(initialize=1.0, mutable=False)

    # 变量定义
    model.V2 = pyo.Var(model.BUS, within=pyo.NonNegativeReals)  # 移除了相维度  #因为是单相系统
    model.Pf = pyo.Var(model.LINE, within=pyo.Reals)  # 单相潮流
    model.Qf = pyo.Var(model.LINE, within=pyo.Reals)
    model.I2 = pyo.Var(model.LINE, within=pyo.NonNegativeReals)  # 单相电流
    model.Pn = pyo.Var(model.BUS, within=pyo.Reals)  # 节点注入功率
    model.Qn = pyo.Var(model.BUS, within=pyo.Reals)
    model.var_proj = pyo.Var(range(2), within=pyo.Reals)  #聚合变量。通常表示平衡节点的注入功率 [Pn[1], Qn[1]]。用于定义可行域的投影

    model.constraints = pyo.ConstraintList()  #约束列表初始化。作用：创建一个空的约束集合。用途：后续通过model.constraints.add()动态添加约束。优势：灵活，不需要预先知道约束数量
    # 电压约束 (pu^2)
    V2max = {i: bus_data[bus_ids.index(i), 11]**2 for i in model.BUS}  #创建字典，i是键，bus_data[bus_ids.index(i), 11]**2是值
    V2min = {i: bus_data[bus_ids.index(i), 12]**2 for i in model.BUS}
    # 电压边界
    for i in model.BUS:
        model.constraints.add(model.V2[i] >= V2min[i])
        model.constraints.add(model.V2[i] <= V2max[i])

    # 单相电压下降方程 (简化)
    for l in model.LINE:
        i = model.from_bus[l]
        j = model.to_bus[l]
        # 单相版本去除了相循环和相间耦合项
        lin_loss = 2 * (model.R[l] * model.Pf[l] + model.X[l] * model.Qf[l])
        quad_loss = (model.R[l] ** 2 + model.X[l] ** 2) * model.I2[l]
        model.constraints.add(model.V2[j] == model.V2[i] - lin_loss + quad_loss)   # Vj² = Vi² - 2*(R*P + X*Q) + (R²+X²)*I²

    # 电流-功率关系
    for l in model.LINE:
        i = model.from_bus[l]
        # model.constraints.add(model.I2[l] * model.V2[i] >= model.Pf[l] ** 2 + model.Qf[l] ** 2)
        # model.constraints.add(model.I2[l] * model.V2[i] <= model.Pf[l] ** 2 + model.Qf[l] ** 2)
        model.constraints.add(model.I2[l] * model.V2[i] == model.Pf[l] ** 2 + model.Qf[l] ** 2)  #S² = P² + Q² = V² * I²


    # 功率平衡 (单相)
    for n in model.BUS:
        if n != 1:  # 非根节点
            inflow_P = sum(model.Pf[l] for l in model.LINE if model.to_bus[l] == n)
            loss_P = sum(model.R[l] * model.I2[l] for l in model.LINE if model.to_bus[l] == n)
            outflow_P = sum(model.Pf[l] for l in model.LINE if model.from_bus[l] == n)
            model.constraints.add(inflow_P - loss_P - outflow_P + model.Pn[n] == 0.0)  #流入功率 - 线路损耗 - 流出功率 + 节点注入功率 = 0

            inflow_Q = sum(model.Qf[l] for l in model.LINE if model.to_bus[l] == n)
            loss_Q = sum(model.X[l] * model.I2[l] for l in model.LINE if model.to_bus[l] == n)
            outflow_Q = sum(model.Qf[l] for l in model.LINE if model.from_bus[l] == n)
            model.constraints.add(inflow_Q - loss_Q - outflow_Q + model.Qn[n] == 0.0)

            # 灵活负荷处理
            node_flex_info = node_flex_dict.get(n, 0)  #字典.get(键, 默认值)。作用：在字典中查找指定的键；如果键存在，返回对应的值；如果键不存在，返回默认值。
            if not node_flex_info['type']:      # 固定负荷:注入功率 = -负荷功率
                model.constraints.add(model.Pn[n] == -model.Pd[n])
                model.constraints.add(model.Qn[n] == -model.Qd[n])
            elif node_flex_info['type'] == 1:   # 有功可调负荷
                model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'] * abs(model.Pd[n]))
                model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'] * abs(model.Pd[n]))
                model.constraints.add(model.Qn[n] == -model.Qd[n])
            elif node_flex_info['type'] == 2:   # 有功无功都可调负荷
                model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'][0] * abs(model.Pd[n]))
                model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'][0] * abs(model.Pd[n]))
                model.constraints.add(model.Qn[n] <= -model.Qd[n] + node_flex_info['rate'][1] * abs(model.Qd[n]))
                model.constraints.add(model.Qn[n] >= -model.Qd[n] - node_flex_info['rate'][1] * abs(model.Qd[n]))
            elif node_flex_info['type'] == 3:
                model.constraints.add((model.Pn[n]+model.Pd[n])**2 + (model.Qn[n]+model.Qd[n])**2 <= 0.3*(model.Pd[n]**2+model.Qd[n]**2))

    # 平衡节点约束
    outflow_P = sum(model.Pf[l] for l in model.LINE if model.from_bus[l] == 1)
    outflow_Q = sum(model.Qf[l] for l in model.LINE if model.from_bus[l] == 1)
    model.constraints.add(model.Pn[1] - outflow_P == model.Pd[1])  #Pn[1]：平衡节点注入的有功功率（正值表示发电注入），不算负荷功率。Pd[1]：平衡节点 本地的有功负荷。
    model.constraints.add(model.Qn[1] - outflow_Q == model.Qd[1])
    model.constraints.add(model.V2[bus_ids[0]] == model.V_root ** 2)  # 参考电压

    # 聚合变量更新
    model.constraints.add(model.var_proj[0] == model.Pn[1])
    model.constraints.add(model.var_proj[1] == model.Qn[1])
    class CaseData(Dataset):  #一个继承自PyTorch Dataset 的类，专门用于生成电力系统训练数据。
        def __init__(self, size=total_samples):  #初始化  size：数据集大小（样本数量），默认使用外部的 total_samples
            self.size = size
            #self.noise_range = noise_range  #noise_range：噪声范围，用于生成随机电压波动

        def __len__(self):
            return self.size  #作用：返回数据集的总样本数，这是PyTorch Dataset必须实现的方法。

        def __getitem__(self, idx):
            return {  #dtheta顺序按照如下来
                    'Pd_meta':torch.rand(dim_n, device=device) - torch.tensor(Pd_meta_init, device=device).float(),  # 归一化值 ∈ [-0.5,0.5]
                    'Qd_meta':torch.rand(dim_n, device=device) - torch.tensor(Qd_meta_init, device=device).float(),
            }

    dim = 2
    # 生成n个等间隔的[0, 2π) 角度 ，一维数组
    num = 36 # 边数/角度的数量
    theta_num = np.linspace(0, 2 * np.pi, num, endpoint=False)  # endpoint=False确保不包含2π

    # 转化成 方向 ，二维数组
    A_hat_normalized = np.column_stack((np.cos(theta_num), np.sin(theta_num)))  # np.column_stack() 是垂直方向（列方向）的堆叠





    errorcalculator = ErrorCalculator(  #初始化一个 ErrorCalculator 对象
        original_model={'model': model},
        A_hat=A_hat_normalized,
        solver='ipopt',  #ipopt，非线性优化求解器
    )

    case_name = casedata['casename']
    figure_folder = f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{case_name}\\A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)\\figures'
    os.makedirs(figure_folder, exist_ok=True)


    n_train = 501

    if plot_flag:  #只有当 plot_flag=True 时才执行可视化
        plt.figure(figsize=(8, 6))  #创建8×6英寸的图形窗口
        xlim = np.array([sum(bus_P) - 0.5*abs(sum(bus_P)),sum(bus_P) + 0.8*abs(sum(bus_P))])/ baseMVA  #设置x轴和y轴范围
        ylim = np.array([sum(bus_Q) - 0.5*abs(sum(bus_Q)),sum(bus_Q) + 0.8*abs(sum(bus_Q))])/ baseMVA
        plotter = ShapeDrawer_2D()  #创建一个2D形状绘制器（是自定义类）
        plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,  #定义可行域多边形的线性不等式（Ax ≤ b）
                             facecolor='green', xlim=xlim, ylim=ylim,
                             label=f'Approximation',
                             title=f'Training step = {0}'  #图表标题，显示训练步数
                             )

        os.makedirs(figure_folder + f'/pretrain_process', exist_ok=True)  #创建保存图片的目录
        plotter.save(figure_folder + f'/pretrain_process/step0{0}.png')  #保存为PNG格式图片。文件名：step00.png（训练步数0）
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

    # 训练参数配置
    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 5,  #5
            "training_callback": training_callback,
            "optimizer": 'sgd',
            "lr": 1e-2,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.99},  #学习率调度器。"type": "StepLR"：阶梯式学习率衰减。"step_size": 100：每100步衰减一次学习率。"gamma": 0.99：衰减系数，学习率变为原来的0.99倍
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            'feas_tol': 1e-10,  #可行性误差的收敛阈值 #1e-10
            'opt_tol': 1e-10,  #最优性误差的收敛阈值 #1e-10
            "rate_opt_feas": 0.6
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": training_callback,
            "optimizer": "adam",
            # "optimizer": "sgd",
            "lr": 5e-5,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            'feas_tol': 1e-10,  #1e-10
            'opt_tol': 1e-10,   #1e-10
            "rate_opt_feas": 0.6,
        }
    params_dict, param_count = pyomo_params_to_numpy(model)  #将 Pyomo 模型中的参数转换为 NumPy 格式，便于神经网络处理。
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,  #电力系统的可调参数
        'dataloader': DataLoader(  #CaseData() → 生成电压波动样本 → DataLoader → 批量训练数据
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,  # 维度信息
    }
    return {
        'casename': case_name,
        'A_hat': A_hat_normalized,
        'b_hat': errorcalculator.b_hat,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'params': params,
        'result_path': f'{PROJECT_ROOT}\\results\\ds_proj_paper\\{case_name}\\A(36,2)_type3(2,29)_lr1(3e-5)_lr2(1e-5)_rate(1e-4)\\{model_type}_weights.pth',
        'n_train': n_train,
        'metadata': {
            'dscasedata': casedata,
        }
    }


def disagg_DS(P_target, Q_target, v_target, dscasedata):
    model = pyo.ConcreteModel()
    DScase(model, dscasedata)
    model.constraints.add(v_target ** 2 == model.V2[1])
    model.obj = pyo.Objective(
        expr=(P_target-model.Pn[1]) ** 2 + (Q_target-model.Qn[1]) ** 2,
        sense=pyo.minimize
    )
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model, tee=True)  #tee=True改成False

    return model.obj()

def build_acopf_model(ppc):
    baseMVA = ppc['baseMVA']
    bus = ppc['bus']
    gen = ppc['gen']
    branch = ppc['branch']
    gencost = ppc['gencost']

    nb = bus.shape[0]     # number of buses
    nl = branch.shape[0]  # number of lines
    ng = gen.shape[0]     # number of generators

    model = pyo.ConcreteModel()

    model.N = pyo.RangeSet(0, nb - 1)
    model.L = pyo.RangeSet(0, nl - 1)
    model.G = pyo.RangeSet(0, ng - 1)

    # Parameters
    Pd_value = {int(i): bus[i, 2] / baseMVA for i in model.N}
    Qd_value = {int(i): bus[i, 3] / baseMVA for i in model.N}

    bus_type = {int(i): int(bus[i, 1]) for i in model.N}  # 1 = PQ, 2 = PV, 3 = Slack

    # Generator location mapping: gen_id -> bus_id
    gen_bus_map = {int(i): int(gen[i, 0]) - 1 for i in model.G}

    # Bounds
    Vmax = {int(i): bus[i, 11] for i in model.N}
    Vmin = {int(i): bus[i, 12] for i in model.N}

    Vgen = {int(i): gen[i, 5] for i in model.G}


    Pmax = {int(i): gen[i, 8] / baseMVA for i in model.G}
    Pmin = {int(i): gen[i, 9] / baseMVA for i in model.G}
    Qmax = {int(i): gen[i, 3] / baseMVA for i in model.G}
    Qmin = {int(i): gen[i, 4] / baseMVA for i in model.G}

    # Cost coefficients
    cost_a = {int(i): gencost[i, 4] for i in model.G}  # quadratic
    cost_b = {int(i): gencost[i, 5] for i in model.G}  # linear
    cost_c = {int(i): gencost[i, 6] for i in model.G}  # constant

    # Admittance matrix elements
    from scipy.sparse import lil_matrix
    import cmath
    Ybus = lil_matrix((nb, nb), dtype=complex)
    for k in range(nl):
        i, j = int(branch[k, 0]) - 1, int(branch[k, 1]) - 1
        r, x, b = branch[k, 2], branch[k, 3], branch[k, 4]
        z = complex(r, x)
        y = 1 / z
        Ybus[i, j] -= y
        Ybus[j, i] -= y
        Ybus[i, i] += y + complex(0, b / 2)
        Ybus[j, j] += y + complex(0, b / 2)

    G = Ybus.real.toarray()
    B = Ybus.imag.toarray()

    # Variables
    model.V = pyo.Var(model.N, within=pyo.NonNegativeReals, bounds=lambda m, i: (Vmin[i], Vmax[i]), initialize=1.0)
    model.theta = pyo.Var(model.N, bounds=(-np.pi, np.pi), initialize=0.0)

    model.Pg = pyo.Var(model.G, bounds=lambda m, i: (Pmin[i], Pmax[i]), initialize=1.0)
    model.Qg = pyo.Var(model.G, bounds=lambda m, i: (Qmin[i], Qmax[i]), initialize=0.0)

    model.Pd = pyo.Var(model.N, initialize=lambda m, i: Pd_value[i])
    model.Qd = pyo.Var(model.N, initialize=lambda m, i: Qd_value[i])


    # Reference bus (Slack): fix angle and voltage
    for i in model.N:
        model.Pd[i].fix(Pd_value[i])
        model.Qd[i].fix(Qd_value[i])
        if bus_type[i] == 3:
            i_gen = np.where(gen[:, 0] == i+1)[0][0]
            model.V[i].fix(Vgen[i_gen])
            model.theta[i].fix(0.0)
        elif  bus_type[i] == 2:
            i_gen = np.where(gen[:,0]==i+1)[0][0]
            model.V[i].fix(Vgen[i_gen])
    # Objective: minimize generation cost
    model.obj = pyo.Objective(
        expr=sum(cost_a[g] * model.Pg[g]**2 + cost_b[g] * model.Pg[g] + cost_c[g] for g in model.G),
        sense=pyo.minimize
    )

    # Power balance constraints
    def real_power_balance_rule(m, i):
        return sum(
            m.V[i] * m.V[j] * (G[i, j] * pyo.cos(m.theta[i] - m.theta[j]) + B[i, j] * pyo.sin(m.theta[i] - m.theta[j]))
            for j in model.N
        ) + m.Pd[i] == sum(m.Pg[g] for g in model.G if gen_bus_map[g] == i)

    def reactive_power_balance_rule(m, i):
        return sum(
            m.V[i] * m.V[j] * (G[i, j] * pyo.sin(m.theta[i] - m.theta[j]) - B[i, j] * pyo.cos(m.theta[i] - m.theta[j]))
            for j in model.N
        ) + m.Qd[i] == sum(m.Qg[g] for g in model.G if gen_bus_map[g] == i)


    model.real_power_balance = pyo.Constraint(model.N, rule=real_power_balance_rule)
    model.reactive_power_balance = pyo.Constraint(model.N, rule=reactive_power_balance_rule)


    return model

def case136ds(root_voltage = 1):
    """PyPower case format of case118zh (translated from MATPOWER format)"""
    ppc = {"version": '2'}

    ## system MVA base
    ppc["baseMVA"] = 10.0
    ppc["basekV"] = 13.8

    # 转换后的NumPy数组
    ppc["bus"] = np.array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [2, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [3, 1, 47.78, 19.01, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [4, 1, 42.55, 16.93, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [5, 1, 87.02, 34.62, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [6, 1, 311.31, 123.855, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [7, 1, 148.869, 59.23, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [8, 1, 238.672, 94.96, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [9, 1, 62.3, 24.79, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [10, 1, 124.598, 49.57, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [11, 1, 140.175, 55.77, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [12, 1, 116.813, 46.47, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [13, 1, 249.203, 99.15, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [14, 1, 291.447, 115.952, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [15, 1, 303.72, 120.835, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [16, 1, 215.396, 85.7, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [17, 1, 198.586, 79.01, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [18, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [19, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [20, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [21, 1, 30.13, 14.73, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [22, 1, 230.972, 112.92, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [23, 1, 60.26, 29.46, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [24, 1, 230.972, 112.92, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [25, 1, 120.507, 58.92, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [26, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [27, 1, 56.98, 27.86, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [28, 1, 364.665, 178.281, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [29, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [30, 1, 124.647, 60.94, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [31, 1, 56.98, 27.86, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [32, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [33, 1, 85.47, 41.79, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [34, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [35, 1, 396.735, 193.96, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [36, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [37, 1, 181.152, 88.56, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [38, 1, 242.172, 118.395, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [39, 1, 75.32, 36.82, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [40, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [41, 1, 1.25, 0.53, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [42, 1, 6.27, 2.66, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [43, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [44, 1, 117.88, 49.97, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [45, 1, 62.67, 26.57, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [46, 1, 172.285, 73.03, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [47, 1, 458.556, 194.388, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [48, 1, 262.962, 111.473, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [49, 1, 235.761, 99.94, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [50, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [51, 1, 109.215, 46.3, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [52, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [53, 1, 72.81, 30.87, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [54, 1, 258.473, 109.57, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [55, 1, 69.17, 29.32, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [56, 1, 21.84, 9.26, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [57, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [58, 1, 20.53, 8.7, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [59, 1, 150.548, 63.82, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [60, 1, 220.687, 93.55, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [61, 1, 92.38, 39.16, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [62, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [63, 1, 226.693, 96.1, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [64, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [65, 1, 294.016, 116.974, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [66, 1, 83.02, 33.03, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [67, 1, 83.02, 33.03, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [68, 1, 103.77, 41.29, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [69, 1, 176.408, 70.18, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [70, 1, 83.02, 33.03, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [71, 1, 217.917, 86.7, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [72, 1, 23.29, 9.27, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [73, 1, 5.08, 2.02, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [74, 1, 72.64, 28.9, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [75, 1, 405.99, 161.523, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [76, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [77, 1, 100.182, 42.47, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [78, 1, 142.523, 60.42, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [79, 1, 96.04, 40.71, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [80, 1, 300.454, 127.366, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [81, 1, 141.238, 59.87, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [82, 1, 279.847, 118.631, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [83, 1, 87.31, 37.01, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [84, 1, 243.849, 103.371, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [85, 1, 247.75, 105.025, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [86, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [87, 1, 89.88, 38.1, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [88, 1, 1137.28, 482.108, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [89, 1, 458.339, 194.296, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [90, 1, 385.197, 163.29, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [91, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [92, 1, 79.61, 33.75, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [93, 1, 87.31, 37.01, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [94, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [95, 1, 74, 31.37, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [96, 1, 232.05, 98.37, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [97, 1, 141.819, 60.12, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [98, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [99, 1, 76.45, 32.41, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [100, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [101, 1, 51.32, 21.76, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [102, 1, 59.87, 25.38, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [103, 1, 9.07, 3.84, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [104, 1, 2.09, 0.89, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [105, 1, 16.735, 7.09, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [106, 1, 1506.522, 638.634, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [107, 1, 313.023, 132.694, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [108, 1, 79.83, 33.84, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [109, 1, 51.32, 21.76, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [110, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [111, 1, 202.435, 85.82, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [112, 1, 60.82, 25.78, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [113, 1, 45.62, 19.34, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [114, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [115, 1, 157.07, 66.58, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [116, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [117, 1, 250.148, 106.041, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [118, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [119, 1, 69.81, 29.59, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [120, 1, 32.07, 13.6, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [121, 1, 61.08, 25.89, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [122, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [123, 1, 94.62, 46.26, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [124, 1, 49.86, 24.38, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [125, 1, 123.164, 60.21, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [126, 1, 78.35, 38.3, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [127, 1, 145.475, 71.12, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [128, 1, 21.37, 10.45, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [129, 1, 74.79, 36.56, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [130, 1, 227.926, 111.431, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [131, 1, 35.61, 17.41, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [132, 1, 249.295, 121.877, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [133, 1, 316.722, 154.842, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [134, 1, 333.817, 163.199, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [135, 1, 249.295, 121.877, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95],
        [136, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.05, 0.95]
    ])


    ## generator data
    # bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin
    ppc["gen"] = np.array([
        [1, 0, 0, 10, -10, 1, 100, root_voltage, 10, 0]
    ])

    ppc["branch"] = np.array([
        [1, 2, 0.33205, 0.76653, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [2, 3, 0.00188, 0.00433, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [3, 4, 0.22324, 0.51535, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [4, 5, 0.09943, 0.22953, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [5, 6, 0.15571, 0.35945, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [6, 7, 0.16321, 0.37677, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [7, 8, 0.11444, 0.26417, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [7, 9, 0.05675, 0.05666, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [9, 10, 0.52124, 0.27418, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [9, 11, 0.10877, 0.10860, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [11, 12, 0.39803, 0.20937, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [11, 13, 0.91744, 0.31469, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [11, 14, 0.11823, 0.11805, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [14, 15, 0.50228, 0.26421, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [14, 16, 0.05675, 0.05666, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [16, 17, 0.29379, 0.15454, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [1, 18, 0.33205, 0.76653, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [18, 19, 0.00188, 0.00433, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [19, 20, 0.22324, 0.51535, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [20, 21, 0.10881, 0.25118, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [21, 22, 0.71078, 0.37388, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [21, 23, 0.18197, 0.42008, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [23, 24, 0.30326, 0.15952, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [23, 25, 0.02439, 0.05630, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [25, 26, 0.04502, 0.10394, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [26, 27, 0.01876, 0.04331, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [27, 28, 0.11823, 0.11805, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [28, 29, 0.02365, 0.02361, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [29, 30, 0.18954, 0.09970, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [30, 31, 0.39803, 0.20937, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [29, 32, 0.05675, 0.05666, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [32, 33, 0.09477, 0.04985, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [33, 34, 0.41699, 0.21934, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [34, 35, 0.11372, 0.05982, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [32, 36, 0.07566, 0.07555, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [36, 37, 0.36960, 0.19442, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [37, 38, 0.26536, 0.13958, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [36, 39, 0.05675, 0.05666, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [1, 40, 0.33205, 0.76653, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [40, 41, 0.11819, 0.27283, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [41, 42, 2.96288, 1.01628, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [41, 43, 0.00188, 0.00433, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [43, 44, 0.06941, 0.16024, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [44, 45, 0.81502, 0.42872, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [44, 46, 0.06378, 0.14724, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [46, 47, 0.13132, 0.30315, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [47, 48, 0.06191, 0.14291, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [48, 49, 0.11444, 0.26417, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [49, 50, 0.28374, 0.28331, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [50, 51, 0.28374, 0.28331, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [49, 52, 0.04502, 0.10394, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [52, 53, 0.02626, 0.06063, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [53, 54, 0.06003, 0.13858, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [54, 55, 0.03002, 0.06929, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [55, 56, 0.02064, 0.04764, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [53, 57, 0.10881, 0.25118, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [57, 58, 0.25588, 0.13460, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [58, 59, 0.41699, 0.21934, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [59, 60, 0.50228, 0.26421, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [60, 61, 0.33170, 0.17448, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [61, 62, 0.20849, 0.10967, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [48, 63, 0.13882, 0.32047, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [1, 64, 0.00750, 0.01732, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [64, 65, 0.27014, 0.62362, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [65, 66, 0.38270, 0.88346, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [66, 67, 0.33018, 0.76220, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [67, 68, 0.32830, 0.75787, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [68, 69, 0.17072, 0.39409, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [69, 70, 0.55914, 0.29412, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [69, 71, 0.05816, 0.13425, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [71, 72, 0.70130, 0.36890, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [72, 73, 1.02352, 0.53839, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [71, 74, 0.06754, 0.15591, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [74, 75, 1.32352, 0.45397, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [1, 76, 0.01126, 0.02598, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [76, 77, 0.72976, 1.68464, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [77, 78, 0.22512, 0.51968, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [78, 79, 0.20824, 0.48071, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [79, 80, 0.04690, 0.10827, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [80, 81, 0.61950, 0.61857, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [81, 82, 0.34049, 0.33998, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [82, 83, 0.56862, 0.29911, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [82, 84, 0.10877, 0.10860, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [84, 85, 0.56862, 0.29911, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [1, 86, 0.01126, 0.02598, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [86, 87, 0.41835, 0.96575, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [87, 88, 0.10499, 0.13641, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [87, 89, 0.43898, 1.01338, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [89, 90, 0.07520, 0.02579, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [90, 91, 0.07692, 0.17756, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [91, 92, 0.33205, 0.76653, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [92, 93, 0.08442, 0.19488, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [93, 94, 0.13320, 0.30748, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [94, 95, 0.29320, 0.29276, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [95, 96, 0.21753, 0.21721, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [96, 97, 0.26482, 0.26443, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [94, 98, 0.10318, 0.23819, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [98, 99, 0.13507, 0.31181, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [1, 100, 0.00938, 0.02165, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [100, 101, 0.16884, 0.38976, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [101, 102, 0.11819, 0.27283, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [102, 103, 2.28608, 0.78414, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [102, 104, 0.45587, 1.05236, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [104, 105, 0.69600, 1.60669, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [105, 106, 0.45774, 1.05669, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [106, 107, 0.20298, 0.26373, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [107, 108, 0.21348, 0.27737, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [108, 109, 0.54967, 0.28914, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [109, 110, 0.54019, 0.28415, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [108, 111, 0.04550, 0.05911, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [111, 112, 0.47385, 0.24926, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [112, 113, 0.86241, 0.45364, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [113, 114, 0.56862, 0.29911, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [109, 115, 0.77711, 0.40878, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [115, 116, 1.08038, 0.56830, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [110, 117, 1.09933, 0.57827, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [117, 118, 0.47385, 0.24926, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [105, 119, 0.32267, 0.74488, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [119, 120, 0.14633, 0.33779, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [120, 121, 0.12382, 0.28583, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [1, 122, 0.01126, 0.02598, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [122, 123, 0.64910, 1.49842, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [123, 124, 0.04502, 0.10394, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [124, 125, 0.52640, 0.18056, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [124, 126, 0.02064, 0.04764, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [126, 127, 0.53071, 0.27917, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [126, 128, 0.09755, 0.22520, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [128, 129, 0.11819, 0.27283, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [128, 130, 0.13882, 0.32047, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [130, 131, 0.04315, 0.09961, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [131, 132, 0.09192, 0.21220, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [132, 133, 0.16134, 0.37244, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [133, 134, 0.37832, 0.37775, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [134, 135, 0.39724, 0.39664, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        [135, 136, 0.29320, 0.29276, 0, 100, 100, 100, 0, 0, 1, -360, 360],
        # [8, 74, 0.13132, 0.30315, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [10, 25, 0.26536, 0.13958, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [16, 84, 0.14187, 0.14166, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [39, 136, 0.08512, 0.08499, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [26, 52, 0.04502, 0.10394, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [51, 97, 0.14187, 0.14166, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [56, 99, 0.14187, 0.14166, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [63, 121, 0.03940, 0.09094, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [67, 80, 0.12944, 0.29882, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [80, 132, 0.01688, 0.03898, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [85, 136, 0.33170, 0.17448, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [92, 105, 0.14187, 0.14166, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [91, 130, 0.07692, 0.17756, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [91, 104, 0.07692, 0.17756, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [93, 105, 0.07692, 0.17756, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [93, 133, 0.07692, 0.17756, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [97, 121, 0.26482, 0.26443, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [111, 48, 0.49696, 0.64567, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [127, 77, 0.17059, 0.08973, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [129, 78, 0.05253, 0.12126, 0, 100, 100, 100, 0, 0, 0, -360, 360],
        # [136, 99, 0.29320, 0.29276, 0, 100, 100, 100, 0, 0, 0, -360, 360]
    ])

    ppc["gencost"] = np.array([
        [2, 0, 0, 3, 0, 20, 0],
    ])
    Vbase = ppc["basekV"]*1e3
    Sbase = ppc["baseMVA"]*1e6
    Zbase = Vbase**2/Sbase
    ppc["branch"][:, 2:4] /= Zbase #阻抗标幺化
    ppc["bus"][:, 2:4]/=1e3#负荷标幺化

    # Create list of (node, load) tuples
    node_loads = [(int(bus_data[0]), bus_data[2]) for bus_data in ppc["bus"]]
    # Sort nodes by load in descending order
    sorted_nodes = sorted(node_loads, key=lambda x: x[1], reverse=True)
    # Determine the top 50% nodes with the highest load
    num_nodes = len(sorted_nodes)
    top_50_percent = num_nodes // 2
    top_nodes = {node for node, _ in sorted_nodes[:top_50_percent]}

    # Create node_flex_dict
    node_flex_dict = {}
    for node, _ in node_loads:
        if node in top_nodes:
            node_flex_dict[node] = {"type": 1, "rate": 0.3}
        else:
            node_flex_dict[node] = {"type": 0, "rate": None}
    ppc["node_flex_dict"] = node_flex_dict
    ppc["root_voltage"] = root_voltage
    return ppc
# Run model

def _find_top_percent_bus(tsppc, percent = 0.5, threshold = 20):
    """
    Find bus numbers where active power demand is in the top percent and its absolute value
    is greater than or equal to the threshold.

    Parameters:
    ppc (dict): PYPOWER case dictionary
    percent (float): Percentage of top buses to select (e.g., 50 for top 50%)
    threshold (float): Minimum absolute value of active power demand (Pd) MW

    Returns:
    dict: Dictionary with bus numbers as keys and 1 as values
    """
    # Extract Pd (active power demand) from bus data (column index 2)
    pd_values = tsppc["bus"][:, 2]
    # Extract bus numbers (column index 0)
    # bus_numbers = tsppc["bus"][:, 0].astype(int)

    # Sort absolute Pd values in descending order and get corresponding indices
    sorted_indices = np.argsort(pd_values)[::-1]

    # Determine number of buses to select based on percent
    num_buses = len(pd_values)
    num_top = int(np.ceil(num_buses * percent))

    # Select top percent indices
    top_indices = sorted_indices[:num_top]

    # Filter buses where absolute Pd is >= threshold
    result = []
    for idx in top_indices:
        if pd_values[idx] >= threshold:
            # result.append(bus_numbers[idx])
            result.append(idx)
    return result

def define_td_case_data(tsppc, dsppc, ds_percent = 0.5, load_threshold = 20):
    top_indices = _find_top_percent_bus(tsppc, percent = ds_percent, threshold = load_threshold)
    total_PQ_ds = sum(dsppc["bus"][:,2:4])
    dscasedata_dict = {}
    for key in top_indices:
        tsppc["bus"][key,2:4] -= total_PQ_ds #
        dscasedata_dict[key] = dsppc
    return dscasedata_dict



if __name__ == '__main__':

    # dsppc = case33bw_ds(flex_percent=0.8, flex_rate=0.3)  #flex_percent = 0.8 改成0.5 对res有影响
    # P_target = 0
    # Q_target = 0
    # v_target = 0.90
    # res = disagg_DS(P_target, Q_target, v_target, dsppc)
    # print(res)
    # v_target = 1.05
    # res = disagg_DS(P_target, Q_target, v_target, dsppc)
    # print(res)
    # print(sum(dsppc['bus'][:,2:4]))  #计算并打印电力系统所有节点的总有功负荷和总无功负荷。
    # print(dsppc['baseMVA'])
    # print(dsppc['bus'])  #二维数组（numpy array），每行代表一个节点，每列代表该节点的一个属性。
    # #ppc = tsppc

    ppc=case33bw_ds()
    #print(ppc['gen'])
    #将gen矩阵转换为浮点数类型
    # ppc['gen'] = ppc['gen'].astype(float)
    # ppc['gen'][0,5]=0.95
    #print(ppc['gen'])

    #潮流计算测试case
    from pypower.api import ppoption, runpf  #从pypower.api模块中导入两个函数：ppoption和runpf。ppoption用于设置潮流计算的选项，runpf用于运行潮流计算。

    # 以下开始测试单向潮流计算
    ppopt = ppoption()  # 使用默认选项
    ppopt['VERBOSE'] = 2 # 控制输出详细程度，0 表示输出较少信息
    ppopt['OUT_ALL'] = 0   # 不输出任何结果（除了错误信息）
    # 或者可以设置输出更多信息，例如：
    ppopt = ppoption(VERBOSE=2, OUT_ALL=1)  # 输出详细结果  #覆盖前面两行对ppopt的设置。
    #
    # 运行潮流计算
    results, success = runpf(ppc, ppopt)  #调用runpf函数进行潮流计算。传入两个参数：电力系统数据ppc和计算选项ppopt。函数返回两个值：results（包含潮流计算结果的数据结构）和success（一个布尔值，表示潮流计算是否收敛）。

    # 检查是否收敛
    if success:
        print("潮流计算收敛！")
        # 输出各节点的电压幅值和相角（角度）
        print("\n节点电压结果：")
        print("节点ID  电压幅值(pu)  电压相角(度)")
        for i in range(len(results["bus"])):
            bus_id = int(results["bus"][i][0])
            voltage_mag = results["bus"][i][7]#*results["bus"][i][9]*1e3
            voltage_angle = results["bus"][i][8]  # 角度，not rad
            print(f"{bus_id:4d}    {voltage_mag:.6f}      {voltage_angle:.6f}")
        # print(ppc['loadvolt'])
    else:
        print("潮流计算未收敛！")

    print(ppc["type3_nodes"])
    print(ppc["type2_nodes"])
    print(ppc["type0_nodes"])
    print(ppc["node_flex_dict"])
    print(ppc["max_P"])
    print(ppc["sorted_nodes"])
    # print(ppc["branch"])