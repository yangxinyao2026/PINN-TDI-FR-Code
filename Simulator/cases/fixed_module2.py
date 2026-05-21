import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from matplotlib.patches import Polygon
import numpy as np
import torch
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pyomo.environ as pyo  # 建模型-->加参数和决策变量-->加约束-->加目标函数-->求解模型-->输出结果


def DScase_train(model,casedata):  # 使用单相算例，如case33
    baseMVA = casedata['baseMVA']  #np.uint8 是 NumPy 提供的一种 无符号 8 位整数类型（Unsigned 8-bit Integer），取值范围为 0~255，占用 1 个字节 内存。它常用于 图像处理、嵌入式数据存储 等需要节省内存的场景。
    bus_data = casedata['bus']  #ndarray
    branch_data = casedata['branch']  #ndarray

    # 单相阻抗改为标量值列表
    branch_R = branch_data[:,2]  #ndarray
    branch_X = branch_data[:,3]

    node_flex_dict = casedata.get('node_flex_dict')
    bus_P = bus_data[:,2]  #有功负荷
    bus_Q = bus_data[:,3]  #无功负荷

    # 节点和支路
    bus_ids = [int(row[0]) for row in bus_data]   # 所有节点编号的列表  #[int(row[0]) for row in bus_data] 遍历所有行，提取每个节点的编号。int(row[0]) 确保节点编号是整数类型。# eg.执行后得到：bus_ids = [1, 2, 3, ...]
    line_ids = list(range(len(branch_data)))  # 支路索引编号  #range(len(branch_data)) 生成从0到(支路数-1)的连续整数。list(...) 将range对象转换为列表。# eg.执行后得到：line_ids = [0, 1, 2, 3, 4]
    dim_n=len(bus_ids)
    total_Pd=sum(bus_P)/baseMVA
    total_Qd=sum(bus_Q)/baseMVA

    # Pyomo 模型

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
    # V_root_min = 0.95 * V_root_1
    # V_root_max = 1.05 * V_root_1

    def _normalize(x, xmin, xmax):
        result = []  #创建一个空列表
        for i in range(len(x)):
            if xmax[i] == xmin[i]:
                result.append(0.5)
            else:
                result.append((x[i] - xmin[i]) / (xmax[i] - xmin[i]))
        return result

    # def V_root_normalize(x, xmin, xmax):
    #     if xmax == xmin:
    #         return 0.5
    #     return (x - xmin) / (xmax - xmin)

    #归一化
    Pd_meta_init = _normalize(Pd_i, Pd_min, Pd_max)
    Qd_meta_init = _normalize(Qd_i, Qd_min, Qd_max)
    #V_root_meta_init=V_root_normalize(V_root_1,V_root_min,V_root_max)

    #与model.BUS相对应
    def Pd_meta_init_bus(model, i):
        idx = bus_ids.index(i)
        return Pd_meta_init[idx]
    def Qd_meta_init_bus(model, i):
        idx = bus_ids.index(i)
        return Qd_meta_init[idx]
                                                       #输入的是列表
    model.Pd_meta = pyo.Param(model.BUS,initialize=Pd_meta_init_bus, mutable=True, domain=pyo.Reals)
    model.Qd_meta = pyo.Param(model.BUS,initialize=Qd_meta_init_bus, mutable=True, domain=pyo.Reals)
    #model.V_root_meta = pyo.Param(initialize=V_root_meta_init, mutable=True, domain=pyo.Reals)

    #反归一化表达式
    def Pd_denormalize_expr(model, i):
        idx = bus_ids.index(i)
        return model.Pd_meta[i] * (Pd_max[idx] - Pd_min[idx]) + Pd_min[idx]

    def Qd_denormalize_expr(model, i):
        idx = bus_ids.index(i)
        return model.Qd_meta[i] * (Qd_max[idx] - Qd_min[idx]) + Qd_min[idx]

    # def _denormalize(x_normalized, xmin, xmax):
    #     return (x_normalized) * (xmax - xmin) + xmin

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
            # elif node_flex_info['type'] == 3:   # 有功可调，且P^2+Q^2<=设定值
            #     model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'] * abs(model.Pd[n]))
            #     model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'] * abs(model.Pd[n]))
            #     model.constraints.add(model.Pn[n]**2 + model.Qn[n]**2 <= 0.25*casedata['max_P']/baseMVA)
            elif node_flex_info['type'] == 3:
                # model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'] * abs(model.Pd[n]))
                # model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'] * abs(model.Pd[n]))
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

    case_name = casedata['casename']

    return {'Pd_meta_init':Pd_meta_init,
            'Qd_meta_init':Qd_meta_init,
            #'V_root_meta_init':V_root_meta_init,
            'casename': case_name,
            }

class ShapeDrawer_2D:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.shapes = []  # 统一存储所有图形对象
        self.ax.grid(False)  # 不显示网格

        self.error_history = {
            'iterations': [],
            'error_feas': [],
            'error_opt': []
        }
    def _add_shape(self, patch, shape_type, ** kwargs):
        """统一添加图形到存储列表"""
        shape_id = len(self.shapes)
        shape_info = {
            'patch': patch,
            'shape': shape_type,
            'id': shape_id,
            ** kwargs  # 存储其他自定义属性
        }
        self.shapes.append(shape_info)
        return shape_id
    def plot_polygon(self, xlim, ylim, alpha=0.2, edgecolor='blue',
                     facecolor='blue', label=None, title=None, A=None, b=None, x_org=None):
        """绘制多边形"""  #改一下输入的判断
        #if A is not None and b is not None and len(A) > 0 and len(b) > 0:
        # 交点未知
        if A is not None and b is not None:
            # 计算交点
            vertices = []
            for i in range(len(A)):  #len(A)=A的行数
                for j in range(i + 1, len(A)): #遍历不等式两两组合（避免重复）,每个不等式对可能定义两条直线的交点
                    try:
                        x, y = np.linalg.solve(np.array([A[i], A[j]]), np.array([b[i], b[j]]))  #使用 NumPy 的线性方程组求解器来找到两条直线的交点。将不等式转为等式：A[i]·[x, y] = b[i] 和 A[j]·[x, y] = b[j]，求解这个2×2线性方程组得到交点
                        if np.all(A @ np.array([x, y]) <= b + 1e-5):  #检查：该交点是否满足 所有 原始不等式，只有满足所有不等式的交点才是可行域的顶点
                            vertices.append((x, y))
                    except np.linalg.LinAlgError:  #当两条直线平行或无解时（系数矩阵奇异），跳过该不等式对
                        pass

            if not vertices:
                return None
        else: # 交点已知
            vertices = x_org

        # 顶点排序
        center = np.mean(vertices, axis=0)
        angles = np.arctan2([v[1] - center[1] for v in vertices],  #使用np.arctan2(y, x)来计算给定y坐标和x坐标的点的反正切值（即点(x,y)与原点连线与x轴正方向的夹角，范围在[-π, π]）。
                            [v[0] - center[0] for v in vertices])
        sorted_vertices = np.array([v for _, v in sorted(zip(angles, vertices))])  #sorted()排序，zip将对应的元素打包成一个个元组

        # 创建并添加多边形
        polygon = Polygon(sorted_vertices, closed=True, alpha=alpha,
                          edgecolor=edgecolor, facecolor=facecolor, label=label)
        patch = self.ax.add_patch(polygon)

        # 统一存储
        shape_id = self._add_shape(
            patch=patch,
            shape_type='polygon',
            vertices=sorted_vertices,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            label=label
        )

        # 设置坐标范围和标题
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # # 移除所有边框和网格
        # self.ax.set_xticks([])  #ax.set_xticks 是 Matplotlib 中用于设置 X 轴刻度位置的方法。它允许你指定刻度的位置，并可选地设置次要刻度。
        # self.ax.set_yticks([])
        # self.ax.spines['top'].set_visible(False)
        # self.ax.spines['right'].set_visible(False)
        # self.ax.spines['bottom'].set_visible(False)
        # self.ax.spines['left'].set_visible(False)
        # self.ax.grid(False)


        if title:
            self.ax.set_title(title)

        return shape_id

    def save(self, filename, dpi=300, transparent=False, format='svg', show_legend=True):
        """保存图形到文件（SVG格式）"""
        if show_legend and any(shape.get('label') for shape in self.shapes):
            self.ax.legend(  # ax.legend()函数用于在图表中添加图例
                loc='upper right',  #图例位置
                bbox_to_anchor=(1, 1), #用于自定义图例位置，特别是当需要将图例放置在坐标轴外时
                frameon=False  # 无边框图例
            )

        # 确保保存为SVG格式
        if not filename.lower().endswith('.'+format):
            filename += '.'+format

        self.fig.savefig(
            filename,
            dpi=dpi,
            transparent=transparent,
            format=format,
            bbox_inches='tight'
        )
        plt.close(self.fig)
        print(f"图形已保存到 {filename}")
