import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from matplotlib.patches import Polygon
import numpy as np
import torch
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pyomo.environ as pyo  # Build a model-->Add parameters and decision variables-->add constraints-->Add objective function-->Solve the model-->Output results


def DScase_train(model,casedata):  # Use a single-phase study such ascase33
    baseMVA = casedata['baseMVA']
    bus_data = casedata['bus']  #ndarray
    branch_data = casedata['branch']  #ndarray

    # Single phase impedance changed to list of scalar values
    branch_R = branch_data[:,2]  #ndarray
    branch_X = branch_data[:,3]

    node_flex_dict = casedata.get('node_flex_dict')
    bus_P = bus_data[:,2]  #active load
    bus_Q = bus_data[:,3]  #Reactive load

    # nodes and branches
    bus_ids = [int(row[0]) for row in bus_data]  # Integer bus identifiers.
    line_ids = list(range(len(branch_data)))  # Zero-based branch identifiers.
    dim_n=len(bus_ids)
    total_Pd=sum(bus_P)/baseMVA
    total_Qd=sum(bus_Q)/baseMVA

    # Pyomo model

    model.BUS = pyo.Set(initialize=bus_ids)
    model.LINE = pyo.Set(initialize=line_ids)

    # Branch start and end mapping
    model.from_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 0]) for l in model.LINE}, within=model.BUS, mutable=False)  #within=model.BUS: The constraint start and end nodes must be inBUSwithin the collection
    model.to_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 1]) for l in model.LINE}, within=model.BUS, mutable=False)

    # Impedance parameter changed to scalar (Single phase without phase coupling)
    model.R = pyo.Param(model.LINE, initialize=lambda model, l: branch_R[l], mutable=False)
    model.X = pyo.Param(model.LINE, initialize=lambda model, l: branch_X[l], mutable=False)

    #After modification
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
        result = []
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

    #normalization
    Pd_meta_init = _normalize(Pd_i, Pd_min, Pd_max)
    Qd_meta_init = _normalize(Qd_i, Qd_min, Qd_max)
    #V_root_meta_init=V_root_normalize(V_root_1,V_root_min,V_root_max)

    #withmodel.BUSCorrespond
    def Pd_meta_init_bus(model, i):
        idx = bus_ids.index(i)
        return Pd_meta_init[idx]
    def Qd_meta_init_bus(model, i):
        idx = bus_ids.index(i)
        return Qd_meta_init[idx]
                                                       #The input is a list
    model.Pd_meta = pyo.Param(model.BUS,initialize=Pd_meta_init_bus, mutable=True, domain=pyo.Reals)
    model.Qd_meta = pyo.Param(model.BUS,initialize=Qd_meta_init_bus, mutable=True, domain=pyo.Reals)
    #model.V_root_meta = pyo.Param(initialize=V_root_meta_init, mutable=True, domain=pyo.Reals)

    #denormalization expression
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

    # variable definition
    model.V2 = pyo.Var(model.BUS, within=pyo.NonNegativeReals)  # Phase dimensions removed  #Because it is a single phase system
    model.Pf = pyo.Var(model.LINE, within=pyo.Reals)  # single phase power flow
    model.Qf = pyo.Var(model.LINE, within=pyo.Reals)
    model.I2 = pyo.Var(model.LINE, within=pyo.NonNegativeReals)  # Single phase current
    model.Pn = pyo.Var(model.BUS, within=pyo.Reals)  # Node injection power
    model.Qn = pyo.Var(model.BUS, within=pyo.Reals)
    model.var_proj = pyo.Var(range(2), within=pyo.Reals)  # Aggregate active/reactive power at the root bus.

    model.constraints = pyo.ConstraintList()  # Constraints are added dynamically below.
    # Voltage constraints (pu^2)
    V2max = {i: bus_data[bus_ids.index(i), 11]**2 for i in model.BUS}  # Squared upper voltage limits by bus.
    V2min = {i: bus_data[bus_ids.index(i), 12]**2 for i in model.BUS}
    # voltage boundary
    for i in model.BUS:
        model.constraints.add(model.V2[i] >= V2min[i])
        model.constraints.add(model.V2[i] <= V2max[i])

    # Single phase voltage drop equation (Simplify)
    for l in model.LINE:
        i = model.from_bus[l]
        j = model.to_bus[l]
        # Single-phase version removes phase cycling and phase-to-phase coupling terms
        lin_loss = 2 * (model.R[l] * model.Pf[l] + model.X[l] * model.Qf[l])
        quad_loss = (model.R[l] ** 2 + model.X[l] ** 2) * model.I2[l]
        model.constraints.add(model.V2[j] == model.V2[i] - lin_loss + quad_loss)   # Vj² = Vi² - 2*(R*P + X*Q) + (R²+X²)*I²

    # current-power relationship
    for l in model.LINE:
        i = model.from_bus[l]
        # model.constraints.add(model.I2[l] * model.V2[i] >= model.Pf[l] ** 2 + model.Qf[l] ** 2)
        # model.constraints.add(model.I2[l] * model.V2[i] <= model.Pf[l] ** 2 + model.Qf[l] ** 2)
        model.constraints.add(model.I2[l] * model.V2[i] == model.Pf[l] ** 2 + model.Qf[l] ** 2)  #S² = P² + Q² = V² * I²


    # Power balance (single phase)
    for n in model.BUS:
        if n != 1:  # non-root node
            inflow_P = sum(model.Pf[l] for l in model.LINE if model.to_bus[l] == n)
            loss_P = sum(model.R[l] * model.I2[l] for l in model.LINE if model.to_bus[l] == n)
            outflow_P = sum(model.Pf[l] for l in model.LINE if model.from_bus[l] == n)
            model.constraints.add(inflow_P - loss_P - outflow_P + model.Pn[n] == 0.0)  #Inflow power - Line loss - Outflow power + Node injection power = 0

            inflow_Q = sum(model.Qf[l] for l in model.LINE if model.to_bus[l] == n)
            loss_Q = sum(model.X[l] * model.I2[l] for l in model.LINE if model.to_bus[l] == n)
            outflow_Q = sum(model.Qf[l] for l in model.LINE if model.from_bus[l] == n)
            model.constraints.add(inflow_Q - loss_Q - outflow_Q + model.Qn[n] == 0.0)

            # Flexible load handling
            node_flex_info = node_flex_dict.get(n, 0)  # Use the fixed-load default for unspecified buses.
            if not node_flex_info['type']:      # fixed load:Injection power = -Load power
                model.constraints.add(model.Pn[n] == -model.Pd[n])
                model.constraints.add(model.Qn[n] == -model.Qd[n])
            elif node_flex_info['type'] == 1:   # Active adjustable load
                model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'] * abs(model.Pd[n]))
                model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'] * abs(model.Pd[n]))
                model.constraints.add(model.Qn[n] == -model.Qd[n])
            elif node_flex_info['type'] == 2:   # Adjustable load for both active and reactive power
                model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'][0] * abs(model.Pd[n]))
                model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'][0] * abs(model.Pd[n]))
                model.constraints.add(model.Qn[n] <= -model.Qd[n] + node_flex_info['rate'][1] * abs(model.Qd[n]))
                model.constraints.add(model.Qn[n] >= -model.Qd[n] - node_flex_info['rate'][1] * abs(model.Qd[n]))
            # elif node_flex_info['type'] == 3:   # Active power is adjustable, andP^2+Q^2<=set value
            #     model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'] * abs(model.Pd[n]))
            #     model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'] * abs(model.Pd[n]))
            #     model.constraints.add(model.Pn[n]**2 + model.Qn[n]**2 <= 0.25*casedata['max_P']/baseMVA)
            elif node_flex_info['type'] == 3:
                # model.constraints.add(model.Pn[n] <= -model.Pd[n] + node_flex_info['rate'] * abs(model.Pd[n]))
                # model.constraints.add(model.Pn[n] >= -model.Pd[n] - node_flex_info['rate'] * abs(model.Pd[n]))
                model.constraints.add((model.Pn[n]+model.Pd[n])**2 + (model.Qn[n]+model.Qd[n])**2 <= 0.3*(model.Pd[n]**2+model.Qd[n]**2))



    # Balance node constraints
    outflow_P = sum(model.Pf[l] for l in model.LINE if model.from_bus[l] == 1)
    outflow_Q = sum(model.Qf[l] for l in model.LINE if model.from_bus[l] == 1)
    model.constraints.add(model.Pn[1] - outflow_P == model.Pd[1])  # Root-bus active-power balance.
    model.constraints.add(model.Qn[1] - outflow_Q == model.Qd[1])
    model.constraints.add(model.V2[bus_ids[0]] == model.V_root ** 2)  # Reference voltage

    # Aggregate variable update
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
        self.shapes = []  # Store all graphics objects uniformly
        self.ax.grid(False)  # Do not show grid

        self.error_history = {
            'iterations': [],
            'error_feas': [],
            'error_opt': []
        }
    def _add_shape(self, patch, shape_type, ** kwargs):
        """Unifiedly add graphics to the storage list"""
        shape_id = len(self.shapes)
        shape_info = {
            'patch': patch,
            'shape': shape_type,
            'id': shape_id,
            ** kwargs  # Store other custom properties
        }
        self.shapes.append(shape_info)
        return shape_id
    def plot_polygon(self, xlim, ylim, alpha=0.2, edgecolor='blue',
                     facecolor='blue', label=None, title=None, A=None, b=None, x_org=None):
        """Draw polygon"""  #Change the input judgment
        #if A is not None and b is not None and len(A) > 0 and len(b) > 0:
        # Intersection point unknown
        if A is not None and b is not None:
            # Calculate intersection point
            vertices = []
            for i in range(len(A)):  #len(A)=Anumber of rows
                for j in range(i + 1, len(A)):  # Each constraint pair may define one vertex.
                    try:
                        x, y = np.linalg.solve(np.array([A[i], A[j]]), np.array([b[i], b[j]]))
                        if np.all(A @ np.array([x, y]) <= b + 1e-5):  # Keep feasible intersections only.
                            vertices.append((x, y))
                    except np.linalg.LinAlgError:  #When two straight lines are parallel or there is no solution (the coefficient matrix is singular), the inequality pair is skipped
                        pass

            if not vertices:
                return None
        else: # The intersection point is known
            vertices = x_org

        # Vertex sorting
        center = np.mean(vertices, axis=0)
        angles = np.arctan2([v[1] - center[1] for v in vertices],
                            [v[0] - center[0] for v in vertices])
        sorted_vertices = np.array([v for _, v in sorted(zip(angles, vertices))])  #sorted()sort, zipPack the corresponding elements into tuples

        # Create and add polygons
        polygon = Polygon(sorted_vertices, closed=True, alpha=alpha,
                          edgecolor=edgecolor, facecolor=facecolor, label=label)
        patch = self.ax.add_patch(polygon)

        # Unified storage
        shape_id = self._add_shape(
            patch=patch,
            shape_type='polygon',
            vertices=sorted_vertices,
            alpha=alpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            label=label
        )

        # Set coordinate range and title
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        # # Remove all borders and grids
        # self.ax.set_xticks([])
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
        """Save graphics to file (SVGFormat) """
        if show_legend and any(shape.get('label') for shape in self.shapes):
            self.ax.legend(  # ax.legend()Function to add a legend to the chart
                loc='upper right',  #legend location
                bbox_to_anchor=(1, 1),  # Place the legend outside the axes.
                frameon=False  # Borderless legend
            )

        # Make sure to save asSVGFormat
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
        print(f"Graphic saved to {filename}")
