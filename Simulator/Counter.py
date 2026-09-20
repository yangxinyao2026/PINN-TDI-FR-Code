from pyomo.environ import *
import numpy as np
import time



# class PolyHausdorffCalculator:
#     """
#     Calculate the distance between two polyhedraHausdorffDistance and its sensitivity to constraint parameters
#
#     Properties:
#         model: Pyomomodel object
#         solver_name: Solver name to use
#         M: BigMconstant
#         delta_A: AThe perturbation amplitude of the matrix
#         delta_b: bvector perturbation amplitude
#     """
#
#     def __init__(self, A_target, b_target, solver_name='gurobi', M=1e3, delta_A=1e-5, delta_b=1e-3):
#         """
#         initializationHausdorffcalculator
#
#         parameters:
#             solver_name (str): Solver name (Default: 'gurobi')
#             M (float): BigMconstant (Default: 1e3)
#             delta_A (float): AThe perturbation amplitude of the matrix (Default: 1e-5)
#             delta_b (float): bvector perturbation amplitude (Default: 1e-3)
#         """
#         self.solver_name = solver_name
#         self.M = M
#         self.delta_A = delta_A
#         self.delta_b = delta_b
#         self.create_model(A_target, b_target)
#
#     def create_model(self, C, d):
#         """
#         Create the Hausdorff-distance model.
#
#         parameters:
#             C (np.array): Constraint matrix for fixed polyhedron (k×n)
#             d (np.array): Constraint vectors for fixed polyhedron (k)
#             n (int): variable dimension
#
#         Return:
#             ConcreteModel: CreatedPyomomodel
#         """
#         k = C.shape[0]
#         n = C.shape[1]
#
#         model = ConcreteModel()
#
#         # willAandbDefine as parameter (make variable, mutable=True)
#         model.A = Param(range(k), range(n), mutable=True, initialize=0.0)
#         model.b = Param(range(k), mutable=True, initialize=0.0)
#
#         # Define variables
#         model.x = Var(range(n), domain=Reals)  # Points in a variable polyhedron
#         model.y = Var(range(n), domain=Reals)  # Fixed points in a polyhedron
#         model.lambda_AtoC = Var(range(k), domain=NonNegativeReals)  # Lagrange multiplier
#         model.z_AtoC = Var(range(k), domain=Binary)  # Binary variables for complementary conditions
#         model.lambda_CtoA = Var(range(k), domain=NonNegativeReals)  # Lagrange multiplier
#         model.z_CtoA = Var(range(k), domain=Binary)  # Binary variables for complementary conditions
#
#         # Objective: maximize squared distance.
#         def objective_rule(model):
#             return sum((model.x[i] - model.y[i]) ** 2 for i in range(n))
#
#         model.obj = Objective(rule=objective_rule, sense=maximize)
#
#         # Constraints
#         # 1. Ax ≤ b
#         def constraint_Ax_rule(model, i):
#             return sum(model.A[i, j] * model.x[j] for j in range(n)) <= model.b[i]
#
#         model.constr_Ax = Constraint(range(k), rule=constraint_Ax_rule)
#
#         # 2. Cy ≤ d (fixed constraints)
#         def constraint_Cy_rule(model, i):
#             return sum(C[i, j] * model.y[j] for j in range(n)) <= d[i]
#
#         model.constr_Cy = Constraint(range(k), rule=constraint_Cy_rule)
#
#         # 3. y - x + C^T λ = 0
#         def constraint_eq_rule_AtoC(model, j):
#             return model.y[j] - model.x[j] + sum(C[i, j] * model.lambda_AtoC[i] for i in range(k)) == 0
#
#         model.constr_eq_AtoC = Constraint(range(n), rule=constraint_eq_rule_AtoC)
#
#         # 4. The complementary relaxation condition is largeMlegal treatment
#         # 4a: λ_i ≤ z_i * M
#         def constraint_lambda_rule_AtoC(model, i):
#             return model.lambda_AtoC[i] <= model.z_AtoC[i] * self.M
#
#         model.constr_lambda_AtoC = Constraint(range(k), rule=constraint_lambda_rule_AtoC)
#
#         # 4b: d_i - C_i y ≤ (1-z_i)*M
#         def constraint_comp_rule_AtoC(model, i):
#             return d[i] - sum(C[i, j] * model.y[j] for j in range(n)) <= (1 - model.z_AtoC[i]) * self.M
#
#         model.constr_comp_AtoC = Constraint(range(k), rule=constraint_comp_rule_AtoC)
#
#         # 5. x - y + A^T λ = 0
#         def constraint_eq_rule_CtoA(model, j):
#             return model.x[j] - model.y[j] + sum(model.A[i, j] * model.lambda_CtoA[i] for i in range(k)) == 0
#
#         model.constr_eq_CtoA = Constraint(range(n), rule=constraint_eq_rule_CtoA)
#
#         # 6. The complementary relaxation condition is largeMlegal treatment
#         # 6a: λ_i ≤ z_i * M
#         def constraint_lambda_rule_CtoA(model, i):
#             return model.lambda_CtoA[i] <= model.z_CtoA[i] * self.M
#
#         model.constr_lambda_CtoA = Constraint(range(k), rule=constraint_lambda_rule_CtoA)
#
#         # 6b: b_i - A_i x ≤ (1-z_i)*M
#         def constraint_comp_rule_CtoA(model, i):
#             return model.b[i] - sum(model.A[i, j] * model.x[j] for j in range(n)) <= (1 - model.z_CtoA[i]) * self.M
#
#         model.constr_comp_CtoA = Constraint(range(k), rule=constraint_comp_rule_CtoA)
#
#         self.model = model
#         return
#
#     def solve(self):
#         """
#         SolveHausdorffdistance model
#
#         Return:
#             tuple: (Distance squared, Solve success status)
#         """
#         if self.model is None:
#             raise ValueError("The model has not been created yet, please call firstcreate_modelmethod")
#
#         solver = SolverFactory(self.solver_name)
#
#         # Solve firstAArriveCdirection
#         self.model.constr_eq_CtoA.deactivate()
#         self.model.constr_lambda_CtoA.deactivate()
#         self.model.constr_comp_CtoA.deactivate()
#
#         self.model.constr_eq_AtoC.activate()
#         self.model.constr_lambda_AtoC.activate()
#         self.model.constr_comp_AtoC.activate()
#
#         results = solver.solve(self.model)
#
#         distance_sq = []
#         if results.solver.termination_condition == TerminationCondition.optimal:
#             distance_sq.append(value(self.model.obj()))
#         else:
#             return None, False
#
#         # Solve againCArriveAdirection
#         self.model.constr_eq_AtoC.deactivate()
#         self.model.constr_lambda_AtoC.deactivate()
#         self.model.constr_comp_AtoC.deactivate()
#
#         self.model.constr_eq_CtoA.activate()
#         self.model.constr_lambda_CtoA.activate()
#         self.model.constr_comp_CtoA.activate()
#
#         results = solver.solve(self.model)
#
#         if results.solver.termination_condition == TerminationCondition.optimal:
#             distance_sq.append(value(self.model.obj()))
#         else:
#             return None, False
#
#         return max(distance_sq), True
#
#     def compute_sensitivity(self, A, b):
#         """
#         CalculateHausdorffdistance and its pairAandbsensitivity
#
#         parameters:
#             A (np.array): Constraint matrix for variable polyhedron (k×n)
#             b (np.array): Constraint vectors for variable polyhedron (k)
#             C (np.array): Constraint matrix for fixed polyhedron (k×n)
#             d (np.array): Constraint vectors for fixed polyhedron (k)
#
#         Return:
#             dict: Results containing distances, sensitivity matrices, etc.
#         """
#         k, n = A.shape
#         start_time = time.time()
#
#         # settingsAandbinitial value
#         for i in range(k):
#             self.model.b[i] = b[i]
#             for j in range(n):
#                 self.model.A[i, j] = A[i, j]
#
#         # Solving for the base case
#         base_distance_sq, success = self.solve()
#         if not success:
#             return {'distance': None, 'sensitivity_A': None, 'sensitivity_b': None,
#                     'solver_status': 'solve_failed'}
#
#         base_distance = np.sqrt(base_distance_sq)
#         solve_time = time.time() - start_time
#
#         print(f"Benchmark solution completed, distance: {base_distance:.6f}, Time consuming: {solve_time:.2f}seconds")
#
#         # Perturbation method to calculate sensitivity
#         sensitivity_A = np.zeros_like(A, dtype=float)
#         sensitivity_b = np.zeros_like(b, dtype=float)
#
#         # YesAperturb each element of
#         perturb_start = time.time()
#         for i in range(k):
#             original_A_i = np.array([self.model.A[i, jx].value for jx in range(n)])
#             for j in range(n):
#                 # Forward perturbation and normalization
#                 self.model.A[i, j] = original_A_i[j] + self.delta_A
#                 row_norm = np.sqrt(
#                     np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model.A[i, j]) ** 2)
#                 for jx in range(n):
#                     self.model.A[i, jx] = self.model.A[i, jx] / row_norm if row_norm > 1e-10 else 0
#                 pos_distance_sq, _ = self.solve()
#
#                 # negative perturbation
#                 self.model.A[i, j] = original_A_i[j] - self.delta_A
#                 row_norm = np.sqrt(
#                     np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model.A[i, j]) ** 2)
#                 for jx in range(n):
#                     self.model.A[i, jx] = self.model.A[i, jx] / row_norm if row_norm > 1e-10 else 0
#                 neg_distance_sq, _ = self.solve()
#
#                 # Restore original value
#                 for jx in range(n):
#                     self.model.A[i, jx] = original_A_i[jx]
#
#                 if pos_distance_sq is not None and neg_distance_sq is not None:
#                     pos_distance = np.sqrt(pos_distance_sq)
#                     neg_distance = np.sqrt(neg_distance_sq)
#                     if pos_distance > base_distance and neg_distance > base_distance:
#                         sensitivity_A[i, j] = 0.0
#                     else:
#                         sensitivity_A[i, j] = (pos_distance - neg_distance) / (2 * self.delta_A)
#                 else:
#                     sensitivity_A[i, j] = np.nan
#
#         # Yesbperturb each element of
#         for i in range(k):
#             # Back up the original value
#             original_b_i = self.model.b[i].value
#
#             # forward perturbation
#             self.model.b[i] = original_b_i + self.delta_b
#             pos_distance_sq, _ = self.solve()
#
#             # negative perturbation
#             self.model.b[i] = original_b_i - self.delta_b
#             neg_distance_sq, _ = self.solve()
#
#             # Restore original value
#             self.model.b[i] = original_b_i
#
#             if pos_distance_sq is not None and neg_distance_sq is not None:
#                 pos_distance = np.sqrt(pos_distance_sq)
#                 neg_distance = np.sqrt(neg_distance_sq)
#                 if pos_distance > base_distance and neg_distance > base_distance:
#                     sensitivity_b[i] = 0.0
#                 else:
#                     sensitivity_b[i] = (pos_distance - neg_distance) / (2 * self.delta_b)
#             else:
#                 sensitivity_b[i] = np.nan
#
#         total_time = time.time() - start_time
#         perturb_time = total_time - solve_time
#
#         print(f"Sensitivity calculation completed in {total_time:.2f} seconds")
#
#         return {
#             'distance': base_distance,
#             'sensitivity_A': sensitivity_A,
#             'sensitivity_b': sensitivity_b,
#             'solve_time': solve_time,
#             'perturb_time': perturb_time,
#             'total_time': total_time,
#             'solver_status': 'success'
#         }
# # Example usage
# if __name__ == "__main__":
#     # Define problem data
#     C = np.array([[1, 0], [0, 1], [-1, 0], [0, -1],
#                   [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
#     d = np.array([1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5], dtype=float)
#
#     A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
#     b = np.array([1.0, 1.0, 1, 1], dtype=float) - 0.50 / (np.sqrt(2) + 2)
#
#     # Create a calculator instance
#     calculator = PolyHausdorffCalculator(
#         solver_name='gurobi',  # UseGurobisolver
#         A_target = C, b_target = d,
#         delta_A=1e-4,  # AThe amplitude of the disturbance
#         delta_b=1e-4  # bThe amplitude of the disturbance
#     )
#
#     # CalculateHausdorffDistance and sensitivity
#     result = calculator.compute_sensitivity(A, b)
#
#     # Output results
#     print("\nCalculation result:")
#     print(f"Hausdorffdistance: {result['distance']:.6f}")
#     print("\nYesbsensitivity:")
#     print(result['sensitivity_b'])
#     print("\nYesAThe sensitivity matrix of:")
#     print(result['sensitivity_A'])
#     print(f"\nTotal calculation time: {result['total_time']:.2f} seconds")



class PolyBallHausdorffCalculator:
    """
    Calculate the distance between the polyhedron and the unit sphereHausdorffdistance

    Properties:
        model: Pyomomodel object
        solver_name: Solver name to use
        M: BigMconstant
        delta_A: AThe perturbation amplitude of the matrix
        delta_b: bvector perturbation amplitude
    """

    def __init__(self, A,b, R = 1.0, solver_name='gurobi', M=1e3, delta_A=1e-5, delta_b=1e-3):
        """
        Initialize the calculator

        parameters:
            solver_name (str): Solver name (Default: 'gurobi')
            M (float): BigMconstant (Default: 1e3)
            delta_A (float): AThe perturbation amplitude of the matrix (Default: 1e-5)
            delta_b (float): bvector perturbation amplitude (Default: 1e-3)
        """
        self.solver_name = solver_name

        self.ncons, self.dim = A.shape
        self.A = A
        self.b = b
        self.R = R
        self.M = M
        self.delta_A = delta_A
        self.delta_b = delta_b

    def create_model(self):
        """
        createHausdorffdistance calculation model

        parameters:
            A (np.array): Constraint matrix for polyhedron (k×n)
            b (np.array): Constraint vector of a polyhedron (k)
        """
        k, n = self.ncons, self.dim
        model_P2B = ConcreteModel()

        # willAandbDefine as parameter (make variable, mutable=True)
        model_P2B.A = Param(range(k), range(n), mutable=True, initialize=0.0)
        model_P2B.b = Param(range(k), mutable=True, initialize=0.0)

        # Define variables
        model_P2B.x = Var(range(n), domain=Reals)  # Points in a polyhedron

        # ==================== Part One: Polyhedron to Unit Sphere ====================
        # target: max ||x||^2 (The center of the sphere is at the origin)
        model_P2B.obj = Objective(
            expr=sum(model_P2B.x[i] ** 2 for i in range(n)),
            sense=maximize
        )

        # polyhedral constraints: Ax <= b
        def poly_constraint_rule(model, i):
            return sum( model_P2B.A[i, j] * model.x[j] for j in range(n)) <=  model_P2B.b[i]

        model_P2B.poly_constr = Constraint(range(k), rule=poly_constraint_rule)

        # ==================== Part 2: Unit sphere to polyhedron ====================
        #Theoretical calculations, no optimization required
        self.model_P2B = model_P2B


    def solve(self):
        """
        SolveHausdorffdistance

        Return:
            tuple: (distance value, Solution status)
        """
        solver = SolverFactory(self.solver_name)

        # ===== Part One: Polyhedron to Unit Sphere =====
        results = solver.solve(self.model_P2B,tee=False)
        if results.solver.termination_condition != TerminationCondition.optimal:
            return None, False

        # Calculate actual distance (||x|| - R, The minimum is0)
        d1 = max(0.0,np.sqrt(value(self.model_P2B.obj))-self.R)

        # ===== Part 2: Unit sphere to polyhedron =====
        row_norms = np.linalg.norm(self.A, axis=1)
        # The processing norm is0rows (avoid dividing by0)
        zero_norm_mask = row_norms == 0
        row_norms[zero_norm_mask] = 1.0 # If the norm is0, The scaling factor is set to1 (Keep original value)
        # normalized matrixA: Divide each row by the corresponding norm
        A_norm = self.A / row_norms[:, np.newaxis]  # Passnp.newaxisKeep dimensions aligned
        # normalized vectorb: Divide each element by the norm of the corresponding row
        b_norm = self.b / row_norms
        # print(b_norm)
        d2 = np.max([0,np.max(self.R-b_norm)])


        # eventuallyHausdorffdistance
        # hausdorff_dist = max(d1, d2)
        hausdorff_dist = d1**2+d2**2 #true objective function
        return hausdorff_dist, {'feas':d1, 'opt':d2}, True

    def compute_hausdorff(self, sensitivity = False):
        """
        CalculateHausdorffdistance and its pairAandbsensitivity

        parameters:
            A (np.array): Constraint matrix for variable polyhedron (k×n)
            b (np.array): Constraint vectors for variable polyhedron (k)

        Return:
            dict: Results containing distances, sensitivity matrices, etc.
        """
        A = self.A
        b = self.b
        k, n = A.shape
        start_time = time.time()

        # settingsAandbinitial value
        for i in range(k):
            self.model_P2B.b[i] = b[i]
            for j in range(n):
                self.model_P2B.A[i, j] = A[i, j]
        # Solving for the base case
        base_distance, single_distances, success = self.solve()
        if not success:
            return {'distance': None, 'sensitivity_A': None, 'sensitivity_b': None,
                    'solver_status': 'solve_failed'}

        solve_time = time.time() - start_time

        print(f"Benchmark solution completed, distance: {base_distance:.6f}, Time consuming: {solve_time:.2f}seconds")
        sensitivity_A = None
        sensitivity_b = None
        perturb_time = None
        total_time = solve_time
        if sensitivity:
            # Perturbation method to calculate sensitivity
            sensitivity_A = np.zeros_like(A, dtype=float)
            sensitivity_b = np.zeros_like(b, dtype=float)

            # YesAperturb each element of
            perturb_start = time.time()
            for i in range(k):
                original_A_i = np.array([self.model_P2B.A[i, jx].value for jx in range(n)])
                for j in range(n):
                    # Forward perturbation and normalization
                    self.model_P2B.A[i, j] = original_A_i[j] + self.delta_A

                    row_norm = np.sqrt(
                        np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model_P2B.A[i, j]) ** 2)
                    for jx in range(n):
                        self.model_P2B.A[i, jx] = self.model_P2B.A[i, jx] / row_norm if row_norm > 1e-10 else 0
                    pos_distance, _, _ = self.solve()

                    # negative perturbation
                    self.model_P2B.A[i, j] = original_A_i[j] - self.delta_A
                    row_norm = np.sqrt(
                        np.linalg.norm(original_A_i) ** 2 - original_A_i[j] ** 2 + value(self.model_P2B.A[i, j]) ** 2)
                    for jx in range(n):
                        self.model_P2B.A[i, jx] = self.model_P2B.A[i, jx] / row_norm if row_norm > 1e-10 else 0
                    neg_distance, _, _ = self.solve()

                    # Restore original value
                    self.model_P2B.A[i, j] = original_A_i[j]

                    if pos_distance is not None and neg_distance is not None:
                        if pos_distance > base_distance and neg_distance > base_distance:
                            sensitivity_A[i, j] = 0.0
                        else:
                            sensitivity_A[i, j] = (pos_distance - neg_distance) / (2 * self.delta_A)
                    else:
                        sensitivity_A[i, j] = np.nan

            # Yesbperturb each element of
            for i in range(k):
                # Back up the original value
                original_b_i = self.model_P2B.b[i].value

                # forward perturbation
                self.model_P2B.b[i] = original_b_i + self.delta_b
                pos_distance, _, _ = self.solve()

                # negative perturbation
                self.model_P2B.b[i] = original_b_i - self.delta_b
                neg_distance, _, _ = self.solve()

                # Restore original value
                self.model_P2B.b[i] = original_b_i

                if pos_distance is not None and neg_distance is not None:
                    if pos_distance > base_distance and neg_distance > base_distance:
                        sensitivity_b[i] = 0.0
                    else:
                        sensitivity_b[i] = (pos_distance - neg_distance) / (2 * self.delta_b)
                else:
                    sensitivity_b[i] = np.nan

            total_time = time.time() - start_time
            perturb_time = total_time - solve_time

            print(f"Sensitivity calculation completed, total time taken: {total_time:.2f}seconds (baseline solution: {solve_time:.2f}seconds, perturbation solution: {perturb_time:.2f}seconds) ")

        return {
            'distance': base_distance,
            'single_distances':single_distances,
            # 'distance_ideal': (np.sqrt(self.dim)-1)/(np.sqrt(self.dim)+1),
            'distance_ideal': (np.sqrt(self.dim)-1)**2/(self.dim+1),
            'sensitivity_A': sensitivity_A,
            'sensitivity_b': sensitivity_b,
            'solve_time': solve_time,
            'perturb_time': perturb_time,
            'total_time': total_time,
            'solver_status': 'success'
        }


# Example usage
if __name__ == "__main__":
    # Define polyhedron (square [-1,1]×[-1,1])
    dim = 50
    A = np.vstack([np.eye(dim),-np.eye(dim)])
    b = 2.0/(np.sqrt(dim)+1)*np.array(np.ones(2*dim),dtype=float)

    # Create a calculator
    calculator = PolyBallHausdorffCalculator(
        A = A,
        b = b,
        R = 1.0,
        solver_name='gurobi',  # It is recommended to use a solver that supports second-order cones
        M=1e3,
        delta_A=1e-4,
        delta_b=1e-4
    )

    # Calculate distance
    calculator.create_model()
    res = calculator.compute_hausdorff()
    print(f"Hausdorffdistance: {res['distance']:.4f} (Calculation status: {'success' if res['solver_status'] else 'failed'})")
    print(f"theoretical distance:{res['distance_ideal']:.4f}")
    # # sensitivity analysis (Optional)
    # sensitivity = calculator.compute_hausdorff()
    # print("\nSensitivity analysis results:")
    # print(f"Yesbsensitivity: {sensitivity['sensitivity_b']}")
    # print(f"Sensitivity matrix for A:\n{sensitivity['sensitivity_A']}")
