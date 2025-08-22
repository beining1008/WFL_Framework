import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint, fsolve
import math

class KKTSolverInterface:
    def solve_kkt_conditions(self, objective_function, constraints, variables):
        raise NotImplementedError

    def verify_kkt_optimality(self, solution, lagrange_multipliers):
        raise NotImplementedError

class BaseOptimalityConditionSolver(KKTSolverInterface):
    def __init__(self, solver_parameters):
        self.tolerance = solver_parameters.get('tolerance', 1e-8)
        self.max_iterations = solver_parameters.get('max_iterations', 1000)
        self.numerical_epsilon = solver_parameters.get('numerical_epsilon', 1e-12)

class KarushKuhnTuckerSolver(BaseOptimalityConditionSolver):
    def __init__(self, solver_parameters):
        super().__init__(solver_parameters)
        self.gradient_tolerance = 1e-10
        self.complementary_slackness_tolerance = 1e-10
        
    def compute_lagrangian_gradient(self, variables, lambda_multipliers, objective_grad_func, constraint_grad_funcs):
        x = variables
        lambdas = lambda_multipliers
        
        objective_gradient = objective_grad_func(x)
        
        constraint_gradient_sum = np.zeros_like(objective_gradient)
        
        for i, (constraint_grad_func, lambda_i) in enumerate(zip(constraint_grad_funcs, lambdas)):
            constraint_gradient = constraint_grad_func(x)
            constraint_gradient_sum += lambda_i * constraint_gradient
        
        lagrangian_gradient = objective_gradient + constraint_gradient_sum
        
        return lagrangian_gradient
    
    def check_stationarity_condition(self, variables, lambda_multipliers, objective_grad_func, constraint_grad_funcs):
        lagrangian_grad = self.compute_lagrangian_gradient(
            variables, lambda_multipliers, objective_grad_func, constraint_grad_funcs
        )
        
        stationarity_violation = np.linalg.norm(lagrangian_grad)
        
        return stationarity_violation < self.gradient_tolerance
    
    def check_primal_feasibility(self, variables, constraint_funcs):
        constraint_violations = []
        
        for constraint_func in constraint_funcs:
            constraint_value = constraint_func(variables)
            
            if isinstance(constraint_value, (list, np.ndarray)):
                violations = [max(0, cv) for cv in constraint_value]
                constraint_violations.extend(violations)
            else:
                constraint_violations.append(max(0, constraint_value))
        
        max_violation = max(constraint_violations) if constraint_violations else 0
        
        return max_violation < self.tolerance
    
    def check_dual_feasibility(self, lambda_multipliers):
        negative_multipliers = [lam for lam in lambda_multipliers if lam < -self.numerical_epsilon]
        
        return len(negative_multipliers) == 0
    
    def check_complementary_slackness(self, variables, lambda_multipliers, constraint_funcs):
        complementary_violations = []
        
        for i, (constraint_func, lambda_i) in enumerate(zip(constraint_funcs, lambda_multipliers)):
            constraint_value = constraint_func(variables)
            
            if isinstance(constraint_value, (list, np.ndarray)):
                for cv in constraint_value:
                    complementary_product = abs(lambda_i * max(0, cv))
                    complementary_violations.append(complementary_product)
            else:
                complementary_product = abs(lambda_i * max(0, constraint_value))
                complementary_violations.append(complementary_product)
        
        max_complementary_violation = max(complementary_violations) if complementary_violations else 0
        
        return max_complementary_violation < self.complementary_slackness_tolerance
    
    def verify_kkt_optimality(self, solution, lagrange_multipliers, objective_grad_func, constraint_grad_funcs, constraint_funcs):
        variables = solution
        lambdas = lagrange_multipliers
        
        stationarity_satisfied = self.check_stationarity_condition(
            variables, lambdas, objective_grad_func, constraint_grad_funcs
        )
        
        primal_feasibility_satisfied = self.check_primal_feasibility(variables, constraint_funcs)
        
        dual_feasibility_satisfied = self.check_dual_feasibility(lambdas)
        
        complementary_slackness_satisfied = self.check_complementary_slackness(
            variables, lambdas, constraint_funcs
        )
        
        kkt_verification = {
            'stationarity': stationarity_satisfied,
            'primal_feasibility': primal_feasibility_satisfied,
            'dual_feasibility': dual_feasibility_satisfied,
            'complementary_slackness': complementary_slackness_satisfied,
            'all_conditions_satisfied': all([
                stationarity_satisfied,
                primal_feasibility_satisfied,
                dual_feasibility_satisfied,
                complementary_slackness_satisfied
            ])
        }
        
        return kkt_verification

class VehicularResourceOptimizationKKT(KarushKuhnTuckerSolver):
    def __init__(self, system_parameters):
        solver_params = {'tolerance': 1e-8, 'max_iterations': 1000}
        super().__init__(solver_params)
        
        self.mu = system_parameters['mu']
        self.kappa = system_parameters['kappa']
        self.B = system_parameters['B']
        self.P_n = system_parameters['P_n']
        self.e_max = system_parameters['e_max']
        self.G_n = system_parameters['G_n']
        self.zeta_n = system_parameters['zeta_n']
        self.D_n = system_parameters['D_n']
    
    def objective_function(self, variables):
        chi_n, delta_tx_n = variables
        
        computation_delay = (self.mu * self.zeta_n) / (chi_n * self.G_n)
        transmission_delay = delta_tx_n
        
        total_delay = computation_delay + transmission_delay
        
        return total_delay
    
    def objective_gradient(self, variables):
        chi_n, delta_tx_n = variables
        
        grad_chi = -(self.mu * self.zeta_n) / (chi_n ** 2 * self.G_n)
        grad_delta = 1.0
        
        return np.array([grad_chi, grad_delta])
    
    def energy_constraint(self, variables, h_n_squared):
        chi_n, delta_tx_n = variables
        
        computation_energy = self.kappa * self.mu * self.zeta_n * (chi_n * self.G_n) ** 2
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        transmission_energy = delta_tx_n * (exponential_term - 1) / h_n_squared
        
        total_energy = computation_energy + transmission_energy
        constraint_value = total_energy - self.e_max
        
        return constraint_value
    
    def energy_constraint_gradient(self, variables, h_n_squared):
        chi_n, delta_tx_n = variables
        
        grad_chi_energy = 2 * self.kappa * self.mu * self.zeta_n * self.G_n ** 2 * chi_n
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        ln_2 = math.log(2)
        
        exponential_derivative = -(self.D_n * ln_2 / (self.B * delta_tx_n ** 2)) * exponential_term
        
        grad_delta_energy = ((exponential_term - 1) / h_n_squared + 
                           delta_tx_n * exponential_derivative / h_n_squared)
        
        return np.array([grad_chi_energy, grad_delta_energy])
    
    def power_constraint(self, variables, h_n_squared):
        chi_n, delta_tx_n = variables
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        rho_n_equivalent = (exponential_term - 1) / (self.P_n * h_n_squared)
        
        constraint_value = rho_n_equivalent - 1.0
        
        return constraint_value
    
    def power_constraint_gradient(self, variables, h_n_squared):
        chi_n, delta_tx_n = variables
        
        grad_chi_power = 0.0
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        ln_2 = math.log(2)
        
        exponential_derivative = -(self.D_n * ln_2 / (self.B * delta_tx_n ** 2)) * exponential_term
        
        grad_delta_power = exponential_derivative / (self.P_n * h_n_squared)
        
        return np.array([grad_chi_power, grad_delta_power])
    
    def solve_kkt_conditions(self, h_n_squared, initial_guess=None):
        if initial_guess is None:
            initial_guess = [0.5, 0.1]
        
        def objective_wrapper(x):
            return self.objective_function(x)
        
        def energy_constraint_wrapper(x):
            return self.energy_constraint(x, h_n_squared)
        
        def power_constraint_wrapper(x):
            return self.power_constraint(x, h_n_squared)
        
        bounds = [(1e-6, 1.0), (1e-6, 10.0)]
        
        constraints = [
            NonlinearConstraint(energy_constraint_wrapper, -np.inf, 0),
            NonlinearConstraint(power_constraint_wrapper, -np.inf, 0)
        ]
        
        try:
            optimization_result = minimize(
                objective_wrapper,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': self.tolerance, 'maxiter': self.max_iterations}
            )
            
            if optimization_result.success:
                optimal_variables = optimization_result.x
                optimal_objective = optimization_result.fun
                
                lagrange_multipliers = []
                if hasattr(optimization_result, 'v') and optimization_result.v is not None:
                    lagrange_multipliers = optimization_result.v
                else:
                    lagrange_multipliers = [0.0, 0.0]
                
                constraint_funcs = [
                    lambda x: self.energy_constraint(x, h_n_squared),
                    lambda x: self.power_constraint(x, h_n_squared)
                ]
                
                constraint_grad_funcs = [
                    lambda x: self.energy_constraint_gradient(x, h_n_squared),
                    lambda x: self.power_constraint_gradient(x, h_n_squared)
                ]
                
                kkt_verification = self.verify_kkt_optimality(
                    optimal_variables,
                    lagrange_multipliers,
                    self.objective_gradient,
                    constraint_grad_funcs,
                    constraint_funcs
                )
                
                return {
                    'optimal_variables': optimal_variables,
                    'optimal_objective': optimal_objective,
                    'lagrange_multipliers': lagrange_multipliers,
                    'kkt_verification': kkt_verification,
                    'optimization_success': True
                }
            else:
                return self._fallback_kkt_solution(h_n_squared)
                
        except Exception as e:
            return self._fallback_kkt_solution(h_n_squared)
    
    def _fallback_kkt_solution(self, h_n_squared):
        chi_n_fallback = 0.5
        delta_tx_fallback = 0.1
        
        return {
            'optimal_variables': [chi_n_fallback, delta_tx_fallback],
            'optimal_objective': float('inf'),
            'lagrange_multipliers': [0.0, 0.0],
            'kkt_verification': {
                'stationarity': False,
                'primal_feasibility': False,
                'dual_feasibility': True,
                'complementary_slackness': False,
                'all_conditions_satisfied': False
            },
            'optimization_success': False
        }

class MultiVehicleKKTSolver:
    def __init__(self, system_parameters, num_vehicles=20):
        self.num_vehicles = num_vehicles
        self.system_params = system_parameters
        self.vehicle_kkt_solvers = []
        
        for vehicle_id in range(num_vehicles):
            vehicle_params = system_parameters.copy()
            vehicle_params['zeta_n'] = np.random.randint(100, 1000)
            
            kkt_solver = VehicularResourceOptimizationKKT(vehicle_params)
            self.vehicle_kkt_solvers.append(kkt_solver)
    
    def solve_multi_vehicle_kkt_system(self, channel_gains_h_squared):
        kkt_solutions = {}
        
        for vehicle_id in range(self.num_vehicles):
            h_n_squared = channel_gains_h_squared.get(vehicle_id, 1.0)
            
            vehicle_solution = self.vehicle_kkt_solvers[vehicle_id].solve_kkt_conditions(h_n_squared)
            kkt_solutions[vehicle_id] = vehicle_solution
        
        global_kkt_metrics = self._compute_global_kkt_metrics(kkt_solutions)
        
        return {
            'individual_solutions': kkt_solutions,
            'global_metrics': global_kkt_metrics
        }
    
    def _compute_global_kkt_metrics(self, kkt_solutions):
        successful_optimizations = sum(1 for sol in kkt_solutions.values() if sol['optimization_success'])
        
        kkt_satisfied_count = sum(1 for sol in kkt_solutions.values() 
                                if sol['kkt_verification']['all_conditions_satisfied'])
        
        total_objective = sum(sol['optimal_objective'] for sol in kkt_solutions.values() 
                            if sol['optimal_objective'] != float('inf'))
        
        return {
            'success_rate': successful_optimizations / self.num_vehicles,
            'kkt_satisfaction_rate': kkt_satisfied_count / self.num_vehicles,
            'total_system_delay': total_objective,
            'average_delay_per_vehicle': total_objective / self.num_vehicles if self.num_vehicles > 0 else 0
        }

def create_kkt_solver_system():
    SYSTEM_PARAMS = {
        'mu': 1e7,
        'kappa': 1e-28,
        'B': 56e6,
        'P_n': 15,
        'e_max': 0.1,
        'G_n': 0.5e9,
        'D_n': 10.08e6,
        'zeta_n': 500
    }
    
    multi_vehicle_kkt = MultiVehicleKKTSolver(SYSTEM_PARAMS, num_vehicles=20)
    
    return multi_vehicle_kkt
