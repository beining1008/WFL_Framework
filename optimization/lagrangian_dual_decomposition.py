import torch
import numpy as np
import scipy.optimize
from scipy.optimize import minimize, NonlinearConstraint
import math

class OptimizationSolverInterface:
    def solve_optimization_problem(self, objective_function, constraints):
        raise NotImplementedError

class BaseResourceAllocator(OptimizationSolverInterface):
    def __init__(self, system_parameters):
        self.mu = system_parameters['mu']
        self.kappa = system_parameters['kappa']
        self.B = system_parameters['B']
        self.P_n = system_parameters['P_n']
        self.e_max = system_parameters['e_max']
        self.G_n = system_parameters['G_n']
        self.zeta_n = system_parameters.get('zeta_n', 500)
        self.D_n = system_parameters['D_n']

class LagrangianSolver(BaseResourceAllocator):
    def __init__(self, system_parameters):
        super().__init__(system_parameters)
        self.convergence_tolerance = 1e-8
        self.max_iterations = 1000
        self.lambda_dual_variables = {}

class KKTConditionSolver(LagrangianSolver):
    def __init__(self, system_parameters):
        super().__init__(system_parameters)
        self.kkt_tolerance = 1e-10
        
    def compute_delay_minimization_objective(self, chi_n, delta_tx_n, h_n_squared):
        computation_delay = (self.mu * self.zeta_n) / (chi_n * self.G_n)
        transmission_delay = delta_tx_n
        
        total_delay = computation_delay + transmission_delay
        return total_delay
    
    def compute_energy_constraint(self, chi_n, delta_tx_n, h_n_squared):
        computation_energy = self.kappa * self.mu * self.zeta_n * (chi_n * self.G_n) ** 2
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        rho_n_equivalent = (exponential_term - 1) / (self.P_n * h_n_squared)
        
        if rho_n_equivalent > 1.0:
            rho_n_equivalent = 1.0
        
        transmission_energy = rho_n_equivalent * self.P_n * delta_tx_n
        
        total_energy = computation_energy + transmission_energy
        return total_energy
    
    def compute_lagrangian_function(self, variables, lambda_vec, h_n_squared):
        chi_n, delta_tx_n = variables
        
        objective = self.compute_delay_minimization_objective(chi_n, delta_tx_n, h_n_squared)
        
        energy_constraint_value = self.compute_energy_constraint(chi_n, delta_tx_n, h_n_squared) - self.e_max
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        rho_constraint_value = (exponential_term - 1) / (self.P_n * h_n_squared) - 1.0
        
        lagrangian = (objective + 
                     lambda_vec[0] * energy_constraint_value + 
                     lambda_vec[1] * max(0, rho_constraint_value) - 
                     lambda_vec[2] * chi_n + 
                     lambda_vec[3] * (chi_n - 1))
        
        return lagrangian
    
    def compute_kkt_gradients(self, variables, lambda_vec, h_n_squared):
        chi_n, delta_tx_n = variables
        
        grad_chi = (-self.mu * self.zeta_n / (chi_n ** 2 * self.G_n) + 
                   2 * lambda_vec[0] * self.kappa * self.mu * self.zeta_n * self.G_n ** 2 * chi_n - 
                   lambda_vec[2] + lambda_vec[3])
        
        exponential_term = 2 ** (self.D_n / (delta_tx_n * self.B))
        ln_2 = math.log(2)
        
        exponential_derivative = -(self.D_n * ln_2 / (self.B * delta_tx_n ** 2)) * exponential_term
        
        grad_delta = (1 + 
                     lambda_vec[0] * ((exponential_term - 1) / h_n_squared + 
                                    delta_tx_n * exponential_derivative / h_n_squared) - 
                     lambda_vec[1] * exponential_derivative / (self.P_n * h_n_squared))
        
        return np.array([grad_chi, grad_delta])
    
    def solve_unconstrained_case(self, h_n_squared):
        chi_n_optimal = 1.0
        
        delta_tx_optimal = self.D_n / (self.B * math.log2(1 + self.P_n * h_n_squared))
        
        energy_check = self.compute_energy_constraint(chi_n_optimal, delta_tx_optimal, h_n_squared)
        
        if energy_check <= self.e_max:
            return chi_n_optimal, delta_tx_optimal, True
        else:
            return None, None, False
    
    def solve_constrained_case(self, h_n_squared):
        def energy_constraint_equation(variables):
            chi_n, delta_tx_n = variables
            return self.compute_energy_constraint(chi_n, delta_tx_n, h_n_squared) - self.e_max
        
        def objective_function(variables):
            chi_n, delta_tx_n = variables
            return self.compute_delay_minimization_objective(chi_n, delta_tx_n, h_n_squared)
        
        initial_guess = [0.5, self.D_n / (self.B * math.log2(1 + 0.5 * self.P_n * h_n_squared))]
        
        bounds = [(1e-6, 1.0), (1e-6, 10.0)]
        
        constraint = NonlinearConstraint(energy_constraint_equation, -1e-6, 1e-6)
        
        try:
            result = minimize(
                objective_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                options={'ftol': self.convergence_tolerance, 'maxiter': self.max_iterations}
            )
            
            if result.success:
                chi_n_optimal, delta_tx_optimal = result.x
                return chi_n_optimal, delta_tx_optimal, True
            else:
                return self._fallback_analytical_solution(h_n_squared)
        except:
            return self._fallback_analytical_solution(h_n_squared)
    
    def _fallback_analytical_solution(self, h_n_squared):
        lambda_1_guess = 1e-6
        
        for iteration in range(100):
            chi_n_candidate = min((1 / (2 * lambda_1_guess * self.kappa * self.G_n ** 3)) ** (1/3), 1.0)
            
            def delta_equation(delta_tx):
                exponential_term = 2 ** (self.D_n / (delta_tx * self.B))
                energy_left = self.e_max - self.kappa * self.mu * self.zeta_n * (chi_n_candidate * self.G_n) ** 2
                energy_right = delta_tx * (exponential_term - 1) / h_n_squared
                return energy_left - energy_right
            
            try:
                from scipy.optimize import fsolve
                delta_tx_candidate = fsolve(delta_equation, 0.1)[0]
                
                if delta_tx_candidate > 0:
                    energy_check = self.compute_energy_constraint(chi_n_candidate, delta_tx_candidate, h_n_squared)
                    if abs(energy_check - self.e_max) < 1e-6:
                        return chi_n_candidate, delta_tx_candidate, True
                
                lambda_1_guess *= 1.1
            except:
                lambda_1_guess *= 1.5
        
        return 0.5, 0.1, False
    
    def solve_optimization_problem(self, h_n_squared):
        chi_optimal, delta_optimal, unconstrained_feasible = self.solve_unconstrained_case(h_n_squared)
        
        if unconstrained_feasible:
            rho_optimal = min((2 ** (self.D_n / (delta_optimal * self.B)) - 1) / (self.P_n * h_n_squared), 1.0)
            return {
                'chi_n': chi_optimal,
                'delta_tx_n': delta_optimal,
                'rho_n': rho_optimal,
                'optimal_delay': self.compute_delay_minimization_objective(chi_optimal, delta_optimal, h_n_squared),
                'energy_consumption': self.compute_energy_constraint(chi_optimal, delta_optimal, h_n_squared),
                'constrained': False
            }
        else:
            chi_constrained, delta_constrained, constrained_feasible = self.solve_constrained_case(h_n_squared)
            
            if constrained_feasible:
                rho_constrained = min((2 ** (self.D_n / (delta_constrained * self.B)) - 1) / (self.P_n * h_n_squared), 1.0)
                return {
                    'chi_n': chi_constrained,
                    'delta_tx_n': delta_constrained,
                    'rho_n': rho_constrained,
                    'optimal_delay': self.compute_delay_minimization_objective(chi_constrained, delta_constrained, h_n_squared),
                    'energy_consumption': self.compute_energy_constraint(chi_constrained, delta_constrained, h_n_squared),
                    'constrained': True
                }
            else:
                return {
                    'chi_n': 0.1,
                    'delta_tx_n': 1.0,
                    'rho_n': 0.1,
                    'optimal_delay': float('inf'),
                    'energy_consumption': self.e_max,
                    'constrained': True,
                    'feasible': False
                }

class HighSNRClosedFormSolver(KKTConditionSolver):
    def __init__(self, system_parameters):
        super().__init__(system_parameters)
    
    def compute_high_snr_solution(self, h_n_squared):
        if self.P_n * h_n_squared < 10:
            return super().solve_optimization_problem(h_n_squared)
        
        chi_n_optimal = 1.0
        
        snr_term = self.e_max * h_n_squared / (self.kappa * self.mu * self.zeta_n * (chi_n_optimal * self.G_n) ** 2)
        
        if snr_term <= 1:
            delta_tx_optimal = float('inf')
            return {
                'chi_n': chi_n_optimal,
                'delta_tx_n': delta_tx_optimal,
                'rho_n': 1.0,
                'optimal_delay': delta_tx_optimal,
                'energy_consumption': self.e_max,
                'constrained': True,
                'high_snr': True
            }
        
        delta_tx_optimal = self.D_n / (self.B * math.log2(1 + snr_term))
        
        rho_optimal = min((2 ** (self.D_n / (delta_tx_optimal * self.B)) - 1) / (self.P_n * h_n_squared), 1.0)
        
        return {
            'chi_n': chi_n_optimal,
            'delta_tx_n': delta_tx_optimal,
            'rho_n': rho_optimal,
            'optimal_delay': self.compute_delay_minimization_objective(chi_n_optimal, delta_tx_optimal, h_n_squared),
            'energy_consumption': self.compute_energy_constraint(chi_n_optimal, delta_tx_optimal, h_n_squared),
            'constrained': True,
            'high_snr': True
        }

class MultiVehicleLagrangianOptimizer:
    def __init__(self, system_parameters, num_vehicles=20):
        self.num_vehicles = num_vehicles
        self.system_params = system_parameters
        self.vehicle_optimizers = []
        
        for vehicle_id in range(num_vehicles):
            vehicle_params = system_parameters.copy()
            vehicle_params['zeta_n'] = np.random.randint(100, 1000)
            
            optimizer = HighSNRClosedFormSolver(vehicle_params)
            self.vehicle_optimizers.append(optimizer)
    
    def solve_multi_vehicle_optimization(self, channel_gains_h_squared):
        optimization_results = {}
        
        total_communication_delay = 0.0
        total_energy_consumption = 0.0
        
        for vehicle_id in range(self.num_vehicles):
            h_n_squared = channel_gains_h_squared.get(vehicle_id, 1.0)
            
            vehicle_result = self.vehicle_optimizers[vehicle_id].compute_high_snr_solution(h_n_squared)
            optimization_results[vehicle_id] = vehicle_result
            
            if vehicle_result['optimal_delay'] != float('inf'):
                total_communication_delay = max(total_communication_delay, vehicle_result['optimal_delay'])
            
            total_energy_consumption += vehicle_result['energy_consumption']
        
        return {
            'individual_results': optimization_results,
            'total_communication_delay': total_communication_delay,
            'total_energy_consumption': total_energy_consumption,
            'feasible_vehicles': sum(1 for r in optimization_results.values() if r.get('feasible', True))
        }

def create_lagrangian_optimization_system():
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
    
    multi_vehicle_optimizer = MultiVehicleLagrangianOptimizer(SYSTEM_PARAMS, num_vehicles=20)
    
    return multi_vehicle_optimizer
