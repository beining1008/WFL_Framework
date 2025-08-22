import torch
import numpy as np
import math
from collections import defaultdict

class FLMDCalculatorInterface:
    def compute_model_drift(self, global_model, local_model):
        raise NotImplementedError

    def evaluate_eligibility_criteria(self, drift_values):
        raise NotImplementedError

class BaseFederatedLearningModelDrift(FLMDCalculatorInterface):
    def __init__(self, flmd_parameters):
        self.lambda_theta = flmd_parameters.get('lambda_theta', 0.1)
        self.beta = flmd_parameters.get('beta', 1.0)
        self.mu_over_L = flmd_parameters.get('mu_over_L', 0.01)
        self.drift_history = defaultdict(list)

class FLMDMetricCalculator(BaseFederatedLearningModelDrift):
    def __init__(self, flmd_parameters):
        super().__init__(flmd_parameters)
        self.numerical_stability_epsilon = 1e-12
        
    def compute_parameter_wise_drift(self, omega_global, omega_local_n):
        if isinstance(omega_global, dict) and isinstance(omega_local_n, dict):
            parameter_drifts = {}
            total_numerator = 0.0
            total_denominator = 0.0
            
            for param_name in omega_global.keys():
                if param_name in omega_local_n:
                    global_param = omega_global[param_name]
                    local_param = omega_local_n[param_name]
                    
                    if isinstance(global_param, torch.Tensor):
                        global_param = global_param.detach().cpu()
                        local_param = local_param.detach().cpu()
                    
                    param_diff = local_param - global_param
                    param_diff_norm = torch.norm(param_diff).item()
                    global_param_norm = torch.norm(global_param).item()
                    
                    parameter_drifts[param_name] = {
                        'diff_norm': param_diff_norm,
                        'global_norm': global_param_norm,
                        'relative_drift': param_diff_norm / (global_param_norm + self.numerical_stability_epsilon)
                    }
                    
                    total_numerator += param_diff_norm ** 2
                    total_denominator += global_param_norm ** 2
            
            overall_drift = math.sqrt(total_numerator) / (math.sqrt(total_denominator) + self.numerical_stability_epsilon)
            
            return overall_drift, parameter_drifts
        else:
            if isinstance(omega_global, torch.Tensor):
                omega_global = omega_global.detach().cpu()
                omega_local_n = omega_local_n.detach().cpu()
            
            diff_tensor = omega_local_n - omega_global
            diff_norm = torch.norm(diff_tensor).item()
            global_norm = torch.norm(omega_global).item()
            
            overall_drift = diff_norm / (global_norm + self.numerical_stability_epsilon)
            
            return overall_drift, {'tensor_drift': overall_drift}
    
    def compute_model_drift(self, omega_global, omega_local_n):
        theta_n, parameter_details = self.compute_parameter_wise_drift(omega_global, omega_local_n)
        return theta_n
    
    def compute_weighted_flmd(self, omega_global, omega_local_n, sample_size_zeta_n):
        base_drift = self.compute_model_drift(omega_global, omega_local_n)
        
        sample_weight = math.log(sample_size_zeta_n + 1) / math.log(1000 + 1)
        weighted_drift = base_drift * sample_weight
        
        return weighted_drift
    
    def compute_temporal_flmd_evolution(self, client_id, omega_global, omega_local_n, communication_round):
        current_drift = self.compute_model_drift(omega_global, omega_local_n)
        
        self.drift_history[client_id].append({
            'round': communication_round,
            'drift': current_drift,
            'timestamp': communication_round
        })
        
        if len(self.drift_history[client_id]) > 1:
            previous_drift = self.drift_history[client_id][-2]['drift']
            drift_velocity = current_drift - previous_drift
            
            if len(self.drift_history[client_id]) > 2:
                prev_prev_drift = self.drift_history[client_id][-3]['drift']
                prev_drift_velocity = previous_drift - prev_prev_drift
                drift_acceleration = drift_velocity - prev_drift_velocity
            else:
                drift_acceleration = 0.0
        else:
            drift_velocity = 0.0
            drift_acceleration = 0.0
        
        temporal_metrics = {
            'current_drift': current_drift,
            'drift_velocity': drift_velocity,
            'drift_acceleration': drift_acceleration,
            'drift_trend': 'increasing' if drift_velocity > 0 else 'decreasing' if drift_velocity < 0 else 'stable'
        }
        
        return temporal_metrics

class EligibilitySetManager(FLMDMetricCalculator):
    def __init__(self, flmd_parameters):
        super().__init__(flmd_parameters)
        self.eligibility_history = defaultdict(list)
        
    def evaluate_eligibility_criteria(self, drift_values):
        eligible_clients = []
        ineligible_clients = []
        
        for client_id, theta_n in drift_values.items():
            if theta_n <= self.lambda_theta:
                eligible_clients.append(client_id)
            else:
                ineligible_clients.append(client_id)
        
        eligibility_ratio = len(eligible_clients) / len(drift_values) if drift_values else 0.0
        
        return {
            'eligible_clients': eligible_clients,
            'ineligible_clients': ineligible_clients,
            'eligibility_ratio': eligibility_ratio,
            'total_clients': len(drift_values)
        }
    
    def compute_adaptive_eligibility_weights(self, drift_values, communication_round):
        adaptive_weights = {}
        
        for client_id, theta_n in drift_values.items():
            if theta_n <= self.lambda_theta:
                adaptive_weights[client_id] = 1.0
            else:
                decay_factor = (1 - self.mu_over_L) ** (-communication_round)
                adaptive_weight = math.exp(-self.beta * theta_n * decay_factor)
                adaptive_weights[client_id] = adaptive_weight
        
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            normalized_weights = {client_id: weight / total_weight for client_id, weight in adaptive_weights.items()}
        else:
            normalized_weights = {client_id: 1.0 / len(adaptive_weights) for client_id in adaptive_weights.keys()}
        
        return normalized_weights
    
    def update_eligibility_history(self, client_id, eligibility_status, communication_round):
        self.eligibility_history[client_id].append({
            'round': communication_round,
            'eligible': eligibility_status,
            'timestamp': communication_round
        })
    
    def compute_client_reliability_score(self, client_id, window_size=10):
        if client_id not in self.eligibility_history:
            return 0.5
        
        recent_history = self.eligibility_history[client_id][-window_size:]
        
        if not recent_history:
            return 0.5
        
        eligible_count = sum(1 for entry in recent_history if entry['eligible'])
        reliability_score = eligible_count / len(recent_history)
        
        return reliability_score

class AdaptiveThresholdManager:
    def __init__(self, initial_lambda_theta=0.1, adaptation_rate=0.01):
        self.lambda_theta = initial_lambda_theta
        self.adaptation_rate = adaptation_rate
        self.threshold_history = [initial_lambda_theta]
        self.performance_history = []
        
    def update_threshold_based_on_performance(self, global_loss, convergence_rate, eligibility_ratio):
        performance_score = self._compute_performance_score(global_loss, convergence_rate, eligibility_ratio)
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) > 1:
            performance_trend = self.performance_history[-1] - self.performance_history[-2]
            
            if performance_trend > 0:
                self.lambda_theta *= (1 + self.adaptation_rate)
            else:
                self.lambda_theta *= (1 - self.adaptation_rate)
            
            self.lambda_theta = max(0.01, min(self.lambda_theta, 1.0))
        
        self.threshold_history.append(self.lambda_theta)
        
        return self.lambda_theta
    
    def _compute_performance_score(self, global_loss, convergence_rate, eligibility_ratio):
        loss_component = 1.0 / (1.0 + global_loss)
        convergence_component = max(0, convergence_rate)
        participation_component = eligibility_ratio
        
        performance_score = 0.4 * loss_component + 0.3 * convergence_component + 0.3 * participation_component
        
        return performance_score
    
    def get_adaptive_threshold(self, communication_round):
        base_threshold = self.lambda_theta
        
        round_factor = 1.0 + 0.1 * math.sin(0.1 * communication_round)
        adaptive_threshold = base_threshold * round_factor
        
        return max(0.01, min(adaptive_threshold, 1.0))

class FLMDAnalyzer:
    def __init__(self, flmd_parameters):
        self.flmd_calculator = FLMDMetricCalculator(flmd_parameters)
        self.eligibility_manager = EligibilitySetManager(flmd_parameters)
        self.threshold_manager = AdaptiveThresholdManager(flmd_parameters.get('lambda_theta', 0.1))
        
    def analyze_federated_round(self, global_model, client_models, sample_sizes, communication_round):
        drift_values = {}
        temporal_metrics = {}
        
        for client_id, local_model in client_models.items():
            theta_n = self.flmd_calculator.compute_model_drift(global_model, local_model)
            
            if client_id in sample_sizes:
                weighted_theta_n = self.flmd_calculator.compute_weighted_flmd(
                    global_model, local_model, sample_sizes[client_id]
                )
            else:
                weighted_theta_n = theta_n
            
            drift_values[client_id] = weighted_theta_n
            
            temporal_metrics[client_id] = self.flmd_calculator.compute_temporal_flmd_evolution(
                client_id, global_model, local_model, communication_round
            )
        
        eligibility_results = self.eligibility_manager.evaluate_eligibility_criteria(drift_values)
        
        adaptive_weights = self.eligibility_manager.compute_adaptive_eligibility_weights(
            drift_values, communication_round
        )
        
        for client_id in drift_values.keys():
            is_eligible = client_id in eligibility_results['eligible_clients']
            self.eligibility_manager.update_eligibility_history(client_id, is_eligible, communication_round)
        
        analysis_results = {
            'drift_values': drift_values,
            'temporal_metrics': temporal_metrics,
            'eligibility_results': eligibility_results,
            'adaptive_weights': adaptive_weights,
            'communication_round': communication_round
        }
        
        return analysis_results
    
    def update_adaptive_threshold(self, global_loss, convergence_rate, eligibility_ratio):
        performance_metrics = {
            'global_loss': global_loss,
            'convergence_rate': convergence_rate,
            'eligibility_ratio': eligibility_ratio
        }

        new_threshold = self.threshold_manager.update_threshold_based_on_performance(
            global_loss, convergence_rate, eligibility_ratio
        )

        self.flmd_calculator.lambda_theta = new_threshold
        self.eligibility_manager.lambda_theta = new_threshold

        return new_threshold

def create_flmd_analysis_system():
    FLMD_PARAMS = {
        'lambda_theta': 0.1,
        'beta': 1.0,
        'mu_over_L': 0.01
    }
    
    flmd_analyzer = FLMDAnalyzer(FLMD_PARAMS)
    
    return flmd_analyzer
