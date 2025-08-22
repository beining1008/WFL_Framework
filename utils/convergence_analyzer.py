import torch
import numpy as np
import math
from collections import deque, defaultdict
import scipy.stats as stats

class ConvergenceAnalysisInterface:
    def analyze_convergence_behavior(self, loss_history, gradient_history):
        raise NotImplementedError

    def detect_convergence_plateau(self, metric_sequence):
        raise NotImplementedError

class BaseConvergenceMetrics(ConvergenceAnalysisInterface):
    def __init__(self, analyzer_config):
        self.window_size = analyzer_config.get('window_size', 20)
        self.plateau_tolerance = analyzer_config.get('plateau_tolerance', 1e-6)
        self.convergence_threshold = analyzer_config.get('convergence_threshold', 1e-5)
        self.statistical_significance_level = analyzer_config.get('significance_level', 0.05)

class FederatedLearningConvergenceAnalyzer(BaseConvergenceMetrics):
    def __init__(self, analyzer_config):
        super().__init__(analyzer_config)
        self.global_loss_history = deque(maxlen=1000)
        self.client_loss_histories = defaultdict(lambda: deque(maxlen=1000))
        self.gradient_norm_history = deque(maxlen=1000)
        self.flmd_history = defaultdict(lambda: deque(maxlen=1000))
        self.convergence_indicators = {}
        
    def update_convergence_metrics(self, communication_round, global_loss, client_losses, gradient_norms, flmd_values):
        self.global_loss_history.append({
            'round': communication_round,
            'loss': global_loss,
            'timestamp': communication_round
        })
        
        for client_id, client_loss in client_losses.items():
            self.client_loss_histories[client_id].append({
                'round': communication_round,
                'loss': client_loss,
                'timestamp': communication_round
            })
        
        if gradient_norms:
            avg_gradient_norm = np.mean(list(gradient_norms.values()))
            self.gradient_norm_history.append({
                'round': communication_round,
                'gradient_norm': avg_gradient_norm,
                'timestamp': communication_round
            })
        
        for client_id, flmd_value in flmd_values.items():
            self.flmd_history[client_id].append({
                'round': communication_round,
                'flmd': flmd_value,
                'timestamp': communication_round
            })
    
    def analyze_convergence_behavior(self, communication_round):
        convergence_analysis = {}
        
        if len(self.global_loss_history) >= self.window_size:
            global_loss_convergence = self._analyze_loss_convergence(
                [entry['loss'] for entry in list(self.global_loss_history)[-self.window_size:]]
            )
            convergence_analysis['global_loss'] = global_loss_convergence
        
        if len(self.gradient_norm_history) >= self.window_size:
            gradient_convergence = self._analyze_gradient_convergence(
                [entry['gradient_norm'] for entry in list(self.gradient_norm_history)[-self.window_size:]]
            )
            convergence_analysis['gradient_norms'] = gradient_convergence
        
        client_convergence_analysis = {}
        for client_id, client_history in self.client_loss_histories.items():
            if len(client_history) >= self.window_size:
                client_losses = [entry['loss'] for entry in list(client_history)[-self.window_size:]]
                client_convergence = self._analyze_loss_convergence(client_losses)
                client_convergence_analysis[client_id] = client_convergence
        
        convergence_analysis['client_losses'] = client_convergence_analysis
        
        flmd_convergence_analysis = {}
        for client_id, flmd_history in self.flmd_history.items():
            if len(flmd_history) >= self.window_size:
                flmd_values = [entry['flmd'] for entry in list(flmd_history)[-self.window_size:]]
                flmd_convergence = self._analyze_flmd_convergence(flmd_values)
                flmd_convergence_analysis[client_id] = flmd_convergence
        
        convergence_analysis['flmd_values'] = flmd_convergence_analysis
        
        overall_convergence = self._compute_overall_convergence_status(convergence_analysis)
        convergence_analysis['overall_status'] = overall_convergence
        
        return convergence_analysis
    
    def _analyze_loss_convergence(self, loss_sequence):
        if len(loss_sequence) < 2:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        loss_differences = np.diff(loss_sequence)
        
        plateau_detected = self.detect_convergence_plateau(loss_sequence)
        
        trend_analysis = self._compute_trend_statistics(loss_sequence)
        
        relative_improvement = abs(loss_differences[-1]) / (abs(loss_sequence[-2]) + 1e-12)
        
        convergence_criteria = {
            'plateau_detected': plateau_detected,
            'relative_improvement_small': relative_improvement < self.convergence_threshold,
            'decreasing_trend': trend_analysis['slope'] < 0,
            'stable_variance': trend_analysis['variance_stable']
        }
        
        converged = (convergence_criteria['plateau_detected'] or 
                    convergence_criteria['relative_improvement_small']) and \
                   convergence_criteria['decreasing_trend']
        
        return {
            'converged': converged,
            'convergence_criteria': convergence_criteria,
            'trend_statistics': trend_analysis,
            'relative_improvement': relative_improvement,
            'loss_variance': np.var(loss_sequence),
            'loss_mean': np.mean(loss_sequence)
        }
    
    def _analyze_gradient_convergence(self, gradient_norm_sequence):
        if len(gradient_norm_sequence) < 2:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        gradient_trend = self._compute_trend_statistics(gradient_norm_sequence)
        
        small_gradients = np.mean(gradient_norm_sequence) < self.convergence_threshold
        
        gradient_plateau = self.detect_convergence_plateau(gradient_norm_sequence)
        
        convergence_criteria = {
            'small_gradient_norms': small_gradients,
            'gradient_plateau': gradient_plateau,
            'decreasing_gradient_trend': gradient_trend['slope'] < 0
        }
        
        converged = convergence_criteria['small_gradient_norms'] or \
                   (convergence_criteria['gradient_plateau'] and 
                    convergence_criteria['decreasing_gradient_trend'])
        
        return {
            'converged': converged,
            'convergence_criteria': convergence_criteria,
            'gradient_statistics': gradient_trend,
            'mean_gradient_norm': np.mean(gradient_norm_sequence),
            'gradient_variance': np.var(gradient_norm_sequence)
        }
    
    def _analyze_flmd_convergence(self, flmd_sequence):
        if len(flmd_sequence) < 2:
            return {'converged': False, 'reason': 'insufficient_data'}
        
        flmd_trend = self._compute_trend_statistics(flmd_sequence)
        
        low_flmd_values = np.mean(flmd_sequence) < 0.1
        
        flmd_plateau = self.detect_convergence_plateau(flmd_sequence)
        
        convergence_criteria = {
            'low_flmd_values': low_flmd_values,
            'flmd_plateau': flmd_plateau,
            'stable_flmd_trend': abs(flmd_trend['slope']) < 0.01
        }
        
        converged = convergence_criteria['low_flmd_values'] and \
                   convergence_criteria['stable_flmd_trend']
        
        return {
            'converged': converged,
            'convergence_criteria': convergence_criteria,
            'flmd_statistics': flmd_trend,
            'mean_flmd': np.mean(flmd_sequence),
            'flmd_variance': np.var(flmd_sequence)
        }
    
    def detect_convergence_plateau(self, metric_sequence):
        if len(metric_sequence) < self.window_size:
            return False
        
        recent_values = metric_sequence[-self.window_size:]
        
        max_difference = max(recent_values) - min(recent_values)
        mean_value = np.mean(recent_values)
        
        relative_plateau_threshold = self.plateau_tolerance * (abs(mean_value) + 1e-12)
        
        plateau_detected = max_difference < relative_plateau_threshold
        
        return plateau_detected
    
    def _compute_trend_statistics(self, sequence):
        if len(sequence) < 2:
            return {'slope': 0, 'r_squared': 0, 'variance_stable': False}

        x = np.arange(len(sequence))
        y = np.array(sequence)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        variance_first_half = np.var(y[:len(y)//2]) if len(y) >= 4 else 0
        variance_second_half = np.var(y[len(y)//2:]) if len(y) >= 4 else 0
        
        variance_ratio = (variance_second_half + 1e-12) / (variance_first_half + 1e-12)
        variance_stable = 0.5 <= variance_ratio <= 2.0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'standard_error': std_err,
            'variance_stable': variance_stable,
            'variance_ratio': variance_ratio
        }
    
    def _compute_overall_convergence_status(self, convergence_analysis):
        convergence_indicators = []
        
        if 'global_loss' in convergence_analysis:
            convergence_indicators.append(convergence_analysis['global_loss']['converged'])
        
        if 'gradient_norms' in convergence_analysis:
            convergence_indicators.append(convergence_analysis['gradient_norms']['converged'])
        
        client_convergence_rates = []
        if 'client_losses' in convergence_analysis:
            for client_analysis in convergence_analysis['client_losses'].values():
                client_convergence_rates.append(client_analysis['converged'])
        
        flmd_convergence_rates = []
        if 'flmd_values' in convergence_analysis:
            for flmd_analysis in convergence_analysis['flmd_values'].values():
                flmd_convergence_rates.append(flmd_analysis['converged'])
        
        global_convergence_rate = np.mean(convergence_indicators) if convergence_indicators else 0
        client_convergence_rate = np.mean(client_convergence_rates) if client_convergence_rates else 0
        flmd_convergence_rate = np.mean(flmd_convergence_rates) if flmd_convergence_rates else 0
        
        overall_convergence_score = (0.4 * global_convergence_rate + 
                                   0.4 * client_convergence_rate + 
                                   0.2 * flmd_convergence_rate)
        
        convergence_status = {
            'overall_converged': overall_convergence_score > 0.7,
            'convergence_score': overall_convergence_score,
            'global_convergence_rate': global_convergence_rate,
            'client_convergence_rate': client_convergence_rate,
            'flmd_convergence_rate': flmd_convergence_rate,
            'convergence_confidence': self._compute_convergence_confidence(convergence_analysis)
        }
        
        return convergence_status
    
    def _compute_convergence_confidence(self, convergence_analysis):
        confidence_factors = []
        
        if 'global_loss' in convergence_analysis:
            loss_analysis = convergence_analysis['global_loss']
            if 'trend_statistics' in loss_analysis:
                r_squared = loss_analysis['trend_statistics']['r_squared']
                confidence_factors.append(r_squared)
        
        if 'gradient_norms' in convergence_analysis:
            gradient_analysis = convergence_analysis['gradient_norms']
            if 'gradient_statistics' in gradient_analysis:
                r_squared = gradient_analysis['gradient_statistics']['r_squared']
                confidence_factors.append(r_squared)
        
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5

class EarlyStoppingManager:
    def __init__(self, patience=20, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait_counter = 0
        self.stopped_round = 0
        
    def should_stop_training(self, current_loss, current_weights, current_round):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait_counter = 0
            
            if self.restore_best_weights:
                self.best_weights = current_weights.copy() if isinstance(current_weights, dict) else current_weights
        else:
            self.wait_counter += 1
        
        if self.wait_counter >= self.patience:
            self.stopped_round = current_round
            return True
        
        return False
    
    def get_best_weights(self):
        return self.best_weights

class ConvergenceVisualizationEngine:
    def __init__(self):
        self.visualization_data = defaultdict(list)
        
    def update_visualization_data(self, round_number, metrics_dict):
        for metric_name, metric_value in metrics_dict.items():
            self.visualization_data[metric_name].append({
                'round': round_number,
                'value': metric_value
            })
    
    def generate_convergence_plots(self, save_directory='./plots/'):

        for metric_name, metric_data in self.visualization_data.items():
            rounds = [entry['round'] for entry in metric_data]
            values = [entry['value'] for entry in metric_data]

            print(f"Plotting {metric_name} convergence data...")
            print(f"Rounds: {len(rounds)}, Values: {len(values)}")

            pass

def create_convergence_analysis_system():
    ANALYZER_CONFIG = {
        'window_size': 20,
        'plateau_tolerance': 1e-6,
        'convergence_threshold': 1e-5,
        'significance_level': 0.05
    }
    
    convergence_analyzer = FederatedLearningConvergenceAnalyzer(ANALYZER_CONFIG)
    
    early_stopping = EarlyStoppingManager(
        patience=20,
        min_delta=1e-6,
        restore_best_weights=True
    )
    
    visualization_engine = ConvergenceVisualizationEngine()
    
    return convergence_analyzer, early_stopping, visualization_engine
