import torch
import numpy as np
import math
import random
from collections import defaultdict

class GradientCompressionInterface:
    def compress_gradients(self, gradient_tensors):
        raise NotImplementedError

    def decompress_gradients(self, compressed_data):
        raise NotImplementedError

class BaseCompressionAlgorithm(GradientCompressionInterface):
    def __init__(self, compression_config):
        self.compression_ratio = compression_config.get('compression_ratio', 0.3)
        self.error_feedback = compression_config.get('error_feedback', True)
        self.momentum_factor = compression_config.get('momentum_factor', 0.9)
        self.compression_history = defaultdict(list)

class TopKSparsificationEngine(BaseCompressionAlgorithm):
    def __init__(self, compression_config):
        super().__init__(compression_config)
        self.adaptive_k = compression_config.get('adaptive_k', True)
        self.k_adaptation_rate = compression_config.get('k_adaptation_rate', 0.01)
        self.current_k_ratios = {}
        
    def compute_adaptive_k(self, gradient_tensor, layer_name):
        if layer_name not in self.current_k_ratios:
            self.current_k_ratios[layer_name] = self.compression_ratio
        
        gradient_variance = torch.var(gradient_tensor).item()
        gradient_norm = torch.norm(gradient_tensor).item()
        
        if gradient_norm > 0:
            signal_to_noise_ratio = gradient_norm / (gradient_variance + 1e-8)
            
            if signal_to_noise_ratio > 10:
                self.current_k_ratios[layer_name] *= (1 + self.k_adaptation_rate)
            else:
                self.current_k_ratios[layer_name] *= (1 - self.k_adaptation_rate)
            
            self.current_k_ratios[layer_name] = np.clip(self.current_k_ratios[layer_name], 0.01, 0.5)
        
        return self.current_k_ratios[layer_name]
    
    def compress_gradients(self, gradient_tensors):
        compressed_data = {}
        compression_statistics = {}
        
        for layer_name, gradient_tensor in gradient_tensors.items():
            if not isinstance(gradient_tensor, torch.Tensor):
                continue
            
            original_shape = gradient_tensor.shape
            flattened_gradient = gradient_tensor.flatten()
            
            if self.adaptive_k:
                k_ratio = self.compute_adaptive_k(gradient_tensor, layer_name)
            else:
                k_ratio = self.compression_ratio
            
            k = max(1, int(len(flattened_gradient) * k_ratio))
            
            gradient_magnitudes = torch.abs(flattened_gradient)
            top_k_values, top_k_indices = torch.topk(gradient_magnitudes, k)
            
            selected_gradients = flattened_gradient[top_k_indices]
            
            compressed_data[layer_name] = {
                'values': selected_gradients,
                'indices': top_k_indices,
                'original_shape': original_shape,
                'k_ratio': k_ratio,
                'compression_method': 'top_k'
            }
            
            original_size = gradient_tensor.numel() * 4
            compressed_size = len(selected_gradients) * 4 + len(top_k_indices) * 4
            
            compression_statistics[layer_name] = {
                'original_size_bytes': original_size,
                'compressed_size_bytes': compressed_size,
                'compression_ratio': compressed_size / original_size,
                'sparsity_level': 1 - k / len(flattened_gradient),
                'selected_elements': k
            }
        
        return compressed_data, compression_statistics
    
    def decompress_gradients(self, compressed_data):
        decompressed_gradients = {}
        
        for layer_name, compressed_info in compressed_data.items():
            values = compressed_info['values']
            indices = compressed_info['indices']
            original_shape = compressed_info['original_shape']
            
            full_gradient = torch.zeros(torch.prod(torch.tensor(original_shape)))
            full_gradient[indices] = values
            
            decompressed_gradients[layer_name] = full_gradient.reshape(original_shape)
        
        return decompressed_gradients

class RandomSparsificationEngine(BaseCompressionAlgorithm):
    def __init__(self, compression_config):
        super().__init__(compression_config)
        self.random_seed = compression_config.get('random_seed', 42)
        self.unbiased_estimation = compression_config.get('unbiased_estimation', True)
        
    def compress_gradients(self, gradient_tensors):
        compressed_data = {}
        compression_statistics = {}
        
        torch.manual_seed(self.random_seed)
        
        for layer_name, gradient_tensor in gradient_tensors.items():
            if not isinstance(gradient_tensor, torch.Tensor):
                continue
            
            original_shape = gradient_tensor.shape
            flattened_gradient = gradient_tensor.flatten()
            
            num_elements = len(flattened_gradient)
            num_selected = max(1, int(num_elements * self.compression_ratio))
            
            random_indices = torch.randperm(num_elements)[:num_selected]
            selected_gradients = flattened_gradient[random_indices]
            
            if self.unbiased_estimation:
                selected_gradients = selected_gradients / self.compression_ratio
            
            compressed_data[layer_name] = {
                'values': selected_gradients,
                'indices': random_indices,
                'original_shape': original_shape,
                'compression_method': 'random_sparsification',
                'scaling_factor': 1.0 / self.compression_ratio if self.unbiased_estimation else 1.0
            }
            
            original_size = gradient_tensor.numel() * 4
            compressed_size = len(selected_gradients) * 4 + len(random_indices) * 4
            
            compression_statistics[layer_name] = {
                'original_size_bytes': original_size,
                'compressed_size_bytes': compressed_size,
                'compression_ratio': compressed_size / original_size,
                'sparsity_level': 1 - num_selected / num_elements,
                'selected_elements': num_selected
            }
        
        return compressed_data, compression_statistics
    
    def decompress_gradients(self, compressed_data):
        decompressed_gradients = {}
        
        for layer_name, compressed_info in compressed_data.items():
            values = compressed_info['values']
            indices = compressed_info['indices']
            original_shape = compressed_info['original_shape']
            scaling_factor = compressed_info.get('scaling_factor', 1.0)
            
            full_gradient = torch.zeros(torch.prod(torch.tensor(original_shape)))
            full_gradient[indices] = values * scaling_factor
            
            decompressed_gradients[layer_name] = full_gradient.reshape(original_shape)
        
        return decompressed_gradients

class QuantizationCompressionEngine(BaseCompressionAlgorithm):
    def __init__(self, compression_config):
        super().__init__(compression_config)
        self.quantization_bits = compression_config.get('quantization_bits', 8)
        self.quantization_method = compression_config.get('quantization_method', 'uniform')
        self.stochastic_rounding = compression_config.get('stochastic_rounding', True)
        
    def uniform_quantization(self, gradient_tensor):
        min_val = torch.min(gradient_tensor)
        max_val = torch.max(gradient_tensor)
        
        quantization_levels = 2 ** self.quantization_bits
        scale = (max_val - min_val) / (quantization_levels - 1)
        
        if scale == 0:
            quantized_gradients = torch.zeros_like(gradient_tensor, dtype=torch.int32)
        else:
            normalized_gradients = (gradient_tensor - min_val) / scale
            
            if self.stochastic_rounding:
                floor_vals = torch.floor(normalized_gradients)
                prob_round_up = normalized_gradients - floor_vals
                random_vals = torch.rand_like(prob_round_up)
                quantized_gradients = floor_vals + (random_vals < prob_round_up).float()
            else:
                quantized_gradients = torch.round(normalized_gradients)
            
            quantized_gradients = quantized_gradients.int()
        
        return quantized_gradients, scale, min_val
    
    def compress_gradients(self, gradient_tensors):
        compressed_data = {}
        compression_statistics = {}
        
        for layer_name, gradient_tensor in gradient_tensors.items():
            if not isinstance(gradient_tensor, torch.Tensor):
                continue
            
            original_shape = gradient_tensor.shape
            
            if self.quantization_method == 'uniform':
                quantized_values, scale, min_val = self.uniform_quantization(gradient_tensor)
            else:
                quantized_values, scale, min_val = self.uniform_quantization(gradient_tensor)
            
            compressed_data[layer_name] = {
                'quantized_values': quantized_values,
                'scale': scale,
                'min_val': min_val,
                'original_shape': original_shape,
                'quantization_bits': self.quantization_bits,
                'compression_method': 'quantization'
            }
            
            original_size = gradient_tensor.numel() * 32
            compressed_size = gradient_tensor.numel() * self.quantization_bits + 64
            
            compression_statistics[layer_name] = {
                'original_size_bits': original_size,
                'compressed_size_bits': compressed_size,
                'compression_ratio': compressed_size / original_size,
                'quantization_levels': 2 ** self.quantization_bits,
                'bits_per_element': self.quantization_bits
            }
        
        return compressed_data, compression_statistics
    
    def decompress_gradients(self, compressed_data):
        decompressed_gradients = {}
        
        for layer_name, compressed_info in compressed_data.items():
            quantized_values = compressed_info['quantized_values']
            scale = compressed_info['scale']
            min_val = compressed_info['min_val']
            original_shape = compressed_info['original_shape']
            
            dequantized_gradients = quantized_values.float() * scale + min_val
            
            decompressed_gradients[layer_name] = dequantized_gradients.reshape(original_shape)
        
        return decompressed_gradients

class ErrorFeedbackCompressionEngine:
    def __init__(self, base_compressor, momentum_factor=0.9):
        self.base_compressor = base_compressor
        self.momentum_factor = momentum_factor
        self.error_accumulator = {}
        self.momentum_buffer = {}
        
    def compress_with_error_feedback(self, gradient_tensors, client_id):
        if client_id not in self.error_accumulator:
            self.error_accumulator[client_id] = {}
            self.momentum_buffer[client_id] = {}
        
        compensated_gradients = {}
        
        for layer_name, gradient_tensor in gradient_tensors.items():
            if layer_name not in self.error_accumulator[client_id]:
                self.error_accumulator[client_id][layer_name] = torch.zeros_like(gradient_tensor)
                self.momentum_buffer[client_id][layer_name] = torch.zeros_like(gradient_tensor)
            
            error_compensated = gradient_tensor + self.error_accumulator[client_id][layer_name]
            
            momentum_updated = (self.momentum_factor * self.momentum_buffer[client_id][layer_name] + 
                              (1 - self.momentum_factor) * error_compensated)
            
            compensated_gradients[layer_name] = momentum_updated
            self.momentum_buffer[client_id][layer_name] = momentum_updated
        
        compressed_data, compression_stats = self.base_compressor.compress_gradients(compensated_gradients)
        
        decompressed_gradients = self.base_compressor.decompress_gradients(compressed_data)
        
        for layer_name in gradient_tensors.keys():
            if layer_name in decompressed_gradients:
                compression_error = compensated_gradients[layer_name] - decompressed_gradients[layer_name]
                self.error_accumulator[client_id][layer_name] = compression_error
        
        return compressed_data, compression_stats

class AdaptiveCompressionManager:
    def __init__(self, compression_engines):
        self.compression_engines = compression_engines
        self.performance_history = defaultdict(list)
        self.current_engine_index = 0
        self.adaptation_frequency = 10
        self.round_counter = 0
        
    def select_compression_engine(self, gradient_statistics):
        self.round_counter += 1
        
        if self.round_counter % self.adaptation_frequency == 0:
            self._adapt_compression_strategy(gradient_statistics)
        
        return self.compression_engines[self.current_engine_index]
    
    def _adapt_compression_strategy(self, gradient_statistics):
        gradient_norm = sum(stats.get('gradient_norm', 0) for stats in gradient_statistics.values())
        gradient_variance = sum(stats.get('gradient_variance', 0) for stats in gradient_statistics.values())
        
        if gradient_norm > 10.0:
            self.current_engine_index = 0
        elif gradient_variance > 1.0:
            self.current_engine_index = 1
        else:
            self.current_engine_index = 2
        
        self.current_engine_index = min(self.current_engine_index, len(self.compression_engines) - 1)
    
    def compress_gradients_adaptively(self, gradient_tensors, client_id):
        gradient_stats = self._compute_gradient_statistics(gradient_tensors)
        
        selected_engine = self.select_compression_engine(gradient_stats)
        
        compressed_data, compression_stats = selected_engine.compress_gradients(gradient_tensors)
        
        self.performance_history[client_id].append({
            'round': self.round_counter,
            'engine_used': type(selected_engine).__name__,
            'compression_stats': compression_stats,
            'gradient_stats': gradient_stats
        })
        
        return compressed_data, compression_stats
    
    def _compute_gradient_statistics(self, gradient_tensors):
        statistics = {}
        
        for layer_name, gradient_tensor in gradient_tensors.items():
            if isinstance(gradient_tensor, torch.Tensor):
                statistics[layer_name] = {
                    'gradient_norm': torch.norm(gradient_tensor).item(),
                    'gradient_variance': torch.var(gradient_tensor).item(),
                    'gradient_mean': torch.mean(gradient_tensor).item(),
                    'gradient_std': torch.std(gradient_tensor).item(),
                    'sparsity': (gradient_tensor == 0).float().mean().item()
                }
        
        return statistics

def create_gradient_compression_system():
    COMPRESSION_CONFIG = {
        'compression_ratio': 0.3,
        'error_feedback': True,
        'momentum_factor': 0.9,
        'adaptive_k': True,
        'quantization_bits': 8,
        'stochastic_rounding': True
    }
    
    top_k_engine = TopKSparsificationEngine(COMPRESSION_CONFIG)
    random_engine = RandomSparsificationEngine(COMPRESSION_CONFIG)
    quantization_engine = QuantizationCompressionEngine(COMPRESSION_CONFIG)
    
    error_feedback_engine = ErrorFeedbackCompressionEngine(top_k_engine)
    
    adaptive_manager = AdaptiveCompressionManager([
        top_k_engine,
        random_engine,
        quantization_engine
    ])
    
    return {
        'top_k': top_k_engine,
        'random': random_engine,
        'quantization': quantization_engine,
        'error_feedback': error_feedback_engine,
        'adaptive_manager': adaptive_manager
    }
