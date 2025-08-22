import torch
import numpy as np
import math
import threading
import queue
import time

class WirelessCommunicationProtocol:
    def transmit_model_parameters(self, source_id, destination_id, model_data):
        raise NotImplementedError

    def compute_channel_capacity(self, channel_state_information):
        raise NotImplementedError

class BaseWirelessCommunicationFramework(WirelessCommunicationProtocol):
    def __init__(self, communication_config):
        self.B = communication_config['bandwidth_mhz'] * 1e6
        self.P_n = communication_config['transmission_power_dbm']
        self.noise_power = communication_config['noise_power_dbm']
        self.path_loss_exponent = communication_config['path_loss_exponent']
        self.shadowing_variance = communication_config['shadowing_variance']
        self.pilot_error_probability = communication_config['pilot_error_probability']

class MMSEChannelEstimationProtocol(BaseWirelessCommunicationFramework):
    def __init__(self, communication_config):
        super().__init__(communication_config)
        self.channel_estimation_error_variance = 0.1
        self.pilot_contamination_factor = 0.05
        
    def generate_channel_state_information(self, vehicle_positions, communication_round):
        num_vehicles = len(vehicle_positions)
        channel_gains = {}
        
        for vehicle_id in range(num_vehicles):
            distance_to_base_station = np.random.uniform(50, 500)
            
            path_loss_db = 32.45 + 20 * math.log10(5.9) + 20 * math.log10(distance_to_base_station / 1000)
            path_loss_linear = 10 ** (-path_loss_db / 10)
            
            shadowing_db = np.random.normal(0, self.shadowing_variance)
            shadowing_linear = 10 ** (shadowing_db / 10)
            
            small_scale_fading = np.random.rayleigh(1.0)
            
            true_channel_gain = path_loss_linear * shadowing_linear * (small_scale_fading ** 2)
            
            pilot_error = np.random.binomial(1, self.pilot_error_probability)
            if pilot_error:
                estimation_error = np.random.normal(0, self.channel_estimation_error_variance)
                estimated_channel_gain = true_channel_gain * (1 + estimation_error)
            else:
                estimated_channel_gain = true_channel_gain
            
            channel_gains[vehicle_id] = {
                'true_gain': true_channel_gain,
                'estimated_gain': max(estimated_channel_gain, 1e-10),
                'estimation_error': pilot_error,
                'distance': distance_to_base_station
            }
        
        return channel_gains
    
    def compute_mmse_channel_estimate(self, received_pilot, true_channel, noise_variance):
        pilot_power = 1.0
        
        mmse_coefficient = (pilot_power * abs(true_channel) ** 2) / (
            pilot_power * abs(true_channel) ** 2 + noise_variance
        )
        
        estimated_channel = mmse_coefficient * received_pilot
        estimation_error_variance = noise_variance * mmse_coefficient
        
        return estimated_channel, estimation_error_variance
    
    def compute_channel_capacity(self, channel_state_information):
        channel_capacities = {}
        
        for vehicle_id, channel_info in channel_state_information.items():
            h_n_squared = channel_info['estimated_gain']
            
            snr = self.P_n * h_n_squared / (10 ** (self.noise_power / 10))
            
            capacity_bps = self.B * math.log2(1 + snr)
            
            channel_capacities[vehicle_id] = {
                'capacity_bps': capacity_bps,
                'snr_db': 10 * math.log10(snr) if snr > 0 else -np.inf,
                'channel_gain_db': 10 * math.log10(h_n_squared) if h_n_squared > 0 else -np.inf
            }
        
        return channel_capacities

class FederatedLearningCommunicationManager:
    def __init__(self, communication_protocol, num_vehicles=20):
        self.communication_protocol = communication_protocol
        self.num_vehicles = num_vehicles
        self.transmission_queues = {i: queue.Queue() for i in range(num_vehicles)}
        self.reception_queues = {i: queue.Queue() for i in range(num_vehicles)}
        self.communication_threads = []
        self.global_model_buffer = None
        self.model_update_lock = threading.Lock()
        
    def initialize_communication_infrastructure(self):
        for vehicle_id in range(self.num_vehicles):
            tx_thread = threading.Thread(
                target=self._transmission_worker,
                args=(vehicle_id,),
                daemon=True
            )
            
            rx_thread = threading.Thread(
                target=self._reception_worker,
                args=(vehicle_id,),
                daemon=True
            )
            
            self.communication_threads.extend([tx_thread, rx_thread])
            tx_thread.start()
            rx_thread.start()
    
    def _transmission_worker(self, vehicle_id):
        while True:
            try:
                transmission_task = self.transmission_queues[vehicle_id].get(timeout=1.0)
                
                if transmission_task is None:
                    break
                
                destination_id = transmission_task['destination']
                model_data = transmission_task['model_data']
                transmission_time = transmission_task['transmission_time']
                
                time.sleep(transmission_time)
                
                self.reception_queues[destination_id].put({
                    'source': vehicle_id,
                    'model_data': model_data,
                    'timestamp': time.time()
                })
                
                self.transmission_queues[vehicle_id].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                continue
    
    def _reception_worker(self, vehicle_id):
        while True:
            try:
                reception_task = self.reception_queues[vehicle_id].get(timeout=1.0)
                
                if reception_task is None:
                    break
                
                with self.model_update_lock:
                    if vehicle_id == 0:
                        self.global_model_buffer = reception_task['model_data']
                
                self.reception_queues[vehicle_id].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                continue
    
    def transmit_model_parameters(self, source_id, model_parameters, channel_capacities):
        model_size_bits = self._estimate_model_size(model_parameters)
        
        if source_id in channel_capacities:
            capacity_bps = channel_capacities[source_id]['capacity_bps']
            transmission_time = model_size_bits / capacity_bps
        else:
            transmission_time = 1.0
        
        transmission_task = {
            'destination': 0,
            'model_data': model_parameters,
            'transmission_time': transmission_time,
            'model_size_bits': model_size_bits
        }
        
        self.transmission_queues[source_id].put(transmission_task)
        
        return transmission_time
    
    def broadcast_global_model(self, global_model_parameters, channel_capacities):
        model_size_bits = self._estimate_model_size(global_model_parameters)
        
        broadcast_times = {}
        
        for vehicle_id in range(1, self.num_vehicles):
            if vehicle_id in channel_capacities:
                capacity_bps = channel_capacities[vehicle_id]['capacity_bps']
                transmission_time = model_size_bits / capacity_bps
            else:
                transmission_time = 1.0
            
            broadcast_task = {
                'destination': vehicle_id,
                'model_data': global_model_parameters,
                'transmission_time': transmission_time,
                'model_size_bits': model_size_bits
            }
            
            self.transmission_queues[0].put(broadcast_task)
            broadcast_times[vehicle_id] = transmission_time
        
        return broadcast_times
    
    def _estimate_model_size(self, model_parameters):
        if isinstance(model_parameters, dict):
            total_parameters = 0
            for param_name, param_tensor in model_parameters.items():
                if isinstance(param_tensor, torch.Tensor):
                    total_parameters += param_tensor.numel()
                else:
                    total_parameters += len(str(param_tensor)) * 8
            
            model_size_bits = total_parameters * 32
        else:
            model_size_bits = 10.08e6
        
        return model_size_bits
    
    def compute_communication_round_delay(self, channel_capacities):
        uplink_delays = []
        downlink_delays = []
        
        for vehicle_id in range(1, self.num_vehicles):
            if vehicle_id in channel_capacities:
                capacity_bps = channel_capacities[vehicle_id]['capacity_bps']
                model_size_bits = 10.08e6
                
                uplink_delay = model_size_bits / capacity_bps
                downlink_delay = model_size_bits / capacity_bps
                
                uplink_delays.append(uplink_delay)
                downlink_delays.append(downlink_delay)
        
        max_uplink_delay = max(uplink_delays) if uplink_delays else 0
        max_downlink_delay = max(downlink_delays) if downlink_delays else 0
        
        total_communication_delay = max_uplink_delay + max_downlink_delay
        
        return {
            'total_delay': total_communication_delay,
            'uplink_delay': max_uplink_delay,
            'downlink_delay': max_downlink_delay,
            'individual_uplink_delays': uplink_delays,
            'individual_downlink_delays': downlink_delays
        }

class GradientCompressionProtocol:
    def __init__(self, compression_ratio=0.3, quantization_bits=16):
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.compression_history = {}
        
    def compress_model_gradients(self, gradient_dict):
        compressed_gradients = {}
        compression_stats = {}
        
        for param_name, gradient_tensor in gradient_dict.items():
            if isinstance(gradient_tensor, torch.Tensor):
                original_size = gradient_tensor.numel() * 4
                
                top_k_compressed = self._top_k_compression(gradient_tensor)
                quantized_compressed = self._quantization_compression(top_k_compressed)
                
                compressed_gradients[param_name] = quantized_compressed
                
                compressed_size = len(quantized_compressed['indices']) * 4 + len(quantized_compressed['values']) * (self.quantization_bits / 8)
                
                compression_stats[param_name] = {
                    'original_size_bytes': original_size,
                    'compressed_size_bytes': compressed_size,
                    'compression_ratio': compressed_size / original_size,
                    'sparsity': 1 - len(quantized_compressed['indices']) / gradient_tensor.numel()
                }
        
        return compressed_gradients, compression_stats
    
    def _top_k_compression(self, gradient_tensor):
        flattened_gradients = gradient_tensor.flatten()
        k = int(len(flattened_gradients) * self.compression_ratio)
        
        top_k_values, top_k_indices = torch.topk(torch.abs(flattened_gradients), k)
        
        selected_values = flattened_gradients[top_k_indices]
        
        return {
            'values': selected_values,
            'indices': top_k_indices,
            'original_shape': gradient_tensor.shape
        }
    
    def _quantization_compression(self, top_k_data):
        values = top_k_data['values']
        
        min_val = torch.min(values)
        max_val = torch.max(values)
        
        quantization_levels = 2 ** self.quantization_bits
        scale = (max_val - min_val) / (quantization_levels - 1)
        
        quantized_values = torch.round((values - min_val) / scale).int()
        
        return {
            'quantized_values': quantized_values,
            'indices': top_k_data['indices'],
            'scale': scale,
            'min_val': min_val,
            'original_shape': top_k_data['original_shape']
        }
    
    def decompress_model_gradients(self, compressed_gradients):
        decompressed_gradients = {}
        
        for param_name, compressed_data in compressed_gradients.items():
            quantized_values = compressed_data['quantized_values']
            indices = compressed_data['indices']
            scale = compressed_data['scale']
            min_val = compressed_data['min_val']
            original_shape = compressed_data['original_shape']
            
            dequantized_values = quantized_values.float() * scale + min_val
            
            full_gradient = torch.zeros(torch.prod(torch.tensor(original_shape)))
            full_gradient[indices] = dequantized_values
            
            decompressed_gradients[param_name] = full_gradient.reshape(original_shape)
        
        return decompressed_gradients

def create_communication_system():
    COMMUNICATION_CONFIG = {
        'bandwidth_mhz': 56,
        'transmission_power_dbm': 23,
        'noise_power_dbm': -174,
        'path_loss_exponent': 2.5,
        'shadowing_variance': 8.0,
        'pilot_error_probability': 0.1
    }
    
    mmse_protocol = MMSEChannelEstimationProtocol(COMMUNICATION_CONFIG)
    
    communication_manager = FederatedLearningCommunicationManager(
        mmse_protocol, 
        num_vehicles=20
    )
    
    gradient_compressor = GradientCompressionProtocol(
        compression_ratio=0.3,
        quantization_bits=16
    )
    
    return mmse_protocol, communication_manager, gradient_compressor
