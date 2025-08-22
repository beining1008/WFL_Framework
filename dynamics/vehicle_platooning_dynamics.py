import numpy as np
import torch
import math

class VehicleDynamicsModelInterface:
    def update_vehicle_state(self, current_state, control_input, time_step):
        raise NotImplementedError

    def compute_safety_constraints(self, vehicle_states):
        raise NotImplementedError

class BaseVehicularPlatooningModel(VehicleDynamicsModelInterface):
    def __init__(self, dynamics_parameters):
        self.a_max = dynamics_parameters['a_max']
        self.b_max = dynamics_parameters['b_max']
        self.v_des = dynamics_parameters['v_des']
        self.d_min = dynamics_parameters['d_min']
        self.t_min = dynamics_parameters['t_min']
        self.delta = dynamics_parameters['delta']
        self.tau = dynamics_parameters.get('tau', 1.0)
        
        self.num_vehicles = dynamics_parameters.get('N', 20)
        self.vehicle_states = self._initialize_vehicle_states()

class IntelligentDriverModel(BaseVehicularPlatooningModel):
    def __init__(self, dynamics_parameters):
        super().__init__(dynamics_parameters)
        self.desired_spacing = dynamics_parameters.get('desired_spacing', [5.0] * self.num_vehicles)
        
    def _initialize_vehicle_states(self):
        vehicle_states = {}
        
        for n in range(self.num_vehicles):
            initial_position = n * 10.0
            initial_velocity = self.v_des * (0.8 + 0.4 * np.random.random())
            
            vehicle_states[n] = {
                'x_n': initial_position,
                'v_n': initial_velocity,
                'a_n': 0.0,
                'd_n': self.desired_spacing[n] if n < len(self.desired_spacing) else 5.0
            }
        
        return vehicle_states
    
    def compute_relative_position(self, vehicle_id, current_states):
        if vehicle_id == 0:
            return float('inf')
        
        x_n = current_states[vehicle_id]['x_n']
        x_n_minus_1 = current_states[vehicle_id - 1]['x_n']
        d_n_minus_1 = current_states[vehicle_id - 1]['d_n']
        
        delta_x_n = x_n_minus_1 - x_n - d_n_minus_1
        
        return delta_x_n
    
    def compute_relative_velocity(self, vehicle_id, current_states):
        if vehicle_id == 0:
            return 0.0
        
        v_n = current_states[vehicle_id]['v_n']
        v_n_minus_1 = current_states[vehicle_id - 1]['v_n']
        
        delta_v_n = v_n - v_n_minus_1
        
        return delta_v_n
    
    def compute_safety_distance_function(self, v_n, delta_v_n):
        if v_n <= 0:
            return self.d_min
        
        safety_term = (v_n * delta_v_n) / (2 * math.sqrt(self.a_max * self.b_max))
        
        H_function = self.d_min + self.t_min * v_n + safety_term
        
        return max(H_function, self.d_min)
    
    def compute_vehicle_acceleration(self, vehicle_id, current_states):
        if vehicle_id == 0:
            leader_acceleration = 0.1 * math.sin(0.1 * sum(current_states[0].values()))
            return min(max(leader_acceleration, -self.b_max), self.a_max)
        
        v_n = current_states[vehicle_id]['v_n']
        delta_x_n = self.compute_relative_position(vehicle_id, current_states)
        delta_v_n = self.compute_relative_velocity(vehicle_id, current_states)
        
        if delta_x_n <= 0:
            return -self.b_max
        
        velocity_ratio_term = (v_n / self.v_des) ** self.delta
        
        H_value = self.compute_safety_distance_function(v_n, delta_v_n)
        spacing_ratio_term = (H_value / delta_x_n) ** 2
        
        acceleration = self.a_max * (1 - velocity_ratio_term - spacing_ratio_term)
        
        acceleration = min(max(acceleration, -self.b_max), self.a_max)
        
        return acceleration
    
    def update_vehicle_state(self, vehicle_id, current_states, time_step):
        current_state = current_states[vehicle_id]
        
        v_n_current = current_state['v_n']
        x_n_current = current_state['x_n']
        a_n_current = self.compute_vehicle_acceleration(vehicle_id, current_states)
        
        v_n_new = v_n_current + a_n_current * time_step
        v_n_new = max(v_n_new, 0.0)
        
        x_n_new = x_n_current + v_n_current * time_step + 0.5 * a_n_current * (time_step ** 2)
        
        updated_state = {
            'x_n': x_n_new,
            'v_n': v_n_new,
            'a_n': a_n_current,
            'd_n': current_state['d_n']
        }
        
        return updated_state
    
    def compute_safety_constraints(self, vehicle_states):
        safety_violations = []
        
        for vehicle_id in range(1, self.num_vehicles):
            delta_x_n = self.compute_relative_position(vehicle_id, vehicle_states)
            
            if delta_x_n < self.d_min:
                safety_violations.append({
                    'vehicle_id': vehicle_id,
                    'violation_type': 'minimum_spacing',
                    'current_spacing': delta_x_n,
                    'required_spacing': self.d_min
                })
            
            v_n = vehicle_states[vehicle_id]['v_n']
            delta_v_n = self.compute_relative_velocity(vehicle_id, vehicle_states)
            required_spacing = self.compute_safety_distance_function(v_n, delta_v_n)
            
            if delta_x_n < required_spacing:
                safety_violations.append({
                    'vehicle_id': vehicle_id,
                    'violation_type': 'safety_distance',
                    'current_spacing': delta_x_n,
                    'required_spacing': required_spacing
                })
        
        return safety_violations
    
    def simulate_platoon_dynamics(self, simulation_time, time_step):
        num_steps = int(simulation_time / time_step)
        trajectory_history = []
        
        current_states = self.vehicle_states.copy()
        
        for step in range(num_steps):
            new_states = {}
            
            for vehicle_id in range(self.num_vehicles):
                new_states[vehicle_id] = self.update_vehicle_state(
                    vehicle_id, current_states, time_step
                )
            
            current_states = new_states
            
            safety_violations = self.compute_safety_constraints(current_states)
            
            step_data = {
                'time': step * time_step,
                'vehicle_states': current_states.copy(),
                'safety_violations': safety_violations
            }
            trajectory_history.append(step_data)
        
        return trajectory_history

class PlatoonPerformanceMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def compute_string_stability(self, trajectory_history):
        string_stability_metrics = []
        
        for step_data in trajectory_history:
            vehicle_states = step_data['vehicle_states']
            
            velocity_amplification = []
            spacing_amplification = []
            
            for vehicle_id in range(1, len(vehicle_states)):
                v_n = vehicle_states[vehicle_id]['v_n']
                v_n_minus_1 = vehicle_states[vehicle_id - 1]['v_n']
                
                if abs(v_n_minus_1) > 1e-6:
                    velocity_ratio = abs(v_n) / abs(v_n_minus_1)
                    velocity_amplification.append(velocity_ratio)
                
                if vehicle_id >= 2:
                    x_n = vehicle_states[vehicle_id]['x_n']
                    x_n_minus_1 = vehicle_states[vehicle_id - 1]['x_n']
                    x_n_minus_2 = vehicle_states[vehicle_id - 2]['x_n']
                    
                    spacing_n = x_n_minus_1 - x_n
                    spacing_n_minus_1 = x_n_minus_2 - x_n_minus_1
                    
                    if abs(spacing_n_minus_1) > 1e-6:
                        spacing_ratio = abs(spacing_n) / abs(spacing_n_minus_1)
                        spacing_amplification.append(spacing_ratio)
            
            step_metrics = {
                'time': step_data['time'],
                'max_velocity_amplification': max(velocity_amplification) if velocity_amplification else 1.0,
                'max_spacing_amplification': max(spacing_amplification) if spacing_amplification else 1.0,
                'string_stable': all(amp <= 1.1 for amp in velocity_amplification + spacing_amplification)
            }
            string_stability_metrics.append(step_metrics)
        
        return string_stability_metrics
    
    def compute_fuel_efficiency(self, trajectory_history):
        fuel_consumption_metrics = []
        
        for step_data in trajectory_history:
            vehicle_states = step_data['vehicle_states']
            
            total_fuel_consumption = 0.0
            
            for vehicle_id, state in vehicle_states.items():
                v_n = state['v_n']
                a_n = state['a_n']
                
                base_consumption = 0.1 + 0.01 * (v_n ** 2)
                acceleration_penalty = 0.05 * (a_n ** 2) if a_n > 0 else 0.02 * (a_n ** 2)
                
                vehicle_fuel = base_consumption + acceleration_penalty
                total_fuel_consumption += vehicle_fuel
            
            fuel_consumption_metrics.append({
                'time': step_data['time'],
                'total_fuel_consumption': total_fuel_consumption,
                'average_fuel_per_vehicle': total_fuel_consumption / len(vehicle_states)
            })
        
        return fuel_consumption_metrics
    
    def compute_platoon_cohesion(self, trajectory_history):
        cohesion_metrics = []
        
        for step_data in trajectory_history:
            vehicle_states = step_data['vehicle_states']
            
            positions = [state['x_n'] for state in vehicle_states.values()]
            velocities = [state['v_n'] for state in vehicle_states.values()]
            
            position_variance = np.var(positions)
            velocity_variance = np.var(velocities)
            
            max_spacing = max(positions) - min(positions)
            velocity_spread = max(velocities) - min(velocities)
            
            cohesion_score = 1.0 / (1.0 + position_variance + velocity_variance)
            
            cohesion_metrics.append({
                'time': step_data['time'],
                'position_variance': position_variance,
                'velocity_variance': velocity_variance,
                'max_spacing': max_spacing,
                'velocity_spread': velocity_spread,
                'cohesion_score': cohesion_score
            })
        
        return cohesion_metrics

def create_vehicle_platooning_system():
    VEHICLE_DYNAMICS = {
        'a_max': 0.73,
        'b_max': 1.67,
        'v_des': 30.0,
        'd_min': 2.0,
        't_min': 1.5,
        'delta': 4,
        'tau': 1.0,
        'N': 20
    }
    
    platoon_model = IntelligentDriverModel(VEHICLE_DYNAMICS)
    performance_metrics = PlatoonPerformanceMetrics()
    
    return platoon_model, performance_metrics
