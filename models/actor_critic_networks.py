import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticNetworkInterface:
    def compute_policy_distribution(self, state_input):
        raise NotImplementedError

    def estimate_state_value(self, state_input):
        raise NotImplementedError

class BaseNeuralNetworkArchitecture(ActorCriticNetworkInterface):
    def __init__(self, network_config):
        self.input_dim = network_config['input_dim']
        self.hidden_dims = network_config['hidden_dims']
        self.output_dim = network_config['output_dim']
        self.activation_function = network_config.get('activation', 'ReLU')

class DeepActorNetwork(BaseNeuralNetworkArchitecture, nn.Module):
    def __init__(self, network_config):
        BaseNeuralNetworkArchitecture.__init__(self, network_config)
        nn.Module.__init__(self)
        
        self.policy_layers = self._build_policy_network()
        self.action_distribution_head = nn.Linear(self.hidden_dims[-1], self.output_dim)
        
        self.policy_regularization = nn.Dropout(0.2)
        self.batch_normalization = nn.BatchNorm1d(self.hidden_dims[-1])
        
        self._initialize_network_weights()
    
    def _build_policy_network(self):
        layers = []
        input_size = self.input_dim
        
        for i, hidden_size in enumerate(self.hidden_dims):
            layers.append(nn.Linear(input_size, hidden_size))
            
            if self.activation_function == 'ReLU':
                layers.append(nn.ReLU())
            elif self.activation_function == 'Tanh':
                layers.append(nn.Tanh())
            elif self.activation_function == 'LeakyReLU':
                layers.append(nn.LeakyReLU(0.2))
            
            if i < len(self.hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))
            
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _initialize_network_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)
    
    def compute_policy_distribution(self, state_input):
        batch_size = state_input.shape[0]
        
        policy_features = self.policy_layers(state_input)
        
        if batch_size > 1:
            normalized_features = self.batch_normalization(policy_features)
        else:
            normalized_features = policy_features
        
        regularized_features = self.policy_regularization(normalized_features)
        
        action_logits = self.action_distribution_head(regularized_features)
        
        return action_logits
    
    def forward(self, state_input, action_mask=None):
        action_logits = self.compute_policy_distribution(state_input)
        
        if action_mask is not None:
            masked_logits = action_logits + (1 - action_mask) * (-1e9)
            policy_probabilities = F.softmax(masked_logits, dim=-1)
        else:
            policy_probabilities = F.softmax(action_logits, dim=-1)
        
        return policy_probabilities, action_logits

class DeepCriticNetwork(BaseNeuralNetworkArchitecture, nn.Module):
    def __init__(self, network_config):
        BaseNeuralNetworkArchitecture.__init__(self, network_config)
        nn.Module.__init__(self)
        
        self.value_layers = self._build_value_network()
        self.state_value_head = nn.Linear(self.hidden_dims[-1], 1)
        
        self.value_regularization = nn.Dropout(0.2)
        self.layer_normalization = nn.LayerNorm(self.hidden_dims[-1])
        
        self._initialize_network_weights()
    
    def _build_value_network(self):
        layers = []
        input_size = self.input_dim
        
        for i, hidden_size in enumerate(self.hidden_dims):
            layers.append(nn.Linear(input_size, hidden_size))
            
            if self.activation_function == 'ReLU':
                layers.append(nn.ReLU())
            elif self.activation_function == 'Tanh':
                layers.append(nn.Tanh())
            elif self.activation_function == 'LeakyReLU':
                layers.append(nn.LeakyReLU(0.2))
            
            if i < len(self.hidden_dims) - 1:
                layers.append(nn.Dropout(0.15))
            
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _initialize_network_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def estimate_state_value(self, state_input):
        value_features = self.value_layers(state_input)
        
        normalized_features = self.layer_normalization(value_features)
        regularized_features = self.value_regularization(normalized_features)
        
        state_value = self.state_value_head(regularized_features)
        
        return state_value.squeeze(-1)
    
    def forward(self, state_input):
        state_value = self.estimate_state_value(state_input)
        return state_value

class SharedBackboneActorCritic(nn.Module):
    def __init__(self, network_config):
        super(SharedBackboneActorCritic, self).__init__()
        
        self.input_dim = network_config['input_dim']
        self.hidden_dims = network_config['hidden_dims']
        self.action_dim = network_config['output_dim']
        
        self.shared_backbone = self._build_shared_layers()
        
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dims[-1] // 2, self.action_dim)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_dims[-1])
        
        self._initialize_weights()
    
    def _build_shared_layers(self):
        layers = []
        input_size = self.input_dim
        
        for hidden_size in self.hidden_dims:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state_input, action_mask=None):
        shared_features = self.shared_backbone(state_input)
        normalized_features = self.layer_norm(shared_features)
        
        action_logits = self.actor_head(normalized_features)
        state_value = self.critic_head(normalized_features).squeeze(-1)
        
        if action_mask is not None:
            masked_logits = action_logits + (1 - action_mask) * (-1e9)
            action_probabilities = F.softmax(masked_logits, dim=-1)
        else:
            action_probabilities = F.softmax(action_logits, dim=-1)
        
        return action_probabilities, state_value, action_logits

class DuelingNetworkArchitecture(nn.Module):
    def __init__(self, network_config):
        super(DuelingNetworkArchitecture, self).__init__()
        
        self.input_dim = network_config['input_dim']
        self.hidden_dims = network_config['hidden_dims']
        self.action_dim = network_config['output_dim']
        
        self.feature_extractor = self._build_feature_layers()
        
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1] // 2, self.action_dim)
        )
        
        self._initialize_weights()
    
    def _build_feature_layers(self):
        layers = []
        input_size = self.input_dim
        
        for hidden_size in self.hidden_dims:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state_input):
        features = self.feature_extractor(state_input)
        
        state_value = self.value_stream(features)
        action_advantages = self.advantage_stream(features)
        
        advantage_mean = torch.mean(action_advantages, dim=-1, keepdim=True)
        normalized_advantages = action_advantages - advantage_mean
        
        q_values = state_value + normalized_advantages
        
        return q_values, state_value.squeeze(-1), normalized_advantages

class VehicularActorCriticEnsemble(nn.Module):
    def __init__(self, network_config, num_vehicles=20):
        super(VehicularActorCriticEnsemble, self).__init__()
        
        self.num_vehicles = num_vehicles
        self.network_config = network_config
        
        self.vehicle_actors = nn.ModuleList()
        self.vehicle_critics = nn.ModuleList()
        
        for vehicle_id in range(num_vehicles):
            actor_network = DeepActorNetwork(network_config)
            critic_network = DeepCriticNetwork(network_config)
            
            self.vehicle_actors.append(actor_network)
            self.vehicle_critics.append(critic_network)
        
        self.global_coordination_layer = nn.Sequential(
            nn.Linear(network_config['output_dim'] * num_vehicles, network_config['hidden_dims'][-1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(network_config['hidden_dims'][-1], network_config['output_dim'])
        )
        
        self.inter_vehicle_attention = nn.MultiheadAttention(
            embed_dim=network_config['hidden_dims'][-1],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, vehicular_states, action_masks=None):
        batch_size, num_vehicles, state_dim = vehicular_states.shape
        
        vehicle_policies = []
        vehicle_values = []
        vehicle_logits = []
        
        for vehicle_id in range(self.num_vehicles):
            vehicle_state = vehicular_states[:, vehicle_id, :]
            vehicle_mask = action_masks[:, vehicle_id, :] if action_masks is not None else None
            
            policy_probs, action_logits = self.vehicle_actors[vehicle_id](vehicle_state, vehicle_mask)
            state_value = self.vehicle_critics[vehicle_id](vehicle_state)
            
            vehicle_policies.append(policy_probs)
            vehicle_values.append(state_value)
            vehicle_logits.append(action_logits)
        
        stacked_policies = torch.stack(vehicle_policies, dim=1)
        stacked_values = torch.stack(vehicle_values, dim=1)
        stacked_logits = torch.stack(vehicle_logits, dim=1)
        
        flattened_logits = stacked_logits.view(batch_size, -1)
        global_coordination = self.global_coordination_layer(flattened_logits)
        
        return {
            'vehicle_policies': stacked_policies,
            'vehicle_values': stacked_values,
            'vehicle_logits': stacked_logits,
            'global_coordination': global_coordination
        }

class ParameterSharedActorCritic(nn.Module):
    def __init__(self, network_config):
        super(ParameterSharedActorCritic, self).__init__()
        
        self.shared_actor_critic = SharedBackboneActorCritic(network_config)
        
        self.vehicle_embedding = nn.Embedding(20, network_config['input_dim'] // 4)
        
        self.state_projection = nn.Linear(
            network_config['input_dim'] + network_config['input_dim'] // 4,
            network_config['input_dim']
        )
    
    def forward(self, vehicular_states, vehicle_ids, action_masks=None):
        batch_size, num_vehicles, state_dim = vehicular_states.shape
        
        vehicle_embeddings = self.vehicle_embedding(vehicle_ids)
        
        if vehicle_embeddings.dim() == 2:
            vehicle_embeddings = vehicle_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        augmented_states = torch.cat([vehicular_states, vehicle_embeddings], dim=-1)
        
        projected_states = self.state_projection(augmented_states)
        
        flattened_states = projected_states.view(-1, projected_states.shape[-1])
        flattened_masks = action_masks.view(-1, action_masks.shape[-1]) if action_masks is not None else None
        
        policies, values, logits = self.shared_actor_critic(flattened_states, flattened_masks)
        
        reshaped_policies = policies.view(batch_size, num_vehicles, -1)
        reshaped_values = values.view(batch_size, num_vehicles)
        reshaped_logits = logits.view(batch_size, num_vehicles, -1)
        
        return reshaped_policies, reshaped_values, reshaped_logits

def create_actor_critic_networks():
    ACTOR_CRITIC_CONFIG = {
        'input_dim': 128,
        'hidden_dims': [128, 128],
        'output_dim': 20,
        'activation': 'ReLU'
    }
    
    actor_network = DeepActorNetwork(ACTOR_CRITIC_CONFIG)
    critic_network = DeepCriticNetwork(ACTOR_CRITIC_CONFIG)
    
    shared_network = SharedBackboneActorCritic(ACTOR_CRITIC_CONFIG)
    
    dueling_network = DuelingNetworkArchitecture(ACTOR_CRITIC_CONFIG)
    
    ensemble_network = VehicularActorCriticEnsemble(ACTOR_CRITIC_CONFIG, num_vehicles=20)
    
    parameter_shared = ParameterSharedActorCritic(ACTOR_CRITIC_CONFIG)
    
    return {
        'actor': actor_network,
        'critic': critic_network,
        'shared': shared_network,
        'dueling': dueling_network,
        'ensemble': ensemble_network,
        'parameter_shared': parameter_shared
    }
