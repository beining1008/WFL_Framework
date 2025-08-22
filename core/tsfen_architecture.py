import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TemporalSpatialFeatureExtractor:
    def extract_features(self, input_tensor):
        raise NotImplementedError

class BaseNeuralArchitecture(TemporalSpatialFeatureExtractor):
    def __init__(self, config_dict):
        self.configuration_parameters = config_dict
        self.device_allocation_strategy = self._initialize_computation_backend()
    
    def _initialize_computation_backend(self):
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

class MultiHeadSelfAttentionModule(BaseNeuralArchitecture):
    def __init__(self, d_model=64, num_heads=8, dropout_rate=0.1, attention_temperature=0.08):
        super().__init__({'d_model': d_model, 'num_heads': num_heads})
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_temperature = attention_temperature
        self.head_dimension = d_model // num_heads
        
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        
        self._initialize_weight_matrices()
    
    def _initialize_weight_matrices(self):
        for module in [self.query_projection, self.key_projection, self.value_projection]:
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
        
    
    def forward(self, input_tensor, attention_mask=None):
        batch_size, sequence_length, embedding_dim = input_tensor.shape
        
        Q = self.query_projection(input_tensor)
        K = self.key_projection(input_tensor)
        V = self.value_projection(input_tensor)
        
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dimension).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dimension).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dimension).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dimension) * self.attention_temperature)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        context_vectors = torch.matmul(attention_weights, V)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        
        output = self.output_projection(context_vectors)
        return output, attention_weights

class LSTMTemporalExtractor(BaseNeuralArchitecture):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2):
        super().__init__({'hidden_size': hidden_size, 'num_layers': num_layers})
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.layer_normalization = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        
        h_0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(input_sequence.device)
        
        c_0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(input_sequence.device)
        
        lstm_output, (hidden_state, cell_state) = self.lstm_layer(input_sequence, (h_0, c_0))
        
        normalized_output = self.layer_normalization(lstm_output)
        final_output = self.dropout_layer(normalized_output)
        
        return final_output, hidden_state

class TSFENArchitecture(nn.Module):
    def __init__(self, mhsa_config, lstm_config, fc_config):
        super(TSFENArchitecture, self).__init__()
        
        self.multi_head_attention = MultiHeadSelfAttentionModule(
            d_model=mhsa_config['dmodel'],
            num_heads=mhsa_config['num_heads'],
            dropout_rate=mhsa_config['dropout_rate'],
            attention_temperature=mhsa_config['attention_temperature']
        )
        
        self.temporal_extractor = LSTMTemporalExtractor(
            input_size=mhsa_config['dmodel'],
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            bidirectional=lstm_config['bidirectional'],
            dropout=lstm_config['dropout']
        )
        
        lstm_output_dim = lstm_config['hidden_size'] * (2 if lstm_config['bidirectional'] else 1)
        
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(lstm_output_dim, fc_config['hidden_dims'][0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_config['hidden_dims'][0], fc_config['hidden_dims'][1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_config['hidden_dims'][1], fc_config['output_dim'])
        )
        
        self.positional_encoding = self._generate_positional_encoding(
            max_length=mhsa_config['input_shape'][2],
            d_model=mhsa_config['dmodel']
        )
    
    def _generate_positional_encoding(self, max_length, d_model):
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, input_tensor):
        batch_size, num_agents, sequence_length, feature_dim = input_tensor.shape
        
        reshaped_input = input_tensor.view(batch_size * num_agents, sequence_length, feature_dim)
        
        if reshaped_input.size(-1) != self.positional_encoding.size(-1):
            projection_matrix = torch.randn(feature_dim, self.positional_encoding.size(-1)).to(reshaped_input.device)
            reshaped_input = torch.matmul(reshaped_input, projection_matrix)
        
        positional_input = reshaped_input + self.positional_encoding[:, :sequence_length, :].to(reshaped_input.device)
        
        attention_output, attention_weights = self.multi_head_attention(positional_input)
        
        temporal_output, hidden_states = self.temporal_extractor(attention_output)
        
        final_hidden = temporal_output[:, -1, :]
        
        policy_logits = self.fully_connected_layers(final_hidden)
        
        output_reshaped = policy_logits.view(batch_size, num_agents, -1)
        
        return output_reshaped, attention_weights

class AdaptiveMaskingStrategy:
    def __init__(self, lambda_theta=0.1, beta=1.0):
        self.lambda_theta = lambda_theta
        self.beta = beta
        self.mu_over_L_ratio = 0.01
    
    def compute_flmd_mask(self, theta_n_values, current_round):
        basic_mask = (theta_n_values <= self.lambda_theta).float()
        
        adaptive_mask = torch.where(
            theta_n_values <= self.lambda_theta,
            torch.ones_like(theta_n_values),
            torch.exp(-self.beta * theta_n_values * (1 - self.mu_over_L_ratio) ** (-current_round))
        )
        
        return basic_mask, adaptive_mask
    
    def apply_policy_masking(self, policy_logits, mask):
        masked_logits = policy_logits * mask.unsqueeze(-1)
        masked_logits = masked_logits + (1 - mask.unsqueeze(-1)) * (-1e9)
        return F.softmax(masked_logits, dim=-1)

class TSFENPolicyNetwork(TSFENArchitecture):
    def __init__(self, mhsa_config, lstm_config, fc_config, masking_strategy):
        super().__init__(mhsa_config, lstm_config, fc_config)
        self.masking_strategy = masking_strategy
        
    def forward(self, state_tensor, theta_n_values, current_round):
        policy_features, attention_weights = super().forward(state_tensor)
        
        basic_mask, adaptive_mask = self.masking_strategy.compute_flmd_mask(theta_n_values, current_round)
        
        masked_policy = self.masking_strategy.apply_policy_masking(policy_features, adaptive_mask)
        
        return masked_policy, attention_weights, basic_mask

def create_tsfen_architecture():
    MHSA_CONFIG = {
        'dmodel': 64,
        'num_heads': 8,
        'input_shape': (32, 5, 20, 64),
        'dropout_rate': 0.1,
        'attention_temperature': 0.08
    }
    
    LSTM_CONFIG = {
        'hidden_size': 128,
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.2,
        'batch_first': True
    }
    
    ACTOR_CRITIC_CONFIG = {
        'input_dim': 128,
        'hidden_dims': [128, 128],
        'output_dim': 20,
        'activation': 'ReLU'
    }
    
    masking_strategy = AdaptiveMaskingStrategy(lambda_theta=0.1, beta=1.0)
    
    tsfen_policy = TSFENPolicyNetwork(
        mhsa_config=MHSA_CONFIG,
        lstm_config=LSTM_CONFIG,
        fc_config=ACTOR_CRITIC_CONFIG,
        masking_strategy=masking_strategy
    )
    
    return tsfen_policy

class TSFENArchitecture(BaseNeuralArchitecture):
    def __init__(self, config):
        super().__init__()

        self.mhsa_config = config['mhsa']
        self.lstm_config = config['lstm']
        self.fc_config = config['actor_critic']

        self.mhsa_layer = MultiHeadSelfAttentionMechanism(
            embedding_dimension=self.mhsa_config['dmodel'],
            num_attention_heads=self.mhsa_config['num_heads'],
            dropout_probability=self.mhsa_config['dropout_rate'],
            attention_temperature=self.mhsa_config['attention_temperature']
        )

        self.lstm_layer = nn.LSTM(
            input_size=self.lstm_config['input_size'],
            hidden_size=self.lstm_config['hidden_size'],
            num_layers=self.lstm_config['num_layers'],
            bidirectional=self.lstm_config['bidirectional'],
            dropout=self.lstm_config['dropout'],
            batch_first=self.lstm_config['batch_first']
        )

        lstm_output_size = self.lstm_config['hidden_size'] * 2 if self.lstm_config['bidirectional'] else self.lstm_config['hidden_size']

        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, self.fc_config['hidden_dims'][0]),
            nn.ReLU(),
            nn.Dropout(self.fc_config['dropout_rate']),
            nn.Linear(self.fc_config['hidden_dims'][0], self.fc_config['hidden_dims'][1]),
            nn.ReLU(),
            nn.Dropout(self.fc_config['dropout_rate']),
            nn.Linear(self.fc_config['hidden_dims'][1], self.fc_config['output_dim'])
        )

        self.adaptive_mask_beta = config.get('adaptive_mask_beta', 1.0)
        self.flmd_threshold = config.get('flmd_threshold', 0.1)

    def extract_features(self, input_tensor):
        batch_size, M, N, feature_dim = input_tensor.shape

        mhsa_output = self.mhsa_layer(input_tensor.view(batch_size * M, N, feature_dim))
        mhsa_output = mhsa_output.view(batch_size, M, N, feature_dim)

        sequence_features = mhsa_output.mean(dim=2)

        lstm_output, (h_M, c_M) = self.lstm_layer(sequence_features)

        final_hidden_state = h_M[-1] if not self.lstm_config['bidirectional'] else torch.cat([h_M[-2], h_M[-1]], dim=1)

        fc_output = self.fc_layers(final_hidden_state)

        return fc_output

    def apply_adaptive_flmd_mask(self, policy_logits, flmd_values, communication_round):
        adaptive_mask = torch.ones_like(policy_logits)

        for i, flmd_val in enumerate(flmd_values):
            if flmd_val <= self.flmd_threshold:
                adaptive_mask[i] = 1.0
            else:
                decay_factor = (1 - 0.01) ** (-communication_round)
                adaptive_mask[i] = torch.exp(-self.adaptive_mask_beta * flmd_val * decay_factor)

        masked_logits = policy_logits * adaptive_mask

        return masked_logits

    def forward(self, state_sequence, flmd_values=None, communication_round=0):
        tsfen_output = self.extract_features(state_sequence)

        if flmd_values is not None:
            tsfen_output = self.apply_adaptive_flmd_mask(tsfen_output, flmd_values, communication_round)

        policy_distribution = F.softmax(tsfen_output, dim=-1)

        return policy_distribution

def create_tsfen_system():
    from config.system_hyperparameters import get_system_configuration

    config = get_system_configuration()

    tsfen_model = TSFENArchitecture(config)

    return tsfen_model
