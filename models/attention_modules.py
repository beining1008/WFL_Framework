import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AttentionMechanismInterface:
    def compute_attention_weights(self, query, key, value):
        raise NotImplementedError

    def apply_attention_mask(self, attention_scores, mask):
        raise NotImplementedError

class BaseMultiHeadAttention(AttentionMechanismInterface):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

class ScaledDotProductAttention(BaseMultiHeadAttention):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, temperature=1.0):
        super().__init__(d_model, num_heads, dropout_rate)
        self.temperature = temperature
        self.attention_dropout = nn.Dropout(dropout_rate)
        
    def compute_attention_weights(self, query, key, value, attention_mask=None):
        batch_size, seq_len, d_model = query.shape
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        
        if attention_mask is not None:
            scores = self.apply_attention_mask(scores, attention_mask)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        context = torch.matmul(attention_weights, value)
        
        return context, attention_weights
    
    def apply_attention_mask(self, attention_scores, mask):
        mask_value = -1e9
        masked_scores = attention_scores.masked_fill(mask == 0, mask_value)
        return masked_scores

class MultiHeadSelfAttentionLayer(nn.Module):
    def __init__(self, d_model=64, num_heads=8, dropout_rate=0.1, attention_temperature=0.08):
        super(MultiHeadSelfAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_temperature = attention_temperature
        
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.attention_mechanism = ScaledDotProductAttention(
            d_model, num_heads, dropout_rate, attention_temperature
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        for linear_layer in [self.query_linear, self.key_linear, self.value_linear]:
            nn.init.xavier_uniform_(linear_layer.weight)
        
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0)
    
    def forward(self, input_tensor, attention_mask=None):
        batch_size, seq_length, d_model = input_tensor.shape
        
        residual_connection = input_tensor
        
        Q = self.query_linear(input_tensor)
        K = self.key_linear(input_tensor)
        V = self.value_linear(input_tensor)
        
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_output, attention_weights = self.attention_mechanism.compute_attention_weights(
            Q, K, V, attention_mask
        )
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )
        
        output = self.output_linear(attention_output)
        output = self.dropout(output)
        
        output = self.layer_norm(output + residual_connection)
        
        return output, attention_weights

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=64, num_heads=8, dropout_rate=0.1):
        super(CrossAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * d_model, d_model)
        )
        
    def forward(self, query_input, key_value_input, attention_mask=None):
        batch_size, query_seq_len, d_model = query_input.shape
        _, kv_seq_len, _ = key_value_input.shape
        
        residual_1 = query_input
        
        Q = self.query_projection(query_input)
        K = self.key_projection(key_value_input)
        V = self.value_projection(key_value_input)
        
        Q = Q.view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context_vectors = torch.matmul(attention_weights, V)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(
            batch_size, query_seq_len, self.d_model
        )
        
        attention_output = self.output_projection(context_vectors)
        attention_output = self.dropout(attention_output)
        
        attention_output = self.layer_norm_1(attention_output + residual_1)
        
        residual_2 = attention_output
        ff_output = self.feed_forward(attention_output)
        ff_output = self.dropout(ff_output)
        
        final_output = self.layer_norm_2(ff_output + residual_2)
        
        return final_output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length=5000):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        pe = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, input_tensor):
        seq_length = input_tensor.size(1)
        positional_embeddings = self.pe[:seq_length, :].transpose(0, 1)
        
        return input_tensor + positional_embeddings

class VehicularAttentionModule(nn.Module):
    def __init__(self, d_model=64, num_heads=8, num_layers=3, dropout_rate=0.1):
        super(VehicularAttentionModule, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.self_attention_layers = nn.ModuleList([
            MultiHeadSelfAttentionLayer(d_model, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.cross_attention_layer = CrossAttentionLayer(d_model, num_heads, dropout_rate)
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        
    def create_vehicular_attention_mask(self, batch_size, seq_length, vehicle_positions):
        attention_mask = torch.ones(batch_size, seq_length, seq_length)
        
        for batch_idx in range(batch_size):
            for i in range(seq_length):
                for j in range(seq_length):
                    if vehicle_positions is not None:
                        distance = abs(vehicle_positions[batch_idx][i] - vehicle_positions[batch_idx][j])
                        if distance > 50.0:
                            attention_mask[batch_idx, i, j] = 0
        
        return attention_mask
    
    def forward(self, input_tensor, vehicle_positions=None, cross_attention_input=None):
        batch_size, seq_length, d_model = input_tensor.shape
        
        positional_input = self.positional_encoding(input_tensor)
        
        attention_mask = self.create_vehicular_attention_mask(
            batch_size, seq_length, vehicle_positions
        )
        
        current_output = positional_input
        all_attention_weights = []
        
        for layer_idx, attention_layer in enumerate(self.self_attention_layers):
            current_output, attention_weights = attention_layer(
                current_output, attention_mask
            )
            all_attention_weights.append(attention_weights)
        
        if cross_attention_input is not None:
            cross_output, cross_attention_weights = self.cross_attention_layer(
                current_output, cross_attention_input
            )
            current_output = cross_output
            all_attention_weights.append(cross_attention_weights)
        
        final_output = self.final_layer_norm(current_output)
        final_output = self.output_projection(final_output)
        
        return final_output, all_attention_weights

class AdaptiveAttentionTemperature(nn.Module):
    def __init__(self, d_model, initial_temperature=1.0):
        super(AdaptiveAttentionTemperature, self).__init__()
        
        self.temperature_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.initial_temperature = initial_temperature
        self.temperature_range = (0.1, 2.0)
    
    def forward(self, input_features):
        batch_size, seq_length, d_model = input_features.shape
        
        pooled_features = torch.mean(input_features, dim=1)
        
        temperature_logits = self.temperature_predictor(pooled_features)
        
        min_temp, max_temp = self.temperature_range
        adaptive_temperature = min_temp + (max_temp - min_temp) * temperature_logits
        
        return adaptive_temperature.squeeze(-1)

def create_vehicular_attention_system():
    ATTENTION_CONFIG = {
        'd_model': 64,
        'num_heads': 8,
        'num_layers': 3,
        'dropout_rate': 0.1,
        'max_sequence_length': 100
    }
    
    vehicular_attention = VehicularAttentionModule(
        d_model=ATTENTION_CONFIG['d_model'],
        num_heads=ATTENTION_CONFIG['num_heads'],
        num_layers=ATTENTION_CONFIG['num_layers'],
        dropout_rate=ATTENTION_CONFIG['dropout_rate']
    )
    
    adaptive_temperature = AdaptiveAttentionTemperature(
        d_model=ATTENTION_CONFIG['d_model']
    )
    
    return vehicular_attention, adaptive_temperature
