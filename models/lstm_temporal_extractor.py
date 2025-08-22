import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalFeatureExtractor:
    def extract_temporal_features(self, sequence_input):
        raise NotImplementedError

    def compute_temporal_dependencies(self, hidden_states):
        raise NotImplementedError

class BaseLSTMArchitecture(TemporalFeatureExtractor):
    def __init__(self, lstm_config):
        self.input_size = lstm_config['input_size']
        self.hidden_size = lstm_config['hidden_size']
        self.num_layers = lstm_config['num_layers']
        self.bidirectional = lstm_config.get('bidirectional', True)
        self.dropout = lstm_config.get('dropout', 0.2)

class BidirectionalLSTMExtractor(BaseLSTMArchitecture, nn.Module):
    def __init__(self, lstm_config):
        BaseLSTMArchitecture.__init__(self, lstm_config)
        nn.Module.__init__(self)
        
        self.lstm_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        self.output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        self.layer_normalization = nn.LayerNorm(self.output_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.temporal_attention = TemporalAttentionMechanism(self.output_size)
        
        self._initialize_lstm_weights()
    
    def _initialize_lstm_weights(self):
        for name, param in self.lstm_layer.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                if self.bidirectional:
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
                else:
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
    
    def initialize_hidden_states(self, batch_size, device):
        num_directions = 2 if self.bidirectional else 1
        
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        
        return h_0, c_0
    
    def extract_temporal_features(self, sequence_input):
        batch_size, sequence_length, input_dim = sequence_input.shape
        
        h_0, c_0 = self.initialize_hidden_states(batch_size, sequence_input.device)
        
        lstm_output, (final_hidden, final_cell) = self.lstm_layer(sequence_input, (h_0, c_0))
        
        normalized_output = self.layer_normalization(lstm_output)
        
        temporal_features = self.dropout_layer(normalized_output)
        
        return temporal_features, (final_hidden, final_cell)
    
    def compute_temporal_dependencies(self, hidden_states):
        lstm_output, (final_hidden, final_cell) = hidden_states
        
        attended_features, attention_weights = self.temporal_attention(lstm_output)
        
        if self.bidirectional:
            forward_hidden = final_hidden[-2, :, :]
            backward_hidden = final_hidden[-1, :, :]
            combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)
        else:
            combined_hidden = final_hidden[-1, :, :]
        
        return attended_features, combined_hidden, attention_weights
    
    def forward(self, sequence_input):
        temporal_features, hidden_states = self.extract_temporal_features(sequence_input)
        
        attended_output, final_representation, attention_weights = self.compute_temporal_dependencies(
            (temporal_features, hidden_states)
        )
        
        return attended_output, final_representation, attention_weights

class TemporalAttentionMechanism(nn.Module):
    def __init__(self, hidden_size, attention_dim=None):
        super(TemporalAttentionMechanism, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim or hidden_size // 2
        
        self.attention_projection = nn.Linear(hidden_size, self.attention_dim, bias=False)
        self.context_vector = nn.Parameter(torch.randn(self.attention_dim))
        
        self.softmax = nn.Softmax(dim=1)
        
        nn.init.xavier_uniform_(self.attention_projection.weight)
        nn.init.normal_(self.context_vector, std=0.1)
    
    def forward(self, lstm_output):
        batch_size, sequence_length, hidden_size = lstm_output.shape
        
        projected_output = self.attention_projection(lstm_output)
        projected_output = torch.tanh(projected_output)
        
        attention_scores = torch.matmul(projected_output, self.context_vector)
        
        attention_weights = self.softmax(attention_scores)
        
        attended_output = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        
        return attended_output, attention_weights

class MultiScaleLSTMExtractor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 128, 256], dropout=0.2):
        super(MultiScaleLSTMExtractor, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_scales = len(hidden_sizes)
        
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        current_input_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            lstm_layer = nn.LSTM(
                input_size=current_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=True
            )
            
            self.lstm_layers.append(lstm_layer)
            
            output_size = hidden_size * 2
            self.layer_norms.append(nn.LayerNorm(output_size))
            
            current_input_size = output_size
        
        self.fusion_layer = nn.Linear(sum(h * 2 for h in hidden_sizes), hidden_sizes[-1])
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, sequence_input):
        batch_size, sequence_length, input_dim = sequence_input.shape
        
        scale_outputs = []
        current_input = sequence_input
        
        for i, (lstm_layer, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            h_0 = torch.zeros(2, batch_size, self.hidden_sizes[i], device=sequence_input.device)
            c_0 = torch.zeros(2, batch_size, self.hidden_sizes[i], device=sequence_input.device)
            
            lstm_output, _ = lstm_layer(current_input, (h_0, c_0))
            normalized_output = layer_norm(lstm_output)
            
            scale_outputs.append(normalized_output)
            current_input = normalized_output
        
        concatenated_features = torch.cat(scale_outputs, dim=-1)
        
        fused_features = self.fusion_layer(concatenated_features)
        final_output = self.dropout_layer(fused_features)
        
        return final_output, scale_outputs

class VehicularTemporalEncoder(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_vehicles=20):
        super(VehicularTemporalEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_vehicles = num_vehicles
        
        self.vehicle_lstm_extractors = nn.ModuleList()
        
        for vehicle_id in range(num_vehicles):
            lstm_config = {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bidirectional': True,
                'dropout': 0.2
            }
            
            vehicle_extractor = BidirectionalLSTMExtractor(lstm_config)
            self.vehicle_lstm_extractors.append(vehicle_extractor)
        
        self.inter_vehicle_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.global_temporal_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 * num_vehicles, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 4, hidden_size * 2)
        )
    
    def forward(self, vehicular_sequences):
        batch_size, num_vehicles, sequence_length, input_dim = vehicular_sequences.shape
        
        vehicle_representations = []
        vehicle_attention_weights = []
        
        for vehicle_id in range(self.num_vehicles):
            vehicle_sequence = vehicular_sequences[:, vehicle_id, :, :]
            
            vehicle_output, vehicle_repr, attention_weights = self.vehicle_lstm_extractors[vehicle_id](
                vehicle_sequence
            )
            
            vehicle_representations.append(vehicle_repr)
            vehicle_attention_weights.append(attention_weights)
        
        stacked_representations = torch.stack(vehicle_representations, dim=1)
        
        inter_vehicle_output, inter_vehicle_weights = self.inter_vehicle_attention(
            stacked_representations,
            stacked_representations,
            stacked_representations
        )
        
        flattened_representations = inter_vehicle_output.view(batch_size, -1)
        
        global_representation = self.global_temporal_fusion(flattened_representations)
        
        return {
            'global_representation': global_representation,
            'vehicle_representations': vehicle_representations,
            'inter_vehicle_attention': inter_vehicle_weights,
            'vehicle_attention_weights': vehicle_attention_weights
        }

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvolutionalNetwork, self).__init__()
        
        self.num_levels = len(num_channels)
        self.tcn_layers = nn.ModuleList()
        
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            tcn_block = TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size, 
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            )
            
            self.tcn_layers.append(tcn_block)
        
        self.output_projection = nn.Linear(num_channels[-1], num_channels[-1])
    
    def forward(self, input_tensor):
        input_tensor = input_tensor.transpose(1, 2)
        
        output = input_tensor
        for tcn_layer in self.tcn_layers:
            output = tcn_layer(output)
        
        output = output.transpose(1, 2)
        
        final_output = self.output_projection(output)
        
        return final_output

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

def create_temporal_extraction_system():
    LSTM_CONFIG = {
        'input_size': 64,
        'hidden_size': 128,
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.2
    }
    
    temporal_extractor = BidirectionalLSTMExtractor(LSTM_CONFIG)
    
    vehicular_encoder = VehicularTemporalEncoder(
        input_size=64,
        hidden_size=128,
        num_layers=2,
        num_vehicles=20
    )
    
    tcn_network = TemporalConvolutionalNetwork(
        input_channels=64,
        num_channels=[128, 128, 64],
        kernel_size=3,
        dropout=0.2
    )
    
    return temporal_extractor, vehicular_encoder, tcn_network
