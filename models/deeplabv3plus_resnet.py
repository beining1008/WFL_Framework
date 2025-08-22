import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResNetBackboneInterface:
    def extract_features(self, input_tensor):
        raise NotImplementedError
    
    def get_feature_channels(self):
        raise NotImplementedError

class ResNet101Backbone(ResNetBackboneInterface, nn.Module):
    def __init__(self, pretrained=True, output_stride=16):
        super(ResNet101Backbone, self).__init__()
        self.output_stride = output_stride
        self.pretrained = pretrained
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)

        if output_stride == 16:
            self.layer3 = self._make_layer(512, 256, 23, stride=2)
            self.layer4 = self._make_layer(1024, 512, 3, stride=1, dilation=2)
        elif output_stride == 8:
            self.layer3 = self._make_layer(512, 256, 23, stride=1, dilation=2)
            self.layer4 = self._make_layer(1024, 512, 3, stride=1, dilation=4)
        else:
            raise ValueError(f"Unsupported output_stride: {output_stride}")
        
        self.feature_channels = [256, 512, 1024, 2048]
        
        if pretrained:
            self._load_pretrained_weights()
            self._freeze_backbone_parameters()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dilation=1):
        layers = []
        
        downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        
        layers.append(ResNetBottleneck(in_channels, out_channels, stride, downsample, dilation))
        
        for i in range(1, blocks):
            layers.append(ResNetBottleneck(out_channels * 4, out_channels, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def _load_pretrained_weights(self):
        pass

    def _freeze_backbone_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def extract_features(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        low_level_features = self.layer1(x)
        x = self.layer2(low_level_features)
        x = self.layer3(x)
        high_level_features = self.layer4(x)
        
        return low_level_features, high_level_features
    
    def get_feature_channels(self):
        return self.feature_channels

class ResNetBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=1):
        super(ResNetBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilate_rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_blocks = nn.ModuleList()
        for rate in dilate_rates:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilate_rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        
        conv1x1_out = self.conv1x1(x)
        
        aspp_outs = []
        for aspp_block in self.aspp_blocks:
            aspp_outs.append(aspp_block(x))
        
        global_pool_out = self.global_avg_pool(x)
        global_pool_out = F.interpolate(global_pool_out, size=size, mode='bilinear', align_corners=False)
        
        concat_features = torch.cat([conv1x1_out] + aspp_outs + [global_pool_out], dim=1)
        
        return self.project(concat_features)

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels=256, high_level_channels=256, decoder_channels=256, num_classes=4):
        super(DeepLabV3PlusDecoder, self).__init__()
        
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(high_level_channels + 48, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
    
    def forward(self, low_level_features, high_level_features):
        low_level_features = self.low_level_conv(low_level_features)
        
        high_level_features = F.interpolate(
            high_level_features, 
            size=low_level_features.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        concat_features = torch.cat([low_level_features, high_level_features], dim=1)
        
        decoder_output = self.decoder_conv(concat_features)
        
        segmentation_logits = self.classifier(decoder_output)
        
        return segmentation_logits

class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone='resnet101', num_classes=4, pretrained=True, output_stride=16, aspp_dilate=[6, 12, 18]):
        super(DeepLabV3Plus, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.output_stride = output_stride
        
        if backbone == 'resnet101':
            self.backbone = ResNet101Backbone(pretrained=pretrained, output_stride=output_stride)
            backbone_channels = 2048
            low_level_channels = 256
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.aspp = ASPPModule(backbone_channels, 256, aspp_dilate)
        
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=low_level_channels,
            high_level_channels=256,
            decoder_channels=256,
            num_classes=num_classes
        )
        
        self._initialize_weights()
        self._setup_federated_learning_parameters()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _setup_federated_learning_parameters(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.aspp.parameters():
            param.requires_grad = True

        for param in self.decoder.parameters():
            param.requires_grad = True
    
    def forward(self, input_tensor):
        input_size = input_tensor.shape[-2:]
        
        low_level_features, high_level_features = self.backbone.extract_features(input_tensor)
        
        aspp_features = self.aspp(high_level_features)
        
        segmentation_logits = self.decoder(low_level_features, aspp_features)
        
        output = F.interpolate(
            segmentation_logits, 
            size=input_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return output
    
    def get_model_parameters_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        aspp_params = sum(p.numel() for p in self.aspp.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        federated_params = aspp_params + decoder_params

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'aspp_parameters': aspp_params,
            'decoder_parameters': decoder_params,
            'federated_learning_parameters': federated_params,
            'federated_params_millions': federated_params / 1_000_000,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'federated_model_size_bits': federated_params * 32,
            'federated_model_size_mbits': federated_params * 32 / 1_000_000
        }

def create_deeplabv3plus_model():
    AI4MARS_CONFIG = {
        'backbone': 'resnet101',
        'num_classes': 4,
        'pretrained': True,
        'output_stride': 16,
        'aspp_dilate_rates': [6, 12, 18],
        'input_size': (513, 513)
    }
    
    model = DeepLabV3Plus(
        backbone=AI4MARS_CONFIG['backbone'],
        num_classes=AI4MARS_CONFIG['num_classes'],
        pretrained=AI4MARS_CONFIG['pretrained'],
        output_stride=AI4MARS_CONFIG['output_stride'],
        aspp_dilate=AI4MARS_CONFIG['aspp_dilate_rates']
    )
    
    return model
