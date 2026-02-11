# ChessNet.py (1,000만 개 데이터 학습 최적화 버전)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False) # bias 제거 (Norm이 수행)
        self.gn1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self, input_channels=19):
        super().__init__()
        # 너비 확장: 128 -> 256 채널 (복잡한 패턴 인식)
        hidden_channels = 256
        
        self.conv_input = nn.Conv2d(input_channels, hidden_channels, 3, padding=1, bias=False)
        self.gn_input = nn.GroupNorm(32, hidden_channels)
        
        # 깊이 확장: 6개 -> 12개 블록 (1,000만 개 데이터의 전략적 깊이 수용)
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_channels) for _ in range(12)])
        
        # Policy Head: 채널 분리를 통해 정보 손실 방지
        self.policy_conv = nn.Conv2d(hidden_channels, 128, 1, bias=False)
        self.policy_gn = nn.GroupNorm(32, 128)
        self.policy_fc = nn.Linear(128 * 8 * 8, 4672)
        
        # Value Head: 더 깊은 MLP를 통해 정교한 승률 예측
        self.value_conv = nn.Conv2d(hidden_channels, 32, 1, bias=False)
        self.value_gn = nn.GroupNorm(32, 32)
        self.value_fc_1 = nn.Linear(32 * 8 * 8, 256) 
        self.value_fc_2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.gn_input(self.conv_input(x)))
        x = self.res_blocks(x)
        
        # Policy 예측
        p = F.relu(self.policy_gn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value 예측
        v = F.relu(self.value_gn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc_1(v)) 
        v = torch.tanh(self.value_fc_2(v))
        
        return p, v