import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 첫 번째 Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 두 번째 Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # GAP-based Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Input shape: (batch_size, 784) -> (batch_size, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        
        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # GAP Classifier
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
#%%
if __name__ == "__main__":
    # 데이터 테스트
    batch_size = 64
    in_dim = 28*28
    x = torch.randn(batch_size, in_dim)
    
    model = CNN()
    output = model(x)

    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # 각 레이어의 파라미터 크기 출력
    print("\nLayer-wise parameter summary:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}, {param.numel()} parameters")

    # 모델 파라미터 및 연산량(MACs) 계산
    import sys
    import os
    from thop import profile

    # Suppress thop output
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    macs, params = profile(model, inputs=(x, ))
    sys.stdout = original_stdout
    print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    # torch.Size([8, 784]) -> torch.Size([8, 10])
