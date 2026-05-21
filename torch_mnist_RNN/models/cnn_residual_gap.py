import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 첫 번째 Convolutional Layer (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual Connection (14x14 -> 7x7)
        # Main path
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Shortcut path to match dimensions (channels and size)
        self.shortcut = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2), # 1x1 conv to match channel size
            nn.BatchNorm2d(64)
        )

        # GAP-based Classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # Input shape: (batch_size, 784) -> (batch_size, 1, 28, 28)
        x = x.view(-1, 1, 28, 28)
        
        # First layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # --- Residual Connection ---
        identity = self.shortcut(x)
        
        out = self.conv2(x)
        out = self.bn2(out)
        
        out += identity # Add the residual
        x = F.relu(out)

        # GAP Classifier
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # 데이터 테스트
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)
    
    model = CNN()
    output = model(x)

    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # 모델 파라미터 및 연산량(MACs) 계산
    try:
        from thop import profile
        macs, params = profile(model, inputs=(torch.randn(1, 1, 28, 28), ))
        print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')
    except ImportError:
        print("thop is not installed. Please install it using 'pip install thop'")

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
