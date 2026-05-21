import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First Convolutional Layer (Depthwise Pointwise)
        self.dw_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, groups=1)
        self.pw_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Using MaxPool

        # Second Convolutional Layer (Depthwise Pointwise)
        self.dw_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=1, groups=32)
        self.pw_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Using MaxPool

        # third Convolutional Layer (Depthwise Poinrwise)
        self.dw_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1, groups=64)
        self.pw_conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Using MaxPool
 
        # Classifier head with Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)

        # Conv layers
        x = self.dw_conv1(x)
        x = self.pw_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.dw_conv2(x)
        x = self.pw_conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.dw_conv3(x)
        x = self.pw_conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Classifier
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
#%%
if __name__ == "__main__":
    # 데이터 테스트
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)

    model = CNN()
    output = model(x)

    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # 모델 파라미터 및 연산량(MACs) 계산 pip install thop
    try:
        from thop import profile
        macs, params = profile(model, inputs=(torch.randn(1, 1, 28, 28), ))
        print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')
    except ImportError:
        print("thop is not installed. Please install it using 'pip install thop'")

        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {output.shape}")