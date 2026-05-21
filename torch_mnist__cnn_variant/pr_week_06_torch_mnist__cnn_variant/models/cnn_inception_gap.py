import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 28x28 -> 14x14
        )

        # --- Inlined Inception Block 1 (in_channels=32, out_channels=64) ---
        # Path 1: 1x1 Convolution
        self.inc1_path1 = nn.Conv2d(32, 16, kernel_size=1)

        # Path 2: 3x3 Convolution
        self.inc1_path2 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=1),
            nn.Conv2d(8, 24, kernel_size=3, padding=1)
        )

        # Path 3: 5x5 Convolution
        self.inc1_path3 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1),
            nn.Conv2d(4, 8, kernel_size=5, padding=2)
        )

        # Path 4: Max Pooling
        self.inc1_path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.pool1 = nn.MaxPool2d(2) # 14x14 -> 7x7

        # --- Inlined Inception Block 2 (in_channels=64, out_channels=128) ---
        # Path 1: 1x1 Convolution
        self.inc2_path1 = nn.Conv2d(64, 32, kernel_size=1)

        # Path 2: 3x3 Convolution
        self.inc2_path2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.Conv2d(16, 48, kernel_size=3, padding=1)
        )

        # Path 3: 5x5 Convolution
        self.inc2_path3 = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1),
            nn.Conv2d(8, 16, kernel_size=5, padding=2)
        )

        # Path 4: Max Pooling
        self.inc2_path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=1)
        )

        # Classifier head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.stem(x)

        # --- Inception 1 Forward ---
        out1 = self.inc1_path1(x)
        out2 = self.inc1_path2(x)
        out3 = self.inc1_path3(x)
        out4 = self.inc1_path4(x)
        x = torch.cat([out1, out2, out3, out4], dim=1) # Concatenate along channel dimension

        x = self.pool1(x)

        # --- Inception 2 Forward ---
        out1 = self.inc2_path1(x)
        out2 = self.inc2_path2(x)
        out3 = self.inc2_path3(x)
        out4 = self.inc2_path4(x)
        x = torch.cat([out1, out2, out3, out4], dim=1) # Concatenate along channel dimension
        
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
        # Correct input shape for thop profile
        macs, params = profile(model, inputs=(torch.randn(1, 1, 28, 28), ))
        print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')
    except ImportError:
        print("thop is not installed. Please install it using 'pip install thop'")

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
