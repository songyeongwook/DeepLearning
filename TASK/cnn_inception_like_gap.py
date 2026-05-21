import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Inception-like 모듈과 Global Average Pooling을 사용한 CNN.
    파라미터 수를 0.038M (38,000개) 미만으로 제한하면서 성능을 극대화하도록 설계되었습니다.
    """
    def __init__(self):
        super().__init__()
        # 초기 특징 추출을 위한 Stem 레이어
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 28x28 -> 14x14
        )

        # --- Inception-like Block 1 (입력 채널: 32, 출력 채널: 64) ---
        # 경로 1: 1x1 Convolution
        self.inc1_path1 = nn.Conv2d(32, 16, kernel_size=1)

        # 경로 2: 3x3 Convolution
        self.inc1_path2 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=1),
            nn.Conv2d(8, 24, kernel_size=3, padding=1)
        )

        # 경로 3: 5x5 Convolution
        self.inc1_path3 = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1),
            nn.Conv2d(4, 8, kernel_size=5, padding=2)
        )

        # 경로 4: Max Pooling
        self.inc1_path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 16, kernel_size=1)
        )

        self.pool1 = nn.MaxPool2d(2) # 14x14 -> 7x7

        # --- Inception-like Block 2 (입력 채널: 64, 출력 채널: 128) ---
        # 경로 1: 1x1 Convolution
        self.inc2_path1 = nn.Conv2d(64, 32, kernel_size=1)

        # 경로 2: 3x3 Convolution
        self.inc2_path2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.Conv2d(16, 48, kernel_size=3, padding=1)
        )

        # 경로 3: 5x5 Convolution
        self.inc2_path3 = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1),
            nn.Conv2d(8, 16, kernel_size=5, padding=2)
        )

        # 경로 4: Max Pooling
        self.inc2_path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=1)
        )

        # 분류기 헤드
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.stem(x)

        # Inception 1
        out1 = self.inc1_path1(x)
        out2 = self.inc1_path2(x)
        out3 = self.inc1_path3(x)
        out4 = self.inc1_path4(x)
        x = torch.cat([out1, out2, out3, out4], dim=1) # 채널 차원에서 합치기

        x = self.pool1(x)

        # Inception 2
        out1 = self.inc2_path1(x)
        out2 = self.inc2_path2(x)
        out3 = self.inc2_path3(x)
        out4 = self.inc2_path4(x)
        x = torch.cat([out1, out2, out3, out4], dim=1) # 채널 차원에서 합치기
        
        # 분류기
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
