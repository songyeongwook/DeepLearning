import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, dimension, hidden_size, num_layers, num_classes, device, dropout=0.1):
        super().__init__()
        
        self.dimension = dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device
        
        # GRU with Dropout (num_layers > 1일 때만 효과)
        self.rnn = nn.GRU(self.dimension, self.hidden_size, self.num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 첫 번째 중간 layer
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 두 번째 중간 layer
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        #세 번쨰 중간 layer
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        # 최종 출력 layer
        self.fc4 = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, x):
        # (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # RNN forward: out shape = (batch_size, sequence_length, hidden_size)
        out, h_out = self.rnn(x, h0)
        
        # 마지막 time step 사용
        out = out[:, -1, :]
        
        # 첫 번째 중간 layer (fc1 → ReLU → Dropout)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 두 번째 중간 layer (fc2 → ReLU → Dropout)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # 최종 분류 layer
        out = self.fc3(out)
        
        return out

#%%
if __name__ == "__main__":
    # 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    sequence_length = 28
    dimension = 28
    hidden_size = 256
    num_layers = 4
    num_classes = 10
    
    # (batch_size, sequence_length, dimension)
    x = torch.randn(batch_size, sequence_length, dimension).to(device)
    
    model = GRU(dimension=dimension, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                num_classes=num_classes, 
                device=device,
                dropout=0.1).to(device)
    
    output = model(x)
    
    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")