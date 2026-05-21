import torch
import torch.nn as nn

#%%
class CNN_Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 28x28 -> 14x14 -> 7x7 -> 1x1 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),  # 7x7 -> 1x1 (64 channels)
            nn.Flatten(),  # (batch, 64, 1, 1) -> (batch, 64)
            nn.Linear(64, latent_dim),  # 64 -> latent_dim
        )
        
        # Decoder: latent_dim -> 64 -> 1x1 -> 7x7 -> 14x14 -> 28x28
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  # latent_dim -> 64
            nn.ReLU(),
            nn.Unflatten(1, (64, 1, 1)),  # (batch, 64) -> (batch, 64, 1, 1)
            nn.ConvTranspose2d(64, 32, 7),  # 1x1 -> 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space to output"""
        return self.decoder(z)

#%%
if __name__ == "__main__":
    # 데이터 테스트
    batch_size = 64
    latent_dim = 32  # 테스트용 latent dimension
    x = torch.randn(batch_size, 1, 28, 28)
    
    model = CNN_Autoencoder(latent_dim=latent_dim)
    output = model(x)

    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Latent dimension: {latent_dim}")
    print(f"Number of trainable parameters: {num_params:,}")

    # 각 레이어의 파라미터 크기 출력
    print("\nLayer-wise parameter summary:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}, {param.numel()} parameters")

    # 모델 파라미터 및 연산량(MACs) 계산 pip install thop
    try:
        from thop import profile
        macs, params = profile(model, inputs=(x, ))
        print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')
    except ImportError:
        print("thop is not installed. Please install it using 'pip install thop'")

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

