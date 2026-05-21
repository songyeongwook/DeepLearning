import os
import argparse
import torch
import torch.nn as nn
from data_loader import load_mnist
from models.autoencoder_cnn import CNN_Autoencoder
from torch.utils.data import DataLoader

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
plt.close("all")

#%%
def add_noise(clean, noise_factor=0.2):
    """Add Gaussian noise to clean images"""
    noisy = clean + torch.randn_like(clean) * noise_factor
    return noisy

#%%
def show_sample(original, noisy, reconstructed, idx: int=0):
    """
    Display original, noisy, and reconstructed images side by side
    """
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Clip values to [0, 1] range for proper display
    axes[0].imshow(np.clip(original[idx], 0, 1), cmap='gray')
    axes[0].set_title(f"Original Image (idx={idx})")
    axes[0].axis('off')
    
    axes[1].imshow(np.clip(noisy[idx], 0, 1), cmap='gray')
    axes[1].set_title(f"Noisy Image")
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(reconstructed[idx], 0, 1), cmap='gray')
    axes[2].set_title(f"Reconstructed Image")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

#%%
def test(model, test_loader, criterion, device=None):
    """Test the CNN autoencoder model"""
    model.eval()
    test_loss = 0
    all_originals = []
    all_noisy = []
    all_reconstructed = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            # Reshape data to (N, 1, 28, 28) for CNN
            data = data.view(-1, 1, 28, 28)
            target = data
            
            if device:
                data, target = data.to(device), target.to(device)
            
            # Add noise
            noisy = add_noise(data)
            
            # Forward pass
            output = model(noisy)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            # Store for visualization (convert to 28x28 for display)
            all_originals.extend(data.view(-1, 28, 28).cpu().numpy())
            all_noisy.extend(noisy.view(-1, 28, 28).cpu().numpy())
            all_reconstructed.extend(output.view(-1, 28, 28).cpu().numpy())
    
    avg_loss = test_loss / len(test_loader)
    return avg_loss, all_originals, all_noisy, all_reconstructed

#%%
def main(args):
    # DataLoader 준비
    _, _, test_ds, in_dim, x_test_raw, y_test = load_mnist(valDB_portion=0.1)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # 모델 파일 찾기
    if args.model_type == "best":
        model_path = os.path.join(args.model_dir, "best_model.pth")
        print(f"Loading best model from: {model_path}")
    elif args.model_type == "checkpoint":
        # Find the latest checkpoint or specific epoch
        if args.target_epoch is not None:
            model_path = os.path.join(args.model_dir, f"checkpoint_epoch_{args.target_epoch}.pth")
        else:
            # Find the latest checkpoint
            checkpoint_files = [f for f in os.listdir(args.model_dir) 
                              if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
            if not checkpoint_files:
                print(f"Error: No checkpoint files found in {args.model_dir}")
                return
            epoch_numbers = [int(f.replace("checkpoint_epoch_", "").replace(".pth", "")) 
                           for f in checkpoint_files]
            latest_epoch = max(epoch_numbers)
            model_path = os.path.join(args.model_dir, f"checkpoint_epoch_{latest_epoch}.pth")
        print(f"Loading checkpoint from: {model_path}")
    else:
        print(f"Error: Invalid model_type '{args.model_type}'. Use 'best' or 'checkpoint'")
        return
    
    # 모델 초기화
    device = torch.device(args.device) if args.device else torch.device('cpu')
    
    model = CNN_Autoencoder(latent_dim=args.latent_dim)
    print(f"Using CNN Autoencoder with latent_dim={args.latent_dim}")
    
    # 가중치 로드
    try:
        if args.model_type == "checkpoint":
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
            print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
        else:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        model.to(device)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 손실 함수
    criterion = nn.L1Loss().to(device)
    
    # 테스트 수행
    test_loss, originals, noisy_imgs, reconstructed = test(model, test_loader, criterion, device=device)
    print(f"Test Loss (L1): {test_loss:.6f}")
    
    # 디버깅 정보 출력
    print(f"\nData statistics:")
    print(f"Original - min: {min([x.min() for x in originals]):.4f}, max: {max([x.max() for x in originals]):.4f}")
    print(f"Noisy    - min: {min([x.min() for x in noisy_imgs]):.4f}, max: {max([x.max() for x in noisy_imgs]):.4f}")
    print(f"Reconst  - min: {min([x.min() for x in reconstructed]):.4f}, max: {max([x.max() for x in reconstructed]):.4f}")
    
    # 샘플 시각화
    show_sample(originals, noisy_imgs, reconstructed, idx=args.sample_idx)

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained CNN autoencoder model on MNIST")
    parser.add_argument("--model_dir", type=str, default="trained_output/cnn_autoencoder_latent2_b32_lr0.001")
    parser.add_argument("--model_type", type=str, default="checkpoint", choices=["best", "checkpoint"])
    parser.add_argument("--target_epoch", type=int, default=None, help="If None, loads latest checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="mini-batch size for testing")
    parser.add_argument("--latent_dim", type=int, default=2, help="latent space dimension (must match trained model)")
    parser.add_argument("--device", type=str, default="cpu", help="device for inference (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--sample_idx", type=int, default=11, help="index of the test sample to visualize")
    
    args, _ = parser.parse_known_args()
    
    main(args)