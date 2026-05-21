import os
import argparse
import torch
import numpy as np
from models.autoencoder_cnn import CNN_Autoencoder
import matplotlib.pyplot as plt

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.close("all")


def visualize_2d_latent_space(model, device, grid_size=10, value_range=(-2.5, 2.5)):
    """
    2D 잠재 공간을 그리드 형태로 시각화합니다.
    가로축은 z[0], 세로축은 z[1]에 해당합니다.
    """
    model.eval()

    # 각 축에 대해 생성할 값의 범위를 설정합니다.
    x_values = np.linspace(value_range[0], value_range[1], grid_size)
    y_values = np.linspace(value_range[0], value_range[1], grid_size)

    # 생성된 이미지를 담을 그리드를 준비합니다.
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    print(f"Generating {grid_size}x{grid_size} grid of images from 2D latent space (range: {value_range})...")

    with torch.no_grad():
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                # (x, y) 좌표로 2D 잠재 벡터를 생성합니다.
                z = torch.FloatTensor([[x, y]]).to(device)
                
                # 디코더로 이미지 생성
                decoded_img = model.decode(z)
                img = decoded_img.view(28, 28).cpu().numpy()
                
                # 그리드에 이미지 표시
                ax = axes[i, j]
                ax.imshow(img, cmap='gray')
                ax.axis('off')
    
    # 축 레이블 설정 (그리드의 가장자리에만)
    fig.text(0.5, 0.06, f'Latent Variable 1 (z[0]) from {value_range[0]} to {value_range[1]}', ha='center', va='center', fontsize=14)
    fig.text(0.06, 0.5, f'Latent Variable 2 (z[1]) from {value_range[0]} to {value_range[1]}', ha='center', va='center', rotation='vertical', fontsize=14)

    plt.suptitle('2D Latent Space Visualization', fontsize=18)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    plt.show()


def main(args):
    # 이 시각화는 latent_dim=2일 때만 의미가 있습니다.
    if args.latent_dim != 2:
        print(f"Error: This visualization is designed for a 2D latent space (latent_dim=2), "
              f"but the specified dimension is {args.latent_dim}.")
        print("Please run with a model trained with --latent_dim 2.")
        return

    # 장치 설정
    device = torch.device(args.device) if args.device else torch.device('cpu')
    
    # 모델 초기화
    model = CNN_Autoencoder(latent_dim=args.latent_dim)
    print(f"Using CNN Autoencoder with latent_dim={args.latent_dim}")
    
    # 저장된 모델 가중치 로드
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        print(f"Model loaded from: {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("-" * 80)
    
    # 2D 잠재 공간 시각화 실행
    visualize_2d_latent_space(model, device, grid_size=args.grid_size, value_range=args.range)
    
    print("\n" + "-" * 80)
    print("Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the 2D latent space of a trained CNN Autoencoder.")
    parser.add_argument("--model_path", type=str, 
                        default="trained_output/cnn_autoencoder_latent2_b32_lr0.001/best_model.pth",
                        help="Path to the trained autoencoder model file (.pth) with latent_dim=2.")
    parser.add_argument("--latent_dim", type=int, default=2, help="Latent space dimension.")
    parser.add_argument("--grid_size", type=int, default=20,help="Number of images to generate for each axis.")
    parser.add_argument("--range", type=float, nargs=2, default=[-2.5, 2.5], help="Min and max values for latent axes.")
    parser.add_argument("--device", type=str, default="cpu")
    
    args, _ = parser.parse_known_args()
    
    main(args)