import os
import argparse
import torch
from models.autoencoder_cnn import CNN_Autoencoder
import matplotlib.pyplot as plt
import numpy as np

# OpenMP runtime 중복 로드 문제 우회 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def generate_image_from_vector(model, device, latent_vector):
    """
    주어진 잠재 공간 벡터로부터 이미지를 생성하고 화면에 표시합니다.
    """
    model.eval()
    
    print(f"Generating image from latent vector: {latent_vector}...")

    with torch.no_grad():
        # 잠재 벡터를 텐서로 변환
        z = torch.FloatTensor([latent_vector]).to(device)
        
        # 디코더로 이미지 생성
        decoded_img = model.decode(z)
        img = decoded_img.view(28, 28).cpu().numpy()
        
        # 이미지 표시
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Generated from z={latent_vector}")
        plt.show()
    
    print("Image displayed.")


def main(args):
    # 이 스크립트는 latent_dim=2일 때 가장 의미가 있지만,
    # 입력 벡터의 차원과 모델의 잠재 공간 차원이 일치하면 모든 차원에서 작동할 수 있습니다.
    if len(args.vector) != args.latent_dim:
        print(f"Error: The dimension of the input vector ({len(args.vector)}) does not match "
              f"the model's latent dimension ({args.latent_dim}).")
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

    # 특정 벡터로부터 이미지 생성 실행
    generate_image_from_vector(model, device, args.vector)
    
    print("\n" + "-" * 80)
    print("Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image from a specific latent vector using a trained CNN Autoencoder.")
    parser.add_argument("--model_path", type=str, 
                        default="trained_output/cnn_autoencoder_latent2_b32_lr0.001/best_model.pth",
                        help="Path to the trained autoencoder model file (.pth).")
    parser.add_argument("--latent_dim", type=int, default=2, 
                        help="Latent space dimension of the model.")
    parser.add_argument('--vector', type=float, default=[1.5,-0.80], help="The latent vector to generate an image from (e.g., --vector 1.5 -0.8).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda').")
    
    args = parser.parse_args()
    
    main(args)
