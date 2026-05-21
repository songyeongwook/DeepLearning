import argparse
import torch
import torch.nn as nn
import os
import time
import csv
from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from data_loader import load_mnist
from models.autoencoder_cnn import CNN_Autoencoder

#%%
def add_noise(clean, noise_factor=0.2):
    """Add Gaussian noise to clean images"""
    noisy = clean + torch.randn_like(clean) * noise_factor
    return noisy

#%%
def train(model, dataloader, criterion, data_len, opti, device=None):
    """Train the autoencoder model for one epoch"""
    train_loss = 0
    model.train()
    
    for data, _ in dataloader:  # We don't need labels for autoencoder
        # Reshape data to (N, 1, 28, 28) for CNN
        data = data.view(-1, 1, 28, 28)
        target = data
        
        if device:
            data = data.to(device)
            target = target.to(device)
        
        # Add noise to input
        noisy = add_noise(data)
        
        # Forward pass
        output = model(noisy)
        loss = criterion(output, target)
        
        # Backward pass
        opti.zero_grad()
        loss.backward()
        opti.step()
        
        train_loss += loss.item()
    
    return train_loss / len(dataloader)

#%%
def evaluate(model, dataloader, criterion, data_len, device=None):
    """Evaluate the autoencoder model"""
    eval_loss = 0
    model.eval()
    
    with torch.no_grad():
        for data, _ in dataloader:  # We don't need labels for autoencoder
            # Reshape data to (N, 1, 28, 28) for CNN
            data = data.view(-1, 1, 28, 28)
            target = data
            
            if device:
                data = data.to(device)
                target = target.to(device)
            
            # Add noise to input
            noisy = add_noise(data)
            
            # Forward pass
            output = model(noisy)
            loss = criterion(output, target)
            eval_loss += loss.item()
    
    return eval_loss / len(dataloader)

#%%
def main(args):
    # 1) DataLoader 준비
    train_ds, val_ds, _, in_dim, x_test_raw, y_test = load_mnist(
        valDB_portion=0.1
    )
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.no_shuffle)
    test_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2) 모델·손실·최적화기 설정
    device = torch.device(args.device) if args.device else torch.device('cpu')
    
    model = CNN_Autoencoder(latent_dim=args.latent_dim).to(device)
    
    print(f"Model: CNN Autoencoder, Latent dimension: {args.latent_dim}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    criterion = nn.L1Loss().to(device)
    opti = Adam(model.parameters(), lr=args.lr)
    
    # Create save directory
    dir_name = f"{args.output_name}_latent{args.latent_dim}_b{args.batch_size}_lr{args.lr}"
    save_dir = os.path.join("trained_output", dir_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Checkpoint and log file paths
    best_model_path = os.path.join(save_dir, "best_model.pth")
    log_file_path = os.path.join(save_dir, "train_log.csv")
    
    # 3) Resume from checkpoint if exists (automatic)
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Find the latest checkpoint file
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if checkpoint_files:
        # Extract epoch numbers and find the latest
        epoch_numbers = [int(f.replace("checkpoint_epoch_", "").replace(".pth", "")) for f in checkpoint_files]
        latest_epoch = max(epoch_numbers)
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{latest_epoch}.pth")
        
        print(f"Checkpoint found! Loading from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            opti.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"✓ Successfully resumed from epoch {start_epoch}")
            print(f"  Best validation loss so far: {best_val_loss:.6f}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
    else:
        print("No checkpoint found. Starting training from scratch...")
    
    # Initialize log file
    log_exists = os.path.exists(log_file_path)
    log_file = open(log_file_path, 'a', newline='')
    csv_writer = csv.writer(log_file)
    
    if not log_exists or start_epoch == 0:
        # Write header only for new training
        csv_writer.writerow(['Timestamp', 'Epoch', 'Train_Loss', 'Val_Loss', 'Epoch_Time(s)', 'Best_Val_Loss', 'Improved'])
        log_file.flush()
    
    # 4) 학습
    print(f"Training CNN Autoencoder with {args.epochs} epochs...")
    print(f"Save directory: {save_dir}")
    print("-" * 80)
    
    for i in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        train_loss = train(model, train_dataloader, criterion, 
                          len(train_dataloader.dataset), opti, device)
        val_loss = evaluate(model, test_dataloader, criterion, 
                           len(test_dataloader.dataset), device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Check if improved
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            
            # Save checkpoint only when improved
            checkpoint = {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opti.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            checkpoint_path_with_epoch = os.path.join(save_dir, f"checkpoint_epoch_{i+1}.pth")
            torch.save(checkpoint, checkpoint_path_with_epoch)
            improvement_mark = "✓ NEW BEST!"
        else:
            improvement_mark = ""
        
        # Print progress
        print(f"[Epoch: {i+1:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Time: {epoch_time:.2f}s {improvement_mark}")
        
        # Log to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([
            timestamp, 
            i+1, 
            f"{train_loss:.6f}", 
            f"{val_loss:.6f}", 
            f"{epoch_time:.2f}",
            f"{best_val_loss:.6f}",
            "Yes" if improved else "No"
        ])
        log_file.flush()
    
    log_file.close()
    
    print("-" * 80)
    print(f"Training finished!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {log_file_path}")

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Autoencoder on MNIST")
    parser.add_argument("--epochs",      type=int,   default=100,     help="number of training epochs")
    parser.add_argument("--batch_size",  type=int,   default=32,     help="mini-batch size for training")
    parser.add_argument("--lr",          type=float, default=1e-3,   help="learning rate for optimizer")
    parser.add_argument("--latent_dim",  type=int,   default=2,     help="latent space dimension")
    parser.add_argument("--no_shuffle",  type=bool,  default=False,  help="disable data shuffling")
    parser.add_argument("--device",      type=str,   default="cpu",  help="device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--output_name", type=str,   default="cnn_autoencoder", help="base name for output directory")
    
    args, _ = parser.parse_known_args()
    
    main(args)