import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split

def train_unet():
    # Configuration
    data_dir = "data/raw"
    mask_dir = "data/ground_truth"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load Data
    images = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
    
    if len(images) != len(masks):
        print(f"Error: Mismatch in image/mask count ({len(images)} vs {len(masks)})")
        return

    # Process Data
    X = []
    Y = []
    
    # Resize to ensure batch consistency (e.g., 256x256)
    target_size = (256, 256) 

    print("Loading data...")
    for img_path, mask_path in zip(images, masks):
        # Image
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)
        img_tensor = TF.to_tensor(img) # [3, H, W], float32, 0-1
        X.append(img_tensor)
        
        # Mask
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(target_size, resample=Image.NEAREST)
        mask_arr = np.array(mask)
        mask_bin = (mask_arr > 0).astype(np.float32) # Binary 0/1
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0) # [1, H, W]
        Y.append(mask_tensor)

    X = torch.stack(X)
    Y = torch.stack(Y)
    
    # Split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # Model
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training Loop
    epochs = 15
    batch_size = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Manual batching
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        for i in range(0, len(X_train), batch_size):
            inputs = X_train_shuffled[i:i+batch_size].to(device)
            targets = Y_train_shuffled[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        avg_train_loss = train_loss / len(X_train)
        
        # Validation
        model.eval()
        val_iou = 0
        with torch.no_grad():
            inputs_val = X_val.to(device) # Small enough to fit in memory
            targets_val = Y_val.to(device)
            
            outputs_val = model(inputs_val)
            preds_val = (torch.sigmoid(outputs_val) > 0.5).float()
            
            # IoU Calculation
            intersection = (preds_val * targets_val).sum()
            union = preds_val.sum() + targets_val.sum() - intersection
            epsilon = 1e-7
            iou = (intersection + epsilon) / (union + epsilon)
            val_iou = iou.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val IoU: {val_iou:.4f}")

    # Save
    save_path = os.path.join(results_dir, "unet_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_unet()
