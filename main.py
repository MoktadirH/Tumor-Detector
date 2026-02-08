import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import time
import copy

# Constants
TRAIN_DIR = r"Training"
TEST_DIR  = r"Testing"
IMG_SIZE = 224
BATCH_SIZE = 32

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self, train_dir=TRAIN_DIR, test_dir=TEST_DIR):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model = None
        self.class_names = []
        
        # Prepare data immediately upon initialization
        self._prepare_data()

    def _prepare_data(self):
        train_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        ])

        # Load datasets
        try:
            train_full = datasets.ImageFolder(self.train_dir, transform=train_transforms)
            self.class_names = train_full.classes
            
            # Split train/val
            val_ratio = 0.15
            n_total = len(train_full)
            n_val = int(val_ratio * n_total)
            n_train = n_total - n_val
            
            train_ds, val_ds = random_split(
                train_full, 
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Fix validation transform
            val_ds.dataset.transform = test_transforms

            # Create loaders
            self.train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            self.val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.train_loader = None
            self.val_loader = None
            self.class_names = ["Unknown"]

    def build_model(self, fine_tune=False):
        print("Building model...")
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        
        # Freeze or Unfreeze layers
        if fine_tune:
            print("Fine-tuning mode: Unfreezing Layer 4 and FC.")
            for name, param in model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print("Feature extraction mode: Freezing all except FC.")
            for param in model.parameters():
                param.requires_grad = False
                
        # Replace final layer
        num_classes = len(self.class_names)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        self.model = model.to(self.device)
        return self.model

    def train(self, epochs=10, learning_rate=0.0001, fine_tune=False, callback=None):
        if not self.train_loader:
            if callback: callback("Error: Data not loaded.", None)
            return

        self.build_model(fine_tune)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train Phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += imgs.size(0)
            
            train_epoch_loss = train_loss / train_total
            train_epoch_acc = train_correct / train_total
            
            # Val Phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in self.val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = self.model(imgs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * imgs.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += imgs.size(0)
            
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            
            # Save best
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                torch.save(self.model.state_dict(), "best_model.pt")
                saved_msg = " [Saved Best]"
            else:
                saved_msg = ""
            
            epoch_data = {
                "epoch": epoch + 1,
                "epochs": epochs,
                "train_loss": train_epoch_loss,
                "train_acc": train_epoch_acc,
                "val_loss": val_epoch_loss,
                "val_acc": val_epoch_acc,
                "message": f"Epoch {epoch+1}/{epochs} | Acc: {val_epoch_acc:.4f}{saved_msg}"
            }
            
            if callback:
                callback(f"Epoch {epoch+1} Complete: Val Acc {val_epoch_acc:.2%}", epoch_data)
            
            print(f"Epoch {epoch+1}: Train Loss={train_epoch_loss:.4f} Acc={train_epoch_acc:.4f} | Val Loss={val_epoch_loss:.4f} Acc={val_epoch_acc:.4f}")

        return best_acc

def load_for_inference(model_path="best_model.pt"):
    device = get_device()
    
    # Needs class names. Quick way is to read from Training dir like before
    # Or simplified logic:
    try:
        train_ds = datasets.ImageFolder(TRAIN_DIR)
        class_names = train_ds.classes
    except:
        class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"] # Fallback

    num_classes = len(class_names)
    
    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    
    # Model Structure
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model = model.to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using untrained model.")
    
    return model, class_names, preprocess, device

if __name__ == "__main__":
    # Test run
    trainer = Trainer()
    trainer.train(epochs=1, fine_tune=False)
