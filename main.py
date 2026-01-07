import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Currently using:", DEVICE)

    TRAIN_DIR = r"C:\Users\mokta\Documents\Tumor Detector\Training"
    TEST_DIR  = r"C:\Users\mokta\Documents\Tumor Detector\Testing"

    IMG_SIZE = 224
    BATCH_SIZE = 32

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

    train_full = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    test_ds    = datasets.ImageFolder(TEST_DIR,  transform=test_transforms)

    class_names = train_full.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Num classes:", num_classes)
    print("Train images:", len(train_full))
    print("Test images:", len(test_ds))

    val_ratio = 0.15
    n_total = len(train_full)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        train_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # validation should NOT use random augmentations, must be clean for tuning in training
    val_ds.dataset.transform = test_transforms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    #resnet CNN for transfer learning
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    #Keep the feature extraction layers frozen
    for param in model.parameters():
        param.requires_grad = False

    #Change the final layer to match the number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    print(model.fc)


    EPOCHS = 8
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, DEVICE, train=False)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet18.pt")

    print("Best validation accuracy:", best_val_acc)

    model.load_state_dict(torch.load("best_resnet18.pt", map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("\nReport:\n", classification_report(all_labels, all_preds, target_names=class_names))

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total

if __name__ == "__main__":
    main()


