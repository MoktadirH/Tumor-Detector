import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMG_SIZE = 224


def _build_test_loader(test_dir, batch_size=32):
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader


def _class_accuracies(conf_mat):
    row_sums = conf_mat.sum(axis=1)
    diag = np.diag(conf_mat)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.divide(diag, row_sums, where=row_sums != 0)
    per_class_acc = np.nan_to_num(per_class_acc)
    return per_class_acc


def evaluate_model_and_write_report(
    model,
    device,
    test_dir,
    class_names,
    report_path="model_metrics_report.txt",
    batch_size=32,
):
    test_loader = _build_test_loader(test_dir=test_dir, batch_size=batch_size)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    labels_idx = list(range(len(class_names)))

    overall_accuracy = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels_idx, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_idx, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0)

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels_idx)
    per_class_accuracy = _class_accuracies(conf_mat)

    overall_score = (overall_accuracy + macro_f1) / 2.0

    lines = []
    lines.append("Model Metrics Report")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Test directory: {os.path.abspath(test_dir)}")
    lines.append("")
    lines.append("Overall Metrics")
    lines.append("-" * 60)
    lines.append(f"Overall Accuracy : {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    lines.append(f"Macro F1 Score   : {macro_f1:.4f}")
    lines.append(f"Weighted F1 Score: {weighted_f1:.4f}")
    lines.append(f"Overall Score    : {overall_score:.4f}  (mean of Accuracy and Macro F1)")
    lines.append("")
    lines.append("Per-Class Metrics")
    lines.append("-" * 60)
    lines.append(f"{'Class':<20} {'Accuracy':>12} {'F1 Score':>12}")
    lines.append("-" * 60)

    for idx, class_name in enumerate(class_names):
        lines.append(f"{class_name:<20} {per_class_accuracy[idx]:>12.4f} {per_class_f1[idx]:>12.4f}")

    lines.append("")
    lines.append("Confusion Matrix (rows=true, cols=pred)")
    lines.append("-" * 60)
    lines.append(np.array2string(conf_mat))

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))

    return {
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "overall_score": overall_score,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": conf_mat.tolist(),
        "report_path": os.path.abspath(report_path),
    }
