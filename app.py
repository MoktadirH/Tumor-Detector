import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from sklearn.metrics import classification_report, confusion_matrix
from tkinter import scrolledtext
import main

import torch
import torch.nn as nn
from torchvision import models, transforms

def predict_image(image_path, model, class_names, preprocess, device):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    return class_names[int(idx)], float(conf), probs.cpu().numpy(), img

def display():
    model, class_names, preprocess, device = main.load_for_inference("best_model.pt")

    root = tk.Tk()
    root.title("MRI Tumor Classifier (Demo)")

    frame = ttk.Frame(root, padding=12)
    frame.grid(sticky="nsew")

    img_label = ttk.Label(frame)
    img_label.grid(row=0, column=0, rowspan=6, padx=(0, 12), sticky="n")

    result_var = tk.StringVar(value="Upload an MRI image to get a prediction.")
    ttk.Label(frame, textvariable=result_var, wraplength=360, justify="left").grid(
        row=0, column=1, sticky="w"
    )

    probs_box = tk.Text(frame, width=45, height=10)
    probs_box.grid(row=1, column=1, sticky="we", pady=(8, 0))
    probs_box.config(state="disabled")

    tk_preview_ref = {"img": None}

    def on_upload():
        path = filedialog.askopenfilename(
            title="Select an MRI image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            return

        pred, conf, probs, pil_img = predict_image(path, model, class_names, preprocess, device)

        result_var.set(f"Prediction: {pred}\nConfidence: {conf*100:.2f}%\n\nFile:\n{path}")

        probs_box.config(state="normal")
        probs_box.delete("1.0", "end")
        for name, p in sorted(zip(class_names, probs), key=lambda t: t[1], reverse=True):
            probs_box.insert("end", f"{name:15s}  {p*100:6.2f}%\n")
        probs_box.config(state="disabled")

        preview = pil_img.copy()
        preview.thumbnail((320, 320))
        tk_img = ImageTk.PhotoImage(preview)
        tk_preview_ref["img"] = tk_img
        img_label.config(image=tk_img)

    ttk.Button(frame, text="Upload MRI Image", command=on_upload).grid(
        row=5, column=1, sticky="we", pady=(12, 0)
    )

    root.mainloop()
