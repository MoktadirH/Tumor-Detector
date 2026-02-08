import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import main
import os
import threading

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(1.1) 

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MRI Tumor Classifier")
        self.geometry("1200x800")
        
        # Custom Font
        self.main_font = ("Segoe UI", 14)
        self.header_font = ("Segoe UI", 20, "bold")

        # Tabview
        self.tab_view = ctk.CTkTabview(self, anchor="nw")
        self.tab_view.pack(fill="both", expand=True, padx=20, pady=20)

        self.pred_tab = self.tab_view.add("Prediction")
        self.train_tab = self.tab_view.add("Training")

        # Data for charts
        self.train_losses = []
        self.val_losses = []

        # Setup Tabs
        self.setup_prediction_tab()
        self.setup_training_tab()
        
        # Load Model
        self.load_model()
        
        # Graceful exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_model(self):
        try:
            self.model, self.class_names, self.preprocess, self.device = main.load_for_inference("best_model.pt")
            print("Model loaded successfully.")
        except Exception as e:
             print(f"Error loading model: {e}")
             self.class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
             self.preprocess = None
             self.device = "cpu"
             self.model = None

    def setup_prediction_tab(self):
        self.pred_tab.grid_columnconfigure(0, weight=0) # Sidebar
        self.pred_tab.grid_columnconfigure(1, weight=1) # Content
        self.pred_tab.grid_rowconfigure(0, weight=1)

        # Sidebar with "Card" look
        sidebar = ctk.CTkFrame(self.pred_tab, width=220, corner_radius=15)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(sidebar, text="Inference", font=self.header_font).pack(padx=20, pady=(20, 10))
        
        ctk.CTkButton(sidebar, text="Upload Image", command=self.upload_image, font=self.main_font, height=40).pack(padx=20, pady=10)

        ctk.CTkLabel(sidebar, text="Theme:", anchor="w", font=self.main_font).pack(padx=20, pady=(20, 0))
        ctk.CTkOptionMenu(sidebar, values=["System", "Light", "Dark"], command=self.change_appearance_mode_event).pack(padx=20, pady=(10, 20))

        # Content Area
        content = ctk.CTkFrame(self.pred_tab, corner_radius=15, fg_color="transparent")
        content.grid(row=0, column=1, retry="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Image Display
        self.img_container = ctk.CTkFrame(content, corner_radius=15)
        self.img_container.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.image_label = ctk.CTkLabel(self.img_container, text="No Image Selected", font=self.main_font)
        self.image_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Stats
        self.stats_container = ctk.CTkFrame(content, corner_radius=15)
        self.stats_container.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.prediction_label = ctk.CTkLabel(self.stats_container, text="Prediction: -", font=("Segoe UI", 24, "bold"))
        self.prediction_label.pack(pady=(40, 10))
        
        self.confidence_label = ctk.CTkLabel(self.stats_container, text="Confidence: -", font=("Segoe UI", 18))
        self.confidence_label.pack(pady=(0, 20))

        # Chart
        self.chart_frame = ctk.CTkFrame(content, corner_radius=15)
        self.chart_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    def setup_training_tab(self):
        self.train_tab.grid_columnconfigure(0, weight=0)
        self.train_tab.grid_columnconfigure(1, weight=1)
        self.train_tab.grid_rowconfigure(0, weight=1)

        # Controls Sidebar
        controls = ctk.CTkFrame(self.train_tab, width=280, corner_radius=15)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(controls, text="Configuration", font=self.header_font).pack(pady=20)

        # Helper Tooltip Function
        def create_tooltip(widget, text):
            # Simple placeholder for tooltip logic or additional help text label
            pass

        # Epochs
        ctk.CTkLabel(controls, text="Epochs", font=self.main_font).pack(pady=(10,0))
        ctk.CTkLabel(controls, text="(Training iterations)", font=("Segoe UI", 10)).pack()
        self.epochs_slider = ctk.CTkSlider(controls, from_=1, to=50, number_of_steps=49)
        self.epochs_slider.set(5)
        self.epochs_slider.pack(pady=5)
        self.epochs_val = ctk.CTkLabel(controls, text="5", font=self.main_font)
        self.epochs_val.pack()
        self.epochs_slider.configure(command=lambda v: self.epochs_val.configure(text=str(int(v))))

        # LR
        ctk.CTkLabel(controls, text="Learning Rate", font=self.main_font).pack(pady=(10,0))
        ctk.CTkLabel(controls, text="(Step size for optimization)", font=("Segoe UI", 10)).pack()
        self.lr_entry = ctk.CTkEntry(controls, placeholder_text="0.0001", font=self.main_font)
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.pack(pady=5)

        # Fine-tune
        self.finetune_var = ctk.CTkCheckBox(controls, text="Fine-tune", font=self.main_font)
        self.finetune_var.pack(pady=20)
        ctk.CTkLabel(controls, text="Unfreezes layers for higher accuracy.\n(Slower training)", font=("Segoe UI", 10), text_color="gray").pack()

        # Start Button
        self.train_btn = ctk.CTkButton(controls, text="Start Training", command=self.start_training_thread, 
                                     fg_color="#2CC985", hover_color="#229966", font=("Segoe UI", 16, "bold"), height=50)
        self.train_btn.pack(pady=30, padx=20)

        # Status
        self.status_label = ctk.CTkLabel(controls, text="Status: Idle", text_color="gray", font=self.main_font)
        self.status_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(controls)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10, has_focus=False)

        # Visuals Area
        visuals = ctk.CTkFrame(self.train_tab, corner_radius=15, fg_color="transparent")
        visuals.grid(row=0, column=1, sticky="nsew")
        visuals.grid_rowconfigure(0, weight=1)
        visuals.grid_rowconfigure(1, weight=1)
        visuals.grid_columnconfigure(0, weight=1)

        # Log Box
        self.log_box = ctk.CTkTextbox(visuals, font=("Consolas", 12), corner_radius=10)
        self.log_box.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 10))
        self.log_box.insert("0.0", "--- Training Log ---\n")

        # Training Chart
        self.train_chart_frame = ctk.CTkFrame(visuals, corner_radius=15)
        self.train_chart_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
        if hasattr(self, 'last_probs'):
             self.draw_probs_chart(self.last_probs)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
        if not file_path:
            return
        self.process_image(file_path)

    def process_image(self, file_path):
        try:
            pil_img = Image.open(file_path)
            display_size = (400, 400)
            img_copy = pil_img.copy()
            img_copy.thumbnail(display_size)
            ctk_img = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)
            self.image_label.configure(image=ctk_img, text="")

            if self.model:
                img = pil_img.convert("RGB")
                input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()

                idx = np.argmax(probs)
                pred_class = self.class_names[idx]
                conf = probs[idx]

                self.prediction_label.configure(text=f"Prediction: {pred_class}")
                self.confidence_label.configure(text=f"Confidence: {conf*100:.2f}%")
                self.last_probs = probs
                self.draw_probs_chart(probs)
        except Exception as e:
            print(f"Error: {e}")

    def draw_probs_chart(self, probs):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        mode = ctk.get_appearance_mode()
        bg_color = "#2b2b2b" if mode == "Dark" else "#dbdbdb" 
        # Note: In a frame, bg might differ. Using None (transparent) can work if fig frame matches 
        text_color = "white" if mode == "Dark" else "black"
        
        # Approximate frame specific colors
        bg_color = "#212121" if mode == "Dark" else "#EBEBEB" # Standard CTk Frame colors

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        y_pos = np.arange(len(self.class_names))
        ax.barh(y_pos, probs, align='center', color="#3B8ED0")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.class_names, color=text_color)
        ax.invert_yaxis()
        ax.set_xlabel('Probability', color=text_color)
        
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)

    def start_training_thread(self):
        self.train_btn.configure(state="disabled", text="Training...")
        self.status_label.configure(text="Status: Training started...")
        self.progress_bar.set(0)
        self.train_losses = []
        self.val_losses = []
        
        epochs = int(self.epochs_slider.get())
        lr = float(self.lr_entry.get())
        fine_tune = bool(self.finetune_var.get())
        
        thread = threading.Thread(target=self.run_training, args=(epochs, lr, fine_tune))
        thread.start()

    def run_training(self, epochs, lr, fine_tune):
        trainer = main.Trainer()
        trainer.train(epochs=epochs, learning_rate=lr, fine_tune=fine_tune, callback=self.training_callback)
        self.load_model()
        self.after(0, lambda: self.finish_training())

    def training_callback(self, msg, data):
        self.after(0, lambda: self.update_training_ui(msg, data))

    def update_training_ui(self, msg, data):
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        
        if data:
            self.progress_bar.set(data['epoch'] / data['epochs'])
            self.train_losses.append(data['train_loss'])
            self.val_losses.append(data['val_loss'])
            self.draw_training_chart()

    def draw_training_chart(self):
        for widget in self.train_chart_frame.winfo_children():
            widget.destroy()

        mode = ctk.get_appearance_mode()
        bg_color = "#212121" if mode == "Dark" else "#EBEBEB"
        text_color = "white" if mode == "Dark" else "black"
        
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        ax.plot(self.train_losses, label="Train Loss", color="#00E5FF", linewidth=2)
        ax.plot(self.val_losses, label="Val Loss", color="#FF4081", linewidth=2, linestyle="--")
        
        ax.legend(facecolor=bg_color, labelcolor=text_color, framealpha=0)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
            
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.train_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)

    def finish_training(self):
        self.train_btn.configure(state="normal", text="Start Training")
        self.status_label.configure(text="Status: Training Complete")
        self.log_box.insert("end", "Training Finished. Model Reloaded.\n")

    def on_closing(self):
        self.quit()
        self.destroy()

def display():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    display()
