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

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MRI Tumor Classifier")
        self.geometry("1200x800")

        # Tabview
        self.tab_view = ctk.CTkTabview(self)
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
        # Layout: Sidebar (Left), Content (Right)
        self.pred_tab.grid_columnconfigure(0, weight=0) # Sidebar
        self.pred_tab.grid_columnconfigure(1, weight=1) # Content
        self.pred_tab.grid_rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self.pred_tab, width=200, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        
        lbl = ctk.CTkLabel(sidebar, text="Inference", font=ctk.CTkFont(size=20, weight="bold"))
        lbl.pack(padx=20, pady=(20, 10))
        
        btn = ctk.CTkButton(sidebar, text="Upload Image", command=self.upload_image)
        btn.pack(padx=20, pady=10)

        # Theme toggle
        self.appearance_mode_label = ctk.CTkLabel(sidebar, text="Theme:", anchor="w")
        self.appearance_mode_label.pack(padx=20, pady=(20, 0))
        self.appearance_mode_menu = ctk.CTkOptionMenu(sidebar, values=["System", "Light", "Dark"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.pack(padx=20, pady=(10, 20))

        # Content Area
        content = ctk.CTkFrame(self.pred_tab)
        content.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Image Display
        self.img_container = ctk.CTkFrame(content, fg_color="transparent")
        self.img_container.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.image_label = ctk.CTkLabel(self.img_container, text="No Image Selected")
        self.image_label.pack(expand=True, fill="both")

        # Stats & Chart
        self.stats_container = ctk.CTkFrame(content, fg_color="transparent")
        self.stats_container.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.prediction_label = ctk.CTkLabel(self.stats_container, text="Prediction: -", font=ctk.CTkFont(size=24, weight="bold"))
        self.prediction_label.pack(pady=(40, 10))
        
        self.confidence_label = ctk.CTkLabel(self.stats_container, text="Confidence: -", font=ctk.CTkFont(size=18))
        self.confidence_label.pack(pady=(0, 20))

        self.chart_frame = ctk.CTkFrame(content)
        self.chart_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    def setup_training_tab(self):
        # Layout
        self.train_tab.grid_columnconfigure(0, weight=0) # Controls
        self.train_tab.grid_columnconfigure(1, weight=1) # Visuals
        self.train_tab.grid_rowconfigure(0, weight=1)

        # Controls Sidebar
        controls = ctk.CTkFrame(self.train_tab, width=250)
        controls.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(controls, text="Training Config", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)

        # Epochs
        ctk.CTkLabel(controls, text="Epochs:").pack(pady=(10,0))
        self.epochs_slider = ctk.CTkSlider(controls, from_=1, to=50, number_of_steps=49)
        self.epochs_slider.set(5)
        self.epochs_slider.pack(pady=5)
        self.epochs_val = ctk.CTkLabel(controls, text="5")
        self.epochs_val.pack()
        self.epochs_slider.configure(command=lambda v: self.epochs_val.configure(text=str(int(v))))

        # LR
        ctk.CTkLabel(controls, text="Learning Rate:").pack(pady=(10,0))
        self.lr_entry = ctk.CTkEntry(controls, placeholder_text="0.0001")
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.pack(pady=5)

        # Fine-tune
        self.finetune_var = ctk.CTkCheckBox(controls, text="Fine-tune (Unfreeze Layers)")
        self.finetune_var.pack(pady=20)

        # Start Button
        self.train_btn = ctk.CTkButton(controls, text="Start Training", command=self.start_training_thread, fg_color="green")
        self.train_btn.pack(pady=20)

        # Status
        self.status_label = ctk.CTkLabel(controls, text="Status: Idle", text_color="gray")
        self.status_label.pack(pady=10)

        self.progress_bar = ctk.CTkProgressBar(controls)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        # Visuals Area
        visuals = ctk.CTkFrame(self.train_tab)
        visuals.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        visuals.grid_rowconfigure(0, weight=1)
        visuals.grid_rowconfigure(1, weight=1)
        visuals.grid_columnconfigure(0, weight=1)

        # Log Box
        self.log_box = ctk.CTkTextbox(visuals, width=400, height=200)
        self.log_box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.log_box.insert("0.0", "Training logs will appear here...\n")

        # Training Chart
        self.train_chart_frame = ctk.CTkFrame(visuals)
        self.train_chart_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

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
        text_color = "white" if mode == "Dark" else "black"
        
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    def start_training_thread(self):
        self.train_btn.configure(state="disabled", text="Training...")
        self.status_label.configure(text="Status: specific training...")
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
        
        # Reload model after training
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
        bg_color = "#2b2b2b" if mode == "Dark" else "#dbdbdb"
        text_color = "white" if mode == "Dark" else "black"
        
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        ax.plot(self.train_losses, label="Train Loss", color="cyan")
        ax.plot(self.val_losses, label="Val Loss", color="orange")
        
        ax.set_title("Training Progress", color=text_color)
        ax.legend(facecolor=bg_color, labelcolor=text_color)
        ax.tick_params(colors=text_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.train_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")

    def finish_training(self):
        self.train_btn.configure(state="normal", text="Start Training")
        self.status_label.configure(text="Status: Training Complete")
        self.log_box.insert("end", "Training Finished. Model Reloaded.\n")

def display():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    display()
