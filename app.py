import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import main
import os
import threading
import tempfile
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ===== COLOR PALETTE =====
COLORS = {
    "primary": "#000000",
    "secondary": "#34A853",
    "danger": "#EA4335",
    "amber": "#F9A825",
    "text_primary": "#1C2B3A",
    "text_secondary": "#5F7080",
}

# ===== CLASS COLORS (for dynamic prediction text) =====
CLASS_COLORS = {
    "No Tumor": COLORS["secondary"],  # green
    "Glioma": COLORS["danger"],        # red
    "Meningioma": COLORS["amber"],     # amber
    "Pituitary": COLORS["primary"],    # blue
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("MRI Tumor Classifier")
        self.geometry("1100x700")
        
        # Configure style
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        
        # Custom Fonts
        self.main_font = ("Segoe UI", 11)
        self.header_font = ("Segoe UI", 16, "bold")
        self.logo_font = ("Segoe UI", 14, "bold")
        self.prediction_font = ("Segoe UI", 20, "bold")
        self.label_font = ("Segoe UI", 10)
        
        # Training metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.current_train_loss = 0.0
        self.current_val_loss = 0.0
        
        # Prediction state
        self.last_probs = None
        self.last_image_pil = None
        self.is_processing = False
        
        # Tabview using ttk.Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.pred_tab = ttk.Frame(self.notebook)
        self.train_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.pred_tab, text="Prediction")
        self.notebook.add(self.train_tab, text="Training")

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

    # ===== PREDICTION TAB =====
    
    def setup_prediction_tab(self):
        self.pred_tab.columnconfigure(0, weight=0, minsize=220)
        self.pred_tab.columnconfigure(1, weight=1)
        self.pred_tab.rowconfigure(0, weight=1)

        # ===== SIDEBAR =====
        sidebar = ttk.LabelFrame(self.pred_tab, text="Controls", padding=10)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Logo Label
        logo_lbl = ttk.Label(sidebar, text="MRI Tumor Detector", font=self.logo_font, foreground=COLORS["primary"])
        logo_lbl.pack(pady=(10, 20))
        
        # Upload Button
        upload_btn = ttk.Button(sidebar, text="Upload Image", command=self.upload_image)
        upload_btn.pack(fill="x", pady=10, padx=10)
        
        # Export Button
        self.export_btn = ttk.Button(sidebar, text="Export Report", command=self.export_pdf_report)
        self.export_btn.pack(fill="x", pady=10, padx=10)

        # ===== MAIN CONTENT AREA =====
        main_area = ttk.Frame(self.pred_tab)
        main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        main_area.columnconfigure(0, weight=1)
        main_area.columnconfigure(1, weight=1)
        main_area.rowconfigure(0, weight=1)
        main_area.rowconfigure(1, weight=1)

        # ===== IMAGE PANEL =====
        img_frame = ttk.LabelFrame(main_area, text="MRI Scan", padding=10)
        img_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        self.image_label = ttk.Label(img_frame, text="No Image Selected", font=self.main_font, foreground=COLORS["text_secondary"], anchor="center")
        self.image_label.pack(expand=True, fill="both")

        # ===== PREDICTION CARD =====
        pred_frame = ttk.LabelFrame(main_area, text="Result", padding=10)
        pred_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        lbl = ttk.Label(pred_frame, text="Prediction:", font=self.label_font, foreground=COLORS["text_secondary"])
        lbl.pack(anchor="w", pady=(5, 0))
        
        self.prediction_label = ttk.Label(pred_frame, text="-", font=self.prediction_font, foreground=COLORS["text_primary"])
        self.prediction_label.pack(anchor="w", pady=(0, 10))
        
        lbl2 = ttk.Label(pred_frame, text="Confidence:", font=self.label_font, foreground=COLORS["text_secondary"])
        lbl2.pack(anchor="w")
        
        self.confidence_label = ttk.Label(pred_frame, text="- %", font=("Segoe UI", 14, "bold"), foreground=COLORS["primary"])
        self.confidence_label.pack(anchor="w", pady=(0, 5))
        
        self.confidence_bar = ttk.Progressbar(pred_frame, orient="horizontal", mode="determinate")
        self.confidence_bar.pack(fill="x", padx=5, pady=5)

        # ===== PROBABILITY CHART =====
        self.chart_frame = ttk.LabelFrame(main_area, text="Class Probabilities", padding=10)
        self.chart_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff")])
        if not file_path:
            return
        self.process_image(file_path)

    def process_image(self, file_path):
        """Load image and run inference in a thread."""
        self.is_processing = True
        thread = threading.Thread(target=self._run_inference, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _run_inference(self, file_path):
        """Background thread for image processing and inference."""
        try:
            pil_img = Image.open(file_path)
            self.last_image_pil = pil_img
            
            # Display image on main thread
            self.after(0, lambda: self._display_image(pil_img))
            
            if not self.model:
                self.after(0, lambda: self._show_inference_error("Model not loaded"))
                return
            
            # Run inference
            img = pil_img.convert("RGB")
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()

            idx = np.argmax(probs)
            pred_class = self.class_names[idx]
            conf = probs[idx]

            # Update UI on main thread
            self.after(0, lambda: self._update_prediction_ui(pred_class, conf, probs))
            
        except Exception as e:
            print(f"Error: {e}")
            self.after(0, lambda: self._show_inference_error(str(e)))
        finally:
            self.is_processing = False

    def _display_image(self, pil_img):
        """Display image in the image label."""
        display_size = (300, 300)
        img_copy = pil_img.copy()
        img_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_copy)
        self.image_label.image = tk_img
        self.image_label.configure(image=tk_img, text="")

    def _update_prediction_ui(self, pred_class, conf, probs):
        """Update prediction card with results."""
        self.prediction_label.configure(text=pred_class)
        pred_color = CLASS_COLORS.get(pred_class, COLORS["text_primary"])
        self.prediction_label.configure(foreground=pred_color)
        
        self.confidence_label.configure(text=f"{conf*100:.1f}%")
        self.confidence_bar['value'] = conf * 100
        
        self.last_probs = probs
        self.draw_probs_chart(probs)

    def _show_inference_error(self, error_msg):
        """Show inference error in prediction label."""
        self.prediction_label.configure(text="Error", foreground=COLORS["danger"])
        self.confidence_label.configure(text="N/A")
        self.confidence_bar['value'] = 0
        print(f"Inference error: {error_msg}")

    def draw_probs_chart(self, probs):
        """Draw probability bar chart with class highlighting."""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        bg_color = "#f0f0f0"
        try:
            bg_color = self.style.lookup('TLabelframe', 'background') or "#f0f0f0"
        except:
            pass
        text_color = COLORS["text_primary"]
        
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        pred_idx = np.argmax(probs)
        colors_list = [
            COLORS["secondary"] if i == pred_idx else COLORS["primary"]
            for i in range(len(self.class_names))
        ]
        
        y_pos = np.arange(len(self.class_names))
        ax.barh(y_pos, probs, align='center', color=colors_list)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.class_names, color=text_color)
        ax.invert_yaxis()
        ax.set_xlabel('Probability', color=text_color, fontsize=10)
        ax.set_title('Class Probabilities', color=text_color, fontsize=11, fontweight='bold', pad=5)
        ax.set_xlim(0, 1)
        
        ax.tick_params(colors=text_color, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#cccccc")
        ax.spines['bottom'].set_color("#cccccc")
        ax.grid(axis='x', alpha=0.2, color="#cccccc")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")
        plt.close(fig)

    def export_pdf_report(self):
        """Export prediction report as PDF."""
        if self.last_image_pil is None or self.last_probs is None:
            messagebox.showwarning("No Prediction", "Please upload an image and run inference first.")
            return
        
        if not HAS_REPORTLAB:
            messagebox.showerror("Missing Dependency", 
                               "Install reportlab: pip install reportlab")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if not file_path:
            return
        
        try:
            self._generate_pdf_report(file_path)
            messagebox.showinfo("Success", f"Report saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF: {e}")

    def _generate_pdf_report(self, file_path):
        """Generate the PDF report using reportlab."""
        doc = SimpleDocTemplate(file_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor(COLORS["primary"]),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor(COLORS["text_secondary"]),
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor(COLORS["text_primary"]),
            spaceAfter=8
        )
        
        story.append(Paragraph("MRI Tumor Classification Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             subtitle_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Patient Scan", styles['Heading2']))
        
        fd, temp_img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        self.last_image_pil.save(temp_img_path)
        
        img = RLImage(temp_img_path, width=2.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
        
        pred_idx = np.argmax(self.last_probs)
        pred_class = self.class_names[pred_idx]
        confidence = self.last_probs[pred_idx]
        
        story.append(Paragraph("Diagnosis Result", styles['Heading2']))
        color_hex = colors.HexColor(CLASS_COLORS.get(pred_class, COLORS["text_primary"]))
        result_text = f"<font color='{color_hex.hexval()}'><b>{pred_class}</b></font>"
        story.append(Paragraph(result_text, body_style))
        story.append(Paragraph(f"Confidence: {confidence*100:.2f}%", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Class Probabilities", styles['Heading2']))
        table_data = [["Class", "Probability"]]
        for idx, class_name in enumerate(self.class_names):
            table_data.append([class_name, f"{self.last_probs[idx]*100:.2f}%"])
        
        table = Table(table_data, colWidths=[2.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(COLORS["primary"])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#D0D9E3")),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph(
            "Generated by MRI Tumor Classifier — For research use only.",
            footer_style := ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor(COLORS["text_secondary"]),
                alignment=TA_CENTER
            )
        ))
        
        doc.build(story)
        
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    # ===== TRAINING TAB =====
    
    def setup_training_tab(self):
        self.train_tab.columnconfigure(0, weight=0, minsize=250)
        self.train_tab.columnconfigure(1, weight=1)
        self.train_tab.rowconfigure(0, weight=1)

        # ===== SIDEBAR =====
        controls = ttk.LabelFrame(self.train_tab, text="Configuration", padding=10)
        controls.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Epochs scale
        ttk.Label(controls, text="Epochs", font=self.main_font, foreground=COLORS["text_primary"]).pack(anchor="w", pady=(10, 0))
        ttk.Label(controls, text="(Training iterations)", font=("Segoe UI", 9), foreground=COLORS["text_secondary"]).pack(anchor="w")
        self.epochs_scale = tk.Scale(controls, from_=1, to=50, orient="horizontal", resolution=1, bd=0, highlightthickness=0)
        self.epochs_scale.set(5)
        self.epochs_scale.pack(fill="x", pady=5)

        # Learning Rate
        ttk.Label(controls, text="Learning Rate", font=self.main_font, foreground=COLORS["text_primary"]).pack(anchor="w", pady=(10, 0))
        ttk.Label(controls, text="(Step size for optimization)", font=("Segoe UI", 9), foreground=COLORS["text_secondary"]).pack(anchor="w")
        self.lr_entry = ttk.Entry(controls)
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.pack(fill="x", pady=5)

        # Fine-tune Checkbox
        self.finetune_var = tk.BooleanVar(value=False)
        self.finetune_cb = ttk.Checkbutton(controls, text="Fine-tune", variable=self.finetune_var)
        self.finetune_cb.pack(anchor="w", pady=10)
        ttk.Label(controls, text="Unfreezes layers for higher accuracy.\n(Slower training)", 
                 font=("Segoe UI", 9), foreground=COLORS["text_secondary"], justify="left").pack(anchor="w")

        # Start Button
        self.train_btn = ttk.Button(controls, text="Start Training", command=self.start_training_thread)
        self.train_btn.pack(fill="x", pady=(20, 10))

        # Status
        self.status_label = ttk.Label(controls, text="Status: Idle", foreground=COLORS["text_secondary"], font=self.main_font)
        self.status_label.pack(pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(controls, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", pady=10)

        # ===== MAIN CONTENT AREA =====
        main_area = ttk.Frame(self.train_tab)
        main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        main_area.columnconfigure(0, weight=1)
        main_area.rowconfigure(0, weight=0, minsize=80)
        main_area.rowconfigure(1, weight=1)
        main_area.rowconfigure(2, weight=1)

        # ===== METRICS SUMMARY BAR =====
        metrics_frame = ttk.LabelFrame(main_area, text="Metrics Summary", padding=5)
        metrics_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        metrics_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        self.metric_labels = {}
        metric_configs = [
            ("Current Epoch", "0", COLORS["primary"]),
            ("Train Loss", "0.0000", COLORS["amber"]),
            ("Val Loss", "0.0000", COLORS["danger"]),
            ("Best Val Loss", "0.0000", COLORS["secondary"]),
        ]
        
        for idx, (metric_name, initial_val, text_color) in enumerate(metric_configs):
            frame = ttk.Frame(metrics_frame)
            frame.grid(row=0, column=idx, sticky="nsew")
            
            ttk.Label(frame, text=metric_name, font=("Segoe UI", 9), foreground=COLORS["text_secondary"]).pack()
            value_lbl = ttk.Label(frame, text=initial_val, font=("Segoe UI", 14, "bold"), foreground=text_color)
            value_lbl.pack()
            
            self.metric_labels[metric_name.lower().replace(" ", "_")] = value_lbl

        # ===== TRAINING LOG =====
        log_frame = ttk.LabelFrame(main_area, text="📋 Training Log", padding=5)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.log_box = ScrolledText(log_frame, font=("Consolas", 10), height=8)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.insert("1.0", "--- Training Log ---\n")

        # ===== LOSS CHART =====
        chart_frame = ttk.LabelFrame(main_area, text="📊 Loss Over Epochs", padding=5)
        chart_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.train_chart_frame = chart_frame

    def update_status(self, text):
        """Update status label with color based on content."""
        self.status_label.configure(text=f"Status: {text}")
        if "Idle" in text:
            color = COLORS["text_secondary"]
        elif "Training" in text:
            color = COLORS["primary"]
        elif "Complete" in text:
            color = COLORS["secondary"]
        elif "Error" in text:
            color = COLORS["danger"]
        else:
            color = COLORS["text_secondary"]
        self.status_label.configure(foreground=color)

    def start_training_thread(self):
        self.train_btn.configure(state="disabled")
        self.update_status("Training started...")
        self.progress_bar['value'] = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", "--- Training Log ---\n")
        
        epochs = int(self.epochs_scale.get())
        lr = float(self.lr_entry.get())
        fine_tune = bool(self.finetune_var.get())
        
        thread = threading.Thread(target=self.run_training, args=(epochs, lr, fine_tune))
        thread.daemon = True
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
            self.current_epoch = data['epoch']
            self.current_train_loss = data['train_loss']
            self.current_val_loss = data['val_loss']
            self.best_val_loss = min(self.best_val_loss, self.current_val_loss)
            
            self.progress_bar['value'] = (data['epoch'] / data['epochs']) * 100
            self.train_losses.append(data['train_loss'])
            self.val_losses.append(data['val_loss'])
            
            # Update metric labels
            self.metric_labels['current_epoch'].configure(text=str(data['epoch']))
            self.metric_labels['train_loss'].configure(text=f"{data['train_loss']:.4f}")
            self.metric_labels['val_loss'].configure(text=f"{data['val_loss']:.4f}")
            self.metric_labels['best_val_loss'].configure(text=f"{self.best_val_loss:.4f}")
            
            self.draw_training_chart()

    def draw_training_chart(self):
        for widget in self.train_chart_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
        bg_color = "#f0f0f0"
        try:
            bg_color = self.style.lookup('TLabelframe', 'background') or "#f0f0f0"
        except:
            pass
        text_color = COLORS["text_primary"]
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        ax.plot(self.train_losses, label="Train Loss", color=COLORS["primary"], linewidth=2)
        ax.plot(self.val_losses, label="Val Loss", color=COLORS["danger"], linewidth=2, linestyle="--")
        
        ax.set_xlabel('Epoch', color=text_color, fontsize=9)
        ax.set_ylabel('Loss', color=text_color, fontsize=9)
        ax.set_title('Loss Over Epochs', color=text_color, fontsize=11, fontweight='bold', pad=5)
        
        ax.legend(facecolor=bg_color, labelcolor=text_color, framealpha=0.95, 
                 edgecolor="#cccccc", loc='upper right')
        ax.tick_params(colors=text_color, labelsize=9)
        
        ax.grid(True, alpha=0.3, color="#cccccc", linestyle='-', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_color("#cccccc")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.train_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")
        plt.close(fig)

    def finish_training(self):
        self.train_btn.configure(state="normal")
        self.update_status("Complete")
        self.log_box.insert("end", "\n✅ Training Finished. Model Reloaded.\n")
        self._flash_status_label(3)

    def _flash_status_label(self, times):
        """Flash the status label green 3 times as success animation."""
        if times <= 0:
            return
        
        current_color = self.status_label.cget("foreground")
        new_color = COLORS["secondary"] if current_color != COLORS["secondary"] else COLORS["text_secondary"]
        self.status_label.configure(foreground=new_color)
        
        self.after(300, lambda: self._flash_status_label(times - 1))

    def on_closing(self):
        self.quit()
        self.destroy()

def display():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    display()
