import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import main
import os
import threading
from datetime import datetime
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, Pagebreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(1.1)

# ===== COLOR PALETTE =====
COLORS = {
    "app_bg": "#F0F4F8",
    "card_bg": "#FFFFFF",
    "primary": "#1A73E8",
    "secondary": "#34A853",
    "danger": "#EA4335",
    "sidebar_bg": "#E8EEF4",
    "text_primary": "#1C2B3A",
    "text_secondary": "#5F7080",
    "border": "#D0D9E3",
    "amber": "#F9A825",
    "light_gray": "#F8F9FA",
}

# ===== CLASS COLORS (for dynamic prediction text) =====
CLASS_COLORS = {
    "No Tumor": COLORS["secondary"],  # green
    "Glioma": COLORS["danger"],        # red
    "Meningioma": COLORS["amber"],     # amber
    "Pituitary": COLORS["primary"],    # blue
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MRI Tumor Classifier")
        self.geometry("1400x900")
        
        # Configure root
        self.configure(fg_color=COLORS["app_bg"])
        
        # Custom Fonts
        self.main_font = ("Segoe UI", 14)
        self.header_font = ("Segoe UI", 20, "bold")
        self.logo_font = ("Segoe UI", 16, "bold")
        self.prediction_font = ("Segoe UI", 26, "bold")
        self.label_font = ("Segoe UI", 11)
        
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
        
        # Tabview
        self.tab_view = ctk.CTkTabview(self, anchor="nw")
        self.tab_view.pack(fill="both", expand=True, padx=20, pady=20)

        self.pred_tab = self.tab_view.add("Prediction")
        self.train_tab = self.tab_view.add("Training")

        # Setup Tabs
        self.setup_prediction_tab()
        self.setup_training_tab()
        
        # Load Model
        self.load_model()
        
        # Graceful exit
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ===== HELPER METHODS =====
    
    def create_card(self, parent, accent_color=COLORS["primary"]):
        """Create a standard white card frame with colored top accent strip."""
        outer = ctk.CTkFrame(parent, fg_color=COLORS["border"], corner_radius=12)
        
        inner = ctk.CTkFrame(outer, fg_color=COLORS["card_bg"], corner_radius=12)
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Accent strip at top
        accent = ctk.CTkFrame(inner, fg_color=accent_color, height=6, corner_radius=0)
        accent.pack(fill="x", padx=0, pady=0)
        accent.configure(height=6)
        
        return outer, inner
    
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
        self.status_label.configure(text_color=color)

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
        self.pred_tab.grid_columnconfigure(0, weight=0, minsize=260)
        self.pred_tab.grid_columnconfigure(1, weight=1)
        self.pred_tab.grid_rowconfigure(0, weight=1)

        # ===== SIDEBAR =====
        sidebar_outer, sidebar_inner = self.create_card(self.pred_tab, accent_color=COLORS["primary"])
        sidebar_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Logo
        logo_frame = ctk.CTkFrame(sidebar_inner, fg_color=COLORS["card_bg"])
        logo_frame.pack(fill="x", padx=16, pady=(16, 0))
        ctk.CTkLabel(logo_frame, text="MRI Classifier", font=self.logo_font, 
                     text_color=COLORS["primary"]).pack(pady=(0, 2))
        
        # Divider
        ctk.CTkFrame(sidebar_inner, fg_color=COLORS["border"], height=2).pack(fill="x", padx=16, pady=12)
        
        # Upload Button
        ctk.CTkButton(sidebar_inner, text="📁 Upload Image", command=self.upload_image, 
                     font=self.main_font, height=40, fg_color=COLORS["primary"],
                     hover_color="#1557b0").pack(padx=16, pady=(0, 10))

        # Drag & Drop Zone
        dnd_frame = ctk.CTkFrame(sidebar_inner, fg_color=COLORS["light_gray"], 
                                 corner_radius=8, border_width=2, 
                                 border_color=COLORS["border"])
        dnd_frame.pack(padx=16, pady=10, fill="x")
        
        ctk.CTkLabel(dnd_frame, text="📂 Drag & Drop\nImage Here", 
                    font=self.label_font, text_color=COLORS["text_secondary"],
                    justify="center").pack(pady=12, padx=8)
        
        # Theme Section
        ctk.CTkLabel(sidebar_inner, text="Theme:", anchor="w", 
                    font=self.main_font, text_color=COLORS["text_primary"]).pack(padx=16, pady=(16, 4))
        ctk.CTkOptionMenu(sidebar_inner, values=["System", "Light", "Dark"], 
                         command=self.change_appearance_mode_event,
                         fg_color=COLORS["primary"]).pack(padx=16, pady=(0, 10), fill="x")
        
        # Divider
        ctk.CTkFrame(sidebar_inner, fg_color=COLORS["border"], height=2).pack(fill="x", padx=16, pady=12)
        
        # Export Button
        ctk.CTkButton(sidebar_inner, text="📄 Export Report", 
                     command=self.export_pdf_report, 
                     font=self.main_font, height=40,
                     fg_color=COLORS["secondary"],
                     hover_color="#2d8650").pack(padx=16, pady=16, fill="x")

        # ===== MAIN CONTENT AREA =====
        content = ctk.CTkFrame(self.pred_tab, fg_color="transparent")
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        # ===== IMAGE PANEL (with shadow effect) =====
        img_shadow_outer, img_inner = self.create_card(content, accent_color=COLORS["primary"])
        img_shadow_outer.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.image_label = ctk.CTkLabel(img_inner, text="No Image Selected", 
                                        font=self.main_font, 
                                        text_color=COLORS["text_secondary"],
                                        fg_color=COLORS["card_bg"])
        self.image_label.pack(expand=True, fill="both", padx=16, pady=16)

        # ===== PREDICTION CARD =====
        prediction_outer, prediction_inner = self.create_card(content, accent_color=COLORS["primary"])
        prediction_outer.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        pred_label = ctk.CTkLabel(prediction_inner, text="Prediction", font=self.label_font,
                                  text_color=COLORS["text_secondary"])
        pred_label.pack(padx=16, pady=(16, 4))
        
        self.prediction_label = ctk.CTkLabel(prediction_inner, text="-", 
                                             font=self.prediction_font,
                                             text_color=COLORS["text_primary"])
        self.prediction_label.pack(pady=(4, 16))
        
        # Divider
        ctk.CTkFrame(prediction_inner, fg_color=COLORS["border"], height=1).pack(fill="x", padx=16)
        
        # Confidence section
        conf_label = ctk.CTkLabel(prediction_inner, text="Confidence", font=self.label_font,
                                 text_color=COLORS["text_secondary"])
        conf_label.pack(padx=16, pady=(12, 4))
        
        self.confidence_label = ctk.CTkLabel(prediction_inner, text="- %", 
                                            font=("Segoe UI", 16, "bold"),
                                            text_color=COLORS["primary"])
        self.confidence_label.pack(pady=(0, 8))
        
        self.confidence_bar = ctk.CTkProgressBar(prediction_inner, fg_color=COLORS["border"],
                                                progress_color=COLORS["primary"])
        self.confidence_bar.pack(padx=16, pady=(0, 16), fill="x")
        self.confidence_bar.set(0)

        # ===== PROBABILITY CHART (bottom right) =====
        chart_outer, chart_inner = self.create_card(content, accent_color=COLORS["primary"])
        chart_inner.configure(fg_color=COLORS["card_bg"])
        chart_outer.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.chart_frame = chart_inner

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
        if self.last_probs is not None:
             self.draw_probs_chart(self.last_probs)

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
        ctk_img = ctk.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)
        self.image_label.configure(image=ctk_img, text="")

    def _update_prediction_ui(self, pred_class, conf, probs):
        """Update prediction card with results."""
        self.prediction_label.configure(text=pred_class)
        # Color the prediction text based on class
        pred_color = CLASS_COLORS.get(pred_class, COLORS["text_primary"])
        self.prediction_label.configure(text_color=pred_color)
        
        self.confidence_label.configure(text=f"{conf*100:.1f}%")
        self.confidence_bar.set(conf)
        
        self.last_probs = probs
        self.draw_probs_chart(probs)

    def _show_inference_error(self, error_msg):
        """Show inference error in prediction label."""
        self.prediction_label.configure(text="Error", text_color=COLORS["danger"])
        self.confidence_label.configure(text="N/A")
        print(f"Inference error: {error_msg}")

    def draw_probs_chart(self, probs):
        """Draw probability bar chart with class highlighting."""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        bg_color = COLORS["card_bg"]
        text_color = COLORS["text_primary"]
        
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Highlight the top prediction with green
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
        ax.set_title('Class Probabilities', color=text_color, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(0, 1)
        
        ax.tick_params(colors=text_color, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS["border"])
        ax.spines['bottom'].set_color(COLORS["border"])
        ax.grid(axis='x', alpha=0.2, color=COLORS["border"])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both", padx=16, pady=16)
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
        
        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor(COLORS["primary"]),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Subtitle style
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor(COLORS["text_secondary"]),
            alignment=TA_CENTER,
            spaceAfter=12
        )
        
        # Body style
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor(COLORS["text_primary"]),
            spaceAfter=8
        )
        
        # Add title
        story.append(Paragraph("MRI Tumor Classification Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             subtitle_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Add image
        story.append(Paragraph("Patient Scan", styles['Heading2']))
        temp_img_path = "/tmp/report_image.png"
        self.last_image_pil.save(temp_img_path)
        img = RLImage(temp_img_path, width=2.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
        
        # Diagnosis result
        pred_idx = np.argmax(self.last_probs)
        pred_class = self.class_names[pred_idx]
        confidence = self.last_probs[pred_idx]
        
        story.append(Paragraph("Diagnosis Result", styles['Heading2']))
        color_hex = colors.HexColor(CLASS_COLORS.get(pred_class, COLORS["text_primary"]))
        result_text = f"<font color='{color_hex.hexval()}'><b>{pred_class}</b></font>"
        story.append(Paragraph(result_text, body_style))
        story.append(Paragraph(f"Confidence: {confidence*100:.2f}%", body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Probability table
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
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(COLORS["light_gray"])),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor(COLORS["border"])),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor(COLORS["text_secondary"]),
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            "Generated by MRI Tumor Classifier — For research use only.",
            footer_style
        ))
        
        doc.build(story)
        
        # Clean up temp image
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    # ===== TRAINING TAB =====
    
    def setup_training_tab(self):
        self.train_tab.grid_columnconfigure(0, weight=0, minsize=300)
        self.train_tab.grid_columnconfigure(1, weight=1)
        self.train_tab.grid_rowconfigure(0, weight=1)

        # ===== SIDEBAR =====
        controls_outer, controls_inner = self.create_card(self.train_tab, accent_color=COLORS["primary"])
        controls_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        ctk.CTkLabel(controls_inner, text="Configuration", font=self.header_font,
                    text_color=COLORS["text_primary"]).pack(pady=(16, 4), padx=16)
        
        # Divider
        ctk.CTkFrame(controls_inner, fg_color=COLORS["border"], height=2).pack(fill="x", padx=16, pady=8)

        # Epochs
        ctk.CTkLabel(controls_inner, text="Epochs", font=self.main_font,
                    text_color=COLORS["text_primary"]).pack(pady=(12, 0), padx=16, anchor="w")
        ctk.CTkLabel(controls_inner, text="(Training iterations)", font=("Segoe UI", 9),
                    text_color=COLORS["text_secondary"]).pack(padx=16, anchor="w")
        self.epochs_slider = ctk.CTkSlider(controls_inner, from_=1, to=50, number_of_steps=49,
                                          fg_color=COLORS["border"],
                                          progress_color=COLORS["primary"])
        self.epochs_slider.set(5)
        self.epochs_slider.pack(pady=5, padx=16, fill="x")
        self.epochs_val = ctk.CTkLabel(controls_inner, text="5", font=self.main_font,
                                      text_color=COLORS["primary"])
        self.epochs_val.pack(padx=16)
        self.epochs_slider.configure(command=lambda v: self.epochs_val.configure(text=str(int(v))))

        # Learning Rate
        ctk.CTkLabel(controls_inner, text="Learning Rate", font=self.main_font,
                    text_color=COLORS["text_primary"]).pack(pady=(12, 0), padx=16, anchor="w")
        ctk.CTkLabel(controls_inner, text="(Step size for optimization)", font=("Segoe UI", 9),
                    text_color=COLORS["text_secondary"]).pack(padx=16, anchor="w")
        self.lr_entry = ctk.CTkEntry(controls_inner, placeholder_text="0.0001", font=self.main_font)
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.pack(pady=5, padx=16, fill="x")

        # Fine-tune Checkbox
        self.finetune_var = ctk.CTkCheckBox(controls_inner, text="Fine-tune", font=self.main_font,
                                           text_color=COLORS["text_primary"],
                                           checkbox_width=20, checkbox_height=20)
        self.finetune_var.pack(pady=(12, 0), padx=16, anchor="w")
        ctk.CTkLabel(controls_inner, text="Unfreezes layers for higher accuracy.\n(Slower training)", 
                    font=("Segoe UI", 9), text_color=COLORS["text_secondary"]).pack(padx=16, anchor="w")

        # Start Button
        self.train_btn = ctk.CTkButton(controls_inner, text="Start Training", 
                                      command=self.start_training_thread, 
                                      fg_color=COLORS["secondary"], 
                                      hover_color="#2d8650", 
                                      font=("Segoe UI", 16, "bold"), 
                                      height=50,
                                      text_color="white")
        self.train_btn.pack(pady=(20, 0), padx=16, fill="x")

        # Status
        self.status_label = ctk.CTkLabel(controls_inner, text="Status: Idle", 
                                        text_color=COLORS["text_secondary"], 
                                        font=self.main_font)
        self.status_label.pack(pady=12, padx=16)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(controls_inner, 
                                              fg_color=COLORS["border"],
                                              progress_color=COLORS["primary"],
                                              height=16)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=(0, 16), padx=16, fill="x")

        # ===== MAIN CONTENT AREA =====
        visuals = ctk.CTkFrame(self.train_tab, fg_color="transparent")
        visuals.grid(row=0, column=1, sticky="nsew")
        visuals.grid_rowconfigure(0, weight=0, minsize=110)
        visuals.grid_rowconfigure(1, weight=1)
        visuals.grid_rowconfigure(2, weight=1)
        visuals.grid_columnconfigure(0, weight=1)

        # ===== METRICS SUMMARY BAR =====
        metrics_frame = ctk.CTkFrame(visuals, fg_color="transparent")
        metrics_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 10))
        metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        self.metric_cards = {}
        metric_configs = [
            ("Current Epoch", "0", COLORS["primary"]),
            ("Train Loss", "0.0000", COLORS["amber"]),
            ("Val Loss", "0.0000", COLORS["danger"]),
            ("Best Val Loss", "0.0000", COLORS["secondary"]),
        ]
        
        for idx, (metric_name, initial_val, accent_color) in enumerate(metric_configs):
            card_outer, card_inner = self.create_card(metrics_frame, accent_color=accent_color)
            card_outer.grid(row=0, column=idx, sticky="nsew", padx=5)
            
            label = ctk.CTkLabel(card_inner, text=metric_name, font=("Segoe UI", 10),
                               text_color=COLORS["text_secondary"])
            label.pack(padx=12, pady=(10, 2))
            
            value_label = ctk.CTkLabel(card_inner, text=initial_val, 
                                      font=("Segoe UI", 16, "bold"),
                                      text_color=accent_color)
            value_label.pack(padx=12, pady=(2, 10))
            
            self.metric_cards[metric_name.lower().replace(" ", "_")] = value_label

        # ===== TRAINING LOG =====
        log_label = ctk.CTkLabel(visuals, text="📋 Training Log", font=("Segoe UI", 12, "bold"),
                                text_color=COLORS["primary"])
        log_label.grid(row=1, column=0, sticky="nw", padx=0, pady=(0, 5))
        
        self.log_box = ctk.CTkTextbox(visuals, font=("Consolas", 11), corner_radius=10,
                                     fg_color=COLORS["light_gray"],
                                     text_color=COLORS["text_primary"],
                                     scrollbar_button_color=COLORS["primary"])
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.log_box.insert("0.0", "--- Training Log ---\n")

        # ===== LOSS CHART =====
        chart_label = ctk.CTkLabel(visuals, text="📊 Loss Over Epochs", font=("Segoe UI", 12, "bold"),
                                  text_color=COLORS["primary"])
        chart_label.grid(row=2, column=0, sticky="nw", padx=0, pady=(10, 5))
        
        self.train_chart_frame = ctk.CTkFrame(visuals, fg_color=COLORS["card_bg"], corner_radius=12)
        self.train_chart_frame.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)

    def start_training_thread(self):
        self.train_btn.configure(state="disabled", text="Training...")
        self.update_status("Training started...")
        self.progress_bar.set(0)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.log_box.delete("0.0", "end")
        self.log_box.insert("0.0", "--- Training Log ---\n")
        
        epochs = int(self.epochs_slider.get())
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
            
            self.progress_bar.set(data['epoch'] / data['epochs'])
            self.train_losses.append(data['train_loss'])
            self.val_losses.append(data['val_loss'])
            
            # Update metric cards
            self.metric_cards['current_epoch'].configure(text=str(data['epoch']))
            self.metric_cards['train_loss'].configure(text=f"{data['train_loss']:.4f}")
            self.metric_cards['val_loss'].configure(text=f"{data['val_loss']:.4f}")
            self.metric_cards['best_val_loss'].configure(text=f"{self.best_val_loss:.4f}")
            
            self.draw_training_chart()

    def draw_training_chart(self):
        for widget in self.train_chart_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        bg_color = COLORS["card_bg"]
        text_color = COLORS["text_primary"]
        
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        ax.plot(self.train_losses, label="Train Loss", color=COLORS["primary"], linewidth=2.5)
        ax.plot(self.val_losses, label="Val Loss", color=COLORS["danger"], linewidth=2.5, linestyle="--")
        
        ax.set_xlabel('Epoch', color=text_color, fontsize=10)
        ax.set_ylabel('Loss', color=text_color, fontsize=10)
        ax.set_title('Loss Over Epochs', color=text_color, fontsize=12, fontweight='bold', pad=10)
        
        ax.legend(facecolor=bg_color, labelcolor=text_color, framealpha=0.95, 
                 edgecolor=COLORS["border"], loc='upper right')
        ax.tick_params(colors=text_color, labelsize=9)
        
        ax.grid(True, alpha=0.3, color=COLORS["border"], linestyle='-', linewidth=0.5)
        
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.train_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both", padx=16, pady=16)
        plt.close(fig)

    def finish_training(self):
        self.train_btn.configure(state="normal", text="Start Training", text_color="white")
        self.update_status("Complete")
        self.log_box.insert("end", "\n✅ Training Finished. Model Reloaded.\n")
        
        # Flash green effect (3 times, 300ms intervals)
        self._flash_status_label(3)

    def _flash_status_label(self, times):
        """Flash the status label green 3 times as success animation."""
        if times <= 0:
            return
        
        current_color = self.status_label.cget("text_color")
        new_color = COLORS["secondary"] if current_color == COLORS["secondary"] else COLORS["text_secondary"]
        self.status_label.configure(text_color=new_color)
        
        self.after(300, lambda: self._flash_status_label(times - 1))

    def on_closing(self):
        self.quit()
        self.destroy()

def display():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    display()
