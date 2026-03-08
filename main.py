import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import shutil
import time
from utils.dpi_scaling import DPIScaler
from core.image_processing import get_image_files, load_image_as_rgb, extract_numbers, create_overlay

# Set random seed
np.random.seed(3)

# Environment variable setting
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# User-customizable parameters
class Config:
    def __init__(self):
        self.input_dir = ""
        self.output_dir = ""
        self.checkpoint = ""
        self.model_cfg = ""
        self.input_points = []
        self.input_labels = []

class SAM2App:
    def __init__(self, master):
        self.master = master
        self.master.title("SAM2 Mask Generator")
        self.config = Config()
        matplotlib.use('TkAgg')
        
        # Centralized DPI scaling
        self.dpi = DPIScaler(master)
        
        self.current_image = None
        self.current_mask = None
        self.plotting_mode = False
        self.processing = False
        self.device = 'cpu'
        self.point_mode = 'foreground'
        self.inference_state = None
        
        # Add predictor management for memory leak prevention
        self.current_predictor = None
        self.current_model_config = None
        
        self.setup_gui()
        self.current_masks = None
        self.processTimes = 0
        self.start_index = 1
        self.end_index = []
        self.start_index_previous = []
        self.end_index_previous = []
    
    def setup_gui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top frame for input/output selection
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self.setup_io_frame(top_frame)

        # Add range selection controls - moved after input/output selection
        range_frame = ttk.Frame(main_frame)
        range_frame.pack(fill=tk.X, pady=(0, 10))  # Modified pady to ensure proper spacing
        ttk.Label(range_frame, text="Start frame#:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_index_entry = ttk.Entry(range_frame, width=5)
        self.start_index_entry.pack(side=tk.LEFT, padx=(0, 5))
        # Add binding events
        self.start_index_entry.bind('<Return>', lambda e: self.show_raw_image())
        self.start_index_entry.bind('<FocusOut>', lambda e: self.show_raw_image())
        
        ttk.Label(range_frame, text="End frame#:").pack(side=tk.LEFT, padx=(0, 5))
        self.end_index_entry = ttk.Entry(range_frame, width=5)
        self.end_index_entry.pack(side=tk.LEFT, padx=(0, 5))

        # Middle frame for controls
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.X, pady=(0, 10))

        # Visualization controls
        viz_frame = ttk.LabelFrame(middle_frame, text="Visualization Controls")
        viz_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.setup_viz_controls(viz_frame)

        # Analysis controls
        analysis_frame = ttk.LabelFrame(middle_frame, text="Analysis Controls")
        analysis_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        self.setup_analysis_controls(analysis_frame)

        # Progress bar and label
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        self.setup_progress_bar(progress_frame)

        # Image frame: original image and mask display
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        self.setup_image_display(image_frame)

    def setup_io_frame(self, parent):
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="Input Directory:").pack(side=tk.LEFT, padx=(0, 5))
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(input_frame, text="Browse", command=self.select_input).pack(side=tk.LEFT, padx=(5, 0))

        output_frame = ttk.Frame(parent)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=(0, 5))
        self.output_entry = ttk.Entry(output_frame)
        self.output_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(output_frame, text="Browse", command=self.select_output).pack(side=tk.LEFT, padx=(5, 0))

    def setup_viz_controls(self, parent):
        self.plot_points_btn = ttk.Button(parent, text="Start Drawing Points",
                                          command=self.toggle_plotting_mode)
        self.plot_points_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.toggle_mode_btn = ttk.Button(parent, text="Mode: Foreground",
                                          command=self.toggle_point_mode)
        self.toggle_mode_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_points_btn = ttk.Button(parent, text="Clear Points", command=self.clear_points)
        self.clear_points_btn.pack(side=tk.LEFT, padx=5, pady=5)

    def setup_analysis_controls(self, parent):
        device_frame = ttk.Frame(parent)
        device_frame.pack(fill=tk.X, pady=5)
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.device_var = tk.StringVar(self.master)
        self.device_var.set("CPU")
        self.device_menu = ttk.Combobox(device_frame, textvariable=self.device_var,
                                        values=["CPU", "GPU"])
        self.device_menu.pack(side=tk.LEFT, padx=(0, 10))
        self.device_menu.bind("<<ComboboxSelected>>", self.on_device_change)

        self.model_var = tk.StringVar(self.master)
        self.model_var.set("SAM2 Hiera Large")
        ttk.Label(device_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_menu = ttk.Combobox(device_frame, textvariable=self.model_var,
                                       values=["SAM2 Hiera Large", "SAM2 Hiera Base Plus",
                                               "SAM2 Hiera Small", "SAM2 Hiera Tiny"])
        self.model_menu.pack(side=tk.LEFT, padx=(0, 10))

        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)
        self.start_btn = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

    def setup_progress_bar(self, parent):
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        self.progress_label = ttk.Label(parent, text="")
        self.progress_label.pack()

    def setup_image_display(self, parent):
        # Use 1x3 layout: Original Image | Generated Mask | Overlay (when available)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Configure subplot spacing and appearance
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
        
        # Set figure background color
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)


    def select_input(self):
        self.config.input_dir = filedialog.askdirectory(title="Select input folder")
        if self.config.input_dir:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.config.input_dir)
            self.cleanup_temp_folders()  # Clean up temporary folders after confirming input folder
            self.show_raw_image()

    def select_output(self):
        self.config.output_dir = filedialog.askdirectory(title="Select output folder")
        if self.config.output_dir:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, self.config.output_dir)

    def on_device_change(self, event):
        selected_device = self.device_var.get()
        if selected_device == "GPU":
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("GPU selected and available. Using CUDA.")
            else:
                self.device = 'cpu'
                self.device_var.set("CPU")
                messagebox.showwarning("GPU Not Available",
                                       "GPU is not available. Switching to CPU.")
                print("GPU not available. Falling back to CPU.")
        else:
            self.device = 'cpu'
            print("CPU selected.")

    def show_raw_image(self):
        image_files = get_image_files(self.config.input_dir)
        if not image_files:
            messagebox.showwarning("Warning", "No supported image files found in input directory.")
            return
            
        try:
            self.start_index = int(self.start_index_entry.get() or 1)
        except ValueError:
            messagebox.showerror("Error", "Invalid start index. Please enter a valid number.")
            return
            
        # Validate start index bounds
        if self.start_index < 1 or self.start_index > len(image_files):
            messagebox.showerror("Error", f"Start index must be between 1 and {len(image_files)}.")
            return
            
        # Calculate safe array index (convert from 1-based to 0-based)
        array_index = self.start_index - 1
        image_path = image_files[array_index]

        image = load_image_as_rgb(image_path)
        if image is None:
            messagebox.showerror("Error", f"Failed to load image: {image_path}")
            return
        self.current_image = image
        self.display_image()

    def display_image(self, keep_points=False):
        if self.current_image is not None:
            # Panel 1: Original Image with Points
            self.ax1.clear()
            self.ax1.imshow(self.current_image)
            sizes = self.dpi.matplotlib_sizes()
            self.ax1.set_title('Original Image', fontsize=sizes['title'], fontweight='bold', pad=10)
            self.ax1.set_xlabel('X', fontsize=sizes['label'])
            self.ax1.set_ylabel('Y', fontsize=sizes['label'])
            
            # Always check if points should be retained
            if self.config.input_points:
                self.update_image_with_points(self.ax1)

            # Panel 2: Generated Mask
            if self.current_masks is None:
                self.ax2.clear()
                self.ax2.set_title('Generated Mask', fontsize=sizes['title'], fontweight='bold', pad=10)
                self.ax2.text(0.5, 0.5, 'Mask will be\ndisplayed here',
                             ha='center', va='center', fontsize=sizes['text'],
                             transform=self.ax2.transAxes, color='gray')
                self.ax2.axis('off')
            else:
                self.ax2.clear()
                self.ax2.imshow(self.current_masks[0], cmap='gray')
                self.ax2.set_title('Generated Mask', fontsize=sizes['title'], fontweight='bold', pad=10)
                self.ax2.axis('off')

            # Panel 3: Overlay View (Original + Mask)
            self.ax3.clear()
            if self.current_masks is not None:
                blended = create_overlay(self.current_image, self.current_masks[0])
                self.ax3.imshow(blended)
                self.ax3.set_title('Overlay View', fontsize=sizes['title'], fontweight='bold', pad=10)
            else:
                self.ax3.set_title('Overlay View', fontsize=sizes['title'], fontweight='bold', pad=10)
                self.ax3.text(0.5, 0.5, 'Overlay will be\ndisplayed here',
                             ha='center', va='center', fontsize=sizes['text'],
                             transform=self.ax3.transAxes, color='gray')

            self.ax3.axis('off')

            # Apply consistent styling with DPI scaling
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.tick_params(labelsize=sizes['tick'])
                
            self.canvas.draw()

    def toggle_plotting_mode(self):
        self.plotting_mode = not self.plotting_mode
        if self.plotting_mode:
            self.plot_points_btn.config(text="Stop Drawing")
            self.canvas.mpl_connect('button_press_event', self.on_click)
        else:
            self.plot_points_btn.config(text="Start Drawing Points")

    def toggle_point_mode(self):
        self.point_mode = 'background' if self.point_mode == 'foreground' else 'foreground'
        self.toggle_mode_btn.config(text=f"Mode: {self.point_mode.capitalize()}")

    def on_click(self, event):
        if event.inaxes == self.ax1 and self.plotting_mode:
            x, y = int(event.xdata), int(event.ydata)
            if len(self.config.input_points) < 100:
                self.config.input_points.append([x, y])
                self.config.input_labels.append(1 if self.point_mode == 'foreground' else 0)
                # Update image and retain points
                self.display_image(keep_points=True)
            else:
                messagebox.showwarning("Warning", "Maximum number of points (100) reached.")

    def clear_points(self):
        self.config.input_points = []
        self.config.input_labels = []
        self.display_image(keep_points=True)
        messagebox.showinfo("Info", "All points have been cleared.")

    def update_image_with_points(self, ax=None):
        if ax is None:
            ax = self.ax1

        ax.clear()  # Clear the axes before redrawing
        ax.imshow(self.current_image)  # Redraw the current image

        sizes = self.dpi.matplotlib_sizes()
        ax.set_title('Original Image', fontsize=sizes['title'], fontweight='bold', pad=10)
        ax.set_xlabel('X', fontsize=sizes['label'])
        ax.set_ylabel('Y', fontsize=sizes['label'])

        if self.config.input_points:
            points = np.array(self.config.input_points)
            labels = np.array(self.config.input_labels)

            # Enhanced point visualization with DPI scaling
            ax.scatter(points[labels == 1, 0], points[labels == 1, 1],
                      color='red', s=sizes['point'], alpha=0.8, edgecolors='white',
                      linewidth=2, label='Foreground', marker='o')
            ax.scatter(points[labels == 0, 0], points[labels == 0, 1],
                      color='blue', s=sizes['point'], alpha=0.8, edgecolors='white',
                      linewidth=2, label='Background', marker='s')

        # Clear the previous legend to avoid repetition
        legend = ax.get_legend()
        if legend:
            legend.remove()

        # Enhanced legend styling with DPI scaling
        if self.config.input_points:
            legend = ax.legend(loc='upper right', fontsize=sizes['legend'], framealpha=0.9,
                             fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')

        ax.tick_params(labelsize=sizes['tick'])
        self.canvas.draw()

    def update_progress(self, progress):
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress:.1f}%")
        self.master.update()

    def start_processing(self):
        if not self.config.input_dir or not self.config.output_dir:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return

        if not self.config.input_points:
            messagebox.showerror("Error", "Please add at least one point.")
            return

        # Set up model
        models = {
            "SAM2 Hiera Large": ("checkpoints/sam2.1_hiera_large.pt", "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"),
            "SAM2 Hiera Base Plus": ("checkpoints/sam2.1_hiera_base_plus.pt", "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"),
            "SAM2 Hiera Small": ("checkpoints/sam2.1_hiera_small.pt", "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"),
            "SAM2 Hiera Tiny": ("checkpoints/sam2.1_hiera_tiny.pt", "sam2/configs/sam2.1/sam2.1_hiera_t.yaml")
        }
        self.config.checkpoint, self.config.model_cfg = models[self.model_var.get()]

        # Start processing
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.processing = True
        self.process_images()

    def stop_processing(self):
        self.processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Processing stopped.")

    def _get_or_create_predictor(self, checkpoint_path, model_cfg_path):
        """Get existing predictor or create new one, preventing memory leaks"""
        current_config = (checkpoint_path, model_cfg_path, self.device)
        
        # Check if we can reuse existing predictor
        if (self.current_predictor is not None and 
            self.current_model_config == current_config):
            return self.current_predictor
        
        # Clean up old predictor if exists
        self._cleanup_predictor()
        
        try:
            # Create new predictor
            self.current_predictor = build_sam2_video_predictor(
                model_cfg_path, checkpoint_path, device=self.device
            )
            self.current_model_config = current_config
            return self.current_predictor
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create SAM2 predictor: {str(e)}")
            return None

    def _cleanup_predictor(self):
        """Properly cleanup predictor to prevent memory leaks"""
        if self.current_predictor is not None:
            try:
                # Reset inference state if exists
                if self.inference_state is not None:
                    self.current_predictor.reset_state(self.inference_state)
                    self.inference_state = None
                
                # Clear GPU cache if using CUDA
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Clear predictor reference
                self.current_predictor = None
                self.current_model_config = None
                
            except Exception as e:
                print(f"Warning: Error during predictor cleanup: {e}")

    def process_images(self):
        """Process images with comprehensive error handling"""
        try:
            # Ensure output directory exists
            os.makedirs(self.config.output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create output directory: {str(e)}")
            self.processing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return
        
        try:
            self._process_images_internal()
        except Exception as e:
            # Comprehensive error handling for unexpected failures
            error_msg = f"Unexpected error during processing: {str(e)}"
            print(error_msg)
            messagebox.showerror("Processing Error", error_msg)
        finally:
            # Ensure UI state is reset regardless of success or failure
            self.processing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _process_images_internal(self):
        """Internal processing method with detailed error handling"""
        # Get parent directory and folder name of input_dir
        parent_dir = os.path.dirname(self.config.input_dir)
        input_folder_name = os.path.basename(self.config.input_dir)
        
        # Get image file list and sort
        image_files = get_image_files(self.config.input_dir)
        total_images = len(image_files)
        
        if total_images == 0:
            messagebox.showerror("Error", "No supported image files found in input directory.")
            return
        
        # Get user input range with proper validation
        try:
            self.start_index = int(self.start_index_entry.get() or 1)
            self.end_index = int(self.end_index_entry.get() or total_images)
        except ValueError:
            messagebox.showerror("Error", "Invalid frame range values. Please enter valid numbers.")
            self.processing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return
        
        # Ensure indices are within valid range (fix boundary condition bug)
        self.start_index = max(1, self.start_index)
        self.end_index = min(total_images, self.end_index)  # Fixed: removed +1 to prevent index overflow
        
        # Validate range consistency
        if self.start_index > self.end_index:
            messagebox.showerror("Error", "Start index cannot be greater than end index.")
            self.processing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return
                
        # Create temporary directory, use input folder name with suffix, if range hasn't changed twice, use the previous jpg_range folder
        if self.start_index != self.start_index_previous or self.end_index != self.end_index_previous:
            self.processTimes = self.processTimes + 1
        temp_dir = os.path.join(parent_dir, f"{input_folder_name}_jpg/{input_folder_name}_jpg_range{self.processTimes}")
        os.makedirs(temp_dir, exist_ok=True)

        self.start_index_previous = self.start_index
        self.end_index_previous = self.end_index

        # Check JPEG files in temporary folder
        existing_jpegs = {os.path.basename(f): f for f in glob.glob(os.path.join(temp_dir, "*.jpg"))}
        
        # Convert all images to JPEG format
        jpeg_files = []
        processing_images = image_files[self.start_index-1:self.end_index]
        num_processing_images = len(processing_images)
        
        # Initialize progress bar for mask prediction only
        self.update_progress(0)
        self.progress_label.config(text="Preparing images for mask prediction...")
        
        for idx, img_path in enumerate(processing_images):
            # Use simple numeric sequence naming for temporary files
            jpeg_name = f"{idx+self.start_index:06d}.jpg"  # Generate filename like "000001.jpg"
            jpeg_path = os.path.join(temp_dir, jpeg_name)
        
            # Check if corresponding JPEG file already exists
            if jpeg_name in existing_jpegs:
                jpeg_files.append(jpeg_path)
                continue
            
            try:
                # First try to read with OpenCV
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                
                # If OpenCV reading fails, try with PIL/Pillow
                if img is None:
                    
                    print(f"OpenCV failed to read image: {img_path}. Trying with PIL...")
                    
                    # Use PIL to read image
                    pil_image = Image.open(img_path)
                    
                    # Extract bit depth information for logging
                    bit_depth = getattr(pil_image, 'bit_depth', 'unknown')
                    print(f"Image format: {pil_image.format}, Mode: {pil_image.mode}, Bit depth: {bit_depth}")
                    
                    # Special handling for 32-bit floating point TIFF and other high bit depth images
                    if pil_image.mode in ['F', 'I', 'I;16', 'I;32']:
                        # Convert to numpy array
                        img_array = np.array(pil_image)
                        
                        # Normalize to 0-255 range
                        min_val = np.min(img_array)
                        max_val = np.max(img_array)
                        
                        if max_val > min_val:  # Avoid division by zero
                            img = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            img = np.zeros_like(img_array, dtype=np.uint8)
                        
                        # If single channel image, convert to RGB
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:
                        # For standard modes, convert to RGB
                        pil_image = pil_image.convert('RGB')
                        img = np.array(pil_image)
                        # PIL's RGB order is different from OpenCV's BGR order, need conversion
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    # OpenCV successfully read, handle different data types
                    if img.dtype == np.uint16:
                        img = (img / 256).astype(np.uint8)
                    elif img.dtype == np.float32 or img.dtype == np.float64:
                        min_val = np.min(img)
                        max_val = np.max(img)
                        if max_val > min_val:
                            img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            img = np.zeros_like(img, dtype=np.uint8)
                    
                    # Ensure image is colored
                    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Save as JPEG
                cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                jpeg_files.append(jpeg_path)
                
            except Exception as e:
                print(f"Failed to process image {img_path}: {str(e)}")
                # If processing fails, create a black image as placeholder
                try:
                    ref_shape = None
                    for existing_jpeg in jpeg_files:
                        ref_img = cv2.imread(existing_jpeg)
                        if ref_img is not None:
                            ref_shape = ref_img.shape
                            break
                    if ref_shape is None:
                        ref_shape = (512, 512, 3)
                    blank_img = np.zeros(ref_shape, dtype=np.uint8)
                    cv2.imwrite(jpeg_path, blank_img)
                    jpeg_files.append(jpeg_path)
                    print(f"Created placeholder ({ref_shape[1]}x{ref_shape[0]}) for {img_path}")
                except Exception as e:
                    print(f"Failed to create placeholder for {img_path}: {e}")
                continue

        sam2_checkpoint = os.path.join(current_dir, self.config.checkpoint)
        model_cfg = os.path.join(current_dir, self.config.model_cfg)
        
        # Initialize SAM2 model using managed predictor (prevents memory leaks)
        predictor = self._get_or_create_predictor(sam2_checkpoint, model_cfg)
        if predictor is None:
            self.processing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return
        
        # Reset previous state if exists
        if self.inference_state is not None:
            predictor.reset_state(self.inference_state)
        
        # Initialize new state
        try:
            self.inference_state = predictor.init_state(video_path=temp_dir)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize inference state: {str(e)}")
            self.processing = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return

        # Convert input points and labels to numpy arrays
        points_array = np.array(self.config.input_points)
        labels_array = np.array(self.config.input_labels)
        
        # Call predictor to add new points or box
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=1,
            points=points_array,  # Use converted points_array
            labels=labels_array,  # Use converted labels_array
        )

        # Overall start time for basic timing
        total_start_time = time.time()
        
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        
        # Create inference time recording generator wrapper
        def timed_propagate_in_video():
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
                yield out_frame_idx, out_obj_ids, out_mask_logits
        
        # Start processing each frame
        frame_count = 0
        self.progress_label.config(text="Starting mask prediction...")
        for out_frame_idx, out_obj_ids, out_mask_logits in timed_propagate_in_video():
            
            # Read image with proper exception handling and index validation
            try:
                # Validate frame index to prevent array out of bounds
                adjusted_frame_idx = out_frame_idx + self.start_index - 1
                if adjusted_frame_idx < 0 or adjusted_frame_idx >= len(image_files):
                    print(f"Warning: Frame index {adjusted_frame_idx} out of bounds. Skipping.")
                    continue
                
                image_path = image_files[adjusted_frame_idx]
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {image_path}. Skipping.")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_name = os.path.basename(image_path)
                
            except Exception as e:
                print(f"Error processing frame {out_frame_idx}: {str(e)}. Skipping.")
                continue

            # Process inference results (this part is the core post-inference processing)
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            out_masks = video_segments[out_frame_idx]
            binary_masks = out_masks[1]
            binary_masks = binary_masks.astype(np.uint8) * 255
            
            frame_count += 1

            if not self.processing:
                print("Processing stopped by user.")
                break

            # Update display with both image and mask in a single render
            self.current_image = image
            self.current_masks = binary_masks
            self.display_image()

            # Save the binary mask with error handling
            try:
                mask1_dir = os.path.join(self.config.output_dir, "mask1")
                os.makedirs(mask1_dir, exist_ok=True)
                mask_path = os.path.join(mask1_dir, image_name)
                success = cv2.imwrite(mask_path, binary_masks[0])
                if not success:
                    print(f"Warning: Failed to save mask for {image_name}")
            except Exception as e:
                print(f"Error saving mask for {image_name}: {str(e)}")
                # Continue processing other frames even if one fails

            # Update progress - show mask prediction progress only
            mask_progress = (out_frame_idx + 1) / len(processing_images) * 100
            self.update_progress(mask_progress)
            self.progress_label.config(
                text=f"Mask Prediction: {out_frame_idx+1}/{len(processing_images)}"
            )
            self.master.update()

        self.processing = False
        final_message = "Processing complete." if frame_count > 0 else "Processing stopped."
        self.progress_label.config(text=final_message)

        # Re-enable start button and disable stop button when processing is done
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def cleanup_temp_folders(self):
        if not self.config.input_dir:
            return
        
        # Get parent directory and folder name of input_dir
        parent_dir = os.path.dirname(self.config.input_dir)
        input_folder_name = os.path.basename(self.config.input_dir)
        
        # jpg folder path
        jpg_dir = os.path.join(parent_dir, f"{input_folder_name}_jpg")
        
        # If jpg folder exists, delete all contents
        if os.path.exists(jpg_dir):
            try:
                for item in os.listdir(jpg_dir):
                    item_path = os.path.join(jpg_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                print(f"Cleaned up temporary folder: {jpg_dir}")
            except Exception as e:
                print(f"Error cleaning up temporary folder: {e}")
        else:
            # If jpg folder doesn't exist, create it
            os.makedirs(jpg_dir)
            print(f"Created temporary folder: {jpg_dir}")

    def __del__(self):
        """Cleanup when application is destroyed"""
        self._cleanup_predictor()

if __name__ == "__main__":
    root = tk.Tk()

    # Make the application DPI aware on Windows
    try:
        import ctypes
        # Tell Windows that this app is DPI aware
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass  # Ignore if not on Windows or if the call fails

    # Configure modern UI styling with DPI awareness
    style = ttk.Style()
    style.theme_use('clam')

    # Configure DPI-aware fonts and window scaling
    dpi_scaler = DPIScaler(root)
    dpi_scaler.configure_ttk_style(style)

    w, h, min_w, min_h = dpi_scaler.scaled_window_size()
    root.geometry(f"{w}x{h}")
    root.minsize(min_w, min_h)
    
    # Set window icon and title styling
    root.title("SAM2 Mask Generator - Professional Edition")
    try:
        # Try to set a professional icon (if available)
        root.iconbitmap(default='')  # You can add an .ico file path here
    except Exception:
        pass  # Ignore if no icon file available
    
    app = SAM2App(root)
    
    # Add cleanup on window close
    def on_closing():
        app._cleanup_predictor()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()