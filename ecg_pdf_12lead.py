"""
12-Lead ECG PDF Visualization
Renders 12-lead ECG on standard graph paper for clinical reports.
"""

import numpy as np
from PIL import Image, ImageDraw
import io
import logging

logger = logging.getLogger(__name__)


class ECG12LeadPDFRenderer:
    """Render 12-lead ECG on graph paper for PDF reports."""
    
    # Lead order for standard display
    LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Lead descriptions for labeling
    LEAD_LABELS = {
        'I': 'I',
        'II': 'II',
        'III': 'III',
        'aVR': 'aVR',
        'aVL': 'aVL',
        'aVF': 'aVF',
        'V1': 'V1',
        'V2': 'V2',
        'V3': 'V3',
        'V4': 'V4',
        'V5': 'V5',
        'V6': 'V6'
    }
    
    def __init__(self, sampling_rate=500):
        """Initialize 12-lead PDF renderer."""
        self.fs = sampling_rate
    
    def create_12lead_image(self, leads_dict, title="12-LEAD ECG"):
        """
        Create 12-lead ECG image on graph paper.
        
        Args:
            leads_dict: Dictionary with 12 lead signals
            title: Title for the ECG
        
        Returns:
            PIL Image object
        """
        try:
            # Standard ECG paper dimensions
            # Each lead: ~1.5 inches wide (approx 400 pixels at 270 DPI)
            # 3 rows x 4 columns layout
            lead_width = 400
            lead_height = 280
            
            # Layout: 3 rows, 4 columns
            cols = 4
            rows = 3
            
            # Margins
            margin_top = 40
            margin_left = 40
            margin_right = 40
            margin_between_rows = 30
            margin_between_cols = 20
            
            # Total size
            total_width = (lead_width * cols) + (margin_between_cols * (cols - 1)) + margin_left + margin_right
            total_height = (lead_height * rows) + (margin_between_rows * (rows - 1)) + margin_top + 40
            
            # Create image
            img = Image.new('RGB', (total_width, total_height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add title
            try:
                from PIL import ImageFont
                # Try to use a nice font, fall back to default
                try:
                    title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                except:
                    title_font = ImageFont.load_default()
            except:
                title_font = ImageFont.load_default()
            
            draw.text((margin_left, 10), title, fill='black', font=title_font)
            
            # Render each lead
            lead_idx = 0
            for row in range(rows):
                for col in range(cols):
                    if lead_idx >= len(self.LEAD_ORDER):
                        break
                    
                    lead_name = self.LEAD_ORDER[lead_idx]
                    
                    # Calculate position
                    x = margin_left + col * (lead_width + margin_between_cols)
                    y = margin_top + row * (lead_height + margin_between_rows)
                    
                    # Get lead signal
                    if lead_name in leads_dict:
                        lead_signal = leads_dict[lead_name]
                    else:
                        lead_signal = np.zeros(1000)
                    
                    # Render lead on graph paper
                    self._render_lead_on_graph(img, lead_signal, x, y, lead_width, lead_height, lead_name)
                    
                    lead_idx += 1
            
            logger.info("✓ 12-lead ECG image created successfully")
            return img
            
        except Exception as e:
            logger.error(f"Error creating 12-lead image: {e}")
            # Return blank image on error
            return Image.new('RGB', (800, 600), color='white')
    
    def _render_lead_on_graph(self, img, signal, x_offset, y_offset, width, height, lead_name):
        """
        Render a single lead on the image with graph paper background.
        
        Args:
            img: PIL Image to draw on
            signal: Lead signal (numpy array)
            x_offset, y_offset: Position on image
            width, height: Dimensions of lead area
            lead_name: Name of lead (for label)
        """
        try:
            draw = ImageDraw.Draw(img)
            
            # Draw background rectangle
            draw.rectangle(
                [(x_offset, y_offset), (x_offset + width, y_offset + height)],
                outline='lightgray',
                fill='white'
            )
            
            # Draw graph paper grid
            # Small grid (1mm = 2 pixels, 5mm squares)
            grid_small_size = 2  # pixels per small square
            grid_large_size = grid_small_size * 5  # pixels per large square
            
            # Small grid lines (light)
            for x in range(x_offset, x_offset + width, grid_small_size):
                draw.line([(x, y_offset), (x, y_offset + height)], fill='#F0F0F0', width=1)
            for y in range(y_offset, y_offset + height, grid_small_size):
                draw.line([(x_offset, y), (x_offset + width, y)], fill='#F0F0F0', width=1)
            
            # Large grid lines (darker)
            for x in range(x_offset, x_offset + width, grid_large_size):
                draw.line([(x, y_offset), (x, y_offset + height)], fill='#E0E0E0', width=1)
            for y in range(y_offset, y_offset + height, grid_large_size):
                draw.line([(x_offset, y), (x_offset + width, y)], fill='#E0E0E0', width=1)
            
            # Draw lead signal
            if signal is not None and len(signal) > 0:
                signal = np.asarray(signal, dtype=np.float32)
                
                # Normalize signal to fit in the area
                sig_min = np.min(signal)
                sig_max = np.max(signal)
                sig_range = sig_max - sig_min
                if sig_range < 1e-6:
                    sig_range = 1
                
                # Leave margins for signal
                padding_top = height * 0.15
                padding_bottom = height * 0.15
                draw_height = height - padding_top - padding_bottom
                
                # Draw ECG trace
                points = []
                for i, val in enumerate(signal):
                    if i >= len(signal):
                        break
                    
                    # Map signal to x coordinate
                    x = x_offset + (i / max(len(signal) - 1, 1)) * width
                    
                    # Map signal value to y coordinate
                    normalized = (val - sig_min) / sig_range if sig_range > 0 else 0.5
                    y = y_offset + padding_top + (1 - normalized) * draw_height
                    
                    points.append((x, y))
                
                # Draw polyline
                if len(points) > 1:
                    draw.line(points, fill='#000000', width=2)
            
            # Draw lead label
            try:
                from PIL import ImageFont
                try:
                    label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
                except:
                    label_font = ImageFont.load_default()
            except:
                label_font = ImageFont.load_default()
            
            # Draw label at bottom-left of lead area
            draw.text(
                (x_offset + 5, y_offset + height - 15),
                self.LEAD_LABELS[lead_name],
                fill='black',
                font=label_font
            )
            
        except Exception as e:
            logger.warning(f"Error rendering lead {lead_name}: {e}")
    
    def get_12lead_image_buffer(self, leads_dict):
        """
        Get 12-lead ECG as PIL Image buffer suitable for PDF embedding.
        
        Args:
            leads_dict: Dictionary with 12 lead signals
        
        Returns:
            BytesIO buffer with PNG image
        """
        img = self.create_12lead_image(leads_dict)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return buf


def generate_12lead_pdf_image(leads_dict, sampling_rate=500):
    """
    Generate 12-lead ECG image for PDF report.
    
    Args:
        leads_dict: Dictionary with 12 lead signals
        sampling_rate: Sampling rate in Hz
    
    Returns:
        BytesIO buffer with PNG image
    """
    renderer = ECG12LeadPDFRenderer(sampling_rate)
    return renderer.get_12lead_image_buffer(leads_dict)
