"""
Clinical ECG Report Generation with PDF export.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER
import io
from datetime import datetime
import numpy as np
from PIL import Image as PILImage, ImageDraw
import logging

logger = logging.getLogger(__name__)

class ECGReportGenerator:
    """Generate clinical ECG reports in PDF format."""
    
    def __init__(self, filename='ecg_report.pdf'):
        self.filename = filename
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2e5c8a'),
            spaceAfter=10,
            spaceBefore=10
        ))
        
        self.styles['Normal'].fontSize = 10
        self.styles['Normal'].spaceAfter = 8
    
    def create_graph_paper_ecg(self, signal, attention_weights=None, waveforms=None, title="ECG Signal"):
        """Create ECG plot on graph paper."""
        width, height = 1100, 300
        img = PILImage.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw grid
        for x in range(0, width, 20):
            draw.line([(x, 0), (x, height)], fill=(240, 240, 240))
        for y in range(0, height, 20):
            draw.line([(0, y), (width, y)], fill=(240, 240, 240))
        
        # Draw major grid
        for x in range(0, width, 100):
            draw.line([(x, 0), (x, height)], fill=(200, 200, 200), width=1)
        for y in range(0, height, 100):
            draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)
        
        # Draw signal
        if len(signal) > 0:
            sig_min = np.min(signal)
            sig_max = np.max(signal)
            sig_range = sig_max - sig_min
            if sig_range < 1e-8:
                sig_range = 1
            
            padding_top = 30
            draw_height = height - (padding_top + 30)
            x_step = width / len(signal)
            
            for i in range(len(signal) - 1):
                normalized = (signal[i] - sig_min) / sig_range
                normalized_next = (signal[i+1] - sig_min) / sig_range
                
                x1 = int(i * x_step)
                y1 = int(padding_top + (1 - normalized) * draw_height)
                x2 = int((i+1) * x_step)
                y2 = int(padding_top + (1 - normalized_next) * draw_height)
                
                draw.line([(x1, y1), (x2, y2)], fill='blue', width=2)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    
    def generate(self, signal, binary_result, multiclass_result, 
                 ecg_metrics=None, attention_weights=None, waveforms=None, 
                 recommendation=None, patient_id='N/A', patient_name='Anonymous',
                 patient_age='N/A', patient_sex='N/A', doctor_name='N/A', 
                 indication='Routine Checkup', classification=None, leads_12=None):
        """
        Generate clinical ECG report PDF - complete format with 12-lead ECG.
        
        Args:
            signal: Single-lead signal
            binary_result: Binary classification result
            multiclass_result: Multi-class classification result
            ecg_metrics: ECG measurements dictionary
            attention_weights: Attention weights (optional)
            waveforms: Detected waveforms (optional)
            recommendation: Clinical recommendation
            patient_id: Patient ID
            patient_name: Patient name
            patient_age: Patient age
            patient_sex: Patient sex (M/F)
            doctor_name: Referring doctor
            indication: Reason for ECG
            classification: Classification result
            leads_12: Dictionary with 12 lead signals (optional)
        """
        
        doc = SimpleDocTemplate(self.filename, pagesize=letter, 
                               topMargin=0.3*inch, bottomMargin=0.3*inch,
                               leftMargin=0.3*inch, rightMargin=0.3*inch)
        story = []
        
        timestamp = datetime.now().strftime("%d-%b-%Y %H:%M")
        
        # TITLE
        story.append(Paragraph("<b>E C G &nbsp; R E P O R T</b>", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.1*inch))
        
        # PATIENT DETAILS
        story.append(Paragraph("<b>PATIENT DETAILS</b>", self.styles['SectionHeader']))
        
        patient_data = [
            ['Date', timestamp, 'Patient Name', patient_name],
            ['ID', patient_id, 'Age / Sex', f"{patient_age} / {patient_sex}"],
            ['Ref. Doctor', doctor_name, 'Indication', indication],
        ]
        
        patient_table = Table(patient_data, colWidths=[1.2*inch, 1.8*inch, 1.2*inch, 1.8*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.15*inch))
        
        # ECG MEASUREMENTS
        story.append(Paragraph("<b>ECG MEASUREMENTS</b>", self.styles['SectionHeader']))
        
        if ecg_metrics:
            measurements_data = [
                ['Measurement', 'Value', 'Measurement', 'Value'],
                ['Heart Rate', f"{ecg_metrics.get('heart_rate', 'N/A')} bpm", 'P Axis', f"{ecg_metrics.get('p_axis', 'N/A')}"],
                ['PR Interval', f"{ecg_metrics.get('pr_interval', 'N/A')} ms", 'QRS Axis', f"{ecg_metrics.get('qrs_axis', 'N/A')}"],
                ['QRS Duration', f"{ecg_metrics.get('qrs_duration', 'N/A')} ms", 'T Axis', f"{ecg_metrics.get('t_axis', 'N/A')}"],
                ['QT Interval', f"{ecg_metrics.get('qt_interval', 'N/A')} ms", 'RR Interval', f"{ecg_metrics.get('rr_interval', 'N/A')} ms"],
            ]
            
            measurements_table = Table(measurements_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            measurements_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
            ]))
            story.append(measurements_table)
        
        story.append(Spacer(1, 0.15*inch))
        
        # ECG SIGNAL
        story.append(Paragraph("<b>ECG WAVEFORM</b>", self.styles['SectionHeader']))
        try:
            ecg_image_buf = self.create_graph_paper_ecg(signal, attention_weights, waveforms)
            img = Image(ecg_image_buf, width=6.8*inch, height=2*inch)
            story.append(img)
        except Exception as e:
            logger.warning(f"Could not generate ECG image: {e}")
        
        story.append(Spacer(1, 0.15*inch))
        
        # 12-LEAD ECG (if available)
        if leads_12 and len(leads_12) == 12:
            story.append(Paragraph("<b>12-LEAD ECG</b>", self.styles['SectionHeader']))
            try:
                from ecg_pdf_12lead import generate_12lead_pdf_image
                leads_image_buf = generate_12lead_pdf_image(leads_12)
                img_12lead = Image(leads_image_buf, width=7.2*inch, height=5.2*inch)
                story.append(img_12lead)
            except Exception as e:
                logger.warning(f"Could not generate 12-lead ECG image: {e}")
        
        story.append(Spacer(1, 0.15*inch))
        
        # INTERPRETATION
        story.append(Paragraph("<b>INTERPRETATION & DIAGNOSIS</b>", self.styles['SectionHeader']))
        
        interp_text = ""
        if multiclass_result:
            interp_text += f"<b>Classification:</b> {multiclass_result.get('class_name', 'Normal')}<br/>"
        if recommendation:
            interp_text += f"<b>Status:</b> {recommendation.get('status', 'N/A')}<br/>"
            interp_text += f"<b>Recommendation:</b> {recommendation.get('action', 'N/A')}<br/>"
            interp_text += f"<b>Follow-up:</b> {recommendation.get('follow_up', 'N/A')}"
        
        story.append(Paragraph(interp_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # FOOTER SECTION
        story.append(Paragraph("<b>REPORT METADATA</b>", self.styles['SectionHeader']))
        
        footer_data = [
            ['Machine Interpretation', 'Paper Speed', 'Reviewed By'],
            ['(Automated Analysis)', '25 mm/s', '__________________'],
            ['(Not for diagnostic use)', '10 mm/mV', '(Signature)'],
            ['1mV = 10mm', 'Date: ________', '']
        ]
        
        footer_table = Table(footer_data, colWidths=[2.0*inch, 2.0*inch, 2.0*inch])
        footer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ]))
        story.append(footer_table)
        
        # Build PDF
        doc.build(story)
        logger.info(f"✓ Report generated: {self.filename}")
        
        return self.filename
