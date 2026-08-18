document.addEventListener('DOMContentLoaded', () => {
  const elements = {
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    dropLink: document.getElementById('dropLink'),
    uploadPanel: document.getElementById('uploadPanel'),
    filePreview: document.getElementById('filePreview'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    resultsPanel: document.getElementById('resultsPanel'),
    errorToast: document.getElementById('errorToast'),
    errorText: document.getElementById('errorText'),
    toastClose: document.getElementById('toastClose'),
    newBtn: document.getElementById('newBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    ecgCanvas: document.getElementById('ecgCanvas'),
    
    diagnosisClass: document.getElementById('diagnosisClass'),
    diagnosisBadge: document.getElementById('diagnosisBadge'),
    diagnosisConfidence: document.getElementById('diagnosisConfidence'),
    
    metricHR: document.getElementById('metricHR'),
    statusHR: document.getElementById('statusHR'),
    metricPR: document.getElementById('metricPR'),
    statusPR: document.getElementById('statusPR'),
    metricQRS: document.getElementById('metricQRS'),
    statusQRS: document.getElementById('statusQRS'),
    metricQT: document.getElementById('metricQT'),
    statusQT: document.getElementById('statusQT'),
    
    binaryLabel: document.getElementById('binaryLabel'),
    binaryConfFill: document.getElementById('binaryConfFill'),
    binaryConf: document.getElementById('binaryConf'),
    
    multiclassLabel: document.getElementById('multiclassLabel'),
    multiclassConfFill: document.getElementById('multiclassConfFill'),
    multiclassConf: document.getElementById('multiclassConf'),
    
    probList: document.getElementById('probList'),
    
    severityStatus: document.getElementById('severityStatus'),
    severityAction: document.getElementById('severityAction'),
    severityFollowup: document.getElementById('severityFollowup')
  };

  let currentFile = null;
  let currentReportData = null;

  // --- File Upload ---
  elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('drag-active');
  });

  elements.dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-active');
  });

  elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-active');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  elements.dropLink.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    elements.fileInput.click();
  });

  elements.dropZone.addEventListener('click', (e) => {
    // Don't trigger if clicking on the button itself (button has its own listener)
    if (e.target !== elements.dropLink && !e.target.closest('button')) {
      elements.fileInput.click();
    }
  });

  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    const validExts = ['.npz', '.npy', '.csv', '.dat', '.txt', '.mat', '.xlsx'];
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    
    if (!validExts.includes(ext)) {
      showError('Unsupported file format. Please upload a valid ECG file.');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      showError('File size exceeds 50MB limit.');
      return;
    }

    currentFile = file;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatBytes(file.size);
    elements.filePreview.classList.remove('hidden');

    uploadFile(file);
  }

  function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  // --- API Call ---
  async function uploadFile(file) {
    elements.uploadPanel.classList.add('hidden');
    elements.loadingOverlay.classList.remove('hidden');
    elements.resultsPanel.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const data = await response.json();
      currentReportData = data;
      displayResults(data);
    } catch (err) {
      elements.loadingOverlay.classList.add('hidden');
      elements.uploadPanel.classList.remove('hidden');
      showError(err.message || 'An error occurred during analysis.');
    }
  }

  // --- Results Display ---
  function displayResults(data) {
    elements.loadingOverlay.classList.add('hidden');
    elements.resultsPanel.classList.remove('hidden');

    const multi = data.multiclass_classification;
    const binary = data.binary_classification;
    const metrics = data.ecg_metrics;
    const status = data.metrics_status;
    const rec = data.recommendation;
    
    // Diagnosis Banner
    elements.diagnosisClass.textContent = multi.class_name;
    
    elements.diagnosisBadge.textContent = rec.status.toUpperCase();
    elements.diagnosisBadge.className = 'diagnosis-badge';
    if (rec.status.toLowerCase() === 'normal') elements.diagnosisBadge.classList.add('badge-normal');
    else if (rec.status.toLowerCase().includes('critical')) elements.diagnosisBadge.classList.add('badge-critical');
    else elements.diagnosisBadge.classList.add('badge-warning');

    // Metrics
    animateValue(elements.metricHR, 0, metrics.heart_rate, 600, false);
    elements.statusHR.textContent = status.heart_rate;
    
    animateValue(elements.metricPR, 0, metrics.pr_interval, 600, false);
    elements.statusPR.textContent = status.pr_interval;
    
    animateValue(elements.metricQRS, 0, metrics.qrs_duration, 600, false);
    elements.statusQRS.textContent = status.qrs_duration;
    
    animateValue(elements.metricQT, 0, metrics.qt_interval, 600, false);
    elements.statusQT.textContent = status.qt_interval;

    // Classification Blocks
    if (binary && binary.class_name) {
      elements.binaryLabel.textContent = binary.class_name.toUpperCase();
    } else {
      elements.binaryLabel.textContent = '—';
    }

    elements.multiclassLabel.textContent = multi.class_name.toUpperCase();

    // Severity
    elements.severityStatus.textContent = rec.status;
    elements.severityAction.textContent = rec.action;
    elements.severityFollowup.textContent = rec.follow_up;

    // ECG Canvas
    if (data.signal_preview && data.signal_preview.length > 0) {
      renderECGCanvas(data.signal_preview, elements.ecgCanvas);
    } else if (data.signal && data.signal.length > 0) {
      renderECGCanvas(data.signal, elements.ecgCanvas);
    } else {
      // Fallback: render a minimal representation
      const fallbackSignal = Array.from({length: 300}, (_, i) => Math.sin(i * 0.1) * 0.5);
      renderECGCanvas(fallbackSignal, elements.ecgCanvas);
    }
  }

  // --- Canvas Rendering ---
  function renderECGCanvas(signal, canvasEl) {
    const ctx = canvasEl.getContext('2d');
    
    // Set logical size for high DPI
    const rect = canvasEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = rect.width * dpr;
    canvasEl.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;

    // Background
    ctx.fillStyle = '#0D0D0D';
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const gridSize = 20;
    for(let x = 0; x <= width; x += gridSize) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
    for(let y = 0; y <= height; y += gridSize) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
    ctx.stroke();

    // Signal
    if (!signal || signal.length === 0) return;

    let min = Math.min(...signal);
    let max = Math.max(...signal);
    let range = max - min;
    if (range === 0) range = 1;

    // Add padding
    const padding = height * 0.2;
    const drawHeight = height - (padding * 2);

    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    const step = width / (signal.length - 1);
    
    for(let i = 0; i < signal.length; i++) {
      const normalized = (signal[i] - min) / range;
      const x = i * step;
      // Invert Y axis for drawing
      const y = height - padding - (normalized * drawHeight);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    
    ctx.stroke();
  }

  // --- Animations ---
  function animateValue(obj, start, end, duration, isPercentage) {
    let startTimestamp = null;
    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      // easeOutQuart
      const ease = 1 - Math.pow(1 - progress, 4);
      const current = (start + (end - start) * ease).toFixed(1);
      
      obj.innerHTML = current + (isPercentage ? '%' : '');
      
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        obj.innerHTML = end.toFixed(1) + (isPercentage ? '%' : '');
      }
    };
    window.requestAnimationFrame(step);
  }

  // --- Error Handling ---
  function showError(msg) {
    elements.errorText.textContent = msg;
    elements.errorToast.classList.remove('hidden');
    setTimeout(() => {
      elements.errorToast.classList.add('hidden');
    }, 5000);
  }

  elements.toastClose.addEventListener('click', () => {
    elements.errorToast.classList.add('hidden');
  });

  // --- Actions ---
  elements.newBtn.addEventListener('click', () => {
    currentFile = null;
    currentReportData = null;
    elements.fileInput.value = '';
    elements.filePreview.classList.add('hidden');
    elements.resultsPanel.classList.add('hidden');
    elements.uploadPanel.classList.remove('hidden');
    
    // Reset bars
    elements.binaryConfFill.style.width = '0%';
    elements.multiclassConfFill.style.width = '0%';
  });

  elements.downloadBtn.addEventListener('click', () => {
    if (!currentReportData) return;
    
    const data = currentReportData;
    const reportText = `
CARDIOSCAN ECG ANALYSIS REPORT
Date: ${new Date().toLocaleString()}
File: ${currentFile ? currentFile.name : 'Unknown'}

--- DIAGNOSIS ---
Primary: ${data.multiclass_classification.class_name}
Screening: ${data.binary_classification.class_name}

--- METRICS ---
Heart Rate: ${data.ecg_metrics.heart_rate} bpm [${data.metrics_status.heart_rate}]
PR Interval: ${data.ecg_metrics.pr_interval} ms [${data.metrics_status.pr_interval}]
QRS Duration: ${data.ecg_metrics.qrs_duration} ms [${data.metrics_status.qrs_duration}]
QT Interval: ${data.ecg_metrics.qt_interval} ms [${data.metrics_status.qt_interval}]

--- CLINICAL ASSESSMENT ---
Status: ${data.recommendation.status}
Recommendation: ${data.recommendation.action}
Follow-up: ${data.recommendation.follow_up}

Generated by CardioScan System v2.0
`;

    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `CardioScan_Report_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
});
