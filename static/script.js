document.addEventListener('DOMContentLoaded', () => {
  // Theme Toggle
  const themeToggle = document.getElementById('themeToggle');
  
  // Initialize theme from localStorage
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  if (savedTheme === 'light') {
    themeToggle.checked = true;
  }
  
  themeToggle.addEventListener('change', () => {
    const newTheme = themeToggle.checked ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  });
  
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
    
    metricHR: document.getElementById('metricHR'),
    statusHR: document.getElementById('statusHR'),
    metricPR: document.getElementById('metricPR'),
    statusPR: document.getElementById('statusPR'),
    metricQRS: document.getElementById('metricQRS'),
    statusQRS: document.getElementById('statusQRS'),
    metricQT: document.getElementById('metricQT'),
    statusQT: document.getElementById('statusQT'),
    
    binaryLabel: document.getElementById('binaryLabel'),
    multiclassLabel: document.getElementById('multiclassLabel'),
    
    severityStatus: document.getElementById('severityStatus'),
    severityAction: document.getElementById('severityAction'),
    severityFollowup: document.getElementById('severityFollowup')
  };

  let currentFile = null;
  let currentReportData = null;
  let patientInfo = {
    name: '',
    id: '',
    age: '',
    sex: '',
    doctor: '',
    indication: 'Routine Checkup'
  };

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
      const fallbackSignal = Array.from({length: 300}, (_, i) => Math.sin(i * 0.1) * 0.5);
      renderECGCanvas(fallbackSignal, elements.ecgCanvas);
    }
    
    // Display 12-lead ECG
    if (data.leads_12) {
      render12LeadECG(data.leads_12);
    }
    
    // Display advanced visualizations
    if (data.waveforms) displayWaveforms(data.waveforms);
    if (data.attention) displayAttention(data.attention, data.attention_regions);
    if (data.feature_importance) displayFeatureImportance(data.feature_importance);
  }

  // --- Canvas Rendering ---
  function renderECGCanvas(signal, canvasEl) {
    const ctx = canvasEl.getContext('2d');
    
    const rect = canvasEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = rect.width * dpr;
    canvasEl.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;

    ctx.fillStyle = '#0D0D0D';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const gridSize = 20;
    for(let x = 0; x <= width; x += gridSize) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
    for(let y = 0; y <= height; y += gridSize) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
    ctx.stroke();

    if (!signal || signal.length === 0) return;

    let min = Math.min(...signal);
    let max = Math.max(...signal);
    let range = max - min;
    if (range === 0) range = 1;

    const padding = height * 0.15;
    const drawHeight = height - (padding * 2);

    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    const step = width / (signal.length - 1 || 1);
    
    for(let i = 0; i < signal.length; i++) {
      const normalized = (signal[i] - min) / range;
      const x = i * step;
      const y = height - padding - (normalized * drawHeight);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    
    ctx.stroke();
  }

  function render12LeadECG(leads) {
    // Standard 12-lead canvas IDs
    const leadIds = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
    
    leadIds.forEach(leadName => {
      const canvasId = 'lead' + (leadName === 'aVR' ? 'AVR' : leadName === 'aVL' ? 'AVL' : leadName === 'aVF' ? 'AVF' : leadName);
      const canvasEl = document.getElementById(canvasId);
      
      if (canvasEl && leads[leadName]) {
        renderLeadCanvas(leads[leadName], canvasEl);
      }
    });
  }

  function renderLeadCanvas(signal, canvasEl) {
    const ctx = canvasEl.getContext('2d');
    
    const rect = canvasEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = rect.width * dpr;
    canvasEl.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;

    // Background with grid
    ctx.fillStyle = '#0D0D0D';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255,255,255,0.03)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    const gridSize = 10;
    for(let x = 0; x <= width; x += gridSize) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
    for(let y = 0; y <= height; y += gridSize) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
    ctx.stroke();

    if (!signal || signal.length === 0) return;

    let min = Math.min(...signal);
    let max = Math.max(...signal);
    let range = max - min;
    if (range === 0) range = 1;

    const padding = height * 0.1;
    const drawHeight = height - (padding * 2);

    // Draw lead signal
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 0.8;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    const step = width / (signal.length - 1 || 1);
    
    for(let i = 0; i < signal.length; i++) {
      const normalized = (signal[i] - min) / range;
      const x = i * step;
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
  });

  elements.downloadBtn.addEventListener('click', () => {
    if (!currentReportData) return;
    document.getElementById('patientModal').classList.remove('hidden');
  });

  // Modal controls
  const modal = document.getElementById('patientModal');
  const modalBackdrop = document.getElementById('modalBackdrop');
  const modalClose = document.getElementById('modalClose');
  const modalCancel = document.getElementById('modalCancel');
  const modalConfirm = document.getElementById('modalConfirm');

  function closeModal() {
    modal.classList.add('hidden');
  }

  modalClose.addEventListener('click', closeModal);
  modalCancel.addEventListener('click', closeModal);
  modalBackdrop.addEventListener('click', closeModal);

  modalConfirm.addEventListener('click', async () => {
    // Validate patient info
    patientInfo.name = document.getElementById('patientName').value.trim();
    patientInfo.age = document.getElementById('patientAge').value.trim();
    patientInfo.doctor = document.getElementById('doctorName').value.trim();
    patientInfo.id = document.getElementById('patientID').value.trim() || 'N/A';
    patientInfo.sex = document.getElementById('patientSex').value || '';
    patientInfo.indication = document.getElementById('indication').value.trim() || 'Routine Checkup';

    if (!patientInfo.name) {
      showError('Patient name is required');
      return;
    }
    if (!patientInfo.age) {
      showError('Patient age is required');
      return;
    }
    if (!patientInfo.doctor) {
      showError('Doctor name is required');
      return;
    }
    
    try {
      modalConfirm.disabled = true;
      modalConfirm.textContent = 'Generating PDF...';
      
      const response = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          signal: currentReportData.signal_preview || [],
          patient_id: patientInfo.id,
          patient_name: patientInfo.name,
          patient_age: patientInfo.age,
          patient_sex: patientInfo.sex,
          doctor_name: patientInfo.doctor,
          indication: patientInfo.indication,
          ecg_metrics: currentReportData.ecg_metrics || {},
          classification: currentReportData.multiclass_classification || {},
          waveforms: currentReportData.waveforms || {},
          attention: currentReportData.attention
        })
      });
      
      if (!response.ok) throw new Error('Report generation failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ECG_Report_${patientInfo.name}_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      closeModal();
      modalConfirm.textContent = 'Generate Report';
    } catch (error) {
      showError('Failed to generate PDF: ' + error.message);
      modalConfirm.textContent = 'Generate Report';
    } finally {
      modalConfirm.disabled = false;
    }
  });

  // --- Advanced Visualization Functions ---
  
  function displayWaveforms(waveforms) {
    if (!waveforms || Object.keys(waveforms).length === 0) return;
    
    const panel = document.getElementById('waveformsPanel');
    if (!panel) return;
    
    const qrsCount = (waveforms.qrs || []).length;
    const pCount = (waveforms.p_wave || []).length;
    const tCount = (waveforms.t_wave || []).length;
    
    document.getElementById('qrsCount').textContent = qrsCount;
    document.getElementById('pCount').textContent = pCount;
    document.getElementById('tCount').textContent = tCount;
    
    if (qrsCount > 0 || pCount > 0 || tCount > 0) {
      panel.classList.remove('hidden');
    }
  }
  
  function displayAttention(attentionData, topRegions) {
    if (!attentionData) return;
    
    const panel = document.getElementById('attentionPanel');
    if (!panel) return;
    
    // Show the panel regardless
    panel.classList.remove('hidden');
    
    // Get diagnosis to provide context-specific description
    const diagnosis = document.getElementById('diagnosisClass')?.textContent || '';
    const subtext = document.getElementById('attentionSubtext');
    
    if (subtext) {
      let description = '';
      if (diagnosis.includes('AV Block') || diagnosis.includes('CHB')) {
        description = 'Model focused on PR intervals and P-QRS relationships to detect atrioventricular conduction delays and dropped beats';
      } else if (diagnosis.includes('LBBB')) {
        description = 'Model concentrated on QRS complex morphology in lateral leads (V5-V6, I, aVL) to identify left bundle branch delay patterns';
      } else if (diagnosis.includes('RBBB')) {
        description = 'Model examined QRS patterns in right precordial leads (V1-V2) looking for rsR\' configuration indicating right bundle delay';
      } else if (diagnosis.includes('PAC')) {
        description = 'Model attended to P wave timing and morphology to detect premature atrial contractions occurring earlier than expected';
      } else if (diagnosis.includes('PVC')) {
        description = 'Model identified wide QRS complexes without preceding P waves and subsequent compensatory pauses characteristic of PVCs';
      } else if (diagnosis.includes('STD')) {
        description = 'Model analyzed ST segment depressions across multiple leads to assess for myocardial ischemia or subendocardial injury';
      } else if (diagnosis.includes('STE')) {
        description = 'Model detected ST segment elevations indicating acute transmural myocardial injury, typically seen in acute MI';
      } else {
        description = 'Highlights which temporal segments of the ECG waveform received the highest neural attention during classification';
      }
      subtext.textContent = description;
    }
    
    const canvas = document.getElementById('attentionCanvas');
    if (canvas) renderAttentionHeatmap(attentionData, canvas);
  }
  
  function renderAttentionHeatmap(attention, canvasEl) {
    const ctx = canvasEl.getContext('2d');
    const rect = canvasEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    canvasEl.width = rect.width * dpr;
    canvasEl.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    
    ctx.fillStyle = '#0D0D0D';
    ctx.fillRect(0, 0, width, height);
    
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const gridSize = 20;
    for(let x = 0; x <= width; x += gridSize) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
    for(let y = 0; y <= height; y += gridSize) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
    ctx.stroke();
    
    if (!attention || attention.length === 0) return;
    
    const barWidth = width / attention.length;
    const maxVal = Math.max(...attention);
    const minVal = Math.min(...attention);
    const range = maxVal - minVal || 1;
    
    for (let i = 0; i < attention.length; i++) {
      const val = attention[i];
      const normalized = (val - minVal) / range;
      const barHeight = normalized * height;
      
      let hue, saturation;
      if (normalized < 0.25) {
        hue = 240;
        saturation = 50 + (normalized / 0.25) * 50;
      } else if (normalized < 0.5) {
        hue = 180 - ((normalized - 0.25) / 0.25) * 60;
        saturation = 100;
      } else if (normalized < 0.75) {
        hue = 120 - ((normalized - 0.5) / 0.25) * 60;
        saturation = 100;
      } else {
        hue = 60 - ((normalized - 0.75) / 0.25) * 60;
        saturation = 100;
      }
      
      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${40 + normalized * 20}%)`;
      ctx.fillRect(i * barWidth, height - barHeight, barWidth, barHeight);
    }
    
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);
  }
  
  function displayFeatureImportance(importance) {
    if (!importance) return;
    
    const panel = document.getElementById('importancePanel');
    if (!panel) return;
    
    panel.classList.remove('hidden');
    
    // Get diagnosis to provide context-specific description
    const diagnosis = document.getElementById('diagnosisClass')?.textContent || '';
    const subtext = document.getElementById('importanceSubtext');
    
    if (subtext) {
      let description = '';
      if (diagnosis.includes('AV Block') || diagnosis.includes('CHB')) {
        description = 'Shows which parts of the ECG contributed to detecting AV conduction delay patterns (PR prolongation, dropped beats)';
      } else if (diagnosis.includes('LBBB')) {
        description = 'Highlights QRS complex regions that revealed left bundle branch delay patterns (wide QRS, notched R waves)';
      } else if (diagnosis.includes('RBBB')) {
        description = 'Identifies QRS features indicating right bundle branch delay (rsR\' pattern in V1, wide S waves in lateral leads)';
      } else if (diagnosis.includes('PAC')) {
        description = 'Points to premature P wave morphology and timing that indicated early atrial contractions';
      } else if (diagnosis.includes('PVC')) {
        description = 'Reveals wide QRS complexes and compensatory pauses that signaled premature ventricular contractions';
      } else if (diagnosis.includes('STD')) {
        description = 'Shows ST segment regions where depression patterns suggested myocardial ischemia or strain';
      } else if (diagnosis.includes('STE')) {
        description = 'Highlights ST segment elevation zones indicating acute myocardial injury or infarction';
      } else {
        description = 'Indicates which temporal regions of the ECG signal most strongly influenced the model\'s diagnostic decision';
      }
      subtext.textContent = description;
    }
    
    const canvas = document.getElementById('importanceCanvas');
    if (canvas) renderFeatureImportance(importance, canvas);
  }
  
  function renderFeatureImportance(importance, canvasEl) {
    const ctx = canvasEl.getContext('2d');
    const rect = canvasEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    canvasEl.width = rect.width * dpr;
    canvasEl.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    
    ctx.fillStyle = '#0D0D0D';
    ctx.fillRect(0, 0, width, height);
    
    if (!importance || importance.length === 0) return;
    
    const lineWidth = width / importance.length;
    const maxVal = Math.max(...importance);
    
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < importance.length; i++) {
      const normalized = importance[i] / maxVal;
      const y = height - (normalized * height * 0.8);
      
      if (i === 0) ctx.moveTo(i * lineWidth, y);
      else ctx.lineTo(i * lineWidth, y);
    }
    
    ctx.stroke();
  }
  
});
