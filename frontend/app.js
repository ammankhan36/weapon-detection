/**
 * Shield AI - Weapon Detection & Surveillance Frontend Controller
 * Connects to FastAPI YOLOv5 backend or runs interactive simulation fallback.
 */

// Configuration
const CONFIG = {
  API_BASE: 'http://127.0.0.1:8000',
  ALERT_COOLDOWN_MS: 3000,
  WEBCAM_FPS_TARGET: 10, // Frame analysis rate for webcam
  SOUND_ENABLED: true,
};

// Global App State
const state = {
  backendConnected: false,
  activeTab: 'image', // 'image' | 'video' | 'webcam'
  confidence: 0.50,
  iou: 0.45,
  currentImageBlob: null,
  currentAnnotatedUrl: null,
  lastDetections: [],
  historyLogs: [],
  threatClassCounts: { Handgun: 0, Knife: 0, Rifle: 0, Sword: 0 },
  totalThreatsLogged: 0,
  lastAlertTimestamp: 0,
  audioContext: null,
  soundMuted: false,
  webcamStream: null,
  webcamInterval: null,
  isProcessingFrame: false,
  streamFps: 0,
  lastFrameTime: performance.now()
};

// DOM Elements
const DOM = {
  // Header & Status
  backendStatusCard: document.getElementById('backend-status-card'),
  backendStatusText: document.getElementById('backend-status-text'),
  btnReconnect: document.getElementById('btn-reconnect'),
  btnToggleSound: document.getElementById('btn-toggle-sound'),
  iconSoundOn: document.getElementById('icon-sound-on'),
  iconSoundOff: document.getElementById('icon-sound-off'),
  soundLabel: document.getElementById('sound-label'),
  offlineToast: document.getElementById('offline-toast'),
  btnRetryToast: document.getElementById('btn-retry-backend-toast'),

  // Alert Banner
  alertBanner: document.getElementById('alert-banner'),
  alertMessage: document.getElementById('alert-message'),
  alertTimestamp: document.getElementById('alert-timestamp'),
  btnDismissAlert: document.getElementById('btn-dismiss-alert'),

  // Stats
  statActiveCount: document.getElementById('stat-active-count'),
  statTotalLogged: document.getElementById('stat-total-logged'),
  statLatency: document.getElementById('stat-latency'),
  statFps: document.getElementById('stat-fps'),
  badgeTotalFound: document.getElementById('badge-total-found'),

  // Mode Tabs
  tabButtons: document.querySelectorAll('.tab-btn'),
  viewportModeTitle: document.getElementById('viewport-mode-title'),

  // Viewport Elements
  previewStage: document.getElementById('preview-stage'),
  scannerLine: document.getElementById('scanner-line'),
  dropzoneEmpty: document.getElementById('dropzone-empty'),
  imageDisplay: document.getElementById('image-display'),
  videoDisplay: document.getElementById('video-display'),
  webcamDisplay: document.getElementById('webcam-display'),
  overlayCanvas: document.getElementById('overlay-canvas'),
  previewLoader: document.getElementById('preview-loader'),
  loaderStatusMsg: document.getElementById('loader-status-msg'),

  // Input & Buttons
  fileInputImage: document.getElementById('file-input-image'),
  fileInputVideo: document.getElementById('file-input-video'),
  videoInputLabel: document.getElementById('video-input-label'),
  btnLoadSample: document.getElementById('btn-load-sample'),
  btnDownloadResult: document.getElementById('btn-download-result'),
  btnReDetect: document.getElementById('btn-re-detect'),
  btnClearPreview: document.getElementById('btn-clear-preview'),

  // Webcam Controls
  webcamControls: document.getElementById('webcam-controls'),
  btnStartWebcam: document.getElementById('btn-start-webcam'),
  btnStopWebcam: document.getElementById('btn-stop-webcam'),
  btnSnapshotWebcam: document.getElementById('btn-snapshot-webcam'),

  // Video Controls
  videoControls: document.getElementById('video-controls'),
  btnProcessVideo: document.getElementById('btn-process-video'),
  imageControls: document.getElementById('image-controls'),

  // Sliders
  confSlider: document.getElementById('conf-slider'),
  confVal: document.getElementById('conf-val'),
  iouSlider: document.getElementById('iou-slider'),
  iouVal: document.getElementById('iou-val'),

  // Detections & Matrix
  detectionsList: document.getElementById('detections-list'),
  btnExportJson: document.getElementById('btn-export-json'),
  countHandgun: document.getElementById('count-handgun'),
  countKnife: document.getElementById('count-knife'),
  countRifle: document.getElementById('count-rifle'),
  countSword: document.getElementById('count-sword'),

  // History & Modal
  historyLogList: document.getElementById('history-log-list'),
  historyEmpty: document.getElementById('history-empty'),
  btnClearHistory: document.getElementById('btn-clear-history'),
  snapshotModal: document.getElementById('snapshot-modal'),
  modalImage: document.getElementById('modal-image'),
  modalDetails: document.getElementById('modal-details'),
  btnModalDownload: document.getElementById('btn-modal-download'),
  btnModalDismiss: document.getElementById('btn-modal-dismiss'),
  btnCloseModal: document.getElementById('btn-close-modal')
};

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  checkBackendHealth();
  // Poll backend status periodically
  setInterval(checkBackendHealth, 15000);
});

function setupEventListeners() {
  // Tabs
  DOM.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Sound Toggle
  DOM.btnToggleSound.addEventListener('click', toggleSound);

  // Reconnect buttons
  DOM.btnReconnect.addEventListener('click', checkBackendHealth);
  DOM.btnRetryToast.addEventListener('click', checkBackendHealth);

  // Sliders
  DOM.confSlider.addEventListener('input', (e) => {
    state.confidence = parseFloat(e.target.value);
    DOM.confVal.textContent = `${Math.round(state.confidence * 100)}%`;
  });
  DOM.confSlider.addEventListener('change', () => {
    if (state.activeTab === 'image' && state.currentImageBlob) {
      runImageInference(state.currentImageBlob);
    }
  });

  DOM.iouSlider.addEventListener('input', (e) => {
    state.iou = parseFloat(e.target.value);
    DOM.iouVal.textContent = `${Math.round(state.iou * 100)}%`;
  });
  DOM.iouSlider.addEventListener('change', () => {
    if (state.activeTab === 'image' && state.currentImageBlob) {
      runImageInference(state.currentImageBlob);
    }
  });

  // File Inputs & Dropzone
  DOM.fileInputImage.addEventListener('change', handleImageSelect);
  DOM.fileInputVideo.addEventListener('change', handleVideoSelect);
  setupDropzone(DOM.previewStage);

  // Action Buttons
  DOM.btnLoadSample.addEventListener('click', loadSampleImage);
  DOM.btnReDetect.addEventListener('click', () => {
    if (state.currentImageBlob) runImageInference(state.currentImageBlob);
  });
  DOM.btnClearPreview.addEventListener('click', clearPreview);
  DOM.btnDownloadResult.addEventListener('click', downloadCurrentResult);
  DOM.btnExportJson.addEventListener('click', exportDetectionsJson);

  // Webcam Actions
  DOM.btnStartWebcam.addEventListener('click', startWebcam);
  DOM.btnStopWebcam.addEventListener('click', stopWebcam);
  DOM.btnSnapshotWebcam.addEventListener('click', snapWebcamPhoto);

  // Video Actions
  DOM.btnProcessVideo.addEventListener('click', toggleVideoProcessing);

  // Alert Dismiss
  DOM.btnDismissAlert.addEventListener('click', () => {
    DOM.alertBanner.classList.add('hidden');
  });

  // History & Modal
  DOM.btnClearHistory.addEventListener('click', clearHistory);
  DOM.btnCloseModal.addEventListener('click', closeModal);
  DOM.btnModalDismiss.addEventListener('click', closeModal);
  DOM.snapshotModal.querySelector('.modal-backdrop').addEventListener('click', closeModal);
}

// =============================================================================
// BACKEND STATUS & SIMULATION FALLBACK
// =============================================================================
async function checkBackendHealth() {
  DOM.backendStatusCard.className = 'status-indicator-badge checking';
  DOM.backendStatusText.textContent = 'Checking Model...';

  try {
    const res = await fetch(`${CONFIG.API_BASE}/api/status`, { method: 'GET', signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      state.backendConnected = true;
      DOM.backendStatusCard.className = 'status-indicator-badge online';
      DOM.backendStatusText.textContent = `Model Ready (${data.device.toUpperCase()})`;
      DOM.offlineToast.classList.add('hidden');
      return true;
    }
  } catch (err) {
    // Backend offline
  }

  state.backendConnected = false;
  DOM.backendStatusCard.className = 'status-indicator-badge offline';
  DOM.backendStatusText.textContent = 'Backend Offline';
  DOM.offlineToast.classList.remove('hidden');
  return false;
}

// =============================================================================
// AUDIO ALERT ENGINE (Web Audio API Synthesizer)
// =============================================================================
function playTacticalAlertSound() {
  if (state.soundMuted) return;

  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!state.audioContext) {
      state.audioContext = new AudioContext();
    }
    const ctx = state.audioContext;
    if (ctx.state === 'suspended') {
      ctx.resume();
    }

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sawtooth';
    // Rapid two-tone alert chirp
    osc.frequency.setValueAtTime(880, now);
    osc.frequency.setValueAtTime(1400, now + 0.1);
    osc.frequency.setValueAtTime(880, now + 0.2);
    osc.frequency.setValueAtTime(1400, now + 0.3);

    gain.gain.setValueAtTime(0.15, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.45);
  } catch (e) {
    console.warn('Audio synthesis error:', e);
  }
}

function toggleSound() {
  state.soundMuted = !state.soundMuted;
  if (state.soundMuted) {
    DOM.iconSoundOn.classList.add('hidden');
    DOM.iconSoundOff.classList.remove('hidden');
    DOM.soundLabel.textContent = 'Audio Alert: OFF';
    DOM.btnToggleSound.classList.add('muted');
  } else {
    DOM.iconSoundOn.classList.remove('hidden');
    DOM.iconSoundOff.classList.add('hidden');
    DOM.soundLabel.textContent = 'Audio Alert: ON';
    DOM.btnToggleSound.classList.remove('muted');
    playTacticalAlertSound(); // brief test chirp
  }
}

// =============================================================================
// TAB SWITCHING
// =============================================================================
function switchTab(targetTab) {
  state.activeTab = targetTab;

  DOM.tabButtons.forEach(btn => {
    const isTarget = btn.dataset.tab === targetTab;
    btn.classList.toggle('active', isTarget);
    btn.setAttribute('aria-selected', isTarget);
  });

  // Stop webcam if leaving webcam tab
  if (targetTab !== 'webcam' && state.webcamStream) {
    stopWebcam();
  }

  // Adjust toolbar groups
  DOM.imageControls.classList.toggle('hidden', targetTab !== 'image');
  DOM.videoControls.classList.toggle('hidden', targetTab !== 'video');
  DOM.webcamControls.classList.toggle('hidden', targetTab !== 'webcam');
  DOM.videoInputLabel.classList.toggle('hidden', targetTab !== 'video');

  // Update title
  if (targetTab === 'image') {
    DOM.viewportModeTitle.textContent = 'IMAGE INFERENCE VIEWPORT';
    clearPreview();
  } else if (targetTab === 'video') {
    DOM.viewportModeTitle.textContent = 'VIDEO SURVEILLANCE SCANNER';
    clearPreview();
  } else if (targetTab === 'webcam') {
    DOM.viewportModeTitle.textContent = 'LIVE SENSOR / WEBCAM FEED';
    DOM.dropzoneEmpty.classList.add('hidden');
    DOM.imageDisplay.classList.add('hidden');
    DOM.videoDisplay.classList.add('hidden');
    DOM.webcamDisplay.classList.remove('hidden');
  }
}

// =============================================================================
// DROPZONE & FILE INPUT HANDLING
// =============================================================================
function setupDropzone(el) {
  ['dragenter', 'dragover'].forEach(name => {
    el.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      DOM.dropzoneEmpty.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    el.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      DOM.dropzoneEmpty.classList.remove('dragover');
    });
  });

  el.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith('image/')) {
        switchTab('image');
        loadImageFile(file);
      } else if (file.type.startsWith('video/')) {
        switchTab('video');
        loadVideoFile(file);
      }
    }
  });
}

function handleImageSelect(e) {
  if (e.target.files && e.target.files[0]) {
    loadImageFile(e.target.files[0]);
  }
}

function handleVideoSelect(e) {
  if (e.target.files && e.target.files[0]) {
    loadVideoFile(e.target.files[0]);
  }
}

function loadImageFile(file) {
  state.currentImageBlob = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    DOM.imageDisplay.src = e.target.result;
    DOM.imageDisplay.classList.remove('hidden');
    DOM.dropzoneEmpty.classList.add('hidden');
    DOM.videoDisplay.classList.add('hidden');
    DOM.webcamDisplay.classList.add('hidden');
    DOM.overlayCanvas.classList.add('hidden');
    DOM.btnReDetect.disabled = false;
    DOM.btnClearPreview.disabled = false;

    // Run detection
    runImageInference(file);
  };
  reader.readAsDataURL(file);
}

// Quick Sample Weapon Loader
async function loadSampleImage() {
  switchTab('image');
  showLoader(true, 'FETCHING SAMPLE WEAPON...');

  try {
    let blob;
    if (state.backendConnected) {
      const res = await fetch(`${CONFIG.API_BASE}/api/sample`);
      if (res.ok) {
        blob = await res.blob();
      }
    }

    if (!blob) {
      // Fallback: create high-res procedural handgun test canvas
      blob = await generateFallbackWeaponBlob();
    }

    loadImageFile(blob);
  } catch (err) {
    const fallbackBlob = await generateFallbackWeaponBlob();
    loadImageFile(fallbackBlob);
  } finally {
    showLoader(false);
  }
}

// Procedural fallback test graphic
function generateFallbackWeaponBlob() {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 420;
    const ctx = canvas.getContext('2d');

    // Background gradient
    const grad = ctx.createLinearGradient(0, 0, 640, 420);
    grad.addColorStop(0, '#1a1e29');
    grad.addColorStop(1, '#0b0e14');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 640, 420);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    for (let x = 0; x < 640; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, 420); ctx.stroke();
    }
    for (let y = 0; y < 420; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(640, y); ctx.stroke();
    }

    // Gun silhouette
    ctx.fillStyle = '#2b394f';
    ctx.fillRect(160, 160, 260, 50); // barrel/slide
    ctx.fillRect(160, 200, 70, 130); // grip
    ctx.fillRect(230, 210, 40, 30);  // trigger guard

    ctx.fillStyle = '#60a5fa';
    ctx.font = 'bold 20px Outfit, sans-serif';
    ctx.fillText('SAMPLE WEAPON TARGET (SIMULATED)', 140, 100);

    canvas.toBlob((b) => resolve(b), 'image/jpeg');
  });
}

// =============================================================================
// DETECTION LOGIC & API INFERENCE
// =============================================================================
async function runImageInference(fileBlob) {
  showLoader(true, 'RUNNING YOLOv5 INFERENCE...');
  DOM.scannerLine.classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', fileBlob, 'upload.jpg');
  formData.append('conf', state.confidence);
  formData.append('iou', state.iou);

  try {
    if (state.backendConnected) {
      const res = await fetch(`${CONFIG.API_BASE}/api/detect/image`, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const result = await res.json();
        handleInferenceSuccess(result);
        return;
      }
    }
  } catch (err) {
    console.warn('Backend inference failed, using simulation mode:', err);
  }

  // Simulation fallback if backend is offline
  setTimeout(() => {
    const simulated = simulateDetectionResult(DOM.imageDisplay);
    handleInferenceSuccess(simulated);
  }, 400);
}

function handleInferenceSuccess(result) {
  showLoader(false);
  DOM.scannerLine.classList.add('hidden');

  state.lastDetections = result.detections;
  state.currentAnnotatedUrl = result.annotated_image;
  DOM.imageDisplay.src = result.annotated_image;
  DOM.btnDownloadResult.disabled = false;

  // Latency
  DOM.statLatency.textContent = `${result.inference_time_ms} ms`;

  // Render Detections
  renderDetections(result.detections);

  // Trigger Alert if weapon detected
  if (result.detections.length > 0) {
    triggerWeaponAlert(result.detections, result.annotated_image, result.timestamp);
  } else {
    DOM.alertBanner.classList.add('hidden');
  }
}

// Simulation logic when server is offline
function simulateDetectionResult(imgEl) {
  const isHighConf = state.confidence <= 0.85;
  const detections = isHighConf ? [
    {
      class: 'Handgun',
      confidence: 0.92,
      percentage: 92.0,
      box: [120, 120, 480, 360],
      color: '#ef4444'
    }
  ] : [];

  // Generate annotated canvas
  const canvas = document.createElement('canvas');
  canvas.width = imgEl.naturalWidth || 640;
  canvas.height = imgEl.naturalHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);

  if (detections.length > 0) {
    const d = detections[0];
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    ctx.fillStyle = '#ef4444';
    ctx.fillRect(x1, y1 - 28, 140, 28);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Outfit, sans-serif';
    ctx.fillText(`${d.class} ${d.percentage}%`, x1 + 6, y1 - 8);
  }

  return {
    detections: detections,
    total_found: detections.length,
    annotated_image: canvas.toDataURL('image/jpeg', 0.85),
    inference_time_ms: 38.4,
    timestamp: new Date().toLocaleTimeString()
  };
}

// =============================================================================
// ALERT NOTIFICATIONS & AUDIT TIMELINE
// =============================================================================
function triggerWeaponAlert(detections, snapshotUrl, timestamp) {
  const highest = detections.reduce((prev, curr) => (prev.confidence > curr.confidence) ? prev : curr, detections[0]);
  const alertText = `WEAPON DETECTED: ${highest.class} (${highest.percentage}%)`;

  DOM.alertMessage.textContent = alertText;
  DOM.alertTimestamp.textContent = timestamp || new Date().toLocaleTimeString();
  DOM.alertBanner.classList.remove('hidden');

  // Cooldown sound check
  const now = Date.now();
  if (now - state.lastAlertTimestamp > CONFIG.ALERT_COOLDOWN_MS) {
    playTacticalAlertSound();
    state.lastAlertTimestamp = now;
  }

  // Record in History Timeline
  recordHistoryItem({
    id: `alert-${Date.now()}`,
    class: highest.class,
    confidence: highest.percentage,
    timestamp: DOM.alertTimestamp.textContent,
    snapshotUrl: snapshotUrl,
    detectionsCount: detections.length
  });
}

function recordHistoryItem(item) {
  state.historyLogs.unshift(item);
  state.totalThreatsLogged++;
  DOM.statTotalLogged.textContent = state.totalThreatsLogged;
  DOM.historyEmpty.classList.add('hidden');

  const card = document.createElement('div');
  card.className = 'history-card';
  card.innerHTML = `
    <img src="${item.snapshotUrl}" class="history-thumb" alt="${item.class}">
    <div class="history-card-details">
      <span class="history-card-time">${item.timestamp}</span>
      <span class="history-card-weapon">${item.class}</span>
      <span class="history-card-conf">${item.confidence}% Match</span>
    </div>
  `;

  card.addEventListener('click', () => openModal(item));
  DOM.historyLogList.prepend(card);
}

function clearHistory() {
  state.historyLogs = [];
  DOM.historyLogList.innerHTML = '';
  DOM.historyLogList.appendChild(DOM.historyEmpty);
  DOM.historyEmpty.classList.remove('hidden');
}

// =============================================================================
// DETECTIONS & STATS RENDERING
// =============================================================================
function renderDetections(detections) {
  DOM.detectionsList.innerHTML = '';
  const total = detections.length;

  DOM.statActiveCount.textContent = total;
  DOM.badgeTotalFound.textContent = `${total} Found`;
  DOM.badgeTotalFound.className = total > 0 ? 'count-badge threat' : 'count-badge zero';

  // Reset matrix counters
  Object.keys(state.threatClassCounts).forEach(k => state.threatClassCounts[k] = 0);

  if (total === 0) {
    DOM.detectionsList.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <p>No weapons currently detected in viewport.</p>
        <span>Upload an image/video or activate webcam feed.</span>
      </div>
    `;
    updateMatrixUI();
    return;
  }

  detections.forEach(d => {
    if (state.threatClassCounts[d.class] !== undefined) {
      state.threatClassCounts[d.class]++;
    }

    const card = document.createElement('div');
    card.className = 'detection-card';
    card.innerHTML = `
      <div class="detection-info">
        <div class="detection-name-row">
          <span class="threat-dot" style="background:${d.color}"></span>
          <span class="detection-name">${d.class}</span>
        </div>
        <span class="detection-coords">BOX: [${d.box.map(n => Math.round(n)).join(', ')}]</span>
      </div>
      <div class="detection-confidence">
        <span class="confidence-number">${d.percentage}%</span>
        <div class="confidence-bar-track">
          <div class="confidence-bar-fill" style="width: ${d.percentage}%"></div>
        </div>
      </div>
    `;
    DOM.detectionsList.appendChild(card);
  });

  updateMatrixUI();
}

function updateMatrixUI() {
  DOM.countHandgun.textContent = state.threatClassCounts.Handgun;
  DOM.countKnife.textContent = state.threatClassCounts.Knife;
  DOM.countRifle.textContent = state.threatClassCounts.Rifle;
  DOM.countSword.textContent = state.threatClassCounts.Sword;
}

// =============================================================================
// WEBCAM LIVE STREAMING ENGINE
// =============================================================================
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });

    state.webcamStream = stream;
    DOM.webcamDisplay.srcObject = stream;
    DOM.webcamDisplay.classList.remove('hidden');
    DOM.dropzoneEmpty.classList.add('hidden');
    DOM.overlayCanvas.classList.remove('hidden');

    DOM.btnStartWebcam.classList.add('hidden');
    DOM.btnStopWebcam.classList.remove('hidden');
    DOM.btnSnapshotWebcam.classList.remove('hidden');

    // Start Real-Time Inference Loop
    DOM.webcamDisplay.onloadedmetadata = () => {
      DOM.overlayCanvas.width = DOM.webcamDisplay.videoWidth;
      DOM.overlayCanvas.height = DOM.webcamDisplay.videoHeight;
      startWebcamLoop();
    };
  } catch (err) {
    alert('Camera Access Denied or No Webcam Found: ' + err.message);
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(t => t.stop());
    state.webcamStream = null;
  }
  if (state.webcamInterval) {
    clearInterval(state.webcamInterval);
    state.webcamInterval = null;
  }

  DOM.webcamDisplay.srcObject = null;
  DOM.webcamDisplay.classList.add('hidden');
  DOM.overlayCanvas.classList.add('hidden');

  DOM.btnStartWebcam.classList.remove('hidden');
  DOM.btnStopWebcam.classList.add('hidden');
  DOM.btnSnapshotWebcam.classList.add('hidden');
  DOM.statFps.textContent = '0.0';
  renderDetections([]);
}

function startWebcamLoop() {
  const offscreenCanvas = document.createElement('canvas');
  const offCtx = offscreenCanvas.getContext('2d');

  state.webcamInterval = setInterval(async () => {
    if (state.isProcessingFrame || !state.webcamStream) return;
    state.isProcessingFrame = true;

    const w = DOM.webcamDisplay.videoWidth;
    const h = DOM.webcamDisplay.videoHeight;
    if (w === 0 || h === 0) {
      state.isProcessingFrame = false;
      return;
    }

    offscreenCanvas.width = w;
    offscreenCanvas.height = h;
    offCtx.drawImage(DOM.webcamDisplay, 0, 0, w, h);

    const frameBase64 = offscreenCanvas.toDataURL('image/jpeg', 0.7);

    // FPS Calculation
    const now = performance.now();
    const fps = 1000 / Math.max(now - state.lastFrameTime, 1);
    state.lastFrameTime = now;
    DOM.statFps.textContent = fps.toFixed(1);

    try {
      if (state.backendConnected) {
        const res = await fetch(`${CONFIG.API_BASE}/api/detect/frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: frameBase64,
            conf: state.confidence,
            iou: state.iou
          })
        });

        if (res.ok) {
          const data = await res.json();
          drawOverlayBoxes(data.detections, w, h);
          renderDetections(data.detections);
          DOM.statLatency.textContent = `${data.inference_time_ms} ms`;

          if (data.detections.length > 0) {
            triggerWeaponAlert(data.detections, data.annotated_image, data.timestamp);
          } else {
            DOM.alertBanner.classList.add('hidden');
          }
        }
      } else {
        // Simulated live tracking
        drawOverlayBoxes([], w, h);
      }
    } catch (e) {
      // frame dropped
    } finally {
      state.isProcessingFrame = false;
    }
  }, 1000 / CONFIG.WEBCAM_FPS_TARGET);
}

function drawOverlayBoxes(detections, w, h) {
  const canvas = DOM.overlayCanvas;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  detections.forEach(d => {
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = d.color || '#ef4444';
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Label banner
    ctx.fillStyle = d.color || '#ef4444';
    ctx.fillRect(x1, Math.max(0, y1 - 24), 130, 24);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 14px Outfit, sans-serif';
    ctx.fillText(`${d.class} ${d.percentage}%`, x1 + 6, Math.max(16, y1 - 6));
  });
}

function snapWebcamPhoto() {
  if (!state.webcamStream) return;
  const canvas = document.createElement('canvas');
  canvas.width = DOM.webcamDisplay.videoWidth;
  canvas.height = DOM.webcamDisplay.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(DOM.webcamDisplay, 0, 0);
  ctx.drawImage(DOM.overlayCanvas, 0, 0);

  const a = document.createElement('a');
  a.href = canvas.toDataURL('image/jpeg', 0.95);
  a.download = `threat_snap_${Date.now()}.jpg`;
  a.click();
}

// =============================================================================
// VIDEO SCANNER HANDLING
// =============================================================================
function loadVideoFile(file) {
  const url = URL.createObjectURL(file);
  DOM.videoDisplay.src = url;
  DOM.videoDisplay.classList.remove('hidden');
  DOM.dropzoneEmpty.classList.add('hidden');
  DOM.imageDisplay.classList.add('hidden');
  DOM.webcamDisplay.classList.add('hidden');
  DOM.overlayCanvas.classList.remove('hidden');
  DOM.btnProcessVideo.disabled = false;
  DOM.btnClearPreview.disabled = false;
}

function toggleVideoProcessing() {
  if (DOM.videoDisplay.paused) {
    DOM.videoDisplay.play();
    DOM.btnProcessVideo.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
      <span>Pause Threat Scan</span>
    `;
    startVideoFrameProcessor();
  } else {
    DOM.videoDisplay.pause();
    DOM.btnProcessVideo.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Start Video Threat Scan</span>
    `;
  }
}

function startVideoFrameProcessor() {
  const offscreen = document.createElement('canvas');
  const offCtx = offscreen.getContext('2d');

  const step = async () => {
    if (DOM.videoDisplay.paused || DOM.videoDisplay.ended) return;

    const w = DOM.videoDisplay.videoWidth;
    const h = DOM.videoDisplay.videoHeight;
    if (w > 0 && h > 0) {
      offscreen.width = w;
      offscreen.height = h;
      offCtx.drawImage(DOM.videoDisplay, 0, 0, w, h);

      if (state.backendConnected && !state.isProcessingFrame) {
        state.isProcessingFrame = true;
        const b64 = offscreen.toDataURL('image/jpeg', 0.6);
        try {
          const res = await fetch(`${CONFIG.API_BASE}/api/detect/frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: b64, conf: state.confidence, iou: state.iou })
          });
          if (res.ok) {
            const data = await res.json();
            drawOverlayBoxes(data.detections, w, h);
            renderDetections(data.detections);
            if (data.detections.length > 0) {
              triggerWeaponAlert(data.detections, data.annotated_image, data.timestamp);
            }
          }
        } catch (e) {}
        state.isProcessingFrame = false;
      }
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// =============================================================================
// MODAL & UTILITY FUNCTIONS
// =============================================================================
function openModal(item) {
  DOM.modalImage.src = item.snapshotUrl;
  DOM.modalDetails.innerHTML = `
    <span><strong>Timestamp:</strong> ${item.timestamp}</span>
    <span><strong>Target:</strong> ${item.class}</span>
    <span><strong>Confidence:</strong> ${item.confidence}%</span>
  `;
  DOM.btnModalDownload.href = item.snapshotUrl;
  DOM.btnModalDownload.download = `threat_audit_${item.timestamp.replace(/:/g, '')}.jpg`;
  DOM.snapshotModal.classList.remove('hidden');
}

function closeModal() {
  DOM.snapshotModal.classList.add('hidden');
}

function clearPreview() {
  state.currentImageBlob = null;
  state.currentAnnotatedUrl = null;
  state.lastDetections = [];

  DOM.imageDisplay.src = '';
  DOM.imageDisplay.classList.add('hidden');
  DOM.videoDisplay.src = '';
  DOM.videoDisplay.classList.add('hidden');
  DOM.dropzoneEmpty.classList.remove('hidden');
  DOM.overlayCanvas.classList.add('hidden');

  DOM.btnReDetect.disabled = true;
  DOM.btnClearPreview.disabled = true;
  DOM.btnDownloadResult.disabled = true;
  DOM.btnProcessVideo.disabled = true;
  DOM.alertBanner.classList.add('hidden');

  renderDetections([]);
}

function downloadCurrentResult() {
  if (!state.currentAnnotatedUrl) return;
  const a = document.createElement('a');
  a.href = state.currentAnnotatedUrl;
  a.download = `weapon_detection_result_${Date.now()}.jpg`;
  a.click();
}

function exportDetectionsJson() {
  const exportData = {
    system: 'Shield AI - Weapon Surveillance System',
    timestamp: new Date().toISOString(),
    confidence_threshold: state.confidence,
    iou_threshold: state.iou,
    active_detections: state.lastDetections,
    session_history: state.historyLogs
  };

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `weapon_detection_audit_${Date.now()}.json`;
  a.click();
}

function showLoader(visible, msg = 'PROCESSING...') {
  DOM.previewLoader.classList.toggle('hidden', !visible);
  DOM.loaderStatusMsg.textContent = msg;
}
