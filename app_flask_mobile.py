"""
NHẬN DIỆN HÌNH HỌC - Mobile PWA Flask App
==========================================
Cách chạy:
1. pip install tensorflow opencv-python flask numpy
2. Đặt hinh_hoc_model.tflite cùng thư mục
3. python app_flask_mobile.py
4. Mở điện thoại trỏ vào: http://<IP_MÁY_TÍNH>:5000
   (VD: http://192.168.1.5:5000)

Để tìm IP máy tính:
  - Windows: ipconfig  (tìm IPv4)
  - Mac/Linux: ifconfig hoặc ip addr
"""

from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import base64
import json
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = 'hinh_hoc_model.tflite'

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Không tìm thấy {MODEL_PATH}")
    interpreter = None
else:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("✅ Load TFLite model xong!")

CLASS_LABELS_VI = {
    0: "Hình Bình Hành",
    1: "Hình Chữ Nhật",
    2: "Hình Thang",
    3: "Hình Tròn",
    4: "Hình Vuông",
    5: "Tam Giác"
}
CLASS_EMOJI = {0: "▱", 1: "▬", 2: "⏢", 3: "⬤", 4: "■", 5: "▲"}

# ── Manifest PWA ──────────────────────────────────────────
MANIFEST = {
    "name": "Nhận Diện Hình Học",
    "short_name": "HìnhHọc",
    "description": "App nhận diện hình học cho bé",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#0a0a14",
    "theme_color": "#7c3aed",
    "orientation": "portrait",
    "icons": [
        {"src": "/icon-192.svg", "sizes": "192x192", "type": "image/svg+xml"},
        {"src": "/icon-512.svg", "sizes": "512x512", "type": "image/svg+xml"}
    ]
}

ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">
  <rect width="192" height="192" rx="40" fill="#0a0a14"/>
  <polygon points="96,30 162,150 30,150" fill="#7c3aed"/>
  <rect x="55" y="60" width="50" height="50" rx="4" fill="#ec4899" opacity="0.7"/>
  <circle cx="135" cy="120" r="28" fill="#38bdf8" opacity="0.7"/>
</svg>"""

HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<meta name="theme-color" content="#7c3aed">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="HìnhHọc">
<link rel="manifest" href="/manifest.json">
<link rel="apple-touch-icon" href="/icon-192.svg">
<title>Nhận Diện Hình Học</title>
<link href="https://fonts.googleapis.com/css2?family=Fredoka+One&family=Quicksand:wght@500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0a14;
  --surface: #12121f;
  --surface2: #1a1a2e;
  --purple: #7c3aed;
  --pink: #ec4899;
  --cyan: #22d3ee;
  --yellow: #fbbf24;
  --green: #34d399;
  --text: #f1f0ff;
  --muted: rgba(241,240,255,0.45);
  --radius: 20px;
  --safe-top: env(safe-area-inset-top, 0px);
  --safe-bot: env(safe-area-inset-bottom, 0px);
}

* { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }

html, body {
  height: 100%; overflow: hidden;
  background: var(--bg);
  font-family: 'Quicksand', sans-serif;
  color: var(--text);
}

/* ── LAYOUT ── */
.app {
  height: 100dvh;
  display: flex; flex-direction: column;
  padding-top: var(--safe-top);
  padding-bottom: var(--safe-bot);
  overflow: hidden;
}

/* ── HEADER ── */
.header {
  padding: 16px 20px 12px;
  display: flex; align-items: center; gap: 10px;
  flex-shrink: 0;
  background: linear-gradient(180deg, var(--bg) 60%, transparent);
}
.header-icon {
  width: 38px; height: 38px;
  background: linear-gradient(135deg, var(--purple), var(--pink));
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  box-shadow: 0 4px 16px rgba(124,58,237,0.4);
}
.header-title {
  font-family: 'Fredoka One', cursive;
  font-size: 1.3rem; line-height: 1;
  background: linear-gradient(135deg, #c4b5fd, #f9a8d4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-sub { font-size: 0.7rem; color: var(--muted); margin-top: 1px; }

/* ── SCROLLABLE BODY ── */
.body {
  flex: 1; overflow-y: auto;
  padding: 0 16px 16px;
  -webkit-overflow-scrolling: touch;
}

/* ── CAMERA / UPLOAD ZONE ── */
.upload-zone {
  position: relative;
  border-radius: var(--radius);
  overflow: hidden;
  background: var(--surface);
  border: 2px dashed rgba(124,58,237,0.35);
  aspect-ratio: 4/3;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 8px;
  transition: border-color 0.2s, background 0.2s;
  cursor: pointer;
}
.upload-zone.has-image { border-style: solid; border-color: rgba(124,58,237,0.6); }
.upload-zone input[type=file] {
  position: absolute; inset: 0; opacity: 0; cursor: pointer;
  width: 100%; height: 100%;
}
.upload-zone img {
  position: absolute; inset: 0;
  width: 100%; height: 100%; object-fit: contain;
}
.upload-placeholder {
  text-align: center; pointer-events: none; z-index: 1; padding: 20px;
}
.upload-placeholder .icon { font-size: 3rem; display: block; }
.upload-placeholder p { color: var(--muted); font-size: 0.85rem; font-weight: 700; margin-top: 6px; }
.upload-placeholder small { color: rgba(241,240,255,0.25); font-size: 0.72rem; }

/* Camera badge */
.cam-badge {
  position: absolute; bottom: 10px; right: 10px; z-index: 10;
  background: rgba(0,0,0,0.6); backdrop-filter: blur(8px);
  border-radius: 50px; padding: 6px 12px;
  font-size: 0.72rem; color: var(--muted); font-weight: 700;
  display: flex; align-items: center; gap: 4px;
}

/* ── ACTION BUTTONS ── */
.actions {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 10px; margin-top: 12px;
}
.btn {
  border: none; border-radius: 14px;
  font-family: 'Fredoka One', cursive; font-size: 1rem;
  padding: 14px 8px; cursor: pointer;
  transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
  display: flex; align-items: center; justify-content: center; gap: 6px;
  -webkit-appearance: none;
}
.btn:active { transform: scale(0.95); }
.btn-camera {
  background: var(--surface2);
  color: var(--cyan);
  border: 1px solid rgba(34,211,238,0.2);
}
.btn-predict {
  background: linear-gradient(135deg, var(--purple), var(--pink));
  color: white;
  box-shadow: 0 6px 20px rgba(124,58,237,0.4);
  grid-column: 1 / -1;
}
.btn-predict:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }
.btn-gallery {
  background: var(--surface2);
  color: var(--yellow);
  border: 1px solid rgba(251,191,36,0.2);
}

/* ── RESULT CARD ── */
.result-card {
  margin-top: 14px;
  background: var(--surface);
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid rgba(124,58,237,0.2);
  display: none;
  animation: slideUp 0.4s cubic-bezier(0.22,1,0.36,1);
}
@keyframes slideUp {
  from { transform: translateY(30px); opacity: 0; }
  to   { transform: translateY(0);    opacity: 1; }
}

.result-hero {
  padding: 24px 20px 18px;
  text-align: center;
  background: linear-gradient(160deg, rgba(124,58,237,0.12), rgba(236,72,153,0.08));
}
.result-shape-preview {
  width: 80px; height: 80px; margin: 0 auto 12px;
  display: flex; align-items: center; justify-content: center;
}
.result-shape-preview svg { width: 100%; height: 100%; }
.result-name {
  font-family: 'Fredoka One', cursive;
  font-size: 2rem; letter-spacing: 0.5px;
  background: linear-gradient(135deg, #c4b5fd, #f9a8d4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.result-conf-text { color: var(--muted); font-size: 0.82rem; margin-top: 4px; }

.conf-bar-wrap { padding: 0 20px; margin-top: 12px; }
.conf-bar-bg {
  background: rgba(255,255,255,0.08); border-radius: 999px; height: 8px; overflow: hidden;
}
.conf-bar-fill {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--purple), var(--pink));
  transition: width 0.9s cubic-bezier(0.22,1,0.36,1);
  width: 0%;
}

.probs-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 8px; padding: 14px;
}
.prob-cell {
  background: rgba(255,255,255,0.04);
  border-radius: 12px; padding: 10px;
}
.prob-cell.active {
  background: rgba(124,58,237,0.15);
  border: 1px solid rgba(124,58,237,0.3);
}
.prob-row {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.75rem; margin-bottom: 5px; font-weight: 700;
}
.prob-label { color: rgba(255,255,255,0.7); }
.prob-pct { }
.prob-mini-bg { background: rgba(255,255,255,0.1); border-radius: 999px; height: 4px; overflow: hidden; }
.prob-mini-fill { height: 100%; border-radius: 999px; transition: width 0.8s 0.15s ease; width: 0%; }

/* ── SHAPE GUIDE ── */
.guide { margin-top: 14px; }
.guide-title {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px;
  color: var(--muted); text-align: center; margin-bottom: 10px; font-weight: 700;
}
.guide-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
}
.guide-tile {
  background: var(--surface);
  border-radius: 14px; padding: 12px 6px 8px;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.05);
  transition: background 0.2s, transform 0.15s;
}
.guide-tile:active { transform: scale(0.95); }
.guide-tile svg { width: 36px; height: 32px; }
.guide-tile-name { font-size: 0.65rem; color: var(--muted); margin-top: 5px; font-weight: 700; }

/* ── SPINNER ── */
.spin {
  display: inline-block; width: 16px; height: 16px;
  border: 2.5px solid rgba(255,255,255,0.3);
  border-top-color: white; border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── TOAST ── */
.toast {
  position: fixed; bottom: calc(16px + var(--safe-bot));
  left: 50%; transform: translateX(-50%) translateY(80px);
  background: rgba(30,30,50,0.95); backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 50px; padding: 10px 20px;
  font-size: 0.82rem; font-weight: 700; color: var(--text);
  transition: transform 0.3s cubic-bezier(0.22,1,0.36,1);
  z-index: 999; pointer-events: none; white-space: nowrap;
}
.toast.show { transform: translateX(-50%) translateY(0); }

/* ── INSTALL BANNER ── */
.install-banner {
  display: none;
  margin-bottom: 12px;
  background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(236,72,153,0.15));
  border: 1px solid rgba(124,58,237,0.3);
  border-radius: 14px; padding: 12px 14px;
  display: flex; align-items: center; gap: 10px;
}
.install-banner .ib-text { flex: 1; font-size: 0.78rem; color: rgba(255,255,255,0.8); font-weight: 700; }
.install-banner .ib-sub { font-size: 0.68rem; color: var(--muted); font-weight: 500; }
.install-btn {
  border: none; border-radius: 10px; padding: 8px 14px;
  background: linear-gradient(135deg, var(--purple), var(--pink));
  color: white; font-family: 'Fredoka One', cursive;
  font-size: 0.85rem; cursor: pointer; flex-shrink: 0;
  -webkit-appearance: none;
}
</style>
</head>
<body>
<div class="app">
  <div class="header">
    <div class="header-icon">🔷</div>
    <div>
      <div class="header-title">Nhận Diện Hình Học</div>
      <div class="header-sub">Chụp hoặc tải ảnh lên để nhận diện ✨</div>
    </div>
  </div>

  <div class="body" id="scrollBody">

    <!-- Install banner (hiện khi có thể cài PWA) -->
    <div class="install-banner" id="installBanner" style="display:none">
      <span style="font-size:1.4rem">📲</span>
      <div class="ib-text">
        Cài app về máy<br>
        <span class="ib-sub">Dùng được offline, nhanh hơn</span>
      </div>
      <button class="install-btn" id="installBtn">Cài App</button>
    </div>

    <!-- Upload zone -->
    <div class="upload-zone" id="uploadZone">
      <input type="file" id="fileInput" accept="image/*">
      <div class="upload-placeholder" id="placeholder">
        <span class="icon">🖼️</span>
        <p>Bấm để chọn ảnh<br>hoặc chụp từ camera</p>
        <small>JPG, PNG, WEBP</small>
      </div>
      <img id="preview" src="" alt="" style="display:none">
      <div class="cam-badge" id="camBadge" style="display:none">📷 Đổi ảnh</div>
    </div>

    <!-- Buttons -->
    <div class="actions">
      <button class="btn btn-camera" id="btnCamera" onclick="openCamera()">
        📷 Camera
      </button>
      <button class="btn btn-gallery" onclick="document.getElementById('fileInput').click()">
        🖼️ Thư viện
      </button>
      <button class="btn btn-predict" id="btnPredict" disabled onclick="predict()">
        🔍 Nhận Diện Ngay
      </button>
    </div>

    <!-- Result -->
    <div class="result-card" id="resultCard">
      <div class="result-hero">
        <div class="result-shape-preview" id="resShapePreview"></div>
        <div class="result-name" id="resName">—</div>
        <div class="result-conf-text" id="resConf">—</div>
      </div>
      <div class="conf-bar-wrap">
        <div class="conf-bar-bg"><div class="conf-bar-fill" id="confBar"></div></div>
      </div>
      <div class="probs-grid" id="probsGrid"></div>
    </div>

    <!-- Shape guide -->
    <div class="guide">
      <div class="guide-title">Hình có thể nhận diện</div>
      <div class="guide-grid">
        <div class="guide-tile">
          <svg viewBox="0 0 100 70"><polygon points="20,58 75,58 82,12 27,12" fill="#f472b6"/></svg>
          <div class="guide-tile-name">Bình Hành</div>
        </div>
        <div class="guide-tile">
          <svg viewBox="0 0 100 70"><rect x="8" y="18" width="84" height="34" rx="3" fill="#34d399"/></svg>
          <div class="guide-tile-name">Chữ Nhật</div>
        </div>
        <div class="guide-tile">
          <svg viewBox="0 0 100 70"><polygon points="18,58 82,58 68,12 32,12" fill="#fbbf24"/></svg>
          <div class="guide-tile-name">Hình Thang</div>
        </div>
        <div class="guide-tile">
          <svg viewBox="0 0 80 80"><circle cx="40" cy="40" r="32" fill="#a78bfa"/></svg>
          <div class="guide-tile-name">Hình Tròn</div>
        </div>
        <div class="guide-tile">
          <svg viewBox="0 0 80 80"><rect x="8" y="8" width="64" height="64" rx="3" fill="#fb923c"/></svg>
          <div class="guide-tile-name">Hình Vuông</div>
        </div>
        <div class="guide-tile">
          <svg viewBox="0 0 100 80"><polygon points="50,8 92,72 8,72" fill="#38bdf8"/></svg>
          <div class="guide-tile-name">Tam Giác</div>
        </div>
      </div>
    </div>

  </div><!-- /body -->
</div><!-- /app -->

<div class="toast" id="toast"></div>

<!-- Hidden camera input -->
<input type="file" id="cameraInput" accept="image/*" capture="environment" style="display:none">

<script>
const COLORS  = ['#f472b6','#34d399','#fbbf24','#a78bfa','#fb923c','#38bdf8'];
const LABELS  = ['Hình Bình Hành','Hình Chữ Nhật','Hình Thang','Hình Tròn','Hình Vuông','Tam Giác'];
const SHAPES_SVG = [
  `<svg viewBox="0 0 100 70"><polygon points="20,58 75,58 82,12 27,12" fill="#f472b6"/></svg>`,
  `<svg viewBox="0 0 100 70"><rect x="8" y="18" width="84" height="34" rx="3" fill="#34d399"/></svg>`,
  `<svg viewBox="0 0 100 70"><polygon points="18,58 82,58 68,12 32,12" fill="#fbbf24"/></svg>`,
  `<svg viewBox="0 0 80 80"><circle cx="40" cy="40" r="32" fill="#a78bfa"/></svg>`,
  `<svg viewBox="0 0 80 80"><rect x="8" y="8" width="64" height="64" rx="3" fill="#fb923c"/></svg>`,
  `<svg viewBox="0 0 100 80"><polygon points="50,8 92,72 8,72" fill="#38bdf8"/></svg>`
];

let currentB64 = null;
let deferredPrompt = null;

// ── File chọn từ thư viện ──
document.getElementById('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// ── Camera ──
document.getElementById('cameraInput').addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function openCamera() {
  document.getElementById('cameraInput').click();
}

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    currentB64 = e.target.result;
    const img = document.getElementById('preview');
    img.src = currentB64;
    img.style.display = 'block';
    document.getElementById('placeholder').style.display = 'none';
    document.getElementById('camBadge').style.display = 'flex';
    document.getElementById('uploadZone').classList.add('has-image');
    document.getElementById('resultCard').style.display = 'none';
    document.getElementById('btnPredict').disabled = false;
    showToast('✅ Ảnh đã tải lên!');
  };
  reader.readAsDataURL(file);
}

// ── Predict ──
async function predict() {
  if (!currentB64) return;
  const btn = document.getElementById('btnPredict');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span> Đang nhận diện...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image: currentB64})
    });
    const data = await res.json();
    if (data.error) { showToast('❌ ' + data.error); return; }
    showResult(data);

    // Scroll to result
    setTimeout(() => {
      document.getElementById('resultCard').scrollIntoView({behavior:'smooth', block:'nearest'});
    }, 100);

  } catch(err) {
    showToast('❌ Lỗi kết nối server');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🔍 Nhận Diện Ngay';
  }
}

function showResult(data) {
  document.getElementById('resShapePreview').innerHTML = SHAPES_SVG[data.idx];
  document.getElementById('resName').textContent = data.label;
  document.getElementById('resConf').textContent = `Độ tin cậy: ${data.conf}%`;

  setTimeout(() => {
    document.getElementById('confBar').style.width = data.conf + '%';
  }, 100);

  const grid = document.getElementById('probsGrid');
  grid.innerHTML = '';
  data.probs.forEach((p, i) => {
    const pct = (p * 100).toFixed(1);
    const isTop = i === data.idx;
    const cell = document.createElement('div');
    cell.className = 'prob-cell' + (isTop ? ' active' : '');
    cell.innerHTML = `
      <div class="prob-row">
        <span class="prob-label">${LABELS[i]}</span>
        <span class="prob-pct" style="color:${COLORS[i]}">${pct}%</span>
      </div>
      <div class="prob-mini-bg">
        <div class="prob-mini-fill" id="pm${i}" style="background:${COLORS[i]}"></div>
      </div>`;
    grid.appendChild(cell);
  });
  setTimeout(() => {
    data.probs.forEach((p, i) => {
      const el = document.getElementById('pm' + i);
      if (el) el.style.width = (p * 100).toFixed(1) + '%';
    });
  }, 200);

  document.getElementById('resultCard').style.display = 'block';
}

// ── Toast ──
let toastTimer;
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2200);
}

// ── PWA Install ──
window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById('installBanner').style.display = 'flex';
});

document.getElementById('installBtn').addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;
  if (outcome === 'accepted') {
    document.getElementById('installBanner').style.display = 'none';
    showToast('🎉 Đã cài app thành công!');
  }
  deferredPrompt = null;
});

window.addEventListener('appinstalled', () => {
  document.getElementById('installBanner').style.display = 'none';
  showToast('🎉 App đã được cài!');
});

// ── Service Worker (offline) ──
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}
</script>
</body>
</html>"""

# ── Service Worker ────────────────────────────────────────
SW_JS = """
const CACHE = 'hinh-hoc-v1';
const ASSETS = ['/'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys =>
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  if (e.request.url.includes('/predict')) return;
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
"""

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return HTML

@app.route('/manifest.json')
def manifest():
    return Response(json.dumps(MANIFEST), mimetype='application/manifest+json')

@app.route('/sw.js')
def service_worker():
    return Response(SW_JS, mimetype='application/javascript')

@app.route('/icon-192.svg')
@app.route('/icon-512.svg')
def icon():
    return Response(ICON_SVG, mimetype='image/svg+xml')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({"error": "Model chưa load!"}), 500

    data    = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    nparr   = np.frombuffer(img_data, np.uint8)
    img     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img     = cv2.resize(img, (64, 64))
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img     = img.astype('float32') / 255.0
    img     = np.expand_dims(img, 0)

    interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

pred = interpreter.get_tensor(output_details[0]['index'])[0]

idx = int(np.argmax(pred))
conf = float(pred[idx]) * 100

    return jsonify({
        "label": CLASS_LABELS_VI[idx],
        "emoji": CLASS_EMOJI[idx],
        "conf":  round(conf, 1),
        "idx":   idx,
        "probs": pred.tolist()
    })

# ── Run ───────────────────────────────────────────────────
if __name__ == '__main__':
    import socket
    # Tự lấy IP local để in ra
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"

    print("\n" + "="*50)
    print("🚀  SERVER ĐÃ KHỞI ĐỘNG!")
    print("="*50)
    print(f"💻  Máy tính : http://localhost:5000")
    print(f"📱  Điện thoại: http://{local_ip}:5000")
    print("\n📌  Đảm bảo điện thoại & máy tính cùng WiFi!")
    print("="*50 + "\n")

    import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)

