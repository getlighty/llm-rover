"""web_ui.py — Web server + dashboard.

Extracted from rover_brain_llm.py. Receives shared state refs at init time.
No circular imports — uses callbacks/refs passed in via init().
"""

import base64
import glob as globmod
import json
import os
import random
import shutil
import time
import threading
import traceback
import yaml
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import lessons

# ── Shared state refs (set via init()) ──────────────────────────────

_camera_ref = None
_ser_ref = None
_detector_ref = None
_get_log_events_since = None
_get_map_state = None
_log_event = None
_set_provider = None
_get_providers = None    # () -> {"current": {...}, "available": {...}, "desk_mode": bool, "stt_enabled": bool}
_set_desk_mode = None    # (bool) -> None
_set_stt_enabled = None  # (bool) -> None
_set_tts_enabled = None  # (bool) -> None
_set_gimbal_pan_enabled = None  # (bool) -> None
_set_killed = None       # (bool) -> None
_get_killed = None       # () -> bool
_plan_active = None      # threading.Event
_stop_event = None       # threading.Event
_command_queue = None     # queue.Queue
_interrupt_queue = None   # queue.Queue
_classify_interrupt = None  # fn(text) -> str

# ── Annotation / training state ─────────────────────────────────
_training_thread = None
_training_status = {"state": "idle"}
_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def init(*, camera, serial, detector=None, get_log_events_since, get_map_state,
         log_event, set_provider, get_providers, set_desk_mode, set_stt_enabled,
         set_tts_enabled, set_gimbal_pan_enabled, set_yolo_enabled, set_killed, get_killed,
         plan_active, stop_event, command_queue, interrupt_queue,
         classify_interrupt):
    """Wire up shared state references. Call once from main()."""
    global _camera_ref, _ser_ref, _detector_ref, _get_log_events_since
    global _get_map_state
    global _log_event, _set_provider, _get_providers
    global _set_desk_mode, _set_stt_enabled, _set_tts_enabled, _set_gimbal_pan_enabled
    global _set_yolo_enabled, _set_killed, _get_killed
    global _plan_active, _stop_event, _command_queue, _interrupt_queue
    global _classify_interrupt
    _camera_ref = camera
    _ser_ref = serial
    _detector_ref = detector
    _get_log_events_since = get_log_events_since
    _get_map_state = get_map_state
    _log_event = log_event
    _set_provider = set_provider
    _get_providers = get_providers
    _set_desk_mode = set_desk_mode
    _set_stt_enabled = set_stt_enabled
    _set_tts_enabled = set_tts_enabled
    _set_gimbal_pan_enabled = set_gimbal_pan_enabled
    _set_yolo_enabled = set_yolo_enabled
    _set_killed = set_killed
    _get_killed = get_killed
    _plan_active = plan_active
    _stop_event = stop_event
    _command_queue = command_queue
    _interrupt_queue = interrupt_queue
    _classify_interrupt = classify_interrupt


# ── Dashboard HTML ──────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rover Control</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Courier New',monospace;height:100vh;display:flex;flex-direction:column}
#topbar{display:flex;gap:12px;padding:8px 12px;background:#16213e;align-items:center;flex-wrap:wrap}
#topbar label{font-size:12px;color:#8888aa}
#topbar select{background:#0f3460;color:#e0e0e0;border:1px solid #333;padding:4px 8px;border-radius:4px;font-family:inherit;font-size:13px}
#llm{min-width:180px}
#main{display:flex;flex:1;overflow:hidden}
#leftcol{flex:0 0 auto;padding:8px;display:flex;flex-direction:column;gap:8px}
#leftcol img{max-width:640px;width:100%;border-radius:4px;border:1px solid #333}
#gimbal-wrap{width:100%;max-width:640px}
#gimbal{width:100%;aspect-ratio:2/1;border-radius:4px;border:1px solid #333;background:#0a0a1a;cursor:crosshair;touch-action:none;display:block}
#gimbal-info{display:flex;align-items:center;gap:8px;padding:4px 0}
#gimbal-readout{font-size:12px;color:#8888aa;flex:1}
#gimbal-center{background:#0f3460;color:#e0e0e0;border:1px solid #444;padding:4px 12px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px}
#gimbal-center:hover{background:#1a4a80}
#room-map-wrap{width:100%;max-width:640px;background:#0a0a1a;border:1px solid #333;border-radius:4px;padding:8px}
#room-map{width:100%;aspect-ratio:16/10;border-radius:4px;border:1px solid #333;background:#050511;display:block}
#room-guess{font-size:12px;color:#88d1d1;margin-top:6px}
#logpanel{flex:1;display:flex;flex-direction:column;min-width:0}
#log{flex:1;overflow-y:auto;padding:8px;font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-all}
#log .ts{color:#666}
#log .cat-llm{color:#5dade2}
#log .cat-serial{color:#f39c12}
#log .cat-heard{color:#2ecc71}
#log .cat-speak{color:#e74c3c}
#log .cat-plan{color:#9b59b6}
#log .cat-stuck{color:#ff6b6b;font-weight:bold}
#log .cat-interrupt{color:#e67e22}
#log .cat-system{color:#888}
#log .cat-error{color:#ff4444;font-weight:bold}
#log .cat-nav{color:#1abc9c}
#log .cat-orchestrator{color:#8e44ad}
#log .cat-observe{color:#3498db}
#log .cat-decide{color:#27ae60}
#log .cat-bash{color:#d4ac0d}
#log .cat-file{color:#d4ac0d}
#log .cat-follow{color:#e91e63}
#log .cat-battery{color:#f57f17;font-weight:bold}
#log .cat-backup{color:#ff9800}
#log .cat-detect{color:#00bcd4}
#log .cat-yolo_corr{color:#00bcd4}
#log .cat-imu{color:#78909c}
#log .cat-track{color:#ab47bc}
#log .cat-floor_nav{color:#607d8b}
#log .cat-room{color:#26a69a}
#log-filters{padding:4px 8px;background:#16213e;display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid #333}
#log-filters label{font-size:11px;color:#aaa;cursor:pointer;user-select:none}
#log-filters input{margin-right:2px;cursor:pointer}
#cmdbar{display:flex;padding:8px 12px;background:#16213e;gap:8px}
#cmdbar input{flex:1;background:#0f3460;color:#e0e0e0;border:1px solid #333;padding:8px 12px;border-radius:4px;font-family:inherit;font-size:14px}
#cmdbar input::placeholder{color:#555}
#cmdbar button{background:#e94560;color:#fff;border:none;padding:8px 20px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:14px}
#cmdbar button:hover{background:#c73e54}
#annotateCanvas{max-width:640px;width:100%;border-radius:4px;border:1px solid #333;display:none;cursor:crosshair}
#annotateBar{display:none;flex-wrap:wrap;gap:6px;align-items:center;max-width:640px;padding:6px 0}
#annotateBar button{background:#0f3460;color:#e0e0e0;border:1px solid #444;padding:4px 10px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px}
#annotateBar button:hover{background:#1a4a80}
#annotateBar select,#annotateBar input[type=text]{background:#0f3460;color:#e0e0e0;border:1px solid #444;padding:4px 8px;border-radius:4px;font-family:inherit;font-size:12px}
#annotateBar input[type=text]{width:120px}
#annotateBar .sep{width:1px;height:20px;background:#444}
#trainStatus{font-size:12px;color:#888;margin-left:4px}
.ann-btn-active{background:#e94560!important;border-color:#e94560!important}
#killBtn{background:#aa0000;color:#fff;border:2px solid #ff0000;padding:4px 16px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:13px;font-weight:bold;letter-spacing:1px}
#killBtn:hover{background:#cc0000}
#killBtn.killed{background:#ff0000;border-color:#ff4444;animation:killPulse 1s infinite}
@keyframes killPulse{0%,100%{opacity:1}50%{opacity:0.6}}
#restartBtn{background:#aa6600;color:#fff;border:2px solid #ff8800;padding:4px 16px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:13px;font-weight:bold;letter-spacing:1px;margin-left:4px}
#restartBtn:hover{background:#cc7700}
</style>
</head>
<body>
<div id="topbar">
  <label>STT <select id="stt"></select></label>
  <label>LLM <select id="llm"></select></label>
  <label>TTS <select id="tts"></select></label>
  <label>Orch <select id="orch"></select></label>
  <label style="margin-left:auto;cursor:pointer"><input type="checkbox" id="sttmode" checked> STT</label>
  <label style="cursor:pointer"><input type="checkbox" id="ttsmode" checked> TTS</label>
  <label style="cursor:pointer"><input type="checkbox" id="deskmode" checked> Desk Mode</label>
  <label style="cursor:pointer"><input type="checkbox" id="panmode"> Gimbal Pan</label>
  <label style="cursor:pointer"><input type="checkbox" id="yolomode" checked> YOLO</label>
  <button id="annToggle" style="margin-left:8px;background:#0f3460;color:#e0e0e0;border:1px solid #444;padding:4px 12px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">Annotate</button>
  <button id="killBtn" style="margin-left:auto">KILL</button>
  <button id="restartBtn">RESTART</button>
</div>
<div id="main">
  <div id="leftcol">
    <img id="streamImg" src="/stream" alt="Camera">
    <canvas id="annotateCanvas" width="640" height="480"></canvas>
    <div id="annotateBar">
      <button id="annCapture">Capture</button>
      <button id="annDelete">Delete Box</button>
      <button id="annSave">Save</button>
      <div class="sep"></div>
      <select id="annClass"></select>
      <input type="text" id="annNewClass" placeholder="New class...">
      <button id="annApply">Apply Label</button>
      <div class="sep"></div>
      <button id="annTrain">Train YOLO26s</button>
      <span id="trainStatus">idle</span>
    </div>
    <div id="gimbal-wrap">
      <canvas id="gimbal"></canvas>
      <div id="gimbal-info">
        <span id="gimbal-readout">Pan: 0°  Tilt: 0°</span>
        <button id="gimbal-center">Center</button>
      </div>
    </div>
    <div id="room-map-wrap">
      <canvas id="room-map"></canvas>
      <div id="room-guess">Room guess: unknown</div>
    </div>
    <div id="gridSection" style="max-width:640px;margin-top:8px">
      <button id="gridToggle" style="background:#0f3460;color:#e0e0e0;border:1px solid #444;
        padding:4px 12px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px;width:100%">
        &#9632; Exploration Grid</button>
      <div id="gridPanel" style="display:none;background:#0a0a1a;border:1px solid #333;
        border-radius:0 0 4px 4px;padding:8px;text-align:center">
        <img id="gridImg" style="width:320px;height:320px;image-rendering:pixelated;border-radius:4px;border:1px solid #333">
        <div style="display:flex;justify-content:center;gap:12px;margin-top:6px;font-size:11px">
          <span style="color:#2ecc71">&#9632; visited</span>
          <span style="color:#3498db">&#9632; free</span>
          <span style="color:#e74c3c">&#9632; occupied</span>
          <span style="color:#ccc">&#9650; rover</span>
        </div>
      </div>
    </div>
    <div id="calibSection" style="max-width:640px;margin-top:8px">
      <button id="calibToggle" style="background:#0f3460;color:#e0e0e0;border:1px solid #444;
        padding:4px 12px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px;width:100%">
        &#9881; Turn Calibration</button>
      <div id="calibPanel" style="display:none;background:#0a0a1a;border:1px solid #333;
        border-radius:0 0 4px 4px;padding:12px">
        <div style="font-size:12px;color:#8888aa;margin-bottom:8px">
          Current TURN_RATE_DPS: <span id="calibCurrent" style="color:#2ecc71">200.0</span> deg/s
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
          <button class="calib-btn" data-deg="90" data-dir="-1"
            style="background:#0f3460;color:#e0e0e0;border:1px solid #444;padding:6px 12px;
            border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">90&deg; Left</button>
          <button class="calib-btn" data-deg="90" data-dir="1"
            style="background:#0f3460;color:#e0e0e0;border:1px solid #444;padding:6px 12px;
            border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">90&deg; Right</button>
          <button class="calib-btn" data-deg="180" data-dir="1"
            style="background:#0f3460;color:#e0e0e0;border:1px solid #444;padding:6px 12px;
            border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">180&deg; Right</button>
        </div>
        <div id="calibPrompt" style="display:none;margin-top:8px">
          <div style="font-size:12px;color:#e0e0e0;margin-bottom:6px">
            Actual degrees turned?</div>
          <div style="display:flex;gap:8px;align-items:center">
            <input id="calibActual" type="number" min="0" max="360" step="5" value="90"
              style="width:70px;background:#0f3460;color:#e0e0e0;border:1px solid #444;
              padding:4px 6px;border-radius:4px;font-family:inherit;font-size:12px">
            <span style="font-size:12px;color:#8888aa">&deg;</span>
            <button id="calibApply"
              style="background:#2ecc71;color:#fff;border:none;padding:4px 12px;
              border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">Apply</button>
          </div>
          <div id="calibResult" style="font-size:11px;color:#2ecc71;margin-top:6px"></div>
        </div>
        <div style="border-top:1px solid #333;margin-top:10px;padding-top:10px">
          <div style="font-size:12px;color:#8888aa;margin-bottom:6px">
            Gimbal Pan Offset: <span id="gimbalOffCurrent" style="color:#2ecc71">0.0</span>&deg;
          </div>
          <div style="display:flex;gap:8px;align-items:center">
            <button id="gimbalOffMinus" style="background:#0f3460;color:#e0e0e0;border:1px solid #444;
              padding:4px 10px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">-1&deg;</button>
            <input id="gimbalOffVal" type="number" min="-30" max="30" step="0.5" value="0"
              style="width:60px;background:#0f3460;color:#e0e0e0;border:1px solid #444;
              padding:4px 6px;border-radius:4px;font-family:inherit;font-size:12px">
            <button id="gimbalOffPlus" style="background:#0f3460;color:#e0e0e0;border:1px solid #444;
              padding:4px 10px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">+1&deg;</button>
            <button id="gimbalOffSave" style="background:#2ecc71;color:#fff;border:none;
              padding:4px 12px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">Save</button>
            <button id="gimbalOffTest" style="background:#0f3460;color:#e0e0e0;border:1px solid #444;
              padding:4px 10px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px">Test Center</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div id="logpanel"><div id="log-filters"></div><div id="log"></div></div>
</div>
<div id="cmdbar">
  <input id="cmd" type="text" placeholder="Type a command..." autocomplete="off">
  <button onclick="sendCmd()">Send</button>
</div>
<script>
const log=document.getElementById('log');
const cmd=document.getElementById('cmd');
const gimbalCanvas=document.getElementById('gimbal');
const gctx=gimbalCanvas.getContext('2d');
const roomMapCanvas=document.getElementById('room-map');
const roomMapCtx=roomMapCanvas.getContext('2d');
const roomGuessEl=document.getElementById('room-guess');

// Load provider options + desk mode
const deskCb=document.getElementById('deskmode');
const sttCb=document.getElementById('sttmode');
const ttsCb=document.getElementById('ttsmode');
const panCb=document.getElementById('panmode');
const yoloCb=document.getElementById('yolomode');
function llmLabel(v){
  // "ollama/qwen3.5:cloud" → "ollama: qwen3.5:cloud"
  // "groq/meta-llama/llama-4-maverick-17b-128e-instruct" → "groq: llama-4-maverick"
  const i=v.indexOf('/');
  if(i<0)return v;
  const prov=v.substring(0,i);
  let model=v.substring(i+1);
  // Strip org prefix (meta-llama/)
  const j=model.lastIndexOf('/');
  if(j>=0)model=model.substring(j+1);
  // Shorten long model names
  model=model.replace(/-instruct$/,'').replace(/-128e$/,'').replace(/-16e$/,'').replace(/-17b$/,'');
  return prov+': '+model;
}
function updateXaiMode(){
  const sttSel=document.getElementById('stt');
  const isXai=sttSel.value==='xai-realtime';
  document.getElementById('llm').disabled=isXai;
  document.getElementById('tts').disabled=isXai;
  document.getElementById('llm').style.opacity=isXai?'0.4':'1';
  document.getElementById('tts').style.opacity=isXai?'0.4':'1';
}
fetch('/providers').then(r=>r.json()).then(d=>{
  ['stt','llm','tts','orch'].forEach(k=>{
    const sel=document.getElementById(k);
    (d.available[k]||[]).forEach(v=>{
      const o=document.createElement('option');
      o.value=v;
      o.textContent=(k==='llm')?llmLabel(v):v;
      if(v===d.current[k])o.selected=true;
      sel.appendChild(o);
    });
    sel.onchange=()=>{
      const body={};body[k]=sel.value;
      fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      if(k==='stt')updateXaiMode();
    };
  });
  deskCb.checked=d.desk_mode;
  sttCb.checked=d.stt_enabled;
  ttsCb.checked=d.tts_enabled;
  panCb.checked=d.gimbal_pan_enabled;
  yoloCb.checked=d.yolo_enabled;
  updateXaiMode();
});
deskCb.onchange=()=>{
  fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({desk_mode:deskCb.checked})});
};
sttCb.onchange=()=>{
  fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stt_enabled:sttCb.checked})});
};
ttsCb.onchange=()=>{
  fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tts_enabled:ttsCb.checked})});
};
panCb.onchange=()=>{
  fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gimbal_pan_enabled:panCb.checked})});
};
yoloCb.onchange=()=>{
  fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({yolo_enabled:yoloCb.checked})});
};

// Kill switch
const killBtn=document.getElementById('killBtn');
let isKilled=false;
fetch('/kill').then(r=>r.json()).then(d=>{
  isKilled=d.killed;
  killBtn.classList.toggle('killed',isKilled);
  killBtn.textContent=isKilled?'RESUME':'KILL';
});
killBtn.onclick=()=>{
  const engage=!isKilled;
  fetch('/kill',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({engage})}).then(r=>r.json()).then(d=>{
    isKilled=d.killed;
    killBtn.classList.toggle('killed',isKilled);
    killBtn.textContent=isKilled?'RESUME':'KILL';
  });
};

// Restart service
const restartBtn=document.getElementById('restartBtn');
restartBtn.onclick=()=>{
  if(!confirm('Restart the rover service?')) return;
  restartBtn.textContent='RESTARTING...';
  restartBtn.disabled=true;
  fetch('/restart',{method:'POST'}).then(()=>{
    setTimeout(()=>location.reload(), 5000);
  }).catch(()=>{
    setTimeout(()=>location.reload(), 5000);
  });
};

// SSE log stream with filters
const LOG_CATS=['llm','serial','heard','speak','plan','stuck','interrupt','system','error','nav','orchestrator','observe','decide','bash','file','follow','battery','backup','detect','yolo_corr','imu','track','floor_nav','room'];
const logHidden=new Set(['serial']);
const filtersDiv=document.getElementById('log-filters');
LOG_CATS.forEach(cat=>{
  const lbl=document.createElement('label');
  const cb=document.createElement('input');
  cb.type='checkbox';cb.checked=!logHidden.has(cat);
  cb.onchange=()=>{
    if(cb.checked)logHidden.delete(cat);else logHidden.add(cat);
    document.querySelectorAll('#log div').forEach(el=>{
      const c=el.dataset.cat;if(c)el.style.display=logHidden.has(c)?'none':'';
    });
  };
  lbl.appendChild(cb);lbl.appendChild(document.createTextNode(cat));
  filtersDiv.appendChild(lbl);
});
const es=new EventSource('/events');
es.onmessage=e=>{
  const d=JSON.parse(e.data);
  const t=new Date(d.ts*1000).toLocaleTimeString();
  const line=document.createElement('div');
  line.dataset.cat=d.cat;
  line.style.display=logHidden.has(d.cat)?'none':'';
  line.innerHTML='<span class="ts">'+t+'</span> <span class="cat-'+d.cat+'">['+d.cat+']</span> '+escHtml(d.data);
  log.appendChild(line);
  if(log.children.length>1000)log.removeChild(log.firstChild);
  log.scrollTop=log.scrollHeight;
};

function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

function sendCmd(){
  const t=cmd.value.trim();
  if(!t)return;
  fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
  cmd.value='';
}
cmd.addEventListener('keydown',e=>{if(e.key==='Enter')sendCmd()});

// ── Gimbal Trackpad ──
let gimbalPan=0, gimbalTilt=0;
let gimbalDragging=false;
let gimbalLastSend=0;
let gimbalPanOffset=0; // loaded from /nav/config
fetch('/nav/config').then(r=>r.json()).then(d=>{gimbalPanOffset=d.gimbal_pan_offset||0;}).catch(()=>{});
const GIMBAL_PAN_MIN=-180, GIMBAL_PAN_MAX=180;
const GIMBAL_TILT_MIN=-30, GIMBAL_TILT_MAX=90;
const GIMBAL_SEND_INTERVAL=100; // 10Hz throttle

function sizeGimbalCanvas(){
  const rect=gimbalCanvas.getBoundingClientRect();
  gimbalCanvas.width=rect.width;
  gimbalCanvas.height=rect.height;
  drawGimbal(gimbalPan, gimbalTilt);
}
sizeGimbalCanvas();
window.addEventListener('resize', sizeGimbalCanvas);

function drawGimbal(pan, tilt){
  const W=gimbalCanvas.width, H=gimbalCanvas.height;
  gctx.clearRect(0,0,W,H);

  // Grid lines — pan every 45°, tilt every 30°
  gctx.strokeStyle='#1a2a4a';gctx.lineWidth=0.5;
  gctx.fillStyle='#2a2a4a';gctx.font='9px monospace';
  for(let p=-180;p<=180;p+=45){
    const x=((p-GIMBAL_PAN_MIN)/(GIMBAL_PAN_MAX-GIMBAL_PAN_MIN))*W;
    gctx.beginPath();gctx.moveTo(x,0);gctx.lineTo(x,H);gctx.stroke();
    if(p%90===0){gctx.textAlign='center';gctx.fillText(p+'°',x,H-3);}
  }
  for(let t=-30;t<=90;t+=30){
    const y=((GIMBAL_TILT_MAX-t)/(GIMBAL_TILT_MAX-GIMBAL_TILT_MIN))*H;
    gctx.beginPath();gctx.moveTo(0,y);gctx.lineTo(W,y);gctx.stroke();
    gctx.textAlign='left';gctx.fillText(t+'°',3,y-2);
  }

  // Center crosshair
  const cx=W/2, cy=((GIMBAL_TILT_MAX-0)/(GIMBAL_TILT_MAX-GIMBAL_TILT_MIN))*H;
  gctx.strokeStyle='#333';gctx.lineWidth=1;
  gctx.beginPath();gctx.moveTo(cx,0);gctx.lineTo(cx,H);gctx.stroke();
  gctx.beginPath();gctx.moveTo(0,cy);gctx.lineTo(W,cy);gctx.stroke();

  // Position dot
  const px=((pan-GIMBAL_PAN_MIN)/(GIMBAL_PAN_MAX-GIMBAL_PAN_MIN))*W;
  const py=((GIMBAL_TILT_MAX-tilt)/(GIMBAL_TILT_MAX-GIMBAL_TILT_MIN))*H;
  gctx.fillStyle='#e94560';
  gctx.beginPath();gctx.arc(px,py,7,0,Math.PI*2);gctx.fill();
  gctx.fillStyle='#fff';
  gctx.beginPath();gctx.arc(px,py,3,0,Math.PI*2);gctx.fill();
}

function sizeRoomMapCanvas(){
  const rect=roomMapCanvas.getBoundingClientRect();
  roomMapCanvas.width=rect.width;
  roomMapCanvas.height=Math.round(rect.width*0.62);
}

function drawRoomMap(scan){
  const W=roomMapCanvas.width, H=roomMapCanvas.height;
  if(!W||!H)return;
  roomMapCtx.fillStyle='#050511';
  roomMapCtx.fillRect(0,0,W,H);

  const ox=W*0.5, oy=H*0.84;
  const elements=(scan&&Array.isArray(scan.elements))?scan.elements:[];
  const maxDist=Math.max(2.5, ...elements.map(e=>Math.abs(e.distance_m||0)));
  const range=Math.min(6.0, maxDist+0.6);
  const scale=(oy-14)/range;

  // Distance rings
  roomMapCtx.strokeStyle='rgba(88,120,160,0.35)';
  roomMapCtx.lineWidth=1;
  roomMapCtx.fillStyle='#6a86a8';
  roomMapCtx.font='10px monospace';
  for(let m=1;m<=Math.floor(range);m++){
    const r=m*scale;
    roomMapCtx.beginPath();
    roomMapCtx.arc(ox,oy,r,Math.PI,2*Math.PI);
    roomMapCtx.stroke();
    roomMapCtx.fillText(m+'m', ox+r+4, oy-2);
  }

  // Rover marker
  roomMapCtx.fillStyle='#f6f7fb';
  roomMapCtx.beginPath();
  roomMapCtx.moveTo(ox, oy-10);
  roomMapCtx.lineTo(ox-7, oy+8);
  roomMapCtx.lineTo(ox+7, oy+8);
  roomMapCtx.closePath();
  roomMapCtx.fill();

  // Elements
  roomMapCtx.font='11px monospace';
  elements.slice(0,24).forEach(e=>{
    const x=ox+(Number(e.x)||0)*scale;
    const y=oy-(Number(e.y)||0)*scale;
    const span=Math.max(Number(e.width_m)||0.2, Number(e.depth_m)||0.2);
    const r=Math.max(3, Math.min(18, span*scale*0.5));
    const conf=Math.max(0.2, Math.min(1.0, Number(e.confidence)||0.4));
    roomMapCtx.fillStyle='rgba(66,185,131,'+conf.toFixed(2)+')';
    roomMapCtx.beginPath();
    roomMapCtx.arc(x,y,r,0,Math.PI*2);
    roomMapCtx.fill();
    roomMapCtx.fillStyle='#d8fff0';
    const sizeTxt=((Number(e.width_m)||0.2).toFixed(1)+'x'+(Number(e.depth_m)||0.2).toFixed(1)+'m');
    roomMapCtx.fillText((e.name||'obj')+' '+sizeTxt, x+r+3, y-4);
  });

  if(scan&&scan.room_guess){
    const g=scan.room_guess;
    const name=g.name||'unknown';
    const conf=Number(g.confidence||0);
    roomGuessEl.textContent='Room guess: '+name+' ('+(conf*100).toFixed(0)+'%)';
  }else{
    roomGuessEl.textContent='Room guess: unknown';
  }
}

function canvasToGimbal(e){
  const rect=gimbalCanvas.getBoundingClientRect();
  const x=e.clientX-rect.left, y=e.clientY-rect.top;
  const pan=GIMBAL_PAN_MIN+(x/rect.width)*(GIMBAL_PAN_MAX-GIMBAL_PAN_MIN);
  const tilt=GIMBAL_TILT_MAX-(y/rect.height)*(GIMBAL_TILT_MAX-GIMBAL_TILT_MIN);
  return {
    pan: Math.round(Math.max(GIMBAL_PAN_MIN, Math.min(GIMBAL_PAN_MAX, pan))),
    tilt: Math.round(Math.max(GIMBAL_TILT_MIN, Math.min(GIMBAL_TILT_MAX, tilt)))
  };
}

function sendGimbal(pan, tilt){
  const now=Date.now();
  if(now-gimbalLastSend<GIMBAL_SEND_INTERVAL)return;
  gimbalLastSend=now;
  fetch('/esp',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({T:133,X:pan,Y:tilt,SPD:300,ACC:100})});
}

gimbalCanvas.addEventListener('pointerdown',e=>{
  gimbalDragging=true;
  gimbalCanvas.setPointerCapture(e.pointerId);
  const g=canvasToGimbal(e);
  gimbalPan=g.pan; gimbalTilt=g.tilt;
  drawGimbal(gimbalPan, gimbalTilt);
  sendGimbal(gimbalPan, gimbalTilt);
  document.getElementById('gimbal-readout').textContent='Pan: '+gimbalPan+'°  Tilt: '+gimbalTilt+'°';
});

gimbalCanvas.addEventListener('pointermove',e=>{
  if(!gimbalDragging)return;
  const g=canvasToGimbal(e);
  gimbalPan=g.pan; gimbalTilt=g.tilt;
  drawGimbal(gimbalPan, gimbalTilt);
  sendGimbal(gimbalPan, gimbalTilt);
  document.getElementById('gimbal-readout').textContent='Pan: '+gimbalPan+'°  Tilt: '+gimbalTilt+'°';
});

gimbalCanvas.addEventListener('pointerup',e=>{
  gimbalDragging=false;
  // Send final position (bypass throttle)
  gimbalLastSend=0;
  sendGimbal(gimbalPan, gimbalTilt);
});

document.getElementById('gimbal-center').addEventListener('click',()=>{
  gimbalPan=0; gimbalTilt=0;
  drawGimbal(0,0);
  gimbalLastSend=0;
  fetch('/esp',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({T:133,X:0,Y:0,SPD:200,ACC:10})});
  document.getElementById('gimbal-readout').textContent='Pan: 0°  Tilt: 0°';
});

// Poll landmarks for readout (heading, voltage, actual gimbal position)
setInterval(()=>{
  fetch('/landmarks').then(r=>r.json()).then(data=>{
    const rv=data.rover||{};
    const parts=['Pan: '+gimbalPan+'°  Tilt: '+gimbalTilt+'°'];
    if(rv.heading!==undefined)parts.push('Hdg: '+Math.round(rv.heading)+'°');
    if(rv.voltage!==undefined)parts.push(rv.voltage.toFixed(1)+'V');
    document.getElementById('gimbal-readout').textContent=parts.join('  ');
    // Update position from actual gimbal feedback if not dragging
    if(!gimbalDragging && rv.gimbal_pan!==undefined){
      gimbalPan=Math.round(rv.gimbal_pan);
      gimbalTilt=Math.round(rv.gimbal_tilt||0);
      drawGimbal(gimbalPan, gimbalTilt);
    }
    drawRoomMap(data.room_scan||null);
  }).catch(()=>{});
},500);

drawGimbal(0,0);
sizeRoomMapCanvas();
drawRoomMap(null);
window.addEventListener('resize', ()=>{
  sizeRoomMapCanvas();
  fetch('/landmarks').then(r=>r.json()).then(data=>drawRoomMap(data.room_scan||null)).catch(()=>drawRoomMap(null));
});

// ── Annotation Mode ──────────────────────────────────────────────
const annToggle=document.getElementById('annToggle');
const streamImg=document.getElementById('streamImg');
const annCanvas=document.getElementById('annotateCanvas');
const annBar=document.getElementById('annotateBar');
const actx=annCanvas.getContext('2d');
const annClassSel=document.getElementById('annClass');
const annNewClass=document.getElementById('annNewClass');
const trainStatusEl=document.getElementById('trainStatus');

let annMode=false;
let annImage=null;      // base64 of captured image
let annImageObj=null;    // Image element for drawing
let annBoxes=[];         // [{name,bbox:[x1,y1,x2,y2],source:'yolo'|'manual'}]
let annSelected=-1;      // index of selected box
let annDragging=false;
let annDragStart=null;
let annDragEnd=null;
let trainPollId=null;

// Toggle annotate mode
annToggle.onclick=()=>{
  annMode=!annMode;
  annToggle.classList.toggle('ann-btn-active',annMode);
  streamImg.style.display=annMode?'none':'';
  annCanvas.style.display=annMode?'block':'none';
  annBar.style.display=annMode?'flex':'none';
  if(annMode){
    // Load class list
    fetch('/annotate/classes').then(r=>r.json()).then(d=>{
      annClassSel.innerHTML='';
      (d.classes||[]).forEach(c=>{
        const o=document.createElement('option');
        o.value=c;o.textContent=c;
        annClassSel.appendChild(o);
      });
    });
  }
};

// Helper: get mouse position in canvas coordinates (640x480)
function annMousePos(e){
  const r=annCanvas.getBoundingClientRect();
  const sx=640/r.width, sy=480/r.height;
  return {x:Math.round((e.clientX-r.left)*sx), y:Math.round((e.clientY-r.top)*sy)};
}

// Get current class name (new class input takes priority)
function annCurrentClass(){
  const nc=annNewClass.value.trim();
  return nc||annClassSel.value||'object';
}

// Draw everything on canvas
function annRedraw(){
  actx.clearRect(0,0,640,480);
  if(annImageObj) actx.drawImage(annImageObj,0,0,640,480);
  annBoxes.forEach((b,i)=>{
    const [x1,y1,x2,y2]=b.bbox;
    const sel=(i===annSelected);
    actx.strokeStyle=sel?'#ffff00':b.source==='yolo'?'#00ff00':'#4488ff';
    actx.lineWidth=sel?3:2;
    if(b.source==='manual'){actx.setLineDash([6,3]);}else{actx.setLineDash([]);}
    actx.strokeRect(x1,y1,x2-x1,y2-y1);
    actx.setLineDash([]);
    // Label
    const label=b.name+(b.conf!=null?' '+Math.round(b.conf*100)+'%':'');
    actx.font='bold 13px monospace';
    const tw=actx.measureText(label).width;
    const ly=y1>18?y1-4:y2+14;
    actx.fillStyle='rgba(0,0,0,0.6)';
    actx.fillRect(x1,ly-12,tw+6,16);
    actx.fillStyle=sel?'#ffff00':'#fff';
    actx.fillText(label,x1+3,ly);
  });
  // Rubber-band while dragging
  if(annDragging&&annDragStart&&annDragEnd){
    actx.strokeStyle='#ffff00';actx.lineWidth=1;actx.setLineDash([4,4]);
    const x=Math.min(annDragStart.x,annDragEnd.x),y=Math.min(annDragStart.y,annDragEnd.y);
    const w=Math.abs(annDragEnd.x-annDragStart.x),h=Math.abs(annDragEnd.y-annDragStart.y);
    actx.strokeRect(x,y,w,h);
    actx.setLineDash([]);
  }
}

// Hit test: find box containing point (reverse z-order)
function annHitTest(px,py){
  for(let i=annBoxes.length-1;i>=0;i--){
    const [x1,y1,x2,y2]=annBoxes[i].bbox;
    if(px>=x1&&px<=x2&&py>=y1&&py<=y2) return i;
  }
  return -1;
}

// Mouse events on canvas
annCanvas.onmousedown=(e)=>{
  if(!annImageObj) return;
  const p=annMousePos(e);
  const hit=annHitTest(p.x,p.y);
  if(hit>=0){
    annSelected=hit;
    annClassSel.value=annBoxes[hit].name;
    annRedraw();
  } else {
    annSelected=-1;
    annDragging=true;
    annDragStart=p;
    annDragEnd=p;
    annRedraw();
  }
};
annCanvas.onmousemove=(e)=>{
  if(!annDragging) return;
  annDragEnd=annMousePos(e);
  annRedraw();
};
annCanvas.onmouseup=(e)=>{
  if(!annDragging) return;
  annDragging=false;
  annDragEnd=annMousePos(e);
  const x1=Math.min(annDragStart.x,annDragEnd.x),y1=Math.min(annDragStart.y,annDragEnd.y);
  const x2=Math.max(annDragStart.x,annDragEnd.x),y2=Math.max(annDragStart.y,annDragEnd.y);
  if((x2-x1)>10&&(y2-y1)>10){
    const cls=annCurrentClass();
    annBoxes.push({name:cls,bbox:[x1,y1,x2,y2],conf:null,source:'manual'});
    annSelected=annBoxes.length-1;
    // Add new class to dropdown if needed
    if(annNewClass.value.trim()){
      let found=false;
      for(let o of annClassSel.options){if(o.value===cls){found=true;break;}}
      if(!found){const o=document.createElement('option');o.value=cls;o.textContent=cls;annClassSel.appendChild(o);}
      annClassSel.value=cls;
      annNewClass.value='';
    }
  }
  annDragStart=null;annDragEnd=null;
  annRedraw();
};

// Capture button
document.getElementById('annCapture').onclick=()=>{
  fetch('/annotate/snap').then(r=>r.json()).then(d=>{
    annImage=d.image_b64;
    annBoxes=d.detections.map(det=>({name:det.name,bbox:det.bbox,conf:det.conf,source:'yolo'}));
    annSelected=-1;
    // Update class dropdown
    annClassSel.innerHTML='';
    (d.classes||[]).forEach(c=>{
      const o=document.createElement('option');o.value=c;o.textContent=c;annClassSel.appendChild(o);
    });
    // Load image onto canvas
    annImageObj=new Image();
    annImageObj.onload=()=>annRedraw();
    annImageObj.src='data:image/jpeg;base64,'+annImage;
  }).catch(e=>console.error('Capture failed',e));
};

// Delete selected box
document.getElementById('annDelete').onclick=()=>{
  if(annSelected>=0&&annSelected<annBoxes.length){
    annBoxes.splice(annSelected,1);
    annSelected=-1;
    annRedraw();
  }
};

// Apply label to selected box
document.getElementById('annApply').onclick=()=>{
  if(annSelected>=0&&annSelected<annBoxes.length){
    const cls=annCurrentClass();
    annBoxes[annSelected].name=cls;
    // Add to dropdown if new
    if(annNewClass.value.trim()){
      let found=false;
      for(let o of annClassSel.options){if(o.value===cls){found=true;break;}}
      if(!found){const o=document.createElement('option');o.value=cls;o.textContent=cls;annClassSel.appendChild(o);}
      annClassSel.value=cls;
      annNewClass.value='';
    }
    annRedraw();
  }
};

// Save annotations
document.getElementById('annSave').onclick=()=>{
  if(!annImage||annBoxes.length===0){alert('Capture an image and add annotations first.');return;}
  const annotations=annBoxes.map(b=>({name:b.name,bbox:b.bbox}));
  fetch('/annotate/save',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image_b64:annImage,annotations})
  }).then(r=>r.json()).then(d=>{
    if(d.ok) trainStatusEl.textContent='Saved '+d.image+' ('+d.count+' boxes)';
    else trainStatusEl.textContent='Save error: '+d.error;
  }).catch(e=>{trainStatusEl.textContent='Save failed';});
};

// Train button
document.getElementById('annTrain').onclick=()=>{
  if(trainPollId){alert('Training already in progress.');return;}
  fetch('/annotate/train',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({epochs:50,batch:8,base_model:'yolo26s.pt'})
  }).then(r=>r.json()).then(d=>{
    if(!d.ok){trainStatusEl.textContent='Error: '+d.error;return;}
    trainStatusEl.textContent='Starting training...';
    trainPollId=setInterval(()=>{
      fetch('/annotate/train_status').then(r=>r.json()).then(s=>{
        if(s.state==='running') trainStatusEl.textContent='Training: epoch '+s.epoch+'/'+s.total_epochs+' mAP50='+s.map50;
        else if(s.state==='preparing') trainStatusEl.textContent='Preparing dataset...';
        else if(s.state==='done'){
          trainStatusEl.textContent='Done! mAP50='+s.map50;
          clearInterval(trainPollId);trainPollId=null;
        } else if(s.state==='error'){
          trainStatusEl.textContent='Error: '+(s.error||'unknown');
          clearInterval(trainPollId);trainPollId=null;
        }
      });
    },3000);
  });
};
// ── Exploration Grid ──────────────────────────────────────
(function(){
  const tog=document.getElementById('gridToggle');
  const panel=document.getElementById('gridPanel');
  const img=document.getElementById('gridImg');
  let timer=null;
  function refresh(){img.src='/nav/grid?t='+Date.now();}
  tog.onclick=()=>{
    const vis=panel.style.display==='none';
    panel.style.display=vis?'block':'none';
    if(vis){refresh();timer=setInterval(refresh,2000);}
    else if(timer){clearInterval(timer);timer=null;}
  };
})();
// ── Turn Calibration ──────────────────────────────────────
(function(){
  const tog=document.getElementById('calibToggle');
  const panel=document.getElementById('calibPanel');
  const curEl=document.getElementById('calibCurrent');
  const prompt=document.getElementById('calibPrompt');
  const actualIn=document.getElementById('calibActual');
  const resultEl=document.getElementById('calibResult');
  let cmdDeg=0,curDPS=200;
  tog.onclick=()=>{
    const vis=panel.style.display==='none';
    panel.style.display=vis?'block':'none';
    if(vis)fetch('/nav/config').then(r=>r.json()).then(d=>{
      curDPS=d.turn_rate_dps||200;curEl.textContent=curDPS.toFixed(1);});
  };
  document.querySelectorAll('.calib-btn').forEach(b=>{
    b.onclick=()=>{
      cmdDeg=parseInt(b.dataset.deg);
      actualIn.value=cmdDeg;
      prompt.style.display='block';
      resultEl.textContent='Spinning...';
      fetch('/nav/test-spin',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({degrees:cmdDeg*parseInt(b.dataset.dir)})
      }).then(r=>r.json()).then(d=>{
        resultEl.textContent='Done ('+d.time_s.toFixed(2)+'s). Enter actual angle:';
      }).catch(e=>{resultEl.textContent='Error: '+e;});
    };
  });
  document.getElementById('calibApply').onclick=()=>{
    const actual=parseFloat(actualIn.value);
    if(!actual||actual<=0){resultEl.textContent='Enter a valid angle';return;}
    const newDPS=curDPS*(actual/cmdDeg);
    fetch('/nav/config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({turn_rate_dps:Math.round(newDPS*10)/10})
    }).then(r=>r.json()).then(d=>{
      const oldDPS=curDPS;curDPS=d.turn_rate_dps;curEl.textContent=curDPS.toFixed(1);
      resultEl.textContent='Saved! '+oldDPS.toFixed(1)+' -> '+curDPS.toFixed(1)+' deg/s';
      prompt.style.display='none';
    });
  };
})();
// ── Gimbal Pan Offset ─────────────────────────────────────
(function(){
  const curEl=document.getElementById('gimbalOffCurrent');
  const valIn=document.getElementById('gimbalOffVal');
  let off=0;
  function load(){
    fetch('/nav/config').then(r=>r.json()).then(d=>{
      off=d.gimbal_pan_offset||0;curEl.textContent=off.toFixed(1);valIn.value=off;});
  }
  // Load when calibration panel opens (reuse toggle)
  document.getElementById('calibToggle').addEventListener('click',load);
  function save(v){
    fetch('/nav/config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({gimbal_pan_offset:v})
    }).then(r=>r.json()).then(d=>{
      off=d.gimbal_pan_offset||0;curEl.textContent=off.toFixed(1);valIn.value=off;
    });
  }
  document.getElementById('gimbalOffMinus').onclick=()=>{save(off-1);};
  document.getElementById('gimbalOffPlus').onclick=()=>{save(off+1);};
  document.getElementById('gimbalOffSave').onclick=()=>{
    const v=parseFloat(valIn.value)||0;
    save(v);gimbalPanOffset=v;  // update global for trackpad+center
  };
  document.getElementById('gimbalOffTest').onclick=()=>{
    // Send gimbal to 0+offset to test if it looks centered
    fetch('/esp',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({"T":133,"X":parseFloat(valIn.value)||0,"Y":0,"SPD":200,"ACC":10})});
  };
})();
</script>
</body>
</html>"""


# ── Annotation / Training helpers ───────────────────────────────────

def _ensure_dataset_dirs():
    """Create dataset directory structure if needed."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(_DATASET_DIR, sub), exist_ok=True)


def _load_dataset_yaml():
    """Load dataset.yaml, return dict. Creates default if missing."""
    path = os.path.join(_DATASET_DIR, "dataset.yaml")
    _ensure_dataset_dirs()
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    default = {
        "path": _DATASET_DIR,
        "train": "images/train",
        "val": "images/val",
        "nc": 0,
        "names": [],
    }
    with open(path, "w") as f:
        yaml.dump(default, f, default_flow_style=False)
    return default


def _save_dataset_yaml(cfg):
    """Write dataset.yaml (append-only class list — never reorder)."""
    cfg["nc"] = len(cfg.get("names", []))
    cfg["path"] = _DATASET_DIR
    cfg["train"] = "images/train"
    cfg["val"] = "images/val"
    path = os.path.join(_DATASET_DIR, "dataset.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _get_class_id(cfg, name):
    """Return class id for name, appending if new. Mutates cfg in place."""
    names = cfg.setdefault("names", [])
    if name in names:
        return names.index(name)
    names.append(name)
    cfg["nc"] = len(names)
    return len(names) - 1


def _next_img_index():
    """Find next available img_NNNN index in dataset/images/train/."""
    train_dir = os.path.join(_DATASET_DIR, "images", "train")
    existing = globmod.glob(os.path.join(train_dir, "img_*.jpg"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        base = os.path.basename(p)
        try:
            nums.append(int(base.replace("img_", "").replace(".jpg", "")))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def _split_val(ratio=0.2):
    """Copy ~ratio of train images+labels to val/ (non-destructive)."""
    img_train = os.path.join(_DATASET_DIR, "images", "train")
    lbl_train = os.path.join(_DATASET_DIR, "labels", "train")
    img_val = os.path.join(_DATASET_DIR, "images", "val")
    lbl_val = os.path.join(_DATASET_DIR, "labels", "val")
    os.makedirs(img_val, exist_ok=True)
    os.makedirs(lbl_val, exist_ok=True)
    imgs = globmod.glob(os.path.join(img_train, "*.jpg"))
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * ratio))
    for img_path in imgs[:n_val]:
        base = os.path.basename(img_path)
        lbl_name = base.replace(".jpg", ".txt")
        shutil.copy2(img_path, os.path.join(img_val, base))
        lbl_src = os.path.join(lbl_train, lbl_name)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(lbl_val, lbl_name))


def _run_training(epochs, batch, base_model):
    """Training thread target. Updates _training_status dict."""
    global _training_status
    try:
        _training_status = {"state": "preparing", "epoch": 0,
                            "total_epochs": epochs, "map50": 0.0}
        if _log_event:
            _log_event("system", f"YOLO training starting: {base_model}, "
                       f"{epochs} epochs, batch {batch}")

        _split_val()

        # Remove stale val cache
        for cache in globmod.glob(os.path.join(_DATASET_DIR, "**", "*.cache"),
                                  recursive=True):
            os.remove(cache)

        from ultralytics import YOLO
        model = YOLO(base_model)

        def _on_epoch_end(trainer):
            ep = trainer.epoch + 1
            metrics = trainer.metrics or {}
            m50 = metrics.get("metrics/mAP50(B)", 0.0)
            _training_status.update({
                "state": "running", "epoch": ep,
                "total_epochs": epochs, "map50": round(float(m50), 4),
            })
            if _log_event and ep % 5 == 0:
                _log_event("system", f"Training epoch {ep}/{epochs} "
                           f"mAP50={m50:.3f}")

        model.add_callback("on_train_epoch_end", _on_epoch_end)

        yaml_path = os.path.join(_DATASET_DIR, "dataset.yaml")
        model.train(
            data=yaml_path, epochs=epochs, batch=batch,
            imgsz=640, device=0, workers=2, patience=20,
            project=os.path.join(_MODELS_DIR, "runs"),
            name="yolo26s_custom",
            exist_ok=True,
        )

        # Copy best weights
        best_src = os.path.join(_MODELS_DIR, "runs", "yolo26s_custom",
                                "weights", "best.pt")
        best_dst = os.path.join(_MODELS_DIR, "yolo26s-custom.pt")
        if os.path.exists(best_src):
            shutil.copy2(best_src, best_dst)

        final_map = _training_status.get("map50", 0.0)
        _training_status = {
            "state": "done", "epoch": epochs,
            "total_epochs": epochs, "map50": final_map,
        }
        if _log_event:
            _log_event("system", f"Training complete! mAP50={final_map:.3f} "
                       f"→ {best_dst}")

    except Exception as e:
        _training_status = {"state": "error", "error": str(e)}
        if _log_event:
            _log_event("error", f"Training failed: {e}")
        traceback.print_exc()


# ── HTTP Handler ────────────────────────────────────────────────────

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/snap":
            jpg = _camera_ref.get_jpeg() if _camera_ref else None
            if jpg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_error(503)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpg = (_camera_ref.get_overlay_jpeg()
                           if _camera_ref else None)
                    if jpg:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.1)  # ~10 fps stream
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            last_ts = time.time() - 60  # send last 60s of history
            try:
                while True:
                    events = _get_log_events_since(last_ts)
                    for ev in events:
                        data = json.dumps(ev)
                        self.wfile.write(f"data: {data}\n\n".encode())
                        last_ts = ev["ts"]
                    self.wfile.flush()
                    time.sleep(0.3)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/providers":
            body = json.dumps(_get_providers())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/landmarks":
            body = json.dumps(_get_map_state())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/nav/grid":
            import navigator
            grid = navigator._exploration_grid
            if grid:
                jpg = grid.render_image(scale=4)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self.send_error(503, "No exploration grid")
        elif self.path == "/nav/config":
            import navigator
            from navigator import _load_nav_config
            cfg = _load_nav_config()
            cfg.setdefault("turn_rate_dps", navigator.TURN_RATE_DPS)
            cfg.setdefault("gimbal_pan_offset", navigator.GIMBAL_PAN_OFFSET)
            body = json.dumps(cfg)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/lessons":
            body = json.dumps(lessons.load_lessons())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/annotate/snap":
            # Raw JPEG + run YOLO detection on this exact frame
            jpg = _camera_ref.get_jpeg() if _camera_ref else None
            if not jpg:
                self.send_error(503)
                return
            img_b64 = base64.b64encode(jpg).decode("ascii")
            dets = []
            if _detector_ref:
                import numpy as np
                import cv2
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    det_list = _detector_ref.detect(frame)
                    for d in det_list:
                        dets.append({
                            "name": d.get("name", ""),
                            "conf": round(float(d.get("conf", 0)), 3),
                            "bbox": [int(v) for v in d.get("bbox", [0,0,0,0])],
                        })
            # Class list from dataset.yaml
            cfg = _load_dataset_yaml()
            classes = cfg.get("names", [])
            # Also include detector classes if available
            if _detector_ref and hasattr(_detector_ref, "class_names"):
                for cn in _detector_ref.class_names:
                    if cn not in classes:
                        classes.append(cn)
            body = json.dumps({
                "image_b64": img_b64,
                "detections": dets,
                "classes": classes,
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/annotate/classes":
            cfg = _load_dataset_yaml()
            classes = cfg.get("names", [])
            if _detector_ref and hasattr(_detector_ref, "class_names"):
                for cn in _detector_ref.class_names:
                    if cn not in classes:
                        classes.append(cn)
            body = json.dumps({"classes": classes})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/annotate/train_status":
            body = json.dumps(_training_status)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/kill":
            body = json.dumps({"killed": _get_killed() if _get_killed else False})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len) if content_len else b"{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400)
            return

        if self.path == "/command":
            text = data.get("text", "").strip()
            if text:
                _log_event("heard", f"(web) {text}")
                if _plan_active.is_set():
                    kind = _classify_interrupt(text)
                    _log_event("interrupt", f"{kind}: {text}")
                    if kind == "stop":
                        _stop_event.set()
                        if _ser_ref:
                            try:
                                _ser_ref.stop()
                            except Exception:
                                pass
                    elif kind == "cancel":
                        _stop_event.set()
                    elif kind == "override":
                        _stop_event.set()
                        _command_queue.put(text)
                    elif kind == "inject":
                        _interrupt_queue.put(text)
                    elif kind == "feedback_negative":
                        _interrupt_queue.put(text)
                else:
                    _command_queue.put(text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        elif self.path == "/esp":
            # Raw ESP32 command(s) — bypass LLM, send directly to serial
            cmds = data.get("commands", [])
            if not cmds and isinstance(data.get("T"), int):
                cmds = [data]  # single command object
            results = []
            if _ser_ref:
                for cmd in cmds:
                    if isinstance(cmd, dict) and "_pause" in cmd:
                        import time as _t
                        _t.sleep(min(cmd["_pause"], 5.0))
                        results.append({"paused": cmd["_pause"]})
                    else:
                        try:
                            _ser_ref.send(cmd)
                            results.append({"sent": cmd})
                        except Exception as e:
                            results.append({"error": str(e)})
            else:
                results = [{"error": "no serial connection"}]
            body = json.dumps({"ok": True, "results": results})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/providers":
            for kind in ("stt", "llm", "tts", "orch"):
                name = data.get(kind)
                if name:
                    try:
                        _set_provider(kind, name)
                    except Exception as e:
                        _log_event("error", f"Provider switch failed: {e}")
            if "desk_mode" in data:
                _set_desk_mode(bool(data["desk_mode"]))
            if "stt_enabled" in data:
                _set_stt_enabled(bool(data["stt_enabled"]))
            if "tts_enabled" in data:
                _set_tts_enabled(bool(data["tts_enabled"]))
            if "gimbal_pan_enabled" in data:
                _set_gimbal_pan_enabled(bool(data["gimbal_pan_enabled"]))
            if "yolo_enabled" in data:
                _set_yolo_enabled(bool(data["yolo_enabled"]))
            body = json.dumps(_get_providers())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/nav/test-spin":
            import navigator
            degrees = data.get("degrees", 90)
            turn_time = abs(degrees) / navigator.TURN_RATE_DPS
            sign = 1 if degrees > 0 else -1
            if _ser_ref:
                _ser_ref.send({"T": 1,
                               "L": round(navigator.TURN_SPEED * sign, 3),
                               "R": round(-navigator.TURN_SPEED * sign, 3)})
                import time as _t
                _t.sleep(turn_time)
                _ser_ref.send({"T": 1, "L": 0, "R": 0})
            body = json.dumps({"ok": True, "time_s": round(turn_time, 3),
                               "degrees": degrees})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/nav/config":
            import navigator
            from navigator import _load_nav_config, _save_nav_config
            cfg = _load_nav_config()
            changed = False
            if "turn_rate_dps" in data:
                new_dps = float(data["turn_rate_dps"])
                navigator.TURN_RATE_DPS = new_dps
                cfg["turn_rate_dps"] = new_dps
                changed = True
                if _log_event:
                    _log_event("system",
                               f"TURN_RATE_DPS calibrated to {new_dps:.1f}")
            if "gimbal_pan_offset" in data:
                new_off = float(data["gimbal_pan_offset"])
                navigator.GIMBAL_PAN_OFFSET = new_off
                cfg["gimbal_pan_offset"] = new_off
                changed = True
                if _log_event:
                    _log_event("system",
                               f"Gimbal pan offset set to {new_off:+.1f}°")
            if changed:
                cfg["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                _save_nav_config(cfg)
            body = json.dumps({
                "turn_rate_dps": navigator.TURN_RATE_DPS,
                "gimbal_pan_offset": navigator.GIMBAL_PAN_OFFSET,
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/annotate/save":
            # Save annotated image + labels to dataset/
            try:
                img_b64 = data.get("image_b64", "")
                annotations = data.get("annotations", [])
                if not img_b64 or not annotations:
                    self.send_error(400)
                    return
                img_bytes = base64.b64decode(img_b64)
                _ensure_dataset_dirs()
                idx = _next_img_index()
                img_name = f"img_{idx:04d}"
                img_path = os.path.join(_DATASET_DIR, "images", "train",
                                        f"{img_name}.jpg")
                lbl_path = os.path.join(_DATASET_DIR, "labels", "train",
                                        f"{img_name}.txt")
                # Decode image to get dimensions
                import numpy as np
                import cv2
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    self.send_error(400)
                    return
                h, w = img.shape[:2]
                # Save JPEG
                cv2.imwrite(img_path, img)
                # Build YOLO-format labels
                cfg = _load_dataset_yaml()
                lines = []
                for ann in annotations:
                    name = ann.get("name", "").strip()
                    bbox = ann.get("bbox", [])
                    if not name or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h
                    cid = _get_class_id(cfg, name)
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                _save_dataset_yaml(cfg)
                with open(lbl_path, "w") as f:
                    f.write("\n".join(lines) + "\n")
                body = json.dumps({"ok": True, "image": img_name,
                                   "count": len(lines)})
                if _log_event:
                    _log_event("system", f"Annotation saved: {img_name} "
                               f"({len(lines)} boxes)")
            except Exception as e:
                body = json.dumps({"ok": False, "error": str(e)})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/annotate/train":
            global _training_thread
            if (_training_thread is not None
                    and _training_thread.is_alive()):
                body = json.dumps({"ok": False,
                                   "error": "Training already running"})
            else:
                epochs = int(data.get("epochs", 50))
                batch = int(data.get("batch", 8))
                base_model = data.get("base_model", "yolo26s.pt")
                _training_thread = threading.Thread(
                    target=_run_training,
                    args=(epochs, batch, base_model),
                    daemon=True,
                )
                _training_thread.start()
                body = json.dumps({"ok": True, "epochs": epochs,
                                   "batch": batch,
                                   "base_model": base_model})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/kill":
            engage = data.get("engage", True)
            if _set_killed:
                _set_killed(engage)
            body = json.dumps({"ok": True, "killed": engage})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/restart":
            body = json.dumps({"ok": True})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
            # Restart after response is sent — use nohup+setsid so the
            # restart command survives the service stopping itself
            threading.Thread(target=lambda: (
                __import__('time').sleep(0.5),
                __import__('os').system(
                    'setsid systemctl restart rover-brain-llm.service &')
            ), daemon=True).start()
        else:
            self.send_error(404)

    def do_DELETE(self):
        if self.path == "/lessons":
            content_len = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_len) if content_len else b"{}"
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                self.send_error(400)
                return
            lesson_id = data.get("id")
            if lesson_id is not None:
                removed = lessons.delete_lesson(int(lesson_id))
                body = json.dumps({"ok": removed})
            else:
                body = json.dumps({"ok": False, "error": "missing id"})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress request logs


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_server(port=8090):
    """Start the threaded HTTP server. Call init() first."""
    server = ThreadedHTTPServer(("0.0.0.0", port), StreamHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    if _log_event:
        _log_event("system", f"Web UI: http://localhost:{port}/")
