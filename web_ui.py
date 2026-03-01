"""web_ui.py — Web server + dashboard.

Extracted from rover_brain_llm.py. Receives shared state refs at init time.
No circular imports — uses callbacks/refs passed in via init().
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import lessons

# ── Shared state refs (set via init()) ──────────────────────────────

_camera_ref = None
_ser_ref = None
_get_log_events_since = None
_get_map_state = None
_log_event = None
_set_provider = None
_get_providers = None    # () -> {"current": {...}, "available": {...}, "desk_mode": bool, "stt_enabled": bool}
_set_desk_mode = None    # (bool) -> None
_set_stt_enabled = None  # (bool) -> None
_plan_active = None      # threading.Event
_stop_event = None       # threading.Event
_command_queue = None     # queue.Queue
_interrupt_queue = None   # queue.Queue
_classify_interrupt = None  # fn(text) -> str


def init(*, camera, serial, get_log_events_since, get_map_state, log_event,
         set_provider, get_providers, set_desk_mode, set_stt_enabled,
         plan_active, stop_event, command_queue, interrupt_queue,
         classify_interrupt):
    """Wire up shared state references. Call once from main()."""
    global _camera_ref, _ser_ref, _get_log_events_since, _get_map_state
    global _log_event, _set_provider, _get_providers
    global _set_desk_mode, _set_stt_enabled
    global _plan_active, _stop_event, _command_queue, _interrupt_queue
    global _classify_interrupt
    _camera_ref = camera
    _ser_ref = serial
    _get_log_events_since = get_log_events_since
    _get_map_state = get_map_state
    _log_event = log_event
    _set_provider = set_provider
    _get_providers = get_providers
    _set_desk_mode = set_desk_mode
    _set_stt_enabled = set_stt_enabled
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
#main{display:flex;flex:1;overflow:hidden}
#leftcol{flex:0 0 auto;padding:8px;display:flex;flex-direction:column;gap:8px}
#leftcol img{max-width:640px;width:100%;border-radius:4px;border:1px solid #333}
#radar{width:100%;max-width:640px;aspect-ratio:1;border-radius:4px;border:1px solid #333;background:#0a0a1a}
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
#cmdbar{display:flex;padding:8px 12px;background:#16213e;gap:8px}
#cmdbar input{flex:1;background:#0f3460;color:#e0e0e0;border:1px solid #333;padding:8px 12px;border-radius:4px;font-family:inherit;font-size:14px}
#cmdbar input::placeholder{color:#555}
#cmdbar button{background:#e94560;color:#fff;border:none;padding:8px 20px;border-radius:4px;cursor:pointer;font-family:inherit;font-size:14px}
#cmdbar button:hover{background:#c73e54}
</style>
</head>
<body>
<div id="topbar">
  <label>STT <select id="stt"></select></label>
  <label>LLM <select id="llm"></select></label>
  <label>TTS <select id="tts"></select></label>
  <label style="margin-left:auto;cursor:pointer"><input type="checkbox" id="sttmode" checked> STT</label>
  <label style="cursor:pointer"><input type="checkbox" id="deskmode" checked> Desk Mode</label>
</div>
<div id="main">
  <div id="leftcol">
    <img src="/stream" alt="Camera">
    <canvas id="radar"></canvas>
  </div>
  <div id="logpanel"><div id="log"></div></div>
</div>
<div id="cmdbar">
  <input id="cmd" type="text" placeholder="Type a command..." autocomplete="off">
  <button onclick="sendCmd()">Send</button>
</div>
<script>
const log=document.getElementById('log');
const cmd=document.getElementById('cmd');
const radar=document.getElementById('radar');
const rctx=radar.getContext('2d');

// Load provider options + desk mode
const deskCb=document.getElementById('deskmode');
const sttCb=document.getElementById('sttmode');
fetch('/providers').then(r=>r.json()).then(d=>{
  ['stt','llm','tts'].forEach(k=>{
    const sel=document.getElementById(k);
    (d.available[k]||[]).forEach(v=>{
      const o=document.createElement('option');
      o.value=v;o.textContent=v;
      if(v===d.current[k])o.selected=true;
      sel.appendChild(o);
    });
    sel.onchange=()=>{
      const body={};body[k]=sel.value;
      fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    };
  });
  deskCb.checked=d.desk_mode;
  sttCb.checked=d.stt_enabled;
});
deskCb.onchange=()=>{
  fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({desk_mode:deskCb.checked})});
};
sttCb.onchange=()=>{
  fetch('/providers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stt_enabled:sttCb.checked})});
};

// SSE log stream
const es=new EventSource('/events');
es.onmessage=e=>{
  const d=JSON.parse(e.data);
  const t=new Date(d.ts*1000).toLocaleTimeString();
  const line=document.createElement('div');
  line.innerHTML='<span class="ts">'+t+'</span> <span class="cat-'+d.cat+'">['+d.cat+']</span> '+escHtml(d.data);
  log.appendChild(line);
  if(log.children.length>500)log.removeChild(log.firstChild);
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

// ── 2D Cartesian Map ──
const TC={wall:'#e74c3c',door:'#2ecc71',open:'#2ecc71',person:'#f39c12',furniture:'#9b59b6',object:'#5dade2'};
let mapScale=80; // px per meter

function sizeCanvas(){
  const rect=radar.getBoundingClientRect();
  radar.width=rect.width;radar.height=rect.height;
}
sizeCanvas();
window.addEventListener('resize',sizeCanvas);

function drawMap(data){
  const W=radar.width, H=radar.height;
  const rv=data.rover||{x:0,y:0,heading:0,gimbal_pan:0};
  const lms=data.landmarks||[];
  const s=mapScale; // px per meter
  // Camera centers on rover
  const ox=W/2-rv.x*s, oy=H/2+rv.y*s; // world→screen: sx=ox+wx*s, sy=oy-wy*s
  rctx.clearRect(0,0,W,H);

  // Grid lines (1m)
  rctx.strokeStyle='#1a2a4a';rctx.lineWidth=0.5;
  rctx.fillStyle='#2a2a4a';rctx.font='9px monospace';rctx.textAlign='left';
  const x0=Math.floor(-ox/s)-1, x1=Math.ceil((W-ox)/s)+1;
  const y0=Math.floor(-(H-oy)/s)-1, y1=Math.ceil(oy/s)+1;
  for(let gx=x0;gx<=x1;gx++){
    const sx=ox+gx*s;
    rctx.beginPath();rctx.moveTo(sx,0);rctx.lineTo(sx,H);rctx.stroke();
    if(gx%2===0)rctx.fillText(gx+'m',sx+2,H-3);
  }
  for(let gy=y0;gy<=y1;gy++){
    const sy=oy-gy*s;
    rctx.beginPath();rctx.moveTo(0,sy);rctx.lineTo(W,sy);rctx.stroke();
    if(gy%2===0)rctx.fillText(gy+'m',3,sy-2);
  }

  // Sort walls to connect nearby ones
  const walls=lms.filter(l=>(l.type||'object')==='wall');
  const others=lms.filter(l=>(l.type||'object')!=='wall');

  // Connect wall points that are within 1.5m of each other
  if(walls.length>1){
    rctx.strokeStyle=TC.wall;rctx.lineWidth=2.5;
    const used=new Set();
    walls.forEach((w,i)=>{
      walls.forEach((w2,j)=>{
        if(i>=j)return;
        const dx=w.x-w2.x, dy=w.y-w2.y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<1.5){
          const a1=Math.max(0.15,1.0-(w.age||0)/300);
          const a2=Math.max(0.15,1.0-(w2.age||0)/300);
          rctx.globalAlpha=Math.min(a1,a2);
          rctx.beginPath();
          rctx.moveTo(ox+w.x*s,oy-w.y*s);
          rctx.lineTo(ox+w2.x*s,oy-w2.y*s);
          rctx.stroke();
          used.add(i);used.add(j);
        }
      });
    });
    // Isolated wall points as crosses
    walls.forEach((w,i)=>{
      if(used.has(i))return;
      const sx=ox+w.x*s, sy=oy-w.y*s;
      rctx.globalAlpha=Math.max(0.15,1.0-(w.age||0)/300);
      rctx.strokeStyle=TC.wall;rctx.lineWidth=2;
      rctx.beginPath();rctx.moveTo(sx-4,sy-4);rctx.lineTo(sx+4,sy+4);rctx.stroke();
      rctx.beginPath();rctx.moveTo(sx+4,sy-4);rctx.lineTo(sx-4,sy+4);rctx.stroke();
    });
    rctx.globalAlpha=1.0;
  } else {
    walls.forEach(w=>{
      const sx=ox+w.x*s, sy=oy-w.y*s;
      rctx.globalAlpha=Math.max(0.15,1.0-(w.age||0)/300);
      rctx.strokeStyle=TC.wall;rctx.lineWidth=2;
      rctx.beginPath();rctx.moveTo(sx-4,sy-4);rctx.lineTo(sx+4,sy+4);rctx.stroke();
      rctx.beginPath();rctx.moveTo(sx+4,sy-4);rctx.lineTo(sx-4,sy+4);rctx.stroke();
      rctx.globalAlpha=1.0;
    });
  }

  // Wall point dots (small)
  walls.forEach(w=>{
    const sx=ox+w.x*s, sy=oy-w.y*s;
    rctx.globalAlpha=Math.max(0.15,1.0-(w.age||0)/300);
    rctx.fillStyle=TC.wall;
    rctx.beginPath();rctx.arc(sx,sy,2.5,0,Math.PI*2);rctx.fill();
    rctx.globalAlpha=1.0;
  });

  // Non-wall landmarks
  others.forEach(lm=>{
    const sx=ox+lm.x*s, sy=oy-lm.y*s;
    const tp=lm.type||'object';
    const col=TC[tp]||'#5dade2';
    const alpha=Math.max(0.15,1.0-(lm.age||0)/300);
    rctx.globalAlpha=alpha;
    if(tp==='door'||tp==='open'){
      rctx.fillStyle=col;
      rctx.beginPath();rctx.arc(sx,sy,5,0,Math.PI*2);rctx.fill();
      rctx.strokeStyle='#0a0a1a';rctx.lineWidth=2;
      rctx.beginPath();rctx.arc(sx,sy,5,0,Math.PI*2);rctx.stroke();
    } else {
      rctx.fillStyle=col;
      rctx.beginPath();rctx.arc(sx,sy,4,0,Math.PI*2);rctx.fill();
    }
    rctx.font='10px monospace';rctx.textAlign='left';
    rctx.fillText(lm.name,sx+7,sy+3);
    rctx.globalAlpha=1.0;
  });

  // Rover
  const rsx=ox+rv.x*s, rsy=oy-rv.y*s;
  const hRad=-rv.heading*Math.PI/180; // heading: 0=up, CW positive
  rctx.save();rctx.translate(rsx,rsy);rctx.rotate(hRad);
  // Body
  rctx.fillStyle='#e94560';
  rctx.fillRect(-5,-8,10,16);
  // Forward arrow
  rctx.beginPath();rctx.moveTo(0,-14);rctx.lineTo(-5,-8);rctx.lineTo(5,-8);
  rctx.closePath();rctx.fill();
  rctx.restore();
  // Gimbal direction (world angle = heading + gimbal_pan)
  const gRad=-(rv.heading+rv.gimbal_pan)*Math.PI/180;
  rctx.strokeStyle='#5dade2';rctx.lineWidth=2;
  rctx.beginPath();rctx.moveTo(rsx,rsy);
  rctx.lineTo(rsx+Math.sin(-gRad)*25,rsy-Math.cos(-gRad)*25);rctx.stroke();
  // FOV wedge
  const fov=32.5*Math.PI/180;
  const fovR=2.5*s;
  const gAng=Math.PI/2+gRad; // convert to canvas angle
  rctx.fillStyle='rgba(93,173,226,0.06)';
  rctx.beginPath();rctx.moveTo(rsx,rsy);
  rctx.arc(rsx,rsy,fovR,gAng-fov,gAng+fov);
  rctx.closePath();rctx.fill();
}

// Poll map state
setInterval(()=>{
  fetch('/landmarks').then(r=>r.json()).then(drawMap).catch(()=>{});
},500);
</script>
</body>
</html>"""


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
                    jpg = _camera_ref.get_jpeg() if _camera_ref else None
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
        elif self.path == "/lessons":
            body = json.dumps(lessons.load_lessons())
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
        elif self.path == "/providers":
            for kind in ("stt", "llm", "tts"):
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
            body = json.dumps(_get_providers())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
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
