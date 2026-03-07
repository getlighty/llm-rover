"""HTTP server and dashboard for rover_brain_v2."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from rover_brain_v2.models import FollowRequest, NavigationRequest


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Rover Brain V2</title>
  <style>
    :root{
      --bg:#f2eee6;
      --paper:#fffdf8;
      --ink:#1f241f;
      --muted:#6c7368;
      --line:#d7d0c3;
      --accent:#a4552d;
      --accent-strong:#8b3c18;
      --teal:#1f5d58;
      --teal-soft:#d9ebe8;
      --danger:#9a1f1f;
      --danger-soft:#f9d9d9;
      --shadow:0 18px 40px rgba(64,42,27,.08);
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:"IBM Plex Sans","Segoe UI",sans-serif;
      color:var(--ink);
      background:
        radial-gradient(circle at top right, rgba(164,85,45,.12), transparent 24%),
        radial-gradient(circle at top left, rgba(31,93,88,.10), transparent 26%),
        linear-gradient(180deg,#f8f5ee 0%,var(--bg) 100%);
    }
    .shell{max-width:1520px;margin:0 auto;padding:18px}
    .header{
      display:grid;grid-template-columns:1.6fr 1fr;gap:16px;margin-bottom:16px
    }
    .hero,.statusbar,.panel{
      background:var(--paper);
      border:1px solid var(--line);
      border-radius:22px;
      box-shadow:var(--shadow);
    }
    .hero{padding:20px 22px;position:relative;overflow:hidden}
    .hero:before{
      content:"";
      position:absolute;inset:auto -40px -40px auto;width:220px;height:220px;
      background:radial-gradient(circle, rgba(164,85,45,.18), rgba(164,85,45,0));
      pointer-events:none;
    }
    .eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
    h1{margin:8px 0 10px;font-size:34px;line-height:1.05}
    .hero p{margin:0;color:var(--muted);max-width:56ch}
    .statusbar{padding:16px 18px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
    .stat{
      padding:12px 14px;border:1px solid var(--line);border-radius:16px;background:#fff
    }
    .stat .label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
    .stat .value{margin-top:6px;font-size:20px;font-weight:700}
    .layout{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,.95fr);gap:16px}
    .stack{display:grid;gap:16px}
    .panel{padding:16px}
    .panel h2{margin:0 0 12px;font-size:15px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
    .stream-wrap{display:grid;gap:12px}
    .stream{
      width:100%;border-radius:18px;border:1px solid var(--line);background:#e9e3d7;min-height:280px
    }
    .controls-grid,.form-grid,.provider-grid,.room-grid,.toggle-grid,.gimbal-grid{
      display:grid;gap:10px
    }
    .controls-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
    .form-grid{grid-template-columns:1fr auto}
    .provider-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    .room-grid{grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
    .toggle-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    .gimbal-grid{grid-template-columns:repeat(3,minmax(0,1fr));max-width:280px}
    button,input,select{
      width:100%;
      border-radius:14px;
      border:1px solid var(--line);
      font:inherit;
      padding:12px 14px;
      background:#fff;
      color:var(--ink);
    }
    button{
      cursor:pointer;
      font-weight:600;
      transition:transform .08s ease, background .12s ease, border-color .12s ease;
    }
    button:hover{transform:translateY(-1px)}
    .accent{background:var(--accent);border-color:var(--accent);color:#fff}
    .accent:hover{background:var(--accent-strong)}
    .teal{background:var(--teal);border-color:var(--teal);color:#fff}
    .danger{background:var(--danger);border-color:var(--danger);color:#fff}
    .soft{background:#fbf7ef}
    .pill-row{display:flex;flex-wrap:wrap;gap:8px}
    .pill{
      display:inline-flex;align-items:center;gap:8px;
      border-radius:999px;padding:8px 12px;
      border:1px solid var(--line);background:#fff;color:var(--ink);font-size:13px
    }
    .log{
      min-height:260px;max-height:520px;overflow:auto;
      border:1px solid var(--line);border-radius:16px;background:#faf7f2;padding:12px
    }
    .log-line{font-family:"IBM Plex Mono","SFMono-Regular",monospace;font-size:12px;line-height:1.5;padding:4px 0;border-bottom:1px dashed rgba(108,115,104,.15)}
    .log-line:last-child{border-bottom:0}
    .tag{font-weight:700;color:var(--teal)}
    .subtext{font-size:13px;color:var(--muted)}
    .split{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .toggle{
      display:flex;align-items:center;justify-content:space-between;gap:12px;
      padding:12px 14px;border:1px solid var(--line);border-radius:14px;background:#fff
    }
    .toggle span{font-size:13px;font-weight:600}
    .toggle input{width:auto}
    .danger-strip{
      display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px
    }
    @media (max-width: 1100px){
      .header,.layout{grid-template-columns:1fr}
    }
    @media (max-width: 720px){
      .provider-grid,.split,.toggle-grid,.controls-grid{grid-template-columns:1fr}
      h1{font-size:28px}
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <section class="hero">
        <div class="eyebrow">Modular Runtime</div>
        <h1>Rover Brain V2</h1>
        <p>Depth-first navigation, graph-aware room routing, safer follow-me, and a faster operator dashboard. This UI controls only the new <code>rover_brain_v2</code> runtime.</p>
      </section>
      <section class="statusbar">
        <div class="stat"><div class="label">Active Mode</div><div class="value" id="statMode">idle</div></div>
        <div class="stat"><div class="label">Current Room</div><div class="value" id="statRoom">unknown</div></div>
        <div class="stat"><div class="label">Audio</div><div class="value" id="statAudio">checking</div></div>
        <div class="stat"><div class="label">Kill Switch</div><div class="value" id="statKill">released</div></div>
      </section>
    </div>

    <div class="layout">
      <div class="stack">
        <section class="panel">
          <h2>Live View</h2>
          <div class="stream-wrap">
            <img class="stream" src="/api/stream" alt="Live rover stream">
            <div class="pill-row" id="statusPills"></div>
          </div>
        </section>

        <section class="panel">
          <h2>Quick Actions</h2>
          <div class="form-grid" style="margin-bottom:10px">
            <input id="commandInput" placeholder="Type a free-form rover command">
            <button class="accent" onclick="sendCommand()">Send</button>
          </div>
          <div class="split" style="margin-bottom:12px">
            <div>
              <div class="subtext" style="margin-bottom:8px">Navigation</div>
              <div class="form-grid">
                <input id="navTarget" placeholder="e.g. go to kitchen">
                <button class="teal" onclick="startNavigation()">Navigate</button>
              </div>
            </div>
            <div>
              <div class="subtext" style="margin-bottom:8px">Follow</div>
              <div class="form-grid">
                <input id="followTarget" value="person" placeholder="person">
                <button class="accent" onclick="startFollow()">Follow</button>
              </div>
            </div>
          </div>
          <div class="room-grid" id="roomButtons"></div>
          <div class="danger-strip">
            <button class="soft" onclick="post('/api/stop',{})">Stop Active Task</button>
            <button id="killBtn" class="danger" onclick="toggleKill()">Engage Kill</button>
            <button class="accent" onclick="restartService()" id="restartBtn">Restart Service</button>
          </div>
        </section>

        <section class="panel">
          <h2>Teleop</h2>
          <div class="controls-grid">
            <button class="soft" onclick="teleop('forward')">Forward</button>
            <button class="soft" onclick="teleop('left')">Turn Left</button>
            <button class="soft" onclick="teleop('right')">Turn Right</button>
            <button class="soft" onclick="teleop('back')">Back</button>
            <button class="danger" onclick="teleop('stop')">Stop</button>
            <button class="soft" onclick="toggleLights()">Lights</button>
          </div>
          <div class="split" style="margin-top:12px">
            <div>
              <div class="subtext" style="margin-bottom:8px">Gimbal</div>
              <div class="gimbal-grid">
                <div></div>
                <button class="soft" onclick="gimbal(0,20)">Up</button>
                <div></div>
                <button class="soft" onclick="gimbal(-30,0)">Left</button>
                <button class="soft" onclick="gimbal(0,0)">Center</button>
                <button class="soft" onclick="gimbal(30,0)">Right</button>
                <div></div>
                <button class="soft" onclick="gimbal(0,-15)">Down</button>
                <div></div>
              </div>
            </div>
            <div>
              <div class="subtext" style="margin-bottom:8px">Presets</div>
              <div class="controls-grid" style="grid-template-columns:1fr 1fr">
                <button class="soft" onclick="setCommand('scan around')">Scan</button>
                <button class="soft" onclick="setCommand('look around carefully')">Look Around</button>
                <button class="soft" onclick="setCommand('say hello')">Say Hello</button>
                <button class="soft" onclick="setCommand('center camera')">Center Camera</button>
              </div>
            </div>
          </div>
        </section>
      </div>

      <div class="stack">
        <section class="panel">
          <h2>Providers</h2>
          <div class="provider-grid">
            <label><div class="subtext">STT</div><select id="stt"></select></label>
            <label><div class="subtext">Command LLM</div><select id="command_llm"></select></label>
            <label><div class="subtext">Navigator LLM</div><select id="navigator_llm"></select></label>
            <label><div class="subtext">Orchestrator LLM</div><select id="orchestrator_llm"></select></label>
            <label><div class="subtext">TTS</div><select id="tts"></select></label>
          </div>
        </section>

        <section class="panel">
          <h2>Runtime Flags</h2>
          <div class="toggle-grid">
            <label class="toggle"><span>Desk Mode</span><input id="desk_mode" type="checkbox"></label>
            <label class="toggle"><span>Speech Input</span><input id="stt_enabled" type="checkbox"></label>
            <label class="toggle"><span>Speech Output</span><input id="tts_enabled" type="checkbox"></label>
            <label class="toggle"><span>Gimbal Pan</span><input id="gimbal_pan_enabled" type="checkbox"></label>
            <label class="toggle"><span>YOLO Overlay</span><input id="yolo_overlay_enabled" type="checkbox"></label>
          </div>
        </section>

        <section class="panel">
          <h2>Telemetry</h2>
          <div class="pill-row" id="telemetryPills"></div>
        </section>

        <section class="panel">
          <h2>Events</h2>
          <div class="log" id="log"></div>
        </section>
      </div>
    </div>
  </div>

  <script>
    let lastStatus=null;
    let lightsOn=false;

    function esc(s){
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    async function post(url, body){
      const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
      return await res.json();
    }

    async function refreshStatus(){
      const res=await fetch('/api/status');
      const status=await res.json();
      lastStatus=status;
      document.getElementById('statMode').textContent=status.active_mode;
      document.getElementById('statRoom').textContent=status.current_room||'unknown';
      document.getElementById('statAudio').textContent=status.audio_ready?'ready':'disabled';
      document.getElementById('statKill').textContent=status.flags.killed?'engaged':'released';
      document.getElementById('killBtn').textContent=status.flags.killed?'Release Kill':'Engage Kill';
      renderPills(status);
      renderRooms(status.known_rooms||[]);
      bindProviders(status.providers);
      bindFlags(status.flags);
    }

    function renderPills(status){
      const pills=document.getElementById('statusPills');
      pills.innerHTML='';
      const items=[
        ['STT',status.providers.current.stt],
        ['Command',status.providers.current.command_llm],
        ['Navigator',status.providers.current.navigator_llm],
        ['Orchestrator',status.providers.current.orchestrator_llm],
        ['TTS',status.providers.current.tts],
      ];
      items.forEach(([label,value])=>{
        const div=document.createElement('div');
        div.className='pill';
        div.innerHTML='<strong>'+esc(label)+'</strong><span>'+esc(value)+'</span>';
        pills.appendChild(div);
      });
    }

    function renderRooms(rooms){
      const wrap=document.getElementById('roomButtons');
      wrap.innerHTML='';
      rooms.forEach(room=>{
        const button=document.createElement('button');
        button.className='soft';
        button.textContent='Go to '+room.replaceAll('_',' ');
        button.onclick=()=>post('/api/navigate',{target:'go to '+room.replaceAll('_',' ')});
        wrap.appendChild(button);
      });
    }

    function bindProviders(providerInfo){
      ['stt','command_llm','navigator_llm','orchestrator_llm','tts'].forEach(id=>{
        const sel=document.getElementById(id);
        const values=providerInfo.available[id]||[];
        if(sel.dataset.bound==='1')return;
        sel.innerHTML='';
        values.forEach(value=>{
          const opt=document.createElement('option');
          opt.value=value;
          opt.textContent=value;
          if(providerInfo.current[id]===value)opt.selected=true;
          sel.appendChild(opt);
        });
        sel.onchange=()=>post('/api/providers',{[id]:sel.value}).then(refreshStatus);
      });
      ['stt','command_llm','navigator_llm','orchestrator_llm','tts'].forEach(id=>{
        const sel=document.getElementById(id);
        sel.value=providerInfo.current[id];
        sel.dataset.bound='1';
      });
    }

    function bindFlags(flags){
      ['desk_mode','stt_enabled','tts_enabled','gimbal_pan_enabled','yolo_overlay_enabled'].forEach(id=>{
        const cb=document.getElementById(id);
        cb.checked=!!flags[id];
        if(cb.dataset.bound==='1')return;
        cb.onchange=()=>post('/api/flags',{[id]:cb.checked});
        cb.dataset.bound='1';
      });
    }

    async function refreshTelemetry(){
      const res=await fetch('/api/landmarks');
      const data=await res.json();
      const wrap=document.getElementById('telemetryPills');
      wrap.innerHTML='';
      const items=[
        ['Mode',data.active_mode||'idle'],
        ['Room',data.current_room||'unknown'],
      ];
      if(data.imu){
        items.push(['Heading',Math.round(data.imu.heading)+'°']);
        items.push(['Voltage',data.imu.voltage.toFixed(1)+'V']);
        items.push(['Pitch',data.imu.pitch.toFixed(1)+'°']);
      }
      items.forEach(([label,value])=>{
        const div=document.createElement('div');
        div.className='pill';
        div.innerHTML='<strong>'+esc(label)+'</strong><span>'+esc(value)+'</span>';
        wrap.appendChild(div);
      });
    }

    function sendCommand(){
      const input=document.getElementById('commandInput');
      const text=input.value.trim();
      if(!text)return;
      post('/api/command',{text});
      input.value='';
    }

    function setCommand(text){
      document.getElementById('commandInput').value=text;
    }

    function startNavigation(){
      const target=document.getElementById('navTarget').value.trim();
      if(!target)return;
      post('/api/navigate',{target});
    }

    function startFollow(){
      const target=document.getElementById('followTarget').value.trim()||'person';
      post('/api/follow',{target,duration:60.0});
    }

    function teleop(action){post('/api/teleop',{action});}

    function gimbal(pan,tilt){post('/api/gimbal',{pan,tilt});}

    function toggleKill(){
      const engage=!(lastStatus&&lastStatus.flags&&lastStatus.flags.killed);
      post('/api/kill',{engage}).then(refreshStatus);
    }

    function toggleLights(){
      lightsOn=!lightsOn;
      teleop(lightsOn?'lights_on':'lights_off');
    }

    async function restartService(){
      const btn=document.getElementById('restartBtn');
      if(!confirm('Restart the rover service? The page will reload.'))return;
      btn.textContent='Restarting…';
      btn.disabled=true;
      try{await post('/api/restart',{});}catch(e){}
      setTimeout(()=>{
        let attempts=0;
        const poll=setInterval(async()=>{
          attempts++;
          try{const r=await fetch('/api/status');if(r.ok){clearInterval(poll);location.reload();}}catch(e){}
          if(attempts>30){clearInterval(poll);btn.textContent='Restart Service';btn.disabled=false;}
        },2000);
      },3000);
    }

    document.getElementById('commandInput').addEventListener('keydown',e=>{
      if(e.key==='Enter')sendCommand();
    });

    const log=document.getElementById('log');
    const es=new EventSource('/api/events');
    es.onmessage=(event)=>{
      const data=JSON.parse(event.data);
      const line=document.createElement('div');
      line.className='log-line';
      const stamp=new Date(data.ts*1000).toLocaleTimeString();
      line.innerHTML='<span class="tag">['+esc(data.cat)+']</span> <span class="subtext">'+esc(stamp)+'</span> '+esc(data.data);
      log.appendChild(line);
      if(log.children.length>700)log.removeChild(log.firstChild);
      log.scrollTop=log.scrollHeight;
    };

    refreshStatus();
    refreshTelemetry();
    setInterval(refreshStatus,3000);
    setInterval(refreshTelemetry,2000);
  </script>
</body>
</html>
"""


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RoverWebServer:
    def __init__(self, brain, *, host: str, port: int):
        self.brain = brain
        self.host = host
        self.port = port
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    def _make_handler(self):
        brain = self.brain

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self._send_html(DASHBOARD_HTML)
                    return
                if self.path == "/api/status":
                    self._send_json(brain.status())
                    return
                if self.path == "/api/landmarks":
                    self._send_json(brain.landmarks())
                    return
                if self.path == "/api/snap":
                    frame = brain.camera.get_overlay_jpeg()
                    if not frame:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    return
                if self.path == "/api/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            frame = brain.camera.get_overlay_jpeg()
                            if frame:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                                self.wfile.write(frame)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.12)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                if self.path == "/api/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    last_ts = time.time() - 10.0
                    try:
                        while True:
                            for event in brain.events.since(last_ts):
                                last_ts = event["ts"]
                                payload = json.dumps(event)
                                self.wfile.write(f"data: {payload}\n\n".encode())
                            self.wfile.flush()
                            time.sleep(0.3)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                self.send_error(404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    data = {}
                if self.path == "/api/command":
                    self._send_json(brain.handle_text_command(data.get("text", "")))
                    return
                if self.path == "/api/providers":
                    self._send_json(brain.set_providers(**data))
                    return
                if self.path == "/api/flags":
                    brain.update_flags(**data)
                    self._send_json(brain.status()["flags"])
                    return
                if self.path == "/api/teleop":
                    self._send_json(brain.direct_teleop(data.get("action", "")))
                    return
                if self.path == "/api/gimbal":
                    self._send_json(brain.move_gimbal(data.get("pan", 0), data.get("tilt", 0)))
                    return
                if self.path == "/api/follow":
                    brain.start_follow(
                        FollowRequest(
                            target=data.get("target", "person"),
                            duration=float(data.get("duration", 60.0)),
                            target_bw=float(data.get("target_bw", brain.config.follow_target_bw)),
                        )
                    )
                    self._send_json({"ok": True})
                    return
                if self.path == "/api/navigate":
                    brain.start_navigation(
                        NavigationRequest(
                            target=data.get("target", ""),
                            topological=bool(data.get("topological", True)),
                        )
                    )
                    self._send_json({"ok": True})
                    return
                if self.path == "/api/stop":
                    self._send_json(brain.cancel_active_task())
                    return
                if self.path == "/api/kill":
                    self._send_json({"killed": brain.set_killed(bool(data.get("engage", False)))})
                    return
                if self.path == "/api/restart":
                    self._send_json({"ok": True, "restarting": True})
                    threading.Thread(target=self._do_restart, daemon=True).start()
                    return
                self.send_error(404)

            def _send_json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html: str):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _do_restart(self):
                time.sleep(0.5)
                brain.events.publish("system", "Service restart requested from UI")
                subprocess.Popen(
                    ["sudo", "systemctl", "restart", "rover-brain-v2"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            def log_message(self, _format, *_args):
                return

        return Handler
