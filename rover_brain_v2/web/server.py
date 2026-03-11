"""HTTP server and dashboard for rover_brain_v2."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from rover_brain_v2.hardware import bluetooth as bt
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
    .llm-entry{border:1px solid var(--line);border-radius:12px;background:#fff;overflow:hidden}
    .llm-header{display:flex;gap:8px;align-items:center;padding:8px 12px;cursor:pointer;font-size:12px;font-family:"IBM Plex Mono","SFMono-Regular",monospace;user-select:none}
    .llm-header:hover{background:#f5f2eb}
    .llm-header .role{font-weight:700;color:var(--accent);min-width:72px}
    .llm-header .model{color:var(--teal);flex:1}
    .llm-header .elapsed{color:var(--muted)}
    .llm-header .err{color:var(--danger);font-weight:700}
    .llm-header .img-badge{background:var(--teal-soft);color:var(--teal);border-radius:6px;padding:1px 6px;font-size:10px;font-weight:700}
    .llm-body{display:none;padding:0 12px 12px;font-size:12px;font-family:"IBM Plex Mono","SFMono-Regular",monospace;line-height:1.5}
    .llm-body.open{display:block}
    .llm-section{margin-top:8px}
    .llm-section-label{font-weight:700;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px}
    .llm-section pre{margin:0;white-space:pre-wrap;word-break:break-word;background:#faf7f2;border:1px solid var(--line);border-radius:8px;padding:8px;max-height:300px;overflow:auto}
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
        <div class="stat" style="cursor:pointer" onclick="showRoomPicker()"><div class="label">Current Room ✎</div><div class="value" id="statRoom">unknown</div></div>
        <dialog id="roomDialog" style="border-radius:12px;padding:20px;min-width:260px">
          <h3 style="margin:0 0 12px">Set Current Room</h3>
          <div id="roomPickList" style="display:flex;flex-direction:column;gap:6px"></div>
          <div style="margin-top:12px;display:flex;gap:8px">
            <input id="newRoomInput" placeholder="New room name..." style="flex:1;padding:6px;border-radius:6px;border:1px solid #ccc">
            <button class="soft" onclick="setRoom(document.getElementById('newRoomInput').value)">Add</button>
          </div>
          <button class="soft" onclick="document.getElementById('roomDialog').close()" style="margin-top:12px;width:100%">Cancel</button>
        </dialog>
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
                <button class="soft" onclick="gimbalCenter()">Center</button>
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

        <section class="panel">
          <h2>Spatial Map</h2>
          <div style="display:flex;justify-content:center">
            <canvas id="radar" width="320" height="320" style="border:1px solid var(--line);border-radius:50%;background:#faf7f2"></canvas>
          </div>
          <div class="subtext" style="text-align:center;margin-top:6px" id="radarInfo">no data</div>
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
          <details>
            <summary style="cursor:pointer;font-size:15px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:700">Calibration</summary>
            <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
              <label style="font-size:12px">Gimbal Pan <input id="calPan" type="number" step="0.5" style="width:100%;padding:8px;border-radius:10px"></label>
              <label style="font-size:12px">Gimbal Tilt <input id="calTilt" type="number" step="0.5" style="width:100%;padding:8px;border-radius:10px"></label>
              <label style="font-size:12px">Turn °/s <input id="calTurnRate" type="number" step="5" style="width:100%;padding:8px;border-radius:10px"></label>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:6px">
              <button class="soft" onclick="calTest()">Test</button>
              <button class="accent" onclick="calSave()">Save</button>
              <button class="soft" onclick="calTestTurn()">Test 90°</button>
            </div>
          </details>
        </section>
      </div>

      <div class="stack">
        <section class="panel">
          <h2>Events</h2>
          <div class="log" id="log"></div>
        </section>

        <section class="panel">
          <h2>Room Map</h2>
          <div id="roomMapWrap" style="position:relative;width:100%;height:420px;overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#faf7f2;cursor:grab;touch-action:none">
            <svg id="roomMapSvg" width="100%" height="100%" style="display:block"></svg>
          </div>
          <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px">
            <div class="subtext" id="roomMapInfo">loading...</div>
            <div style="display:flex;gap:4px">
              <button class="soft" style="padding:4px 10px;min-width:0;font-size:13px" onclick="roomMapZoom(1.3)">+</button>
              <button class="soft" style="padding:4px 10px;min-width:0;font-size:13px" onclick="roomMapZoom(1/1.3)">−</button>
              <button class="soft" style="padding:4px 10px;min-width:0;font-size:13px" onclick="roomMapReset()">Reset</button>
            </div>
          </div>
        </section>

        <section class="panel">
          <h2>LLM Log</h2>
          <div style="margin-bottom:8px;display:flex;gap:8px;align-items:center">
            <label class="toggle" style="flex:0 0 auto;padding:8px 12px"><span style="font-size:12px">Auto-refresh</span><input id="llmAutoRefresh" type="checkbox" checked></label>
            <button class="soft" style="flex:0 0 auto;padding:8px 14px;font-size:12px" onclick="refreshLLMLog()">Refresh</button>
            <span class="subtext" id="llmCount" style="font-size:11px"></span>
          </div>
          <div id="llmLog" style="max-height:600px;overflow:auto;display:grid;gap:6px"></div>
        </section>

        <section class="panel">
          <h2>Speech</h2>
          <div class="provider-grid">
            <label><div class="subtext">STT</div><select id="stt"></select></label>
            <label><div class="subtext">TTS</div><select id="tts"></select></label>
          </div>
        </section>

        <section class="panel">
          <h2>Bluetooth Speaker</h2>
          <div id="btPaired" style="margin-bottom:8px"></div>
          <div style="display:flex;gap:8px;margin-bottom:8px">
            <button class="teal" onclick="btScan()" id="btScanBtn" style="flex:1">Scan</button>
          </div>
          <div id="btDevices" style="max-height:200px;overflow:auto"></div>
          <div class="subtext" id="btStatus" style="margin-top:6px"></div>
        </section>

        <section class="panel">
          <h2>LLMs</h2>
          <label style="margin-bottom:10px;display:block"><div class="subtext">All LLMs</div><select id="all_llm"></select></label>
          <div class="provider-grid">
            <label><div class="subtext">Command</div><select id="command_llm"></select></label>
            <label><div class="subtext">Navigator</div><select id="navigator_llm"></select></label>
            <label><div class="subtext">Orchestrator</div><select id="orchestrator_llm"></select></label>
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
            <label class="toggle"><span>Reverse Look-Behind</span><input id="reverse_look_behind" type="checkbox"></label>
          </div>
        </section>

        <section class="panel">
          <h2>Telemetry</h2>
          <div class="pill-row" id="telemetryPills"></div>
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
      if(status.calibration){
        const cp=document.getElementById('calPan');
        const ct=document.getElementById('calTilt');
        const cr=document.getElementById('calTurnRate');
        if(document.activeElement!==cp)cp.value=status.calibration.gimbal_pan_center;
        if(document.activeElement!==ct)ct.value=status.calibration.gimbal_tilt_center;
        if(document.activeElement!==cr)cr.value=status.calibration.turn_rate_dps;
      }
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

    function showRoomPicker(){
      const list=document.getElementById('roomPickList');
      list.innerHTML='';
      const rooms=lastStatus?.known_rooms||[];
      const current=lastStatus?.current_room||'';
      rooms.forEach(room=>{
        const btn=document.createElement('button');
        btn.className='soft';
        btn.style.textAlign='left';
        btn.innerHTML=(room===current?'● ':'○ ')+room.replaceAll('_',' ');
        if(room===current)btn.style.fontWeight='bold';
        btn.onclick=()=>setRoom(room);
        list.appendChild(btn);
      });
      document.getElementById('roomDialog').showModal();
    }

    async function setRoom(room){
      if(!room||!room.trim())return;
      room=room.trim().toLowerCase().replaceAll(' ','_');
      document.getElementById('roomDialog').close();
      document.getElementById('statRoom').textContent=room+' (learning...)';
      const res=await post('/api/set_room',{room});
      const data=await res.json();
      if(data.learned_features){
        document.getElementById('statRoom').textContent=room+' ✓';
      }
      setTimeout(refreshStatus,1000);
    }

    function bindProviders(providerInfo){
      const llmIds=['command_llm','navigator_llm','orchestrator_llm'];
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
      // "All LLMs" combo selector
      const allSel=document.getElementById('all_llm');
      const allValues=providerInfo.available['command_llm']||[];
      if(allSel.dataset.bound!=='1'){
        allSel.innerHTML='';
        allValues.forEach(value=>{
          const opt=document.createElement('option');
          opt.value=value;
          opt.textContent=value;
          allSel.appendChild(opt);
        });
        allSel.onchange=()=>{
          const v=allSel.value;
          llmIds.forEach(id=>{document.getElementById(id).value=v;});
          post('/api/providers',{command_llm:v,navigator_llm:v,orchestrator_llm:v}).then(refreshStatus);
        };
        allSel.dataset.bound='1';
      }
      // Reflect current state: show value if all three match, otherwise first option
      const cur=llmIds.map(id=>providerInfo.current[id]);
      allSel.value=(cur[0]===cur[1]&&cur[1]===cur[2])?cur[0]:allValues[0]||'';
    }

    function bindFlags(flags){
      ['desk_mode','stt_enabled','tts_enabled','gimbal_pan_enabled','yolo_overlay_enabled','reverse_look_behind'].forEach(id=>{
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

    function gimbalCenter(){
      post('/api/gimbal',{pan:0,tilt:0});
    }

    function toggleKill(){
      const engage=!(lastStatus&&lastStatus.flags&&lastStatus.flags.killed);
      post('/api/kill',{engage}).then(refreshStatus);
    }

    function toggleLights(){
      lightsOn=!lightsOn;
      teleop(lightsOn?'lights_on':'lights_off');
    }

    function calGetValues(){
      return {
        gimbal_pan_center:parseFloat(document.getElementById('calPan').value)||0,
        gimbal_tilt_center:parseFloat(document.getElementById('calTilt').value)||0,
        turn_rate_dps:parseFloat(document.getElementById('calTurnRate').value)||200,
      };
    }

    function calTest(){
      post('/api/calibration',calGetValues());
    }

    function calSave(){
      post('/api/calibration',calGetValues()).then(()=>{
        alert('Calibration saved');
        refreshStatus();
      });
    }

    function calTestTurn(){
      // Save current turn_rate_dps first, then execute a direct 90° turn
      post('/api/calibration',calGetValues()).then(()=>{
        post('/api/test_turn',{degrees:90});
      });
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
      if(data.cat==='serial')return;
      const line=document.createElement('div');
      line.className='log-line';
      const stamp=new Date(data.ts*1000).toLocaleTimeString();
      line.innerHTML='<span class="tag">['+esc(data.cat)+']</span> <span class="subtext">'+esc(stamp)+'</span> '+esc(data.data);
      log.appendChild(line);
      if(log.children.length>700)log.removeChild(log.firstChild);
      log.scrollTop=log.scrollHeight;
    };

    // LLM Log
    let llmLastId=0;
    async function refreshLLMLog(){
      try{
        const res=await fetch('/api/llm_log?since='+llmLastId);
        const entries=await res.json();
        if(!entries.length)return;
        const wrap=document.getElementById('llmLog');
        const countEl=document.getElementById('llmCount');
        entries.forEach(e=>{
          if(e.id>llmLastId)llmLastId=e.id;
          const div=document.createElement('div');
          div.className='llm-entry';
          const stamp=new Date(e.ts*1000).toLocaleTimeString();
          const imgBadge=e.has_image?'<span class="img-badge">IMG</span>':'';
          const errText=e.error?'<span class="err">ERR</span>':'';
          const hdr=document.createElement('div');
          hdr.className='llm-header';
          hdr.innerHTML=
              '<span class="role">'+esc(e.role)+'</span>'+
              '<span class="model">'+esc(e.model)+'</span>'+
              imgBadge+errText+
              '<span class="elapsed">'+e.elapsed_s+'s</span>'+
              '<span class="subtext">'+esc(stamp)+'</span>';
          const body=document.createElement('div');
          body.className='llm-body';
          body.innerHTML=
              (e.system?'<div class="llm-section"><div class="llm-section-label">System</div><pre>'+esc(e.system)+'</pre></div>':'')+
              '<div class="llm-section"><div class="llm-section-label">Prompt</div><pre>'+esc(e.prompt)+'</pre></div>'+
              '<div class="llm-section"><div class="llm-section-label">Response</div><pre>'+esc(e.response||e.error||'(empty)')+'</pre></div>';
          hdr.onclick=function(){body.classList.toggle('open');};
          div.appendChild(hdr);
          div.appendChild(body);
          wrap.prepend(div);
        });
        // Cap displayed entries
        while(wrap.children.length>60)wrap.removeChild(wrap.lastChild);
        countEl.textContent=wrap.children.length+' entries';
      }catch(e){}
    }
    refreshLLMLog();
    setInterval(()=>{if(document.getElementById('llmAutoRefresh').checked)refreshLLMLog();},2000);

    // Bluetooth Speaker
    function _btBtn(label,cls,mac,action){
      const b=document.createElement('button');
      b.className=cls;b.style.cssText='padding:4px 8px;font-size:11px;margin-left:4px';
      b.textContent=label;
      b.onclick=function(){window['bt'+action](mac);};
      return b;
    }
    async function btRefreshStatus(){
      try{
        const res=await post('/api/bt/status',{});
        const wrap=document.getElementById('btPaired');
        wrap.innerHTML='';
        (res.paired||[]).forEach(d=>{
          const div=document.createElement('div');
          div.className='pill';
          const status=d.connected?'connected':'paired';
          const star=d.preferred?' *':'';
          div.innerHTML='<strong>'+esc(d.name)+star+'</strong><span class="subtext">'+esc(status)+'</span>';
          if(d.connected) div.appendChild(_btBtn('Disconnect','soft',d.mac,'Disconnect'));
          else div.appendChild(_btBtn('Connect','teal',d.mac,'Connect'));
          div.appendChild(_btBtn('Remove','soft',d.mac,'Remove'));
          wrap.appendChild(div);
        });
        const sink=res.bt_sink;
        document.getElementById('btStatus').textContent=sink?'Audio sink: '+sink:'No BT audio sink';
      }catch(e){}
    }
    async function btScan(){
      const btn=document.getElementById('btScanBtn');
      btn.textContent='Scanning...';btn.disabled=true;
      try{
        const res=await post('/api/bt/scan',{duration:8});
        const wrap=document.getElementById('btDevices');
        wrap.innerHTML='';
        (res.devices||[]).forEach(d=>{
          const div=document.createElement('div');
          div.className='pill';div.style.marginBottom='4px';
          if(d.connected){
            div.innerHTML='<strong>'+esc(d.name)+'</strong><span class="subtext">connected</span>';
          }else if(d.paired){
            div.innerHTML='<strong>'+esc(d.name)+'</strong><span class="subtext">paired</span>';
            div.appendChild(_btBtn('Connect','teal',d.mac,'Connect'));
          }else{
            div.innerHTML='<strong>'+esc(d.name)+'</strong><span class="subtext">'+esc(d.mac)+'</span>';
            div.appendChild(_btBtn('Pair','accent',d.mac,'Pair'));
          }
          wrap.appendChild(div);
        });
        if(!(res.devices||[]).length) wrap.innerHTML='<div class="subtext">No devices found</div>';
      }catch(e){}
      btn.textContent='Scan';btn.disabled=false;
    }
    async function btPair(mac){
      document.getElementById('btStatus').textContent='Pairing '+mac+'...';
      const res=await post('/api/bt/pair',{mac});
      document.getElementById('btStatus').textContent=res.connected?'Paired and connected!':'Pair failed';
      btRefreshStatus();
    }
    async function btConnect(mac){
      document.getElementById('btStatus').textContent='Connecting...';
      const res=await post('/api/bt/connect',{mac});
      document.getElementById('btStatus').textContent=res.connected?'Connected!':'Connect failed';
      btRefreshStatus();
    }
    async function btDisconnect(mac){
      await post('/api/bt/disconnect',{mac});
      btRefreshStatus();
    }
    async function btRemove(mac){
      if(!confirm('Remove this device?'))return;
      await post('/api/bt/remove',{mac});
      btRefreshStatus();
    }
    btRefreshStatus();

    // Radar / Spatial Map
    const RADAR_COLORS={door:'#1f5d58',wall:'#6c7368',furniture:'#a4552d',open:'#2d8a4e',obstacle:'#9a1f1f',object:'#555',living:'#8b5cf6'};
    const RADAR_SCALE=1.2; // meters per half-radius
    async function drawRadar(){
      try{
        const res=await fetch('/api/spatial_map');
        const data=await res.json();
        const c=document.getElementById('radar');
        const ctx=c.getContext('2d');
        const cx=c.width/2,cy=c.height/2,r=c.width/2-20;
        ctx.clearRect(0,0,c.width,c.height);
        // Grid circles
        ctx.strokeStyle='#d7d0c3';ctx.lineWidth=0.5;
        for(let i=1;i<=3;i++){
          ctx.beginPath();ctx.arc(cx,cy,r*i/3,0,Math.PI*2);ctx.stroke();
        }
        // Cross hairs
        ctx.beginPath();ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.stroke();
        // Labels
        ctx.fillStyle='#6c7368';ctx.font='10px sans-serif';ctx.textAlign='center';
        ctx.fillText('0°',cx,cy-r-4);ctx.fillText('180°',cx,cy+r+12);
        ctx.textAlign='left';ctx.fillText('90°',cx+r+3,cy+4);
        ctx.textAlign='right';ctx.fillText('-90°',cx-r-3,cy+4);
        // Scale labels
        ctx.fillStyle='#999';ctx.font='9px sans-serif';ctx.textAlign='left';
        for(let i=1;i<=3;i++){
          ctx.fillText((RADAR_SCALE*i/3).toFixed(1)+'m',cx+3,cy-r*i/3+10);
        }
        // Rover triangle (center)
        ctx.fillStyle='var(--accent)';
        ctx.beginPath();ctx.moveTo(cx,cy-8);ctx.lineTo(cx-5,cy+5);ctx.lineTo(cx+5,cy+5);ctx.closePath();ctx.fill();
        // Plot entries
        const entries=data.entries||[];
        entries.forEach(e=>{
          const ang=((-e.bearing+90)*Math.PI/180); // Convert: 0°=up, +=right
          const dist=Math.min(e.dist_m/RADAR_SCALE,1.0)*r;
          const px=cx+dist*Math.cos(ang);
          const py=cy-dist*Math.sin(ang);
          const color=RADAR_COLORS[e.type]||'#555';
          const sz=e.type==='door'?6:e.type==='obstacle'?5:4;
          ctx.fillStyle=color;
          if(e.type==='door'){
            // Diamond for doors
            ctx.beginPath();ctx.moveTo(px,py-sz);ctx.lineTo(px+sz,py);ctx.lineTo(px,py+sz);ctx.lineTo(px-sz,py);ctx.closePath();ctx.fill();
          }else if(e.type==='obstacle'){
            // X for obstacles
            ctx.strokeStyle=color;ctx.lineWidth=2;
            ctx.beginPath();ctx.moveTo(px-sz,py-sz);ctx.lineTo(px+sz,py+sz);ctx.moveTo(px+sz,py-sz);ctx.lineTo(px-sz,py+sz);ctx.stroke();
          }else{
            // Circle for everything else
            ctx.beginPath();ctx.arc(px,py,sz,0,Math.PI*2);ctx.fill();
          }
          // Label
          ctx.fillStyle=color;ctx.font='9px sans-serif';ctx.textAlign='center';
          const lbl=e.label.length>16?e.label.slice(0,15)+'…':e.label;
          ctx.fillText(lbl,px,py-sz-3);
        });
        document.getElementById('radarInfo').textContent=entries.length+' entries | heading '+data.heading+'°';
      }catch(e){}
    }
    drawRadar();
    setInterval(drawRadar,1500);

    // ── Room Map (zoomable, pannable, click-to-expand) ──
    const _rm={layout:null,zoom:1,panX:0,panY:0,dragging:false,dragStart:null,
      selectedRoom:null,data:null,W:580,H:420};
    function _rmApplyView(){
      const svg=document.getElementById('roomMapSvg');
      const vw=_rm.W/_rm.zoom,vh=_rm.H/_rm.zoom;
      const vx=-_rm.panX/_rm.zoom+(_rm.W-vw)/2;
      const vy=-_rm.panY/_rm.zoom+(_rm.H-vh)/2;
      svg.setAttribute('viewBox',vx+' '+vy+' '+vw+' '+vh);
    }
    function roomMapZoom(factor,cx,cy){
      const oldZ=_rm.zoom;
      _rm.zoom=Math.max(0.3,Math.min(5,_rm.zoom*factor));
      if(cx!==undefined&&cy!==undefined){
        _rm.panX+=(cx-_rm.W/2)*(1-factor)*0.5;
        _rm.panY+=(cy-_rm.H/2)*(1-factor)*0.5;
      }
      _rmApplyView();
    }
    function roomMapReset(){_rm.zoom=1;_rm.panX=0;_rm.panY=0;_rm.selectedRoom=null;_rmApplyView();renderRoomMap();}
    // Wheel zoom
    document.getElementById('roomMapWrap').addEventListener('wheel',e=>{
      e.preventDefault();
      const rect=e.currentTarget.getBoundingClientRect();
      roomMapZoom(e.deltaY<0?1.15:1/1.15,e.clientX-rect.left,e.clientY-rect.top);
    },{passive:false});
    // Pan via drag
    document.getElementById('roomMapWrap').addEventListener('pointerdown',e=>{
      if(e.button!==0)return;
      _rm.dragging=true;_rm.dragStart={x:e.clientX-_rm.panX,y:e.clientY-_rm.panY};
      e.currentTarget.setPointerCapture(e.pointerId);
      e.currentTarget.style.cursor='grabbing';
    });
    document.getElementById('roomMapWrap').addEventListener('pointermove',e=>{
      if(!_rm.dragging)return;
      _rm.panX=e.clientX-_rm.dragStart.x;
      _rm.panY=e.clientY-_rm.dragStart.y;
      _rmApplyView();
    });
    document.getElementById('roomMapWrap').addEventListener('pointerup',e=>{
      _rm.dragging=false;
      e.currentTarget.style.cursor='grab';
    });

    function _rmClickRoom(roomId){
      _rm.selectedRoom=_rm.selectedRoom===roomId?null:roomId;
      renderRoomMap();
    }

    async function drawRoomMap(){
      try{
        const res=await fetch('/api/topo_map');
        const data=await res.json();
        if(data.error){document.getElementById('roomMapInfo').textContent=data.error;return;}
        _rm.data=data;
        const svg=document.getElementById('roomMapSvg');
        _rm.W=svg.clientWidth||580;_rm.H=420;

        const nodes=data.nodes||[];
        const edges=data.edges||[];
        const rooms=nodes.filter(n=>n.type==='room');
        const nodeMap={};nodes.forEach(n=>{nodeMap[n.id]=n;});

        // Build room-to-room edges through transitions
        const roomEdges=[];const seen=new Set();
        rooms.forEach(r=>{
          edges.filter(e=>e.a===r.id||e.b===r.id).forEach(e=>{
            const tId=e.a===r.id?e.b:e.a;const tNode=nodeMap[tId];
            if(!tNode||tNode.type!=='transition')return;
            edges.filter(e2=>e2.a===tId||e2.b===tId).forEach(e2=>{
              const oId=e2.a===tId?e2.b:e2.a;
              if(oId===r.id)return;const oNode=nodeMap[oId];
              if(!oNode||oNode.type!=='room')return;
              const key=[r.id,oId].sort().join('|');
              if(seen.has(key))return;seen.add(key);
              roomEdges.push({from:r.id,to:oId,via:tNode});
            });
          });
        });
        _rm.roomEdges=roomEdges;

        // Force layout (once)
        if(!_rm.layout||_rm.layout._roomCount!==rooms.length){
          const pad=90,cx=_rm.W/2,cy=_rm.H/2;
          const pos={};
          rooms.forEach((r,i)=>{
            const a=(2*Math.PI*i)/rooms.length-Math.PI/2;
            pos[r.id]={x:cx+Math.cos(a)*(_rm.W/2-pad),y:cy+Math.sin(a)*(_rm.H/2-pad)};
          });
          const deg={};roomEdges.forEach(re=>{deg[re.from]=(deg[re.from]||0)+1;deg[re.to]=(deg[re.to]||0)+1;});
          let hub=rooms[0]?.id,hubD=0;
          for(const[k,v]of Object.entries(deg)){if(v>hubD){hubD=v;hub=k;}}
          pos[hub]={x:cx,y:cy};
          const placed=new Set([hub]);
          const hubE=roomEdges.filter(re=>re.from===hub||re.to===hub);
          hubE.forEach((re,i)=>{
            const o=re.from===hub?re.to:re.from;
            const az=re.via.azimuth_from;
            let a=(2*Math.PI*i)/hubE.length-Math.PI/2;
            if(az&&az[hub]!==undefined)a=(az[hub]-90)*Math.PI/180;
            const dist=Math.min(_rm.W,_rm.H)/2-pad;
            pos[o]={x:cx+Math.cos(a)*dist,y:cy+Math.sin(a)*dist};placed.add(o);
          });
          rooms.forEach(r=>{
            if(placed.has(r.id))return;
            const ce=roomEdges.find(re=>re.from===r.id&&placed.has(re.to)||re.to===r.id&&placed.has(re.from));
            if(ce){const p=ce.from===r.id?ce.to:ce.from;const pp=pos[p];
              const az=ce.via.azimuth_from;let a=Math.random()*Math.PI*2;
              if(az&&az[p]!==undefined)a=(az[p]-90)*Math.PI/180;
              pos[r.id]={x:pp.x+Math.cos(a)*130,y:pp.y+Math.sin(a)*130};}
            placed.add(r.id);
          });
          for(let it=0;it<60;it++){
            const f={};rooms.forEach(r=>{f[r.id]={x:0,y:0};});
            for(let i=0;i<rooms.length;i++){
              for(let j=i+1;j<rooms.length;j++){
                const a=rooms[i].id,b=rooms[j].id;
                let dx=pos[b].x-pos[a].x,dy=pos[b].y-pos[a].y;
                const d=Math.sqrt(dx*dx+dy*dy)||1;
                if(d<160){const ff=(160-d)*0.3;dx/=d;dy/=d;f[a].x-=dx*ff;f[a].y-=dy*ff;f[b].x+=dx*ff;f[b].y+=dy*ff;}
              }
            }
            roomEdges.forEach(re=>{
              let dx=pos[re.to].x-pos[re.from].x,dy=pos[re.to].y-pos[re.from].y;
              const d=Math.sqrt(dx*dx+dy*dy)||1;const ideal=170;
              if(Math.abs(d-ideal)>10){const ff=(d-ideal)*0.02;dx/=d;dy/=d;
                f[re.from].x+=dx*ff;f[re.from].y+=dy*ff;f[re.to].x-=dx*ff;f[re.to].y-=dy*ff;}
            });
            rooms.forEach(r=>{f[r.id].x+=(cx-pos[r.id].x)*0.01;f[r.id].y+=(cy-pos[r.id].y)*0.01;});
            rooms.forEach(r=>{
              if(r.id===hub)return;
              pos[r.id].x=Math.max(pad,Math.min(_rm.W-pad,pos[r.id].x+f[r.id].x));
              pos[r.id].y=Math.max(40,Math.min(_rm.H-40,pos[r.id].y+f[r.id].y));
            });
          }
          _rm.layout={pos,hub,_roomCount:rooms.length};
        }
        renderRoomMap();
      }catch(e){document.getElementById('roomMapInfo').textContent='error: '+e.message;}
    }

    function renderRoomMap(){
      if(!_rm.data||!_rm.layout)return;
      const svg=document.getElementById('roomMapSvg');
      const data=_rm.data,layout=_rm.layout.pos;
      const nodes=data.nodes||[];const rooms=nodes.filter(n=>n.type==='room');
      const currentRoom=data.current_room||'';
      const sel=_rm.selectedRoom;
      const roomEdges=_rm.roomEdges||[];

      function azLabel(deg){
        if(deg===undefined||deg===null)return'';const d=parseFloat(deg);
        if(d>=-20&&d<=20)return'straight';if(d>20&&d<=70)return'slight R';
        if(d>70&&d<=110)return'right';if(d>110)return'behind R';
        if(d>=-70&&d<-20)return'slight L';if(d>=-110&&d<-70)return'left';return'behind L';
      }

      let html='<defs>'+
        '<marker id="ah" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="var(--teal)"/></marker>'+
        '<filter id="roomShadow"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.15"/></filter>'+
        '<filter id="expandGlow"><feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#a4552d" flood-opacity="0.3"/></filter>'+
        '</defs>';

      // Edges
      roomEdges.forEach(re=>{
        const fp=layout[re.from],tp=layout[re.to];if(!fp||!tp)return;
        const mx=(fp.x+tp.x)/2,my=(fp.y+tp.y)/2;
        const dx=tp.x-fp.x,dy=tp.y-fp.y,len=Math.sqrt(dx*dx+dy*dy)||1;
        const nx=-dy/len,ny=dx/len;
        const dimmed=sel&&re.from!==sel&&re.to!==sel;
        const op=dimmed?'0.25':'1';
        html+='<line x1="'+fp.x+'" y1="'+fp.y+'" x2="'+tp.x+'" y2="'+tp.y+'" stroke="var(--line)" stroke-width="2" stroke-dasharray="6,4" opacity="'+op+'"/>';
        const ds=7;
        html+='<polygon points="'+mx+','+(my-ds)+' '+(mx+ds)+','+my+' '+mx+','+(my+ds)+' '+(mx-ds)+','+my+'" fill="var(--teal-soft)" stroke="var(--teal)" stroke-width="1.5" opacity="'+op+'"/>';
        const label=re.via.label||re.via.id;
        const shortLabel=label.length>22?label.slice(0,20)+'…':label;
        html+='<text x="'+(mx+nx*14)+'" y="'+(my+ny*14-4)+'" text-anchor="middle" font-size="9" fill="var(--teal)" font-weight="600" opacity="'+op+'">'+esc(shortLabel)+'</text>';
        const az=re.via.azimuth_from||{};
        const aOff=30;
        if(az[re.from]!==undefined){
          const ax1=fp.x+dx/len*aOff,ay1=fp.y+dy/len*aOff;
          const ax2=fp.x+dx/len*(aOff+20),ay2=fp.y+dy/len*(aOff+20);
          html+='<line x1="'+ax1+'" y1="'+ay1+'" x2="'+ax2+'" y2="'+ay2+'" stroke="var(--teal)" stroke-width="1.5" marker-end="url(#ah)" opacity="'+(dimmed?0.15:0.7)+'"/>';
          html+='<text x="'+(ax1+nx*10)+'" y="'+(ay1+ny*10)+'" text-anchor="middle" font-size="8" fill="var(--accent)" font-weight="600" opacity="'+op+'">'+azLabel(az[re.from])+'</text>';
        }
        if(az[re.to]!==undefined){
          const ax1=tp.x-dx/len*aOff,ay1=tp.y-dy/len*aOff;
          const ax2=tp.x-dx/len*(aOff+20),ay2=tp.y-dy/len*(aOff+20);
          html+='<line x1="'+ax1+'" y1="'+ay1+'" x2="'+ax2+'" y2="'+ay2+'" stroke="var(--teal)" stroke-width="1.5" marker-end="url(#ah)" opacity="'+(dimmed?0.15:0.7)+'"/>';
          html+='<text x="'+(ax1-nx*10)+'" y="'+(ay1-ny*10)+'" text-anchor="middle" font-size="8" fill="var(--accent)" font-weight="600" opacity="'+op+'">'+azLabel(az[re.to])+'</text>';
        }
      });

      // Room nodes
      rooms.forEach(r=>{
        const p=layout[r.id];if(!p)return;
        const isCur=r.id===currentRoom;
        const isExp=r.id===sel;
        const dimmed=sel&&!isExp&&!roomEdges.some(re=>(re.from===sel&&re.to===r.id)||(re.to===sel&&re.from===r.id));

        // Expanded room: larger box with full feature list
        if(isExp){
          const feats=r.features||[];
          const lineH=14,padV=16,padH=12;
          const headerH=38;
          const boxH=headerH+padV+feats.length*lineH+padV;
          const boxW=200;
          const bx=p.x-boxW/2,by=p.y-headerH/2;
          // Card background
          html+='<g onclick="_rmClickRoom(\\x27'+r.id+'\\x27)" style="cursor:pointer" filter="url(#expandGlow)">';
          html+='<rect x="'+bx+'" y="'+by+'" width="'+boxW+'" height="'+boxH+'" rx="14" fill="'+(isCur?'var(--accent)':'#fff')+'" stroke="'+(isCur?'var(--accent-strong)':'var(--teal)')+'" stroke-width="2"/>';
          // Room name
          html+='<text x="'+p.x+'" y="'+(by+18)+'" text-anchor="middle" font-size="13" font-weight="700" fill="'+(isCur?'#fff':'var(--ink)')+'">'+esc(r.label||r.id)+'</text>';
          // Floor type
          if(r.floor_type){
            html+='<text x="'+p.x+'" y="'+(by+32)+'" text-anchor="middle" font-size="9" fill="'+(isCur?'rgba(255,255,255,0.7)':'var(--muted)')+'">'+esc(r.floor_type)+'</text>';
          }
          // Separator line
          const sepY=by+headerH;
          html+='<line x1="'+(bx+8)+'" y1="'+sepY+'" x2="'+(bx+boxW-8)+'" y2="'+sepY+'" stroke="'+(isCur?'rgba(255,255,255,0.3)':'var(--line)')+'" stroke-width="1"/>';
          // Feature list
          feats.forEach((f,i)=>{
            const fy=sepY+padV+i*lineH;
            const txt=f.length>26?f.slice(0,24)+'…':f;
            html+='<text x="'+(bx+padH)+'" y="'+fy+'" font-size="10" fill="'+(isCur?'rgba(255,255,255,0.9)':'var(--ink)')+'">'+esc(txt)+'</text>';
          });
          // Nav hints
          if(r.nav_hints){
            const hy=sepY+padV+feats.length*lineH+6;
            const hint=r.nav_hints.length>50?r.nav_hints.slice(0,48)+'…':r.nav_hints;
            html+='<text x="'+(bx+padH)+'" y="'+hy+'" font-size="8" font-style="italic" fill="'+(isCur?'rgba(255,255,255,0.6)':'var(--muted)')+'">'+esc(hint)+'</text>';
          }
          html+='</g>';
        }else{
          // Normal collapsed room
          const op=dimmed?'0.3':'1';
          const rx=68,ry=26;
          html+='<g onclick="_rmClickRoom(\\x27'+r.id+'\\x27)" style="cursor:pointer" opacity="'+op+'" filter="url(#roomShadow)">';
          html+='<ellipse cx="'+p.x+'" cy="'+p.y+'" rx="'+rx+'" ry="'+ry+'" fill="'+(isCur?'var(--accent)':'#fff')+'" stroke="'+(isCur?'var(--accent-strong)':'var(--line)')+'" stroke-width="'+(isCur?2.5:1.5)+'"/>';
          html+='<text x="'+p.x+'" y="'+(p.y-3)+'" text-anchor="middle" font-size="12" font-weight="700" fill="'+(isCur?'#fff':'var(--ink)')+'">'+esc(r.label||r.id)+'</text>';
          if(r.floor_type){
            html+='<text x="'+p.x+'" y="'+(p.y+11)+'" text-anchor="middle" font-size="8" fill="'+(isCur?'rgba(255,255,255,0.7)':'var(--muted)')+'">'+esc(r.floor_type)+'</text>';
          }
          if(r.features&&r.features.length>0&&!dimmed){
            const feats=r.features.slice(0,3).join(', ');
            const short=feats.length>30?feats.slice(0,28)+'…':feats;
            html+='<text x="'+p.x+'" y="'+(p.y+ry+12)+'" text-anchor="middle" font-size="7" fill="var(--muted)">'+esc(short)+'</text>';
          }
          html+='</g>';
        }
      });

      svg.innerHTML=html;
      _rmApplyView();
      const info=rooms.length+' rooms, '+roomEdges.length+' connections | current: '+(currentRoom||'unknown');
      document.getElementById('roomMapInfo').textContent=sel?info+' | viewing: '+sel:info;
    }
    drawRoomMap();
    setInterval(drawRoomMap,10000);

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
                    frame = brain.camera.get_overlay_jpeg() if brain.flags.yolo_overlay_enabled or brain.camera.get_nav_dot() is not None else brain.camera.get_jpeg()
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
                            frame = brain.camera.get_overlay_jpeg() if brain.flags.yolo_overlay_enabled or brain.camera.get_nav_dot() is not None else brain.camera.get_jpeg()
                            if frame:
                                self.wfile.write(b"--frame\r\n")
                                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                                self.wfile.write(frame)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.12)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                if self.path.startswith("/api/clip_query?"):
                    query = ""
                    for part in self.path.split("?", 1)[1].split("&"):
                        if part.startswith("q="):
                            from urllib.parse import unquote
                            query = unquote(part.split("=", 1)[1])
                    if brain.clip_map and query:
                        results = brain.clip_map.query(query, top_k=5)
                        self._send_json({"query": query, "results": results,
                                         "stats": brain.clip_map.get_stats()})
                    else:
                        self._send_json({"error": "CLIP map not available or empty query"})
                    return
                if self.path == "/api/clip_stats":
                    if brain.clip_map:
                        self._send_json(brain.clip_map.get_stats())
                    else:
                        self._send_json({"error": "CLIP map not available"})
                    return
                if self.path == "/api/vlm_describe":
                    if brain.vlm and brain.vlm.is_alive():
                        frame = brain.camera.get_jpeg()
                        if frame:
                            result = brain.vlm.describe_scene(frame)
                            self._send_json(result)
                        else:
                            self._send_json({"error": "No camera frame"})
                    else:
                        self._send_json({"error": "VLM server not available"})
                    return
                if self.path == "/api/spatial_map":
                    try:
                        arr = brain.navigator.spatial_map.to_array()
                        heading = brain.navigator.spatial_map._heading_deg
                        self._send_json({"entries": arr, "heading": round(heading)})
                    except Exception:
                        self._send_json({"entries": [], "heading": 0})
                    return
                if self.path == "/api/llm_log" or self.path.startswith("/api/llm_log?"):
                    since_id = 0
                    if "?" in self.path:
                        for part in self.path.split("?", 1)[1].split("&"):
                            if part.startswith("since="):
                                try:
                                    since_id = int(part.split("=", 1)[1])
                                except ValueError:
                                    pass
                    self._send_json(brain.llm_log.entries(since_id))
                    return
                if self.path == "/api/topo_map":
                    topo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "topo_map.json")
                    try:
                        with open(topo_path) as f:
                            topo_data = json.load(f)
                        # Also include current_room from brain status
                        try:
                            topo_data["current_room"] = brain.status().get("room", topo_data.get("current_room", "unknown"))
                        except Exception:
                            pass
                        self._send_json(topo_data)
                    except FileNotFoundError:
                        self._send_json({"error": "topo_map.json not found"})
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
                if self.path == "/api/calibration":
                    self._send_json(brain.set_calibration(**data))
                    return
                if self.path == "/api/test_turn":
                    self._send_json(brain.test_turn(float(data.get("degrees", 90))))
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
                if self.path == "/api/navigate_vlm":
                    brain.start_vlm_navigation(data.get("target", ""))
                    self._send_json({"ok": True})
                    return
                if self.path == "/api/set_room":
                    room = data.get("room", "").strip().lower().replace(" ", "_")
                    if room:
                        result = brain.set_current_room(room)
                        self._send_json(result)
                    else:
                        self._send_json({"error": "no room specified"})
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
                if self.path == "/api/bt/scan":
                    devices = bt.scan(duration=int(data.get("duration", 8)))
                    paired = [d["mac"] for d in bt.paired_devices()]
                    for d in devices:
                        d["paired"] = d["mac"] in paired
                        d["connected"] = bt.is_connected(d["mac"])
                    self._send_json({"devices": devices})
                    return
                if self.path == "/api/bt/pair":
                    mac = data.get("mac", "").strip()
                    if not mac:
                        self._send_json({"error": "missing mac"})
                        return
                    result = bt.pair_and_connect(mac)
                    self._send_json(result)
                    return
                if self.path == "/api/bt/connect":
                    mac = data.get("mac", "").strip()
                    if not mac:
                        self._send_json({"error": "missing mac"})
                        return
                    result = bt.connect(mac)
                    self._send_json(result)
                    return
                if self.path == "/api/bt/disconnect":
                    mac = data.get("mac", "").strip()
                    if not mac:
                        self._send_json({"error": "missing mac"})
                        return
                    result = bt.disconnect(mac)
                    self._send_json(result)
                    return
                if self.path == "/api/bt/remove":
                    mac = data.get("mac", "").strip()
                    if not mac:
                        self._send_json({"error": "missing mac"})
                        return
                    result = bt.remove(mac)
                    self._send_json(result)
                    return
                if self.path == "/api/bt/status":
                    paired = bt.paired_devices()
                    prefs = bt.load_preferred()
                    for d in paired:
                        d["connected"] = bt.is_connected(d["mac"])
                        d["preferred"] = d["mac"] == prefs.get("mac", "")
                    sink = bt.get_bt_audio_sink()
                    self._send_json({"paired": paired, "bt_sink": sink})
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
