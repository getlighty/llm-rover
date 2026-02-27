#!/usr/bin/env python3
"""Test xAI Realtime server_vad with explicit threshold settings."""
import asyncio, base64, json, os, subprocess, sys, numpy as np

env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
import websockets

async def main():
    subprocess.run(["amixer", "-c", "0", "cset", "numid=2", "on"], capture_output=True)
    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}

    async with websockets.connect("wss://api.x.ai/v1/realtime",
                                   additional_headers=headers, ssl=True) as ws:
        print("Connected!")

        # Try server_vad with explicit threshold
        config = {
            "type": "session.update",
            "session": {
                "voice": "Rex",
                "instructions": "Say hello when you hear the user.",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                },
            },
        }
        await ws.send(json.dumps(config))
        print("Session configured (server_vad with threshold=0.5)")

        # Wait for config
        while True:
            msg = json.loads(await ws.recv())
            t = msg["type"]
            if t == "session.updated":
                vad = msg.get("session", {}).get("turn_detection", {})
                xvad = msg.get("session", {}).get("xvad_settings", {})
                print(f"  VAD: {vad}")
                print(f"  xVAD min_rms: {xvad.get('audio_floor_rms_vad_config', {})}")
                break
            elif t != "ping":
                print(f"  [rx] {t}")

        # Stream mic audio for 15 seconds
        print("\n>>> Streaming mic for 15s — SPEAK at any time...")
        arecord = subprocess.Popen(
            ["arecord", "-D", "plughw:0,0", "-f", "S16_LE", "-r", "24000", "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        loop = asyncio.get_event_loop()
        running = True
        chunk_count = 0

        async def send_audio():
            nonlocal chunk_count
            while running:
                data = await loop.run_in_executor(None, arecord.stdout.read, 4800)
                if not data:
                    break
                b64 = base64.b64encode(data).decode("ascii")
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                chunk_count += 1
                if chunk_count % 20 == 1:
                    s = np.frombuffer(data, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(s.astype(np.float64) ** 2)))
                    print(f"  [mic] #{chunk_count} rms={rms:.1f}")

        async def recv_events():
            async for raw in ws:
                msg = json.loads(raw)
                t = msg["type"]
                if t == "ping":
                    continue
                if "transcript" in msg and msg.get("transcript"):
                    print(f"  >>> {t}: {msg['transcript']}")
                elif t == "response.output_audio.delta":
                    print(f"  [audio] delta")
                elif t == "input_audio_buffer.speech_started":
                    print(f"  >>> SPEECH STARTED!")
                elif t == "input_audio_buffer.speech_stopped":
                    print(f"  >>> SPEECH STOPPED!")
                else:
                    print(f"  [rx] {t}")

        async def timer():
            nonlocal running
            await asyncio.sleep(15)
            running = False
            arecord.kill()
            print("\n--- timeout ---")

        try:
            await asyncio.gather(send_audio(), recv_events(), timer())
        except:
            pass
        finally:
            arecord.kill()

asyncio.run(main())
