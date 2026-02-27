#!/usr/bin/env python3
"""Minimal xAI Realtime WebSocket test — sends 5s of mic audio, prints all events."""
import asyncio
import base64
import json
import os
import subprocess
import sys

# Load .env
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
if not XAI_API_KEY:
    print("ERROR: XAI_API_KEY not set")
    sys.exit(1)

import websockets

WS_URL = "wss://api.x.ai/v1/realtime"

async def main():
    # Unmute card 0 mic
    subprocess.run(["amixer", "-c", "0", "cset", "numid=2", "on"], capture_output=True)

    headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
    print(f"Connecting to {WS_URL}...")

    async with websockets.connect(WS_URL, additional_headers=headers, ssl=True) as ws:
        print("Connected!")

        # Configure session
        config = {
            "type": "session.update",
            "session": {
                "voice": "Rex",
                "instructions": "You are a test assistant. Say hello when you hear someone.",
                "turn_detection": {"type": "server_vad"},
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                },
            },
        }
        await ws.send(json.dumps(config))
        print("Session config sent")

        # Send a text message first to verify the API responds
        print("Sending text hello to test response...")
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hi"}],
            },
        }))
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]},
        }))

        # Start mic recording
        print("Starting mic on plughw:0,0 at 24kHz...")
        arecord = subprocess.Popen(
            ["arecord", "-D", "plughw:0,0", "-f", "S16_LE", "-r", "24000", "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        chunks_sent = 0
        running = True

        async def send_audio():
            nonlocal chunks_sent
            loop = asyncio.get_event_loop()
            chunk_size = 24000 * 2 * 100 // 1000  # 100ms = 4800 bytes
            while running:
                data = await loop.run_in_executor(None, arecord.stdout.read, chunk_size)
                if not data:
                    break
                b64 = base64.b64encode(data).decode("ascii")
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
                chunks_sent += 1
                if chunks_sent % 10 == 1:
                    import numpy as np
                    samples = np.frombuffer(data, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    print(f"  [mic] chunk #{chunks_sent}, {len(data)} bytes, RMS={rms:.1f}")

        async def recv_events():
            nonlocal running
            async for msg in ws:
                event = json.loads(msg)
                etype = event.get("type", "")
                if etype == "ping":
                    continue
                if etype == "session.updated":
                    print(f"  [rx] {etype}: {json.dumps(event, indent=2)}")
                else:
                    print(f"  [rx] {etype}: {json.dumps(event)[:300]}")
                if etype == "response.output_audio.delta":
                    print(f"  [rx] (audio delta, {len(event.get('delta',''))} b64 chars)")

        async def timeout():
            nonlocal running
            await asyncio.sleep(10)
            print("\n--- 10s timeout ---")
            running = False
            arecord.kill()

        try:
            await asyncio.gather(send_audio(), recv_events(), timeout())
        except Exception as e:
            print(f"Error: {e}")
        finally:
            arecord.kill()

asyncio.run(main())
