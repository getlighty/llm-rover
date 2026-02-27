#!/usr/bin/env python3
"""Test xAI Realtime with manual turn (commit buffer after 3s of audio)."""
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

        # Use MANUAL turn detection (null) — we'll commit the buffer ourselves
        config = {
            "type": "session.update",
            "session": {
                "voice": "Rex",
                "instructions": "Say exactly what you heard the user say, then say hello.",
                "turn_detection": None,  # manual mode
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                },
            },
        }
        await ws.send(json.dumps(config))
        print("Session configured (manual turn mode)")

        # Wait for session.updated
        while True:
            msg = json.loads(await ws.recv())
            print(f"  [rx] {msg['type']}")
            if msg["type"] == "session.updated":
                break

        # Record 3 seconds of mic audio
        print("\n>>> SPEAK NOW (recording 3 seconds)...")
        arecord = subprocess.Popen(
            ["arecord", "-D", "plughw:0,0", "-f", "S16_LE", "-r", "24000", "-c", "1",
             "-d", "3", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        audio_data = arecord.stdout.read()
        arecord.wait()

        samples = np.frombuffer(audio_data, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        print(f"Recorded {len(audio_data)} bytes ({len(samples)} samples), RMS={rms:.1f}, max={np.max(np.abs(samples))}")

        # Send audio in chunks
        chunk_size = 4800
        chunks_sent = 0
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            b64 = base64.b64encode(chunk).decode("ascii")
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
            chunks_sent += 1
        print(f"Sent {chunks_sent} chunks")

        # Commit the buffer (manual turn)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("Buffer committed, requesting response...")
        await ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["text", "audio"]}}))

        # Listen for response
        print("\nWaiting for response...")
        for _ in range(60):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                etype = msg["type"]
                if etype == "ping":
                    continue
                if "transcript" in msg and msg.get("transcript"):
                    print(f"  [rx] {etype}: transcript={msg['transcript']}")
                elif etype == "response.output_audio.delta":
                    print(f"  [rx] audio delta ({len(msg.get('delta',''))} b64 chars)")
                elif etype == "response.done":
                    print(f"  [rx] response.done")
                    break
                elif etype == "error":
                    print(f"  [rx] ERROR: {msg.get('error', {})}")
                    break
                else:
                    print(f"  [rx] {etype}")
            except asyncio.TimeoutError:
                print("  (timeout waiting for event)")
                break

asyncio.run(main())
