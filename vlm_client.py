"""
VLM Client — talks to Ollama running qwen2.5vl:3b (or any vision model).
Sends camera frames with structured prompts, returns JSON scene descriptions.
"""

import base64
import json
import time
import requests
import logging

log = logging.getLogger("vlm_client")

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:3b"


class VLMClient:
    """Client for Ollama VLM inference."""

    def __init__(self, base_url=DEFAULT_URL, model=DEFAULT_MODEL, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def describe_scene(self, jpeg_bytes, prompt=None):
        """Send a camera frame and get a scene description as JSON.

        Args:
            jpeg_bytes: JPEG image bytes
            prompt: Custom prompt (default: structured obstacle/path query)

        Returns:
            dict with parsed JSON response, or {"raw": text} if not JSON
        """
        if prompt is None:
            prompt = (
                "Describe this scene from a ground rover's perspective. "
                "Output JSON with these fields:\n"
                '  "obstacles": [{"label": str, "position": "left"|"center"|"right", '
                '"distance": "near"|"mid"|"far"}],\n'
                '  "free_path": "left"|"center"|"right"|"none",\n'
                '  "landmarks": [{"label": str, "position": "left"|"center"|"right"}],\n'
                '  "room_type": str (e.g. "kitchen", "hallway", "workshop"),\n'
                '  "summary": str (one sentence)\n'
                "Output ONLY the JSON object, no other text."
            )

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }

        try:
            t0 = time.monotonic()
            resp = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            elapsed = time.monotonic() - t0
            resp.raise_for_status()
            data = resp.json()
            text = data["message"]["content"].strip()
            log.debug("VLM response in %.1fs: %s", elapsed, text[:200])

            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text, "elapsed": round(elapsed, 2)}

        except requests.RequestException as e:
            log.error("VLM request failed: %s", e)
            return {"error": str(e)}

    def query(self, jpeg_bytes, question):
        """Ask a free-form question about an image.

        Args:
            jpeg_bytes: JPEG image bytes
            question: Natural language question

        Returns:
            str response text
        """
        result = self.describe_scene(jpeg_bytes, prompt=question)
        if isinstance(result, dict):
            return result.get("raw", result.get("summary", json.dumps(result)))
        return str(result)

    def is_alive(self):
        """Check if Ollama is running and the model is available."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/tags", timeout=3
            )
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(m.get("name", "").startswith(self.model.split(":")[0])
                       for m in models)
        except requests.RequestException:
            return False


if __name__ == "__main__":
    import sys

    client = VLMClient()
    if not client.is_alive():
        print(f"Ollama not running or {DEFAULT_MODEL} not available.")
        print("Run: ollama pull qwen2.5vl:3b")
        sys.exit(1)

    # Grab a frame
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            jpeg = f.read()
    else:
        try:
            resp = requests.get("http://localhost:8765/api/snap", timeout=5)
            jpeg = resp.content
        except Exception:
            print("No image source available")
            sys.exit(1)

    print("Querying VLM via Ollama...")
    t0 = time.monotonic()
    result = client.describe_scene(jpeg)
    elapsed = time.monotonic() - t0
    print(f"\nResult ({elapsed:.1f}s):")
    print(json.dumps(result, indent=2))
