"""tools.py — Single source of truth for all rover tool definitions.

Defines tools once in a neutral format and provides converters for each
voice/LLM provider (OpenAI/xAI, Gemini, ElevenLabs).

Adding a new tool: add it to TOOLS below, then add the handler in
rover_brain_llm._xai_tool_dispatch() (or rover_brain.xai_tool_dispatch).
That's it — all providers pick it up automatically.
"""

# ── Canonical tool definitions ────────────────────────────────────────

TOOLS = [
    {
        "name": "send_rover_commands",
        "description": (
            "Send raw ESP32 JSON commands to control wheels, gimbal, lights, "
            "and OLED. Use this for ALL physical actions: moving, turning, "
            "nodding, shaking head, blinking lights. Commands are sent "
            "sequentially. Include a duration (seconds) to auto-stop wheels "
            "after that time."
        ),
        "parameters": [
            {
                "name": "commands",
                "type": "array",
                "items_type": "object",
                "description": "List of ESP32 JSON command objects to send sequentially.",
                "required": True,
            },
            {
                "name": "duration",
                "type": "number",
                "description": (
                    "Optional: seconds to wait before sending wheel-stop "
                    "command. Use for timed movements."
                ),
                "required": False,
            },
        ],
    },
    {
        "name": "look_at_camera",
        "description": (
            "Move the gimbal to a position and describe what the camera sees. "
            "Use this whenever you need to see something. You CANNOT see "
            "without calling this. Returns a text description of the scene."
        ),
        "parameters": [
            {
                "name": "pan",
                "type": "number",
                "description": "Gimbal pan angle (-180 to 180). 0=forward. Default 0.",
                "required": False,
            },
            {
                "name": "tilt",
                "type": "number",
                "description": "Gimbal tilt angle (-30 to 90). 0=level. Default 0.",
                "required": False,
            },
            {
                "name": "question",
                "type": "string",
                "description": (
                    "What to look for or describe. E.g. 'what do you see', "
                    "'is there a person', 'describe the room'."
                ),
                "required": False,
            },
        ],
    },
    {
        "name": "navigate_to",
        "description": (
            "Autonomously navigate toward a named object or location. The "
            "rover will search, find, and drive to the target using camera "
            "vision. Takes 10-60 seconds. Returns success/failure."
        ),
        "parameters": [
            {
                "name": "target",
                "type": "string",
                "description": "Object or location to navigate to. E.g. 'door', 'basket', 'printer', 'desk'.",
                "required": True,
            },
        ],
    },
    {
        "name": "search_for",
        "description": (
            "Systematically sweep the gimbal to search for a named object. "
            "Checks all angles. Returns whether the object was found and "
            "its location."
        ),
        "parameters": [
            {
                "name": "target",
                "type": "string",
                "description": "Object to search for. E.g. 'person', 'cup', 'chair'.",
                "required": True,
            },
        ],
    },
    {
        "name": "remember",
        "description": "Save a note to persistent memory. Use when the user asks you to remember something.",
        "parameters": [
            {
                "name": "note",
                "type": "string",
                "description": "The note to remember.",
                "required": True,
            },
        ],
    },
    {
        "name": "get_status",
        "description": "Get rover status: battery voltage, current pose, tracker state, spatial map summary.",
        "parameters": [],
    },
    {
        "name": "set_speed",
        "description": (
            "Set the rover's speed scale. Level 1 = 10% (very slow), "
            "level 5 = 50%, level 10 = 100% (max). Default is level 2 (20%)."
        ),
        "parameters": [
            {
                "name": "level",
                "type": "integer",
                "description": "Speed level 1-10.",
                "required": True,
            },
        ],
    },
    {
        "name": "track_object",
        "description": (
            "Lock the gimbal onto a named object using YOLO detection and "
            "keep it centered for a duration. Returns the compass heading "
            "toward the object. Gimbal only — does NOT drive wheels."
        ),
        "parameters": [
            {
                "name": "target",
                "type": "string",
                "description": "Object label to track. E.g. 'cup', 'person', 'chair', 'hand'.",
                "required": True,
            },
            {
                "name": "duration",
                "type": "number",
                "description": "Seconds to track the object. Default 10.",
                "required": False,
            },
        ],
    },
    {
        "name": "follow_person",
        "description": (
            "Follow a target (person, dog, car, etc.) maintaining distance. "
            "Launches YOLO visual tracking with collision avoidance. "
            "Just call it and you're done, no further planning needed. "
            "Stops when target is lost or duration expires."
        ),
        "parameters": [
            {
                "name": "target",
                "type": "string",
                "description": "What to follow: 'person', 'dog', 'car', etc. Default 'person'.",
                "required": False,
            },
            {
                "name": "duration",
                "type": "number",
                "description": "Max seconds to follow. Default 60.",
                "required": False,
            },
        ],
    },
    {
        "name": "correct_label",
        "description": (
            "Fix a YOLO detection label that is wrong. Use this when you see "
            "via camera that YOLO has misidentified an object (e.g. YOLO says "
            "'cell phone' but it's actually a 'hand'). The correction persists "
            "across restarts."
        ),
        "parameters": [
            {
                "name": "yolo_label",
                "type": "string",
                "description": "The incorrect label YOLO is currently using.",
                "required": True,
            },
            {
                "name": "correct_label",
                "type": "string",
                "description": "The correct label for the object.",
                "required": True,
            },
        ],
    },
]

# Tools whose results should be read back to the user by voice
DATA_TOOLS = {"look_at_camera", "get_status", "search_for", "navigate_to", "track_object", "follow_person"}


# ── Provider-specific converters ──────────────────────────────────────

def _to_json_schema_type(param):
    """Convert a neutral param dict to a JSON Schema property dict."""
    prop = {
        "type": param["type"],
        "description": param["description"],
    }
    if param["type"] == "array" and "items_type" in param:
        prop["items"] = {"type": param["items_type"]}
    return prop


def to_openai():
    """Convert TOOLS to OpenAI/xAI function calling format.

    Returns list of {"type":"function", "name":..., "parameters":...}
    """
    out = []
    for tool in TOOLS:
        properties = {}
        required = []
        for p in tool["parameters"]:
            properties[p["name"]] = _to_json_schema_type(p)
            if p.get("required"):
                required.append(p["name"])
        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        out.append({
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": schema,
        })
    return out


def to_gemini():
    """Convert TOOLS to Gemini function_declarations format.

    Returns [{"function_declarations": [...]}]
    """
    declarations = []
    for tool in TOOLS:
        properties = {}
        required = []
        for p in tool["parameters"]:
            properties[p["name"]] = _to_json_schema_type(p)
            if p.get("required"):
                required.append(p["name"])
        decl = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }
        if required:
            decl["parameters"]["required"] = required
        declarations.append(decl)
    return [{"function_declarations": declarations}]


def tool_names():
    """Return list of tool name strings (for ElevenLabs registration)."""
    return [t["name"] for t in TOOLS]
