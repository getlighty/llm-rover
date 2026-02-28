# Smart Search & Navigation — Algorithm Blueprint

```
                         ┌─────────────────────┐
                         │  "find blue basket"  │
                         │   User / Voice / AI  │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             Main Loop        xAI Tool         ElevenLabs
            (line 4558)    Dispatch 4171      Tool 4167
                    │           │                   │
                    └───────────┼───────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │  spatial_map.find(q)  │  ◄── 3-tier lookup:
                    │  exact → substr → fuz │      1. exact key
                    └──────────┬──────────┘       2. substring
                               │                   3. difflib 0.6
                      ┌────────┴────────┐
                      │ Fresh hit?      │ Stale/miss?
                      ▼                 ▼
                 Gimbal move      systematic_search()
                 (0 LLM calls)    ════════════════
                 → done                 │
                                        ▼
```

## Phase Flow

```
═══════════════════════════════════════════════════════════════
 PHASE 1 — Spatial Map (0 LLM calls, instant)
═══════════════════════════════════════════════════════════════

  spatial_map.find(target)
      │
      ├── Fresh (< 5 min) ──► gimbal to stored pan/tilt
      │                        → _found_and_center()
      │                        → return True
      │
      └── Stale ──► save hint_pan for Phase 3
           │
           └── Miss ──► continue to Phase 2


═══════════════════════════════════════════════════════════════
 PHASE 2 — Scout (up to 3 LLM calls, ~10s)
═══════════════════════════════════════════════════════════════

  3 wide-angle positions, tilt=0:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │   pan=-90°          pan=0°          pan=+90°     │
  │      ◄── scout ──►    ◄── scout ──►    ◄──      │
  │                                                  │
  └──────────────────────────────────────────────────┘

  Each position:
    1. _move_gimbal(pan, 0, spd=300)     ◄── 0.6s settle
    2. tracker.get_jpeg()                 ◄── 640×480 USB cam
    3. _tracker_vision(scout_prompt, jpg) ◄── Gemini 3.1 Pro
    4. parse JSON response

  Scout prompt asks for:
    {"found": bool, "objects": [...], "hint": "left"/"right"/"behind"/"unknown"}

  ├── found=true at any position?
  │     → _found_and_center() → return True
  │
  └── found=false
        → collect hints (directional clues)
        → map ALL objects seen to spatial_map
        → continue to Phase 3


═══════════════════════════════════════════════════════════════
 PHASE 3 — Targeted Checks (1-4 LLM calls, ~15s)
═══════════════════════════════════════════════════════════════

  Position priority:
    1. LLM hints:  "left"→-135°  "right"→+135°  "behind"→180°
    2. Stale spatial map position (hint_pan from Phase 1)
    3. Unchecked intermediates: -45°, +45°, -135°, +135°
    4. Max 4 positions (deduplicated)

  Each position:
    → _check_position(pan, tilt, prompt_key="check")
    → Focused prompt: "Is {target} visible? List all objects."

  ├── found=true? → _found_and_center() → return True
  └── not found?  → continue to Phase 4


═══════════════════════════════════════════════════════════════
 PHASE 4 — Fallback Sweep (7-14+ LLM calls)
═══════════════════════════════════════════════════════════════

  FORWARD HEMISPHERE (skip already-checked positions):

  Tilt 0°:
    -135°  -90°  -45°   0°   +45°  +90°  +135°
      ●      ●     ●     ●     ●     ●      ●

  Tilt 30°:
    -135°  -90°  -45°   0°   +45°  +90°  +135°
      ●      ●     ●     ●     ●     ●      ●

  ├── found? → _found_and_center() → return True
  │
  └── not found? → BODY ROTATION
        │
        ▼
  ┌─────────────────────────────────────┐
  │  Rotate body 180° LEFT              │
  │  wheels: L=-0.35  R=+0.35 m/s      │
  │  time: 180° / 120 dps = 1.5s       │
  │  gimbal centers to (0,0)            │
  │                                     │
  │  TRANSLATE SPATIAL MAP:             │
  │  all mapped objects' pan -= 180°    │
  │  normalize to [-180°, +180°]        │
  │  save to disk                       │
  └─────────────────────────────────────┘
        │
        ▼
  REAR HEMISPHERE (full sweep, same pattern):
    7 pans × 2 tilts = 14 positions

  ├── found? → _found_and_center() → return True
  └── not found → speak "Couldn't find X" → return False
```

## _found_and_center() Detail

```
═══════════════════════════════════════════════════════════════
 CENTERING + BODY ALIGNMENT
═══════════════════════════════════════════════════════════════

  Step 1: Fine-tune centering (up to 3 LLM rounds)
  ┌──────────────────────────────────────┐
  │  for i in range(3):                  │
  │    capture frame                     │
  │    if frame unchanged → done         │
  │    LLM: "Is target centered?"        │
  │    ├── centered=true → break         │
  │    └── centered=false                │
  │         → send gimbal adjust cmd     │
  │         → update cam_pan, cam_tilt   │
  └──────────────────────────────────────┘

  Step 2: Align body to gimbal heading
  ┌──────────────────────────────────────┐
  │  if |cam_pan| > 5°:                  │
  │                                      │
  │  SIMULTANEOUS:                       │
  │    Wheels: turn toward cam_pan       │
  │    Gimbal: counter-rotate to 0°      │
  │    (target stays in view!)           │
  │                                      │
  │  time = |cam_pan| / 120 dps          │
  │  → wheels stop                       │
  │  → gimbal at 0°, body now faces obj  │
  └──────────────────────────────────────┘

  Step 3: Save to spatial_map
    spatial_map.update([target], pan=0, tilt, world_pan)
```

## Navigation (after search finds target)

```
═══════════════════════════════════════════════════════════════
 NAVIGATE_TO_OBJECT — LLM-guided drive loop
═══════════════════════════════════════════════════════════════

  Step 1: Find target (search if needed)
  Step 2: Center gimbal on target (incremental PAN_STEP=15°, TILT_STEP=10°)
  Step 3: Align body (same as centering above)
  Step 4: Drive loop

  ┌─────────────────────────────────────────────┐
  │  Start wheels: L=0.12, R=0.12 m/s (creep)  │
  │                                             │
  │  for step in range(30):     ◄── max 30s     │
  │    wait NAV_LLM_INTERVAL (1s)               │
  │    check emergency_event                    │
  │    capture frame                            │
  │                                             │
  │    if frame unchanged → keep course         │
  │      (skip LLM call, save API cost)         │
  │                                             │
  │    LLM: "Is target visible? Direction?"     │
  │    ┌─────────────────────────────────┐      │
  │    │ Response:                       │      │
  │    │  v: visible?                    │      │
  │    │  close: within 0.5m?            │      │
  │    │  dir: left/right/center         │      │
  │    │  size: small/medium/large       │      │
  │    └─────────────┬───────────────────┘      │
  │                  │                          │
  │    close=true ───► STOP → "I'm at X" ✓     │
  │    v=false ──────► STOP → "Lost X"   ✗     │
  │    dir=left ─────► L=0.06  R=0.18          │
  │    dir=right ────► L=0.18  R=0.06          │
  │    dir=center ───► L=0.12  R=0.12          │
  │                                             │
  └─────────────────────────────────────────────┘
```

## Technology Stack

```
┌──────────────────────────────────────────────────────┐
│                    LLM LAYER                         │
│                                                      │
│  Gemini 3.1 Pro (gemini-3.1-pro-preview)            │
│  └─ _tracker_vision(): stateless vision call         │
│     ├── Image: JPEG resized to 512px, quality 60     │
│     ├── max_tokens: 250                              │
│     ├── Endpoint: generativelanguage.googleapis.com  │
│     └── Timeout: 30s                                 │
│                                                      │
│  Prompt templates (hot-swappable .md files):         │
│  ├── prompts/search.md  (scout, check, center)       │
│  └── prompts/nav.md     (steer, assess)              │
│      loaded via load_prompts() → {section: template} │
├──────────────────────────────────────────────────────┤
│                  PERCEPTION LAYER                    │
│                                                      │
│  USB Camera 640×480 wide-angle (~65° FOV)            │
│  └─ Background capture thread @ 30fps               │
│     └─ tracker.get_jpeg() → latest JPEG bytes        │
│                                                      │
│  Frame diff (nav only):                              │
│  └─ 80×60 grayscale, cv2.absdiff, 20% threshold     │
├──────────────────────────────────────────────────────┤
│                  STATE LAYER                         │
│                                                      │
│  SpatialMap (spatial_map.json, 459+ objects):         │
│  └─ {name: {pan, tilt, time, world_pan}}             │
│     ├── 3-tier lookup: exact → substr → fuzzy        │
│     ├── Staleness: 300s (5 min)                      │
│     └── Pan translation on body rotation             │
│                                                      │
│  PoseEstimator:                                      │
│  └─ body_yaw, cam_pan, cam_tilt, world_pan           │
├──────────────────────────────────────────────────────┤
│                 HARDWARE LAYER                       │
│                                                      │
│  ESP32 via UART serial (/dev/ttyTHS1 @ 115200)      │
│  ├── T:133  Gimbal   pan/tilt/spd/acc               │
│  ├── T:1    Wheels   L/R m/s (max ~1.3)             │
│  ├── T:132  Lights   IO4=base IO5=head (0-255)      │
│  └── T:130  Feedback (battery, encoder)              │
│                                                      │
│  Key constants:                                      │
│  ├── TURN_SPEED      = 0.35 m/s                     │
│  ├── TURN_RATE_DPS   = 120° /s (calibrated)         │
│  ├── NAV_DRIVE_SPEED = 0.12 m/s (slow creep)        │
│  ├── NAV_STEER_OFFSET= 0.06 m/s (gentle)            │
│  └── SEARCH_GIMBAL_SPD = 300                         │
└──────────────────────────────────────────────────────┘
```

## LLM Call Budget

```
Scenario                          LLM Calls    Time
─────────────────────────────────────────────────────
Spatial map fresh hit                  0        ~1s
Found during scout (Phase 2)        1-3       ~10s
Found via targeted hint (Phase 3)   4-7       ~25s
Found in forward sweep (Phase 4)    8-14      ~50s
Found after body rotation          15-28     ~100s
Not found (full 360° exhaustive)   ~28       ~120s

Typical search (object in room):    3-7       ~25s
─────────────────────────────────────────────────────
Old brute-force sweep:             14-28    ~60-120s
```

## Entry Points (callers)

| Caller | Location | Trigger |
|--------|----------|---------|
| Main loop | line 4558 | User says "find X" / "look at X", spatial map miss |
| xAI tool dispatch | line 4171 | `search_for` tool call from voice AI |
| xAI tool dispatch | line 4164 | `navigate_to` → search if not in map |
| navigate_to_object | line 3862 | Pre-drive object finding |
