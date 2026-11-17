# Sign Bridge

Train your own signs on a webcam. Static poses + keyframe sequences (motion).

## Modes

- **Static** — one pose per sign. Hold to commit.
- **Keyframe** — 2–5 ordered poses. Do them in sequence to commit (e.g. hello).
- **1 or 2 hands** — both work automatically (no separate 1-hand mode).

## Run

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Train keyframe sign

1. Train tab → **Keyframe**
2. Set 2–5 steps, type name
3. Record each step in order
4. **Train model**
5. Live: perform poses 1→2→… quickly

## Keys

R record · T train · S speak · U undo · Esc cancel
