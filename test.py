import torch
import cv2
import numpy as np
from main import R3DClassifier

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "models/model_pretrained_best.pth"
NUM_FRAMES      = 16       # must match FlowDatensatz training
RECORD_SECONDS  = 2.5
CAMERA_INDEX    = 0
LOAD_SIZE       = 128      # resize before crop — must match training
CROP_SIZE       = 112

LABEL_MAP = {
    0: "geste_0",  1: "geste_1",  2: "geste_2",
    3: "class_1",  4: "class_2",
    5: "Gesture01", 6: "Gesture02",
    7: "hand_turn", 8: "ok_sign", 9: "thumb_up",
}

# ── Model ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = R3DClassifier(10, (2, 2, 2, 2))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()
print(f"Loaded {MODEL_PATH} on {device}")


# ── Optical flow preprocessing (mirrors FlowDatensatz exactly) ────────────────
def frames_to_flow_tensor(frames_buffer):
    """Convert a list of RGB frames into a (1, 3, T, H, W) flow tensor."""
    indices = np.linspace(0, len(frames_buffer) - 1, NUM_FRAMES + 1, dtype=int)

    # Resize all sampled frames to LOAD_SIZE
    raw = [cv2.resize(frames_buffer[i], (LOAD_SIZE, LOAD_SIZE)) for i in indices]

    # Centre crop coordinates
    ci = cj = (LOAD_SIZE - CROP_SIZE) // 2

    flow_frames = []
    for t in range(NUM_FRAMES):
        g1 = cv2.cvtColor(raw[t],     cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(raw[t + 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )  # (H, W, 2)

        flow = flow[ci:ci + CROP_SIZE, cj:cj + CROP_SIZE]
        fx, fy = flow[..., 0], flow[..., 1]
        mag    = np.sqrt(fx**2 + fy**2)
        scale  = mag.max() + 1e-6

        ch = np.stack([
            np.clip(fx / scale * 0.5 + 0.5, 0, 1),
            np.clip(fy / scale * 0.5 + 0.5, 0, 1),
            np.clip(mag / scale,             0, 1),
        ], axis=0).astype(np.float32)

        flow_frames.append(torch.from_numpy(ch))

    # (T, 3, H, W) → (1, T, 3, H, W) → (1, 3, T, H, W)
    return torch.stack(flow_frames).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)


def classify(frames_buffer):
    video = frames_to_flow_tensor(frames_buffer)
    with torch.no_grad():
        probs = torch.softmax(model(video), dim=1)[0]
    top3 = probs.argsort(descending=True)[:3].tolist()
    return [(LABEL_MAP[i], float(probs[i])) for i in top3]


# ── UI helpers ────────────────────────────────────────────────────────────────
def put_text(img, text, pos, scale=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ── Main loop ─────────────────────────────────────────────────────────────────
IDLE, COUNTDOWN, RECORDING, RESULT = "idle", "countdown", "recording", "result"
COUNTDOWN_SECONDS = 3

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

state           = IDLE
frames_buffer   = []
countdown_start = 0.0
record_start    = 0.0
top3_results    = []

print("SPACE = start countdown  |  Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w    = display.shape[:2]
    now     = cv2.getTickCount() / cv2.getTickFrequency()

    if state == IDLE:
        put_text(display, "Press SPACE to record a gesture", (20, 50), scale=0.9)
        put_text(display, "Q: quit", (20, h - 20), scale=0.65, color=(180, 180, 180))

    elif state == COUNTDOWN:
        elapsed   = now - countdown_start
        remaining = max(0, COUNTDOWN_SECONDS - int(elapsed))
        label     = str(remaining) if remaining > 0 else "GO!"
        color     = (0, 200, 255) if remaining > 0 else (0, 255, 80)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
        put_text(display, label, ((w - tw) // 2, (h + th) // 2), scale=4.0, color=color, thickness=6)
        put_text(display, "Get ready...", (20, 50), scale=0.85, color=(220, 220, 220))

        if elapsed >= COUNTDOWN_SECONDS:
            state         = RECORDING
            frames_buffer = []
            record_start  = now

    elif state == RECORDING:
        elapsed   = now - record_start
        remaining = max(0.0, RECORD_SECONDS - elapsed)
        frames_buffer.append(rgb)

        cv2.rectangle(display, (4, 4), (w - 5, h - 5), (0, 0, 220), 6)
        put_text(display, f"Recording  {remaining:.1f}s", (20, 55), scale=1.2, color=(80, 80, 255), thickness=2)
        put_text(display, f"Frames: {len(frames_buffer)}", (20, 100), scale=0.75, color=(200, 200, 255))

        if elapsed >= RECORD_SECONDS:
            top3_results  = classify(frames_buffer)
            state         = RESULT
            frames_buffer = []

    elif state == RESULT:
        top_label, top_conf = top3_results[0]
        header_color = (0, int(200 * top_conf), 0) if top_conf > 0.5 else (0, 140, 255)

        put_text(display, f"Gesture: {top_label}", (20, 55), scale=1.3, color=header_color, thickness=3)

        bar_x, bar_y0, bar_w_max, bar_h = 20, 90, w - 40, 28
        for rank, (lbl, conf) in enumerate(top3_results):
            y         = bar_y0 + rank * (bar_h + 12)
            filled    = int(bar_w_max * conf)
            bar_color = (0, 180, 0) if rank == 0 else (0, 120, 200)
            cv2.rectangle(display, (bar_x, y), (bar_x + bar_w_max, y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(display, (bar_x, y), (bar_x + filled,    y + bar_h), bar_color,    -1)
            put_text(display, f"{lbl}  {conf * 100:.1f}%",
                     (bar_x + 6, y + bar_h - 6), scale=0.65, color=(255, 255, 255), thickness=1)

        put_text(display, "SPACE: record again  |  Q: quit", (20, h - 20), scale=0.65, color=(180, 180, 180))

    cv2.imshow("Gesture Classifier", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        if state in (IDLE, RESULT):
            state           = COUNTDOWN
            countdown_start = cv2.getTickCount() / cv2.getTickFrequency()

cap.release()
cv2.destroyAllWindows()
