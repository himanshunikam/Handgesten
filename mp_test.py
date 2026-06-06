import torch
import cv2
import numpy as np
import mediapipe
import torch.nn as nn

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH           = "models/mp_model.pth"
NUM_FRAMES           = 30
RECORD_SECONDS       = 2.5
CAMERA_INDEX         = 0
MEDIAPIPE_MODEL_PATH = r"C:\Users\nikam\ML2\v1\models\hand_landmarker.task"

LABEL_MAP = {
    0: "geste_0",  1: "geste_1",  2: "geste_2",
    3: "class_1",  4: "class_2",
    5: "Gesture01", 6: "Gesture02",
    7: "hand_turn", 8: "ok_sign",  9: "thumb_up",
}

# ── Model (must match exp05) ──────────────────────────────────────────────────
class LandmarkLSTM(nn.Module):
    def __init__(self, input_dim=63, hidden=128, layers=2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers,
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden * 2, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ── Setup ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = LandmarkLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()
print(f"Loaded {MODEL_PATH} on {device}")

BaseOptions       = mediapipe.tasks.BaseOptions
HandLandmarker    = mediapipe.tasks.vision.HandLandmarker
HandLandmarkerOptions = mediapipe.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mediapipe.tasks.vision.RunningMode

mp_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)


# ── Preprocessing (must exactly match exp05 normalize_landmarks) ──────────────
def normalize_landmarks(seq):
    seq   = seq.copy()
    wrist = seq[:, 0:1, :]
    seq  -= wrist
    scale = np.linalg.norm(seq[:, 9, :], axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-6)[:, :, None]
    seq  /= scale
    return seq


def classify(landmark_buffer):
    indices = np.linspace(0, len(landmark_buffer) - 1, NUM_FRAMES, dtype=int)
    seq = np.array([landmark_buffer[i] for i in indices], dtype=np.float32)  # (T, 21, 3)
    seq = normalize_landmarks(seq)
    x   = torch.from_numpy(seq.reshape(NUM_FRAMES, -1)).unsqueeze(0).to(device)  # (1, T, 63)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top3 = probs.argsort(descending=True)[:3].tolist()
    return [(LABEL_MAP[i], float(probs[i])) for i in top3]


# ── UI helpers ────────────────────────────────────────────────────────────────
def put_text(img, text, pos, scale=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hand(img, hand_landmarks):
    """Draw 21 landmark dots on the frame."""
    h, w = img.shape[:2]
    for lm in hand_landmarks:
        cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)


# ── Main loop ─────────────────────────────────────────────────────────────────
IDLE, COUNTDOWN, RECORDING, RESULT = "idle", "countdown", "recording", "result"
COUNTDOWN_SECONDS = 3

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

state           = IDLE
landmark_buffer = []
last_valid_lm   = np.zeros((21, 3), dtype=np.float32)
countdown_start = 0.0
record_start    = 0.0
top3_results    = []

print("SPACE = start countdown  |  Q = quit")

with HandLandmarker.create_from_options(mp_options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w    = display.shape[:2]
        now     = cv2.getTickCount() / cv2.getTickFrequency()

        # Run MediaPipe every frame so skeleton is always visible
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb)
        results  = landmarker.detect(mp_image)
        detected = bool(results.hand_landmarks)

        if state == IDLE:
            if detected:
                draw_hand(display, results.hand_landmarks[0])
            put_text(display, "Press SPACE to record a gesture", (20, 50), scale=0.9)
            put_text(display, "Q: quit", (20, h - 20), scale=0.65, color=(180, 180, 180))

        elif state == COUNTDOWN:
            if detected:
                draw_hand(display, results.hand_landmarks[0])
            elapsed   = now - countdown_start
            remaining = max(0, COUNTDOWN_SECONDS - int(elapsed))
            label     = str(remaining) if remaining > 0 else "GO!"
            color     = (0, 200, 255) if remaining > 0 else (0, 255, 80)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
            put_text(display, label, ((w - tw) // 2, (h + th) // 2), scale=4.0, color=color, thickness=6)
            put_text(display, "Get ready...", (20, 50), scale=0.85, color=(220, 220, 220))

            if elapsed >= COUNTDOWN_SECONDS:
                state           = RECORDING
                landmark_buffer = []
                last_valid_lm   = np.zeros((21, 3), dtype=np.float32)
                record_start    = now

        elif state == RECORDING:
            elapsed   = now - record_start
            remaining = max(0.0, RECORD_SECONDS - elapsed)

            if detected:
                coords        = np.array([[l.x, l.y, l.z] for l in results.hand_landmarks[0]],
                                         dtype=np.float32)
                last_valid_lm = coords
                landmark_buffer.append(coords)
                draw_hand(display, results.hand_landmarks[0])
            else:
                landmark_buffer.append(last_valid_lm.copy())  # forward-fill missing detection

            cv2.rectangle(display, (4, 4), (w - 5, h - 5), (0, 0, 220), 6)
            put_text(display, f"Recording  {remaining:.1f}s", (20, 55),
                     scale=1.2, color=(80, 80, 255), thickness=2)
            put_text(display, f"Frames: {len(landmark_buffer)}", (20, 100),
                     scale=0.75, color=(200, 200, 255))

            if elapsed >= RECORD_SECONDS:
                if len(landmark_buffer) >= NUM_FRAMES:
                    top3_results = classify(landmark_buffer)
                else:
                    top3_results = [("no hand detected", 0.0), ("--", 0.0), ("--", 0.0)]
                state           = RESULT
                landmark_buffer = []

        elif state == RESULT:
            top_label, top_conf = top3_results[0]
            header_color = (0, int(200 * top_conf), 0) if top_conf > 0.5 else (0, 140, 255)

            put_text(display, f"Gesture: {top_label}", (20, 55),
                     scale=1.3, color=header_color, thickness=3)

            bar_x, bar_y0, bar_w_max, bar_h = 20, 90, w - 40, 28
            for rank, (lbl, conf) in enumerate(top3_results):
                y         = bar_y0 + rank * (bar_h + 12)
                filled    = int(bar_w_max * conf)
                bar_color = (0, 180, 0) if rank == 0 else (0, 120, 200)
                cv2.rectangle(display, (bar_x, y), (bar_x + bar_w_max, y + bar_h), (60, 60, 60), -1)
                cv2.rectangle(display, (bar_x, y), (bar_x + filled,    y + bar_h), bar_color,    -1)
                put_text(display, f"{lbl}  {conf * 100:.1f}%",
                         (bar_x + 6, y + bar_h - 6), scale=0.65,
                         color=(255, 255, 255), thickness=1)

            put_text(display, "SPACE: record again  |  Q: quit",
                     (20, h - 20), scale=0.65, color=(180, 180, 180))

        cv2.imshow("Gesture Classifier (MediaPipe)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            if state in (IDLE, RESULT):
                state           = COUNTDOWN
                countdown_start = cv2.getTickCount() / cv2.getTickFrequency()

cap.release()
cv2.destroyAllWindows()
