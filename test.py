import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from main import R3DClassifier
# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = "exp04_best.pth"
NUM_FRAMES     = 16         # frames sampled from recording (must match training)
RECORD_SECONDS = 2.5       # how long to record each gesture
CAMERA_INDEX   = 0

LABEL_MAP = {
    0: "geste_0",
    1: "geste_1",
    2: "geste_2",
    3: "class_1",
    4: "class_2",
    5: "Gesture01",
    6: "Gesture02",
    7: "hand_turn",
    8: "ok_sign",
    9: "thumb_up",
}

# ── Model (must match exp_03 training definition) ─────────────────────────────
class mobilenet_handgesten(nn.Module):
    def __init__(self, num_classes=10, hidden_dim=128, num_layers=2):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.feature_extractor = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(
            input_size=576,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.avgpool(self.feature_extractor(x)).flatten(1)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        return self.classifier(out[:, -1, :])


# ── Setup ─────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

model = R3DClassifier(10, (2,2,2,2))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()
print(f"Loaded {MODEL_PATH} on {device}")

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])


def classify(frames_buffer):
    indices = np.linspace(0, len(frames_buffer) - 1, NUM_FRAMES, dtype=int)
    tensors = [transform(Image.fromarray(frames_buffer[i])) for i in indices]
    video = torch.stack(tensors).unsqueeze(0).permute(0,2,1,3,4).to(device)   # (1, T, C, H, W)
    with torch.no_grad():
        probs = torch.softmax(model(video), dim=1)[0]
    top3_idx = probs.argsort(descending=True)[:3].tolist()
    return [(LABEL_MAP[i], float(probs[i])) for i in top3_idx]


def put_text(img, text, pos, scale=1.0, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ── Main loop ─────────────────────────────────────────────────────────────────
IDLE, COUNTDOWN, RECORDING, RESULT = "idle", "countdown", "recording", "result"
COUNTDOWN_SECONDS = 3

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

state          = IDLE
frames_buffer  = []
countdown_start = 0.0
record_start   = 0.0
top3_results   = []   # list of (label, conf)

print("SPACE = start countdown  |  Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
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
        # big centered number
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 6)
        put_text(display, label, ((w - tw) // 2, (h + th) // 2), scale=4.0, color=color, thickness=6)
        put_text(display, "Get ready...", (20, 50), scale=0.85, color=(220, 220, 220))

        if elapsed >= COUNTDOWN_SECONDS:
            state        = RECORDING
            frames_buffer = []
            record_start = now

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

        # top-3 confidence bars
        bar_x, bar_y0, bar_w_max, bar_h = 20, 90, w - 40, 28
        for rank, (lbl, conf) in enumerate(top3_results):
            y      = bar_y0 + rank * (bar_h + 12)
            filled = int(bar_w_max * conf)
            bar_color = (0, 180, 0) if rank == 0 else (0, 120, 200)
            cv2.rectangle(display, (bar_x, y), (bar_x + bar_w_max, y + bar_h), (60, 60, 60), -1)
            cv2.rectangle(display, (bar_x, y), (bar_x + filled, y + bar_h), bar_color, -1)
            put_text(display, f"{lbl}  {conf * 100:.1f}%",
                     (bar_x + 6, y + bar_h - 6), scale=0.65, color=(255, 255, 255), thickness=1)

        put_text(display, "SPACE: record again  |  Q: quit", (20, h - 20), scale=0.65, color=(180, 180, 180))

    cv2.imshow("Gesture Classifier", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        if state in (IDLE, RESULT):
            state          = COUNTDOWN
            countdown_start = cv2.getTickCount() / cv2.getTickFrequency()

cap.release()
cv2.destroyAllWindows()
