import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report

# ─── CONFIG ───────────────────────────────────────────────
NUM_FRAMES   = 8      # frames sampled per video
IMG_SIZE     = 112    # resize each frame to 112x112
BATCH_SIZE   = 4
EPOCHS       = 15
LR           = 1e-4
DATA_ROOT    = "new_data"   # folder with train/val/test subfolders
LABEL_MAP    = {"geste_0": 0, "geste_1": 1, "geste_2": 2}
# ──────────────────────────────────────────────────────────


# ─── DATASET ──────────────────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.split     = split          # e.g. "train"
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_frames(self, rel_path):
        # rel_path is like "geste_0/00002.mp4"
        full_path = os.path.join(DATA_ROOT, self.split, rel_path)
        cap = cv2.VideoCapture(full_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        cap.release()
        return frames   # list of 8 numpy arrays (H, W, 3)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = LABEL_MAP[row["label"]]
        frames = self._load_frames(row["path"])  # list of numpy (H, W, 3) uint8

        # Convert numpy → tensor manually, BEFORE passing to transforms
        tensors = []
        for f in frames:
            t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0  # (3, H, W)
            if self.transform:
                t = self.transform(t)
            tensors.append(t)

        video_tensor = torch.stack(tensors).mean(dim=0)  # (3, H, W)
        return video_tensor, label


# ─── TRANSFORMS ───────────────────────────────────────────
# ─── TRANSFORMS ───────────────────────────────────────────
# No ToPILImage or ToTensor — we handle conversion manually
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])


# ─── DATALOADERS ──────────────────────────────────────────
train_ds = GestureDataset(f"{DATA_ROOT}/train/train.csv", "train", train_transform)
val_ds   = GestureDataset(f"{DATA_ROOT}/val/val.csv",     "val",   val_transform)
test_ds  = GestureDataset(f"{DATA_ROOT}/test/test.csv",   "test",  val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")


# ─── MODEL ────────────────────────────────────────────────
# model = models.mobilenet_v3_small(weights="DEFAULT")

# # Freeze all layers first
# for p in model.parameters():
#     p.requires_grad = False

# # Only train the classifier head
# model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
# # (head parameters are unfrozen by default since it's a new layer)

# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)


# # ─── TRAINING LOOP ────────────────────────────────────────
# def run_epoch(loader, training=True):
#     model.train() if training else model.eval()
#     total_loss, correct, total = 0, 0, 0

#     with torch.set_grad_enabled(training):
#         for imgs, labels in loader:
#             preds = model(imgs)
#             loss  = F.cross_entropy(preds, labels)

#             if training:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             total_loss += loss.item()
#             correct    += (preds.argmax(1) == labels).sum().item()
#             total      += len(labels)

#     return total_loss / len(loader), correct / total


# print("\nStarting training...\n")
# for epoch in range(1, EPOCHS + 1):
#     train_loss, train_acc = run_epoch(train_loader, training=True)
#     val_loss,   val_acc   = run_epoch(val_loader,   training=False)
#     print(f"Epoch {epoch:02d}/{EPOCHS} | "
#           f"Train Loss: {train_loss:.3f}  Acc: {train_acc:.0%} | "
#           f"Val Loss: {val_loss:.3f}  Acc: {val_acc:.0%}")

# torch.save(model.state_dict(), "gesture_model.pth")
# print("\nModel saved to gesture_model.pth")


# # ─── TEST ─────────────────────────────────────────────────
# model.load_state_dict(torch.load("gesture_model.pth"))
# model.eval()

# all_preds, all_labels = [], []
# with torch.no_grad():
#     for imgs, labels in test_loader:
#         preds = model(imgs).argmax(1)
#         all_preds.extend(preds.tolist())
#         all_labels.extend(labels.tolist())

# print("\n── Test Results ──────────────────────────────")
# print(classification_report(all_labels, all_preds,
#       target_names=["geste_0", "geste_1", "geste_2"]))