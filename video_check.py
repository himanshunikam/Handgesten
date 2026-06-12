import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

data_dir = Path("data")
frame_counts = defaultdict(list)

for video_path in sorted(data_dir.rglob("*.MP4")):
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_counts[count].append(str(video_path))

counts = [c for c, paths in frame_counts.items() for _ in paths]

if len(frame_counts) == 1:
    print(f"All {len(counts)} videos have {counts[0]} frames.")
else:
    print(f"Inconsistent frame counts across {len(counts)} videos:")
    for count, paths in sorted(frame_counts.items()):
        print(f"  {count} frames: {len(paths)} videos")

plt.figure(figsize=(10, 5))
plt.hist(counts, bins=range(min(counts), max(counts) + 2), edgecolor="black", color="steelblue")
plt.xlabel("Frame Count")
plt.ylabel("Number of Videos")
plt.title("Distribution of Frame Counts Across Videos")
plt.xticks(sorted(frame_counts.keys()))
plt.tight_layout()
plt.show()