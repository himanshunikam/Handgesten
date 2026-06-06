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

import math

import torch.nn as nn
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.bn(self.temporal_spatial_conv(x))
        x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class R3DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock, pretrained=False):
        super(R3DClassifier, self).__init__()

        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)

        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def get_1x_lr_params(model):
        """
        This generator returns all the parameters for the conv layer of the net.
        """
        b = [model.res3d]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k


    def get_10x_lr_params(model):
        """
        This generator returns all the parameters for the fc layer of the net.
        """
        b = [model.linear]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

if __name__ == "__main__":
    import torch
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = R3DClassifier(101, (2, 2, 2, 2), pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())
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