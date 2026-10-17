# UAV-Crack Segmentation — U-Net (BCE + Dice)

Baseline implementation and experimental evidence for the UAV-Crack track on the
[Visual Language Perception (VLP) platform](http://vlp.chd.edu.cn/tasks/4).

**Method:** `UNet-BCE-Dice` | **Submission:** 17 October 2026

## Results

| Metric | Internal validation | Official leaderboard |
|---|---:|---:|
| mIoU | 78.51% | 79.47% |
| Crack IoU | 60.00% | 61.81% |
| Crack F1 | 75.00% | 76.40% |
| Precision | 75.32% | 77.52% |
| Recall | 74.68% | 75.31% |
| Pixel accuracy | 97.15% | 97.27% |

Official results are from the platform leaderboard; internal results use a source-group-separated validation set.

## Data and experimental setup

The official dataset has 1,200 labeled images and 300 evaluation images at 672 × 378 pixels.
Images are organized across X4, X8 and X16 zoom levels. The internal split has 993
training patches (157 source groups) and 207 validation patches (40 source groups), with
zero shared source groups. See [`data/README.md`](data/README.md) for dataset access.

The baseline uses PyTorch U-Net, BCE + Dice loss, AdamW, learning rate 0.0003,
batch size 8, 20 epochs, and selects the best checkpoint by validation Crack IoU (epoch 17).
Training image size is 512 × 288; output masks are restored to 672 × 378.

## Run

```bash
pip install -r requirements.txt
python code/uav_crack_pipeline.py train --data-root dataset --train-csv data/train_split.csv --val-csv data/val_split.csv --output uav_results
python code/uav_crack_pipeline.py visualize --data-root dataset --val-csv data/val_split.csv --checkpoint uav_results/best_model.pt --output uav_results/examples
python code/uav_crack_pipeline.py predict --checkpoint uav_results/best_model.pt --val-images dataset/leftImg8bit/val --output uav_submission
```

**Note:** The `--val-images` path must point directly to the folder containing the 300 evaluation JPGs. Adjust it to match the downloaded dataset layout.

The prediction command creates `uav_submission/result.zip` with 300 grayscale masks containing binary pixel values 0 and 1.

## Repository contents

- `code/`: training, evaluation, visualization, and submission code
- `data/`: group-separated split CSVs and dataset access instructions
- `results/`: training history, best validation metrics, learning curves and six qualitative comparisons

## Limitations

Thin crack branches may be missed, and textured pavement can generate false positives.
This is a baseline, not a claim of state-of-the-art performance. Separate zoom-level
analysis and stronger segmentation methods remain future work.

The original dataset, model checkpoint, and official evaluation masks are intentionally
not redistributed here. No public source-code license is granted by default.
