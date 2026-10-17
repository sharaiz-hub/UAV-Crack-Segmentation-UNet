# Dataset

Download UAV-Crack from the official VLP competition platform (subject to its access and redistribution rules):
http://vlp.chd.edu.cn/tasks/4

Extract the dataset into `dataset/` at the repository root so it contains:
- `dataset/leftImg8bit/train/UAV-CrackX4/` (and X8, X16)
- `dataset/gtFine/train/UAV-CrackX4/` (and X8, X16)
- `dataset/leftImg8bit/val/` (300 evaluation JPG images, in the appropriate downloaded directory)

`train_split.csv` and `val_split.csv` contain relative image/mask paths and source-group IDs.
The source-grouped split has 993 training and 207 validation images.
The full dataset is not redistributed in this repository.
