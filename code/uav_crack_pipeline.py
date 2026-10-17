"""UAV-Crack baseline: train, validate, visualize, predict and package submission.
Run --help for commands. Requires torch, numpy, pandas, pillow, matplotlib, tqdm.
"""
import argparse
import csv
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CrackDataset(Dataset):
    def __init__(self, frame, size=(288, 512), augment=False):
        self.rows = frame.to_dict('records')
        self.size = size  # H,W
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['image']) as im:
            image = im.convert('RGB').resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
            x = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(row['mask']) as im:
            mask = im.convert('L').resize((self.size[1], self.size[0]), Image.Resampling.NEAREST)
            y = (np.asarray(mask) > 0).astype(np.float32)
        if self.augment:
            if random.random() < .5:
                x, y = np.flip(x, axis=1), np.flip(y, axis=1)
            if random.random() < .5:
                x, y = np.flip(x, axis=0), np.flip(y, axis=0)
            if random.random() < .5:
                x, y = np.rot90(x, 2), np.rot90(y, 2)
            if random.random() < .35:
                x = np.clip(x * random.uniform(.85, 1.15), 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y[None, ...]))
        return x, y


class ConvBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(b), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        b = base
        self.e1 = ConvBlock(3, b)
        self.e2 = ConvBlock(b, b*2)
        self.e3 = ConvBlock(b*2, b*4)
        self.e4 = ConvBlock(b*4, b*8)
        self.mid = ConvBlock(b*8, b*16)
        self.d4 = ConvBlock(b*16+b*8, b*8)
        self.d3 = ConvBlock(b*8+b*4, b*4)
        self.d2 = ConvBlock(b*4+b*2, b*2)
        self.d1 = ConvBlock(b*2+b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        m = self.mid(F.max_pool2d(e4, 2))
        def up(a, skip, block):
            a = F.interpolate(a, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            return block(torch.cat([a, skip], 1))
        return self.out(up(up(up(up(m, e4, self.d4), e3, self.d3), e2, self.d2), e1, self.d1))


def loss_fn(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    axes = (1, 2, 3)
    intersection = (probs * targets).sum(axes)
    dice = (2 * intersection + 1) / (probs.sum(axes) + targets.sum(axes) + 1)
    return bce + (1 - dice).mean()


def metrics(tp, fp, fn, tn):
    eps = 1e-9
    iou = tp / (tp + fp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)
    return {'crack_iou': iou, 'crack_f1': 2*tp/(2*tp+fp+fn+eps),
            'precision': tp/(tp+fp+eps), 'recall': tp/(tp+fn+eps),
            'miou': (iou+bg_iou)/2, 'pixel_accuracy': (tp+tn)/(tp+fp+fn+tn+eps)}


@torch.no_grad()
def evaluate(model, loader, device, threshold=.5):
    model.eval()
    total_loss, n, tp, fp, fn, tn = 0., 0, 0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.shape[0]
        n += x.shape[0]
        p = torch.sigmoid(logits) >= threshold
        t = y > .5
        tp += (p & t).sum().item()
        fp += (p & ~t).sum().item()
        fn += (~p & t).sum().item()
        tn += (~p & ~t).sum().item()
    return {'loss': total_loss / max(n, 1), **metrics(tp, fp, fn, tn)}


def make_loader(csv_path, batch, size, augment, workers, shuffle, data_root):
    df = pd.read_csv(csv_path)
    for col in ('image', 'mask', 'source'):
        if col not in df.columns:
            raise ValueError(f'Missing CSV column {col}: {csv_path}')
    for col in ('image', 'mask'):
        df[col] = df[col].map(lambda p: str(Path(data_root) / p) if not Path(p).is_absolute() else p)
    missing = [p for p in df['image'].tolist()+df['mask'].tolist() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f'CSV paths invalid. First missing: {missing[0]}')
    ds = CrackDataset(df, size=size, augment=augment)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available()), df


def save_graphs(history, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in history]
    for keys, filename, ylabel in [(['train_loss','val_loss'], 'loss_curve.png', 'Loss'),
                                  (['crack_iou','crack_f1','precision','recall','miou'], 'metric_curves.png', 'Score')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in keys:
            ax.plot(epochs, [r[key] for r in history], label=key)
        ax.set(xlabel='Epoch', ylabel=ylabel, title='UAV-Crack baseline')
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)


def run_train(args):
    seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train_loader, train_df = make_loader(args.train_csv, args.batch, (args.height,args.width), True, args.workers, True, args.data_root)
    val_loader, val_df = make_loader(args.val_csv, args.batch, (args.height,args.width), False, args.workers, False, args.data_root)
    overlap = set(train_df.source) & set(val_df.source)
    if overlap:
        raise ValueError(f'Data leakage: {len(overlap)} source groups overlap')
    print(f'Train={len(train_df)}, Val={len(val_df)}, Source overlap=0', flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    history = []
    best = -1
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.shape[0]
            if batch_idx % 25 == 0:
                print(f'Epoch {epoch}/{args.epochs}, batch {batch_idx}/{len(train_loader)}', flush=True)
        result = evaluate(model, val_loader, device)
        scheduler.step(result['crack_iou'])
        record = {'epoch':epoch, 'train_loss':running/len(train_df), 'val_loss':result.pop('loss'), **result}
        history.append(record)
        print(json.dumps(record), flush=True)
        pd.DataFrame(history).to_csv(out/'history.csv', index=False)
        if result['crack_iou'] > best:
            best = result['crack_iou']
            torch.save({'model':model.state_dict(), 'base':args.base,
                        'height':args.height,'width':args.width,
                        'epoch':epoch,'metrics':record,'threshold':.5}, out/'best_model.pt')
            with open(out/'best_metrics.json', 'w', encoding='utf8') as f:
                json.dump(record, f, indent=2)
            print(f'  Saved best checkpoint (Crack IoU={best:.4f})', flush=True)
    save_graphs(history, out)
    print(f'FINISHED. Results in {out.resolve()}', flush=True)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = UNet(ckpt['base']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_probability(model, image_path, size, device):
    with Image.open(image_path) as im:
        original_size = im.size
        resized = im.convert('RGB').resize((size[1],size[0]), Image.Resampling.BILINEAR)
        x = np.asarray(resized, dtype=np.float32).transpose(2,0,1) / 255.
    tensor = torch.from_numpy(x.copy()).unsqueeze(0).to(device)
    logits = model(tensor)
    # Upsample probabilities to original resolution BEFORE thresholding.
    probs = torch.sigmoid(F.interpolate(logits, size=(original_size[1],original_size[0]),
                                        mode='bilinear', align_corners=False))
    return probs.squeeze().cpu().numpy(), original_size


def run_predict(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_model(args.checkpoint, device)
    image_dir = Path(args.val_images)
    files = sorted(image_dir.glob('*.jpg'))
    if len(files) != 300:
        raise ValueError(f'Expected exactly 300 JPGs directly in {image_dir}; found {len(files)}')
    output = Path(args.output)
    result_dir = output/'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    if list(result_dir.iterdir()):
        raise ValueError(f'{result_dir} is not empty. Use a new output folder.')
    for idx, image_path in enumerate(files, 1):
        prob, original_size = predict_probability(model, image_path,
                                                  (ckpt['height'], ckpt['width']), device)
        if original_size != (672,378):
            raise ValueError(f'Unexpected input dimensions {image_path}: {original_size}')
        mask = (prob >= args.threshold).astype(np.uint8)
        Image.fromarray(mask, mode='L').save(result_dir/(image_path.stem+'.png'))
        if idx % 50 == 0:
            print(f'Predicted {idx}/300', flush=True)
    pngs = sorted(result_dir.glob('*.png'))
    assert len(pngs)==300
    for p in pngs:
        with Image.open(p) as im:
            a = np.asarray(im)
            if im.mode != 'L' or im.size != (672,378) or not np.isin(a, [0,1]).all():
                raise ValueError(f'Invalid submission mask: {p}')
    zip_path = output/'result.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:
            z.write(p, arcname=f'result/{p.name}')
    print(f'VALIDATED: 300 grayscale PNGs, size 672x378, values 0/1. ZIP: {zip_path.resolve()}', flush=True)


def run_visualize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_model(args.checkpoint, device)
    df = pd.read_csv(args.val_csv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(df), size=min(args.count,len(df)), replace=False)
    for number, index in enumerate(indices, 1):
        row = df.iloc[int(index)]
        with Image.open(row['image']) as im:
            rgb = np.asarray(im.convert('RGB'))
        with Image.open(row['mask']) as im:
            gt = np.asarray(im.convert('L')) > 0
        prob, _ = predict_probability(model, row['image'], (ckpt['height'],ckpt['width']), device)
        pred = prob >= args.threshold
        fig, axes = plt.subplots(1, 4, figsize=(15,4))
        for ax, data, title, cmap in zip(axes,
                [rgb, gt, pred, prob],
                ['Input', 'Ground truth', 'Prediction', 'Crack probability'],
                [None, 'gray', 'gray', 'viridis']):
            ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f'Internal validation sample {number} | {Path(row["image"]).name}', fontsize=9)
        fig.tight_layout()
        fig.savefig(out/f'example_{number:02d}.png', dpi=140)
        plt.close(fig)
    print(f'Saved {len(indices)} comparisons to {out.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--data-root', default='dataset')
    train.add_argument('--train-csv', default='data/train_split.csv')
    train.add_argument('--val-csv', default='data/val_split.csv')
    train.add_argument('--output', default='uav_results')
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--batch', type=int, default=8)
    train.add_argument('--width', type=int, default=512)
    train.add_argument('--height', type=int, default=288)
    train.add_argument('--base', type=int, default=24)
    train.add_argument('--lr', type=float, default=3e-4)
    train.add_argument('--workers', type=int, default=0)
    train.add_argument('--seed', type=int, default=42)
    pred = sub.add_parser('predict')
    pred.add_argument('--checkpoint', default='uav_results/best_model.pt')
    pred.add_argument('--val-images', required=True)
    pred.add_argument('--output', default='uav_submission')
    pred.add_argument('--threshold', type=float, default=.5)
    vis = sub.add_parser('visualize')
    vis.add_argument('--checkpoint', default='uav_results/best_model.pt')
    vis.add_argument('--data-root', default='dataset')
    vis.add_argument('--val-csv', default='data/val_split.csv')
    vis.add_argument('--output', default='uav_results/examples')
    vis.add_argument('--count', type=int, default=12)
    vis.add_argument('--threshold', type=float, default=.5)
    vis.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    {'train':run_train, 'predict':run_predict, 'visualize':run_visualize}[args.command](args)


if __name__ == '__main__':
    main()
