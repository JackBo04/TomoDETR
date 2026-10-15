"""Portable volumes and annotations; no site-specific storage dependencies."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .geometry import crop_local, rotate90_z, flip_points


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty JSON list')
    ids = set()
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id'] or Path(row['id']).name != row['id'] or row['id'] in {'.', '..'}:
            raise ValueError('id must be a simple filename component')
        if row['id'] in ids:
            raise ValueError('Duplicate volume id')
        ids.add(row['id'])
        if row.get('split') not in {'train', 'validation', 'test'}:
            raise ValueError('Each record needs train/validation/test split')
        for key in ('volume', 'annotations', 'particle_locations', 'occupancy'):
            if key in row:
                row[key] = str((path.parent / row[key]).resolve())
    # The same physical file may not silently appear under different IDs/splits.
    volumes = [r['volume'] for r in rows]
    if len(volumes) != len(set(volumes)):
        raise ValueError('A volume is listed more than once')
    return rows


def read_volume(path):
    path = Path(path)
    if path.suffix == '.npy':
        volume = np.load(path, mmap_mode='r', allow_pickle=False)
    elif path.suffix in {'.mrc', '.rec', '.map'}:
        import mrcfile
        with mrcfile.open(path, permissive=False) as handle:
            volume = handle.data.copy()
    else:
        raise ValueError('Use a 3D .npy or .mrc/.rec/.map volume')
    if volume.ndim != 3 or min(volume.shape) < 96:
        raise ValueError('Volumes must be 3D with every dimension >= 96')
    # Streaming statistics avoid a full-size float64 temporary.
    total = square = 0.0
    for slab in volume:
        slab = np.asarray(slab, dtype=np.float64)
        if not np.isfinite(slab).all():
            raise ValueError('Nonfinite image values')
        total += slab.sum()
        square += np.square(slab).sum()
    mean = total / volume.size
    std = np.sqrt(max(square / volume.size - mean * mean, 0))
    if std <= 0:
        raise ValueError('Constant volume cannot be standardized')
    return volume, float(mean), float(std)


def read_annotations(path, classes, shape=None):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if not {'z','y','x','class_id'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV requires z,y,x,class_id columns in detector voxels')
        rows = [[float(r['z']), float(r['y']), float(r['x']), float(r['class_id'])] for r in reader]
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(a).all() or np.any(a[:, 3] != np.floor(a[:, 3])):
        raise ValueError('Coordinates must be finite and labels integral')
    if np.any(a[:, 3] < 0) or np.any(a[:, 3] >= classes):
        raise ValueError('Class IDs must lie in [0,C-1]')
    if np.any(a[:, :3] < 0) or (shape is not None and np.any(a[:, :3] > np.asarray(shape)-1)):
        raise ValueError('Annotation outside volume voxel-center extent')
    # Exact duplicates only, preserving input order.
    _, idx = np.unique(a, axis=0, return_index=True)
    a = a[np.sort(idx)]
    return a[:, :3].astype(np.float32), a[:, 3].astype(np.int64)


def augment(image, points, rng):
    if rng.random() < .5:
        k = int(rng.integers(1, 4))
        image = torch.rot90(image, k, (-2,-1))
        points = rotate90_z(points, 96, k)
    for axis in range(3):
        if rng.random() < .5:
            image = torch.flip(image, (axis+1,))
            points = flip_points(points, 96, axis)
    for operation in rng.permutation(4):
        if rng.random() >= .3:
            continue
        if operation == 0:
            image = image * float(rng.uniform(.85, 1.15))
        elif operation == 1:
            image = image + float(rng.uniform(-.15, .15))
        elif operation == 2:
            low, high = image.amin(), image.amax()
            if high > low:
                image = ((image-low)/(high-low)).clamp_min(0).pow(float(rng.uniform(.85,1.15))) * (high-low)+low
        else:
            image = image + torch.from_numpy(rng.normal(0,.5,image.shape).astype(np.float32))
    return image, points


class PatchDataset(Dataset):
    """80% class-balanced centered crops with ±24 jitter, 20% uniform crops."""
    def __init__(self, rows, classes, seed, samples=1024):
        if not rows or any(r['split'] != 'train' for r in rows):
            raise ValueError('Training requires nonempty train records only')
        self.seed, self.samples, self.epoch = seed, samples, 0
        self.records = []
        self.pool = [[] for _ in range(classes)]
        for index, row in enumerate(rows):
            image, mean, std = read_volume(row['volume'])
            xyz, labels = read_annotations(row['annotations'], classes, image.shape)
            self.records.append((image, mean, std, xyz, labels))
            for point, label in enumerate(labels):
                self.pool[label].append((index, point))
        self.present = [c for c, pool in enumerate(self.pool) if pool]
        if not self.present:
            raise ValueError('At least one foreground annotation is required')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        if rng.random() >= .2:
            pool = self.pool[int(rng.choice(self.present))]
            record_idx, point = pool[int(rng.integers(len(pool)))]
            image, mean, std, xyz, labels = self.records[record_idx]
            center = xyz[point] + rng.uniform(-24,24,3)
            origin = np.clip(np.floor(center-48).astype(int), 0, np.asarray(image.shape)-96)
        else:
            image, mean, std, xyz, labels = self.records[int(rng.integers(len(self.records)))]
            origin = np.asarray([rng.integers(n-96+1) for n in image.shape])
        patch = np.asarray(image[tuple(slice(o,o+96) for o in origin)], dtype=np.float32)
        points, keep = crop_local(xyz, origin, 96)
        tensor, points = augment(torch.from_numpy(np.ascontiguousarray((patch-mean)/std))[None], points, rng)
        return tensor, torch.from_numpy(points/95), torch.from_numpy(labels[keep])


def collate(batch):
    images, points, labels = zip(*batch)
    return torch.stack(images), list(points), list(labels)
