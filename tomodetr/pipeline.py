"""Frozen detector export, fixed graph construction, and final selection."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from .data import read_volume
from .graph import build_neighbor_graph


def origins(shape, size=96, stride=48):
    axes = []
    for n in shape:
        if n < size:
            raise ValueError('Volume axis smaller than window')
        axes.append(list(range(0, n-size+1, stride)))
        if axes[-1][-1] != n-size:
            axes[-1].append(n-size)
    return itertools.product(*axes)


@torch.inference_mode()
def export_volume(model, path, device, batch_size=4):
    model.eval().requires_grad_(False)
    volume, mean, std = read_volume(path)
    starts = list(origins(volume.shape))
    parts = {key: [] for key in ('query_feat','local_logits','xyz_global','xyz_local',
                                 'boundary_dist','patch_id','query_id')}
    for start in range(0,len(starts),batch_size):
        batch = starts[start:start+batch_size]
        patches = [np.asarray(volume[tuple(slice(o,o+96) for o in origin)], dtype=np.float32)
                   for origin in batch]
        inputs = torch.from_numpy(np.ascontiguousarray((np.stack(patches)-mean)/std))[:,None].to(device)
        final = model(inputs)['layers'][-1]
        score = final['class_logits'].float().softmax(-1)[...,:-1].amax(-1)
        for b, origin in enumerate(batch):
            keep = score[b] >= .01
            local = final['centers'][b,keep].float() * 95
            values = {
                'query_feat': final['queries'][b,keep].float(),
                'local_logits': final['class_logits'][b,keep].float(),
                'xyz_local': local,
                'xyz_global': local + local.new_tensor(origin),
                'boundary_dist': torch.minimum(local,95-local).amin(-1),
                'patch_id': torch.full((len(local),),start+b,device=device,dtype=torch.long),
                'query_id': torch.where(keep)[0],
            }
            for key,value in values.items():
                parts[key].append(value.cpu().numpy())
    arrays = {k: np.concatenate(v) for k,v in parts.items()}
    arrays['tomo_shape'] = np.asarray(volume.shape, dtype=np.int64)
    return arrays


def save_cache(path, arrays, metadata):
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata,sort_keys=True)))


def load_cache(path, classes):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files if k != 'metadata'}
        meta = json.loads(str(archive['metadata']))
    if meta.get('schema') != 1 or meta.get('coordinate_order') != 'zyx':
        raise ValueError('Unsupported cache schema/coordinate convention')
    n = len(arrays['xyz_global'])
    shapes = {'query_feat': (n,128), 'local_logits': (n,classes+1),
              'xyz_global':(n,3), 'xyz_local':(n,3), 'boundary_dist':(n,),
              'patch_id':(n,), 'query_id':(n,), 'tomo_shape':(3,)}
    for key,shape in shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid cache field: {key}')
    if np.any(arrays['xyz_global'] < 0) or np.any(arrays['xyz_global'] > arrays['tomo_shape']-1):
        raise ValueError('Cache points outside volume')
    if len(set(zip(arrays['patch_id'].tolist(), arrays['query_id'].tolist()))) != n:
        raise ValueError('Duplicate candidate identity')
    return arrays, meta


def refiner_inputs(arrays, radius, device):
    graph = build_neighbor_graph(arrays['xyz_global'], arrays['patch_id'], radius, 32)
    kwargs = {k: torch.as_tensor(arrays[k], device=device, dtype=torch.float32)
              for k in ('query_feat','local_logits','xyz_global','xyz_local','boundary_dist','tomo_shape')}
    kwargs.update(neighbor_idx=torch.as_tensor(graph.neighbor_idx,device=device),
                  neighbor_mask=torch.as_tensor(graph.neighbor_mask,device=device),
                  relative_xyz=torch.as_tensor(graph.relative_xyz,device=device),
                  patch_size=96, graph_radius=radius)
    return kwargs


def nms(xyz, labels, scores, radii, class_aware=True):
    """Stable point NMS; class-agnostic mode requires one shared radius."""
    if not class_aware:
        if not radii or any(radius != radii[0] for radius in radii):
            raise ValueError("Class-agnostic NMS requires identical radii")
        labels = np.zeros(len(labels), dtype=np.int64)
        radii = [radii[0]]
    kept = []
    for c, radius in enumerate(radii):
        idx = np.flatnonzero(labels == c)
        if not len(idx):
            continue
        tree = cKDTree(xyz[idx])
        suppressed = np.zeros(len(idx),dtype=bool)
        for j in np.argsort(-scores[idx],kind='stable'):
            if not suppressed[j]:
                kept.append(idx[j])
                suppressed[tree.query_ball_point(xyz[idx[j]],radius)] = True
    return np.asarray(sorted(kept,key=lambda j:-scores[j]),dtype=np.int64)


@torch.inference_mode()
def predict(arrays, config, model=None, device='cpu'):
    """All final foreground class scores are retained for per-class selection."""
    logits = torch.from_numpy(arrays['local_logits']).to(device)
    xyz = arrays['xyz_global']
    if model is not None:
        model.eval()
        output = model(**refiner_inputs(arrays,config['graph_radius'],device))
        logits, xyz = output['pred_logits'], output['pred_xyz'].cpu().numpy()
    probability = logits.float().softmax(-1)[:,:-1].cpu().numpy()
    # One particle class per query, chosen among foreground classes.
    labels = probability.argmax(-1)
    scores = probability[np.arange(len(probability)),labels]
    idx = nms(xyz,labels,scores,config['nms_radii'],
              class_aware=config.get('nms_class_aware', True))
    return {'xyz':xyz[idx], 'labels':labels[idx], 'scores':scores[idx]}


def threshold(prediction, values):
    keep = prediction['scores'] >= np.asarray(values)[prediction['labels']]
    return {k:v[keep] for k,v in prediction.items()}
