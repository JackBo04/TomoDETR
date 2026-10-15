"""Train, export, calibrate and predict with the paper model."""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import LocalDetector, CWQR
from .config import load_config
from .data import manifest, sha256, PatchDataset, collate, read_annotations
from .losses import local_loss, global_loss
from .pipeline import export_volume, save_cache, load_cache, refiner_inputs, predict, threshold
from .metrics import benchmark_counts, summarize, fbeta
from . import shrec


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(path, kind, config, device):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('kind') != kind or payload.get('config') != config:
        raise ValueError('Checkpoint kind/config mismatch; use its exact saved configuration')
    model = (LocalDetector if kind=='detector' else CWQR)(len(config['classes']))
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device), payload


def selected_rows(args, split):
    rows = [r for r in manifest(args.manifest) if r['split']==split]
    if not rows:
        raise ValueError(f'No records for {split}')
    return rows


def cache_for(row, args, config):
    arrays, meta = load_cache(args.cache/f"{row['id']}.npz",len(config['classes']))
    if meta.get('id') != row['id'] or meta.get('volume') != sha256(row['volume']):
        raise ValueError('Cache is not from the requested volume')
    if meta.get('classes') != config['classes']:
        raise ValueError('Cache class order differs from configuration')
    return arrays, meta


def checkpoint(model,optimizer,config,kind,epoch,provenance,scheduler=None):
    return {'kind':kind, 'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict() if scheduler else None,
            'config':config,'epoch':epoch,'provenance':provenance,
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def train(args, config):
    rows = selected_rows(args,'train')
    seed_all(config['seed'])
    kind = 'detector' if args.command=='train-detector' else 'cwqr'
    model = (LocalDetector if kind=='detector' else CWQR)(len(config['classes'])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler = None
    provenance = {'train_ids':[r['id'] for r in rows],
                  'volumes':{r['id']:sha256(r['volume']) for r in rows},
                  'annotations':{r['id']:sha256(r['annotations']) for r in rows}}
    if kind=='detector':
        epochs = config['detector_epochs']
        dataset = PatchDataset(rows,len(config['classes']),config['seed'],args.samples_per_epoch)
        loader = DataLoader(dataset,batch_size=args.batch_size,collate_fn=collate,num_workers=0)
        if args.batch_size <= 0 or 8 % args.batch_size or args.samples_per_epoch % 8:
            raise ValueError('batch-size must divide 8; samples-per-epoch must be divisible by 8')
        accumulation = 8//args.batch_size
        provenance.update(samples_per_epoch=args.samples_per_epoch,batch_size=args.batch_size)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=.1,total_iters=10),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['schedule_epochs']-10,eta_min=1e-6),
        ],milestones=[10])
    else:
        epochs = 40
        items = []
        detector_hashes = set()
        for row in rows:
            arrays, meta = cache_for(row,args,config)
            detector_hashes.add(meta['detector_sha256'])
            xyz, labels = read_annotations(row['annotations'],len(config['classes']),arrays['tomo_shape'])
            items.append((refiner_inputs(arrays,config['graph_radius'],'cpu'),xyz,labels))
        if len(detector_hashes)!=1:
            raise ValueError('CWQR training caches must come from a single frozen detector')
        provenance['detector_sha256'] = next(iter(detector_hashes))
    start = 0
    if args.resume:
        model, old = load_model(args.resume,kind,config,args.device)
        if old['provenance'] != provenance:
            raise ValueError('Resume data or training settings differ')
        # Rebind optimizer to the restored model before loading optimizer state.
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        if kind=='detector':
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,[
                torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=.1,total_iters=10),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['schedule_epochs']-10,eta_min=1e-6),
            ],milestones=[10])
            scheduler.load_state_dict(old['scheduler'])
        optimizer.load_state_dict(old['optimizer'])
        start = old['epoch']
        torch.set_rng_state(old['torch_rng'])
        if torch.cuda.is_available() and old['cuda_rng']:
            torch.cuda.set_rng_state_all(old['cuda_rng'])
    for epoch in range(start+1,epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        if kind=='detector':
            dataset.epoch = epoch
            for step,(image,xyz,labels) in enumerate(loader,1):
                image = image.to(args.device)
                xyz = [v.to(args.device) for v in xyz]
                labels = [v.to(args.device) for v in labels]
                with torch.autocast(device_type=torch.device(args.device).type,dtype=torch.bfloat16,
                                    enabled=torch.device(args.device).type=='cuda'):
                    loss = local_loss(model(image),xyz,labels)['loss']
                (loss/accumulation).backward()
                if step % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
            scheduler.step()
        else:
            order = np.random.default_rng(config['seed']+epoch).permutation(len(items))
            for i in order:
                inputs,xyz,labels = items[i]
                inputs = {k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
                output = model(**inputs)
                loss = global_loss(output,torch.as_tensor(xyz,device=args.device),
                                   torch.as_tensor(labels,device=args.device),
                                   inputs['tomo_shape'],config['radii'])['loss']
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite CWQR loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
        row = {'epoch':epoch,'loss':float(np.mean(losses))}
        print(json.dumps(row),flush=True)
        with (args.output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        payload = checkpoint(model,optimizer,config,kind,epoch,provenance,scheduler)
        torch.save(payload,args.output/f'epoch_{epoch:03d}.pt')
        torch.save(payload,args.output/'latest.pt')


def export(args, config):
    model,_ = load_model(args.detector,'detector',config,args.device)
    detector_hash = sha256(args.detector)
    for row in manifest(args.manifest):
        arrays = export_volume(model,row['volume'],args.device,args.batch_size)
        meta = {'schema':1,'coordinate_order':'zyx','id':row['id'],
                'volume':sha256(row['volume']), 'classes':config['classes'],
                'detector_sha256':detector_hash}
        save_cache(args.output/f"{row['id']}.npz",arrays,meta)
        print(json.dumps({'id':row['id'],'candidates':len(arrays['xyz_global'])}),flush=True)


def predictions(args,config,split):
    model = payload = None
    if args.cwqr:
        model,payload = load_model(args.cwqr,'cwqr',config,args.device)
    rows = selected_rows(args,split)
    results, hashes = [], set()
    for row in rows:
        arrays,meta = cache_for(row,args,config)
        hashes.add(meta['detector_sha256'])
        if payload and payload['provenance']['detector_sha256'] != meta['detector_sha256']:
            raise ValueError('CWQR and feature cache detector checkpoints differ')
        results.append((row,predict(arrays,config,model,args.device),arrays['tomo_shape']))
    if len(hashes) != 1:
        raise ValueError('Prediction caches mix detectors')
    identity = {'detector_sha256':next(iter(hashes)),
                'cwqr_sha256':sha256(args.cwqr) if args.cwqr else None,'config':config}
    return results,identity


def calibrate(args,config):
    rows,identity = predictions(args,config,'validation')
    if config['dataset']=='shrec':
        data = [(pred,*shrec.load_truth(row)) for row,pred,_ in rows]
        grid = np.arange(.01,1,.01)
        counts = [sum((shrec.counts(threshold(pred,[t]*len(config['classes'])),names,occupancy,config['classes'])
                       for pred,names,occupancy in data),np.zeros(3,dtype=np.int64)) for t in grid]
        best = max(range(len(grid)),key=lambda i:(float(fbeta(counts[i])),grid[i]))
        identity.update(thresholds=[float(grid[best])]*len(config['classes']),
                        validation_ids=[r['id'] for r,_,_ in rows],
                        validation_volume_hashes=[sha256(r['volume']) for r,_,_ in rows],
                        validation_metrics={'counts_tp_fp_fn':counts[best].tolist(),'official_localization_f1':float(fbeta(counts[best]))})
        save_json(args.output/'calibration.json',identity)
        return
    data = [(pred,*read_annotations(row['annotations'],len(config['classes']),shape))
            for row,pred,shape in rows]
    if config['dataset']=='czii':
        # Historical official Val40 coarse-to-fine grid, independently by class.
        from scipy.spatial import cKDTree
        grid = np.unique(np.r_[np.linspace(.01,.02,11),np.linspace(.025,.5,96),np.linspace(.51,.99,49)])
        values, selected_counts = [], []
        for c, radius in enumerate(config['tolerances']):
            coverage = []
            for pred,xyz,labels in data:
                points = pred['xyz'][pred['labels']==c]
                scores = pred['scores'][pred['labels']==c]
                gt = xyz[labels==c]
                best = np.full(len(gt),-np.inf)
                if len(points) and len(gt):
                    for score,neighbors in zip(scores,cKDTree(points).query_ball_tree(cKDTree(gt),radius)):
                        if neighbors:
                            best[neighbors] = np.maximum(best[neighbors],score)
                coverage.append((scores,best))
            def evaluate(t):
                count = sum((np.array([int((best>=t).sum()),int((scores>=t).sum())-int((best>=t).sum()),int((best<t).sum())])
                             for scores,best in coverage),np.zeros(3,dtype=np.int64))
                key = (float(fbeta(count,4)),int(count[0]),-int(count[1]),float(t))
                return key,count
            coarse = [evaluate(t) for t in grid]
            key,count = max(coarse,key=lambda item:item[0])
            fine = np.arange(max(.01,key[-1]-.01),min(.999,key[-1]+.01)+.00025,.00025)
            # Clamp roundoff at the documented upper bound.
            fine = np.minimum(fine,.999)
            key,count = max(coarse+[evaluate(t) for t in fine],key=lambda item:item[0])
            values.append(key[-1]); selected_counts.append(count)
        identity.update(thresholds=values,validation_ids=[r['id'] for r,_,_ in rows],
                        validation_volume_hashes=[sha256(r['volume']) for r,_,_ in rows],
                        validation_metrics=summarize(selected_counts,config),
                        threshold_search='official Val40 coarse grid plus 0.00025 refinement within +/-0.01')
        save_json(args.output/'calibration.json',identity)
        return
    # Strictly validation-only; ties select the higher threshold deterministically.
    grid = np.unique(np.r_[np.arange(.01,1,.01),.995,.999])
    curve = []
    for t in grid:
        curve.append(sum((benchmark_counts(threshold(pred,[t]*len(config['classes'])),xyz,labels,config)
                          for pred,xyz,labels in data),np.zeros((len(config['classes']),3),dtype=np.int64)))
    curve = np.stack(curve)
    # F4 macro decomposes by class; pooled F1 requires a joint threshold search.
    indices = np.zeros(len(config['classes']),dtype=int)
    for c in range(len(indices)):
        values = fbeta(curve[:,c],config['beta'])
        indices[c] = np.flatnonzero(values==values.max())[-1]
    if config['dataset']=='deepict':
        if len(indices)!=2:
            raise ValueError('DeePiCt calibration expects ribosome and FAS')
        best = (-1.,-1,-1)
        for a in range(len(grid)):
            for b in range(len(grid)):
                candidate = (float(fbeta(curve[a,0]+curve[b,1])),a,b)
                best = max(best,candidate)
        indices = np.asarray(best[1:])
    values = grid[indices].tolist()
    count = np.stack([curve[indices[c],c] for c in range(len(indices))])
    identity.update(thresholds=values,validation_ids=[r['id'] for r,_,_ in rows],
                    validation_volume_hashes=[sha256(r['volume']) for r,_,_ in rows],
                    validation_metrics=summarize(count,config),threshold_grid=grid.tolist())
    save_json(args.output/'calibration.json',identity)


def infer(args,config):
    rows,identity = predictions(args,config,args.split)
    calibration = json.loads(args.calibration.read_text())
    if any(calibration.get(k)!=v for k,v in identity.items()):
        raise ValueError('Calibration model/config differs from prediction')
    values = calibration['thresholds']
    if len(values)!=len(config['classes']) or not np.isfinite(values).all() or np.any(np.asarray(values)<0) or np.any(np.asarray(values)>1):
        raise ValueError('Invalid calibrated thresholds')
    if args.split=='test' and any(sha256(r['volume']) in calibration['validation_volume_hashes'] for r,_,_ in rows):
        raise ValueError('Test volume was used for calibration')
    total = np.zeros((len(config['classes']),3),dtype=np.int64)
    shrec_total = np.zeros(3,dtype=np.int64)
    for row,pred,shape in rows:
        pred = threshold(pred,values)
        np.savez_compressed(args.output/f"{row['id']}.npz",**pred)
        with (args.output/f"{row['id']}.csv").open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['z','y','x','class_id','score'])
            for xyz,c,s in zip(pred['xyz'],pred['labels'],pred['scores']):
                writer.writerow([*xyz,int(c),float(s)])
        if config['dataset']=='shrec':
            shrec.write_submission(args.output/f"{row['id']}.txt",pred,config['classes'])
        if args.evaluate and config['dataset']=='shrec':
            names,occupancy = shrec.load_truth(row)
            shrec_total += shrec.counts(pred,names,occupancy,config['classes'])
        elif args.evaluate:
            xyz,labels = read_annotations(row['annotations'],len(config['classes']),shape)
            total += benchmark_counts(pred,xyz,labels,config)
    save_json(args.output/'prediction_metadata.json',dict(identity,thresholds=values,split=args.split))
    if args.evaluate:
        save_json(args.output/'metrics.json',
                  {'counts_tp_fp_fn':shrec_total.tolist(),'official_localization_f1':float(fbeta(shrec_total))}
                  if config['dataset']=='shrec' else summarize(total,config))


def bind_shrec_refit(args,config):
    """Bind already-frozen development thresholds to the mandated 0--8 refit."""
    if config['dataset']!='shrec':
        raise ValueError('This operation is defined only for the SHREC development/refit protocol')
    frozen = json.loads(args.calibration.read_text())
    if frozen.get('config')!=config or frozen.get('validation_ids')!=['model_8']:
        raise ValueError('Use the same selected SHREC settings and model_8-only calibration')
    _,detector = load_model(args.detector,'detector',config,'cpu')
    expected = {f'model_{i}' for i in range(9)}
    if set(detector['provenance']['train_ids'])!=expected:
        raise ValueError('Refit detector must train on exactly model_0 through model_8')
    identity = dict(detector_sha256=sha256(args.detector),cwqr_sha256=None,config=config)
    if bool(args.cwqr)!=bool(frozen.get('cwqr_sha256')):
        raise ValueError('Development and refit must use the same CWQR endpoint')
    if args.cwqr:
        _,refiner = load_model(args.cwqr,'cwqr',config,'cpu')
        if set(refiner['provenance']['train_ids'])!=expected or refiner['provenance']['detector_sha256']!=identity['detector_sha256']:
            raise ValueError('Refit CWQR must use all nine volumes from the refit detector')
        identity['cwqr_sha256']=sha256(args.cwqr)
    frozen.update(identity)
    frozen['development_calibration_sha256']=sha256(args.calibration)
    frozen['binding']='SHREC model_8 thresholds frozen before 0--8 refit; no recalibration'
    save_json(args.output/'calibration.json',frozen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('train-detector','export','train-cwqr','calibrate','predict'):
        p = sub.add_parser(name)
        p.add_argument('--config',type=Path,required=True)
        p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--device',default='cuda:0')
        if name in ('train-detector','export'):
            p.add_argument('--batch-size',type=int,default=4)
        if name in ('train-detector','train-cwqr'):
            p.add_argument('--resume',type=Path)
        if name=='train-detector':
            p.add_argument('--samples-per-epoch',type=int,default=1024)
        if name=='export':
            p.add_argument('--detector',type=Path,required=True)
        if name in ('train-cwqr','calibrate','predict'):
            p.add_argument('--cache',type=Path,required=True)
        if name in ('calibrate','predict'):
            p.add_argument('--cwqr',type=Path,help='Omit only for the without-CWQR endpoint')
        if name=='predict':
            p.add_argument('--calibration',type=Path,required=True)
            p.add_argument('--split',choices=['validation','test'],default='test')
            p.add_argument('--evaluate',action='store_true',help='Read annotations and compute strict reference metrics')
    p = sub.add_parser('bind-shrec-refit',help='Carry frozen model_8 thresholds to the 0--8 refit')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--detector',type=Path,required=True)
    p.add_argument('--cwqr',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if getattr(args,'batch_size',1)<=0:
        parser.error('batch-size must be positive')
    if args.output.exists() and any(args.output.iterdir()) and not getattr(args,'resume',None):
        parser.error('Output directory is nonempty; choose a new directory or resume training')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command.startswith('train-'):
        train(args,config)
    else:
        {'export':export,'calibrate':calibrate,'predict':infer,'bind-shrec-refit':bind_shrec_refit}[args.command](args,config)


if __name__=='__main__':
    main()
