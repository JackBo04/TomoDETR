"""Synthetic end-to-end test. Shortened schedule is NOT a benchmark experiment."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(17)
    rows=[]
    for split in ['train','validation','test']:
        np.save(root/f'{split}.npy',rng.normal(size=(96,96,96)).astype(np.float32))
        with (root/f'{split}.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['z','y','x','class_id'])
            w.writerows([[24,24,24,0],[48,48,48,1],[64,64,64,0]])
        rows.append(dict(id=split,volume=f'{split}.npy',annotations=f'{split}.csv',split=split))
    (root/'manifest.json').write_text(json.dumps(rows))
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/deepict.yaml').read_text())
    config.update(detector_epochs=11,schedule_epochs=11)
    (root/'config.yaml').write_text(yaml.safe_dump(config))
    common=['--config',str(root/'config.yaml'),'--manifest',str(root/'manifest.json'),'--device',args.device]
    commands=[
        ['train-detector','--samples-per-epoch','8','--batch-size','1','--output',str(root/'detector')],
        ['export','--detector',str(root/'detector/latest.pt'),'--output',str(root/'cache'),'--batch-size','1'],
        ['train-cwqr','--cache',str(root/'cache'),'--output',str(root/'cwqr')],
        ['calibrate','--cache',str(root/'cache'),'--cwqr',str(root/'cwqr/latest.pt'),'--output',str(root/'calibration')],
        ['predict','--cache',str(root/'cache'),'--cwqr',str(root/'cwqr/latest.pt'),
         '--calibration',str(root/'calibration/calibration.json'),'--evaluate','--output',str(root/'predictions')],
    ]
    for command in commands:
        print('Running',command[0],flush=True)
        with (root/f'{command[0]}.log').open('w') as f:
            subprocess.run([sys.executable,'-m','tomodetr',*command,*common],stdout=f,stderr=subprocess.STDOUT,check=True)
    metrics=json.loads((root/'predictions/metrics.json').read_text())
    assert np.isfinite(metrics['pooled_f1'])
    print(json.dumps({'completed':True,'synthetic_only':True,'commands':[c[0] for c in commands]},indent=2))


if __name__=='__main__': main()
