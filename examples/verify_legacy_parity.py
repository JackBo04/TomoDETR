import sys,json
from pathlib import Path
import numpy as np
import torch
import argparse
parser=argparse.ArgumentParser(description='Optional parity audit against an external legacy source checkout')
parser.add_argument('--legacy-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.legacy_root.resolve()))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.pure_detr import PureDETR3D
from models.tgr import TomoDETRGlobalRefiner
from models.detr_loss import detr_loss
from models.tgr_loss import global_classification_loss
from tomodetr import LocalDetector,CWQR
from tomodetr.losses import local_loss,global_loss
from tomodetr.pipeline import refiner_inputs
import yaml
config=yaml.safe_load((args.legacy_root/'configs/tomodetr_uniform64_tgr.yaml').read_text())
torch.manual_seed(23)
legacy=PureDETR3D(config['model']).eval()
clean=LocalDetector(6).eval()
clean.load_state_dict(legacy.state_dict(),strict=True)
image=torch.randn(1,1,32,32,32)
with torch.no_grad():
    a,b=legacy(image),clean(image)
for x,y in zip(a['layers'],b['layers']):
    for key in ['queries','centers','class_logits']:
        torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
truth=[torch.rand(4,3)]; labels=[torch.tensor([0,1,4,5])]
x=detr_loss(a,truth,labels,config['loss']);y=local_loss(b,truth,labels)
torch.testing.assert_close(x['total_loss'],y['loss'],rtol=1e-6,atol=1e-7)
legacy_cwqr=TomoDETRGlobalRefiner('tgr',coordinate_refinement=True,max_coordinate_delta_voxels=4).eval()
with torch.no_grad():
    for head in [legacy_cwqr.class_delta,legacy_cwqr.coordinate_delta]:
        head.weight.normal_(0,.01);head.bias.normal_(0,.01)
clean_cwqr=CWQR(6).eval();clean_cwqr.load_state_dict(legacy_cwqr.state_dict(),strict=True)
rng=np.random.default_rng(17);n=37
arrays=dict(query_feat=rng.normal(size=(n,128)).astype('float32'),local_logits=rng.normal(size=(n,7)).astype('float32'),xyz_global=rng.uniform(20,30,(n,3)).astype('float32'),xyz_local=rng.uniform(0,95,(n,3)).astype('float32'),boundary_dist=rng.uniform(0,48,n).astype('float32'),patch_id=np.arange(n)%5,tomo_shape=np.array([96,100,120]))
inputs=refiner_inputs(arrays,9.4614,'cpu')
x,y=legacy_cwqr(**inputs),clean_cwqr(**inputs)
for key in ['pred_xyz','pred_logits','coordinate_delta_voxels']:
    torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
truth=torch.tensor([[21.,22.,23.],[22.,24.,25.]]);labels=torch.tensor([2,5]);radii=[6,6.5,9,15,13,13.5]
a=global_classification_loss(x,truth,labels,inputs['tomo_shape'],coordinate_loss_weight=.1,class_radii_voxels=radii)
b=global_loss(y,truth,labels,inputs['tomo_shape'],radii)
torch.testing.assert_close(a['loss'],b['loss'],rtol=0,atol=0)
report=dict(detector_forward='bit-exact, four layers on 32^3 input',detector_state_dict='strict compatible',cwqr_state_dict='strict compatible',cwqr_forward='bit-exact with nonzero residual weights',local_loss='within 1e-6 relative tolerance',global_loss='bit-exact',trained_benchmark_reproduction=False)
print(json.dumps(report,indent=2))
args.output.write_text(json.dumps(report,indent=2)+'\n')
