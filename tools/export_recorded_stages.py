"""Regenerate the recorded-stage assets; reads research inputs, writes only --output."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
from scipy import ndimage
import mrcfile
import argparse
parser=argparse.ArgumentParser(description='Export recorded scientific stages from the original research workspace')
parser.add_argument('--workspace-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.workspace_root.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'paper/figures/qualitative_design_2026-09-13'))
from data_utils import load_case,validate_crop
W,H=700,540

def fit(ny,nx):
 scale=min(W/nx,H/ny);w,h=round(nx*scale),round(ny*scale);return (W-w)//2,(H-h)//2,w,h
def raw_image(raw):
 lo,hi=np.percentile(raw,[1,99]);a=np.uint8(np.clip((raw-lo)/(hi-lo),0,1)*255);ox,oy,w,h=fit(*raw.shape);base=Image.new('RGB',(W,H),'#182027');base.paste(Image.fromarray(np.flipud(a)).convert('RGB').resize((w,h),Image.Resampling.BILINEAR),(ox,oy));return base
def xy(p,b):
 sx,sy=(b[1]-b[0])[:2];scale=min(W/sx,H/sy);w,h=sx*scale,sy*scale;return ((W-w)/2+(p[0]-b[0,0])*scale,(H-h)/2+h-(p[1]-b[0,1])*scale)
def marks(im,pts,b,color,kind='o',r=9):
 d=ImageDraw.Draw(im)
 for p in pts:
  x,y=xy(p,b)
  if kind=='x':d.line((x-r,y-r,x+r,y+r),fill=color,width=3);d.line((x-r,y+r,x+r,y-r),fill=color,width=3)
  else:d.ellipse((x-r,y-r,x+r,y+r),outline=color,width=3)
 return im
def overlay(im,values,kind='heat'):
 v=values.max(0) if values.ndim==3 else values
 if kind=='mask':a=np.zeros((*v.shape,4),dtype=np.uint8);a[:,:,:3]=[239,155,61];a[:,:,3]=np.uint8(v>0)*130
 elif kind=='labels':
  a=np.zeros((*v.shape,4),dtype=np.uint8);colors=np.array([[238,161,50],[81,175,216],[184,115,210],[218,91,102],[77,176,141],[198,184,67]])
  for j,l in enumerate(np.unique(v[v>0])):a[v==l,:3]=colors[j%len(colors)];a[v==l,3]=160
 else:
  cmap=matplotlib.colormaps['magma' if kind=='heat' else 'viridis'];x=np.clip(v,0,1);a=np.uint8(cmap(x)*255);a[:,:,3]=np.uint8(np.clip(x*1.8,0,.82)*255)
 ox,oy,w,h=fit(*v.shape);tile=Image.fromarray(np.flipud(a)).resize((w,h),Image.Resampling.NEAREST if kind in ['mask','labels'] else Image.Resampling.BILINEAR);lay=Image.new('RGBA',(W,H));lay.paste(tile,(ox,oy));return Image.alpha_composite(im.convert('RGBA'),lay).convert('RGB')
def save(im,name,b):
 sx,sy=(b[1]-b[0])[:2];scale=min(W/sx,H/sy);ox=(W-sx*scale)/2;oy=(H-sy*scale)/2;bottom=H-oy;d=ImageDraw.Draw(im);length=100*scale;d.rectangle((ox+16,bottom-42,ox+length+40,bottom-9),fill='#17242b');d.line((ox+28,bottom-29,ox+28+length,bottom-29),fill='white',width=4);d.text((ox+28,bottom-24),'10 nm',fill='white');im.save(out/name,quality=95)
def final(im,pred,hit,gt,b):
 marks(im,gt,b,'#ffe08c',r=11);marks(im,pred[hit],b,'#33e1b0');marks(im,pred[~hit],b,'#ff607b','x');return im
# Current manuscript's four-particle comparison.
p=root/'paper/figures/qualitative_design_2026-09-13/final_delivery/mechanism_plot_arrays.npz';a=np.load(p);b=a['bounds_xyz_angstrom'];gt=a['gt_xyz_angstrom'];base=raw_image(a['raw_mean']);fg=a['foreground_class1_crop'];assert ndimage.label(fg)[1]==1
save(marks(base.copy(),gt,b,'#ffe08c',r=11),'focus-input.webp',b)
save(marks(overlay(base.copy(),fg,'mask'),gt,b,'#ffe08c',r=11),'focus-mask.webp',b)
save(final(overlay(base.copy(),fg,'mask'),a['DeepFinder_pred_xyz_angstrom'],a['DeepFinder_matched'],gt,b),'focus-centers.webp',b)
save(final(base.copy(),a['DeepFinder_pred_xyz_angstrom'],a['DeepFinder_matched'],gt,b),'focus-final.webp',b)
save(final(base.copy(),a['TomoDETR_pred_xyz_angstrom'],a['TomoDETR_matched'],gt,b),'focus-ours.webp',b)
save(final(base.copy(),a['Local_pred_xyz_angstrom'],a['Local_matched'],gt,b),'focus-local.webp',b)
# True foreground surface for the 3D inset: outward exposed voxel faces, no synthetic bridges.
vox=np.argwhere(fg);faces=[]
for z,y,x in vox:
 z,y,x=int(z),int(y),int(x)
 for dz,dy,dx in [(0,0,-1),(0,0,1),(0,-1,0),(0,1,0),(-1,0,0),(1,0,0)]:
  zz,yy,xx=z+dz,y+dy,x+dx
  if 0<=zz<fg.shape[0] and 0<=yy<fg.shape[1] and 0<=xx<fg.shape[2] and fg[zz,yy,xx]:continue
  if dx:corners=[[x+(dx>0),y,z],[x+(dx>0),y+1,z],[x+(dx>0),y+1,z+1],[x+(dx>0),y,z+1]]
  elif dy:corners=[[x,y+(dy>0),z],[x+1,y+(dy>0),z],[x+1,y+(dy>0),z+1],[x,y+(dy>0),z+1]]
  else:corners=[[x,y,z+(dz>0)],[x+1,y,z+(dz>0)],[x+1,y+1,z+(dz>0)],[x,y+1,z+(dz>0)]]
  faces.append({'v':corners,'n':[dx,dy,dz]})
shape=np.array(fg.shape[::-1]);unit=(b[1]-b[0])/shape
(out/'surface.json').write_text(json.dumps({'shape':shape.tolist(),'faces':faces,'gt':((gt-b[0])/unit).tolist(),'centers':((a['native_centers_xyz_angstrom']-b[0])/unit).tolist()},separators=(',',':')))
# Three native baseline traces in a second crowded region; no substitution of historical TomoDETR outputs.
source=root/'promotion_stage_sources_20261015/intermediate_maps_2026-09-08';dep=np.load(source/'deepetpicker_dense_roi_intermediates.npz');octo=np.load(source/'octopi_dense_roi_intermediates.npz');old=np.load(root/'paper/figures/dense_ferritin_case_2026-09-09/plot_data.npz');spacing=10.01200008392334;b=np.array([[552,230,78],[622,284,114]])*spacing
with mrcfile.mmap(root/'phantom/test_private_test/tomograms/TS_100_4.mrc',permissive=True) as f:volume=f.data[78:114,230:284,552:622].copy()
base=raw_image(volume.mean(0));case=load_case('TS_100_4',0);audit=validate_crop(case,b);gt=case['gt_points_xyz'][audit['gt_indices']];assert len(gt)==6
save(marks(base.copy(),gt,b,'#ffe08c',r=11),'trace-input.webp',b)
def trace_stage(arr,name,kind):
 if arr.shape!=(36,54,70):
  assert arr.shape==(36,54,61);assert not octo['ferritin_argmax_mask'][:,:,61:].any();arr=np.pad(arr,((0,0),(0,0),(0,9)))
 save(marks(overlay(base.copy(),arr,kind),gt,b,'#ffe08c',r=11),name,b)
for arr,n,kind in [('ferritin_probability_mean','deepet-prob','heat'),('ferritin_threshold_fraction','deepet-threshold','heat'),('ferritin_smoothed_mean','deepet-smooth','smooth')]:trace_stage(dep[arr],n+'.webp',kind)
for arr,n,kind in [('ferritin_probability','octopi-prob','heat'),('ferritin_argmax_mask','octopi-mask','mask'),('opened_mask','octopi-open','mask'),('distance_transform','octopi-distance','smooth'),('local_maxima','octopi-maxima','mask'),('watershed_labels','octopi-watershed','labels'),('kept_watershed_mask','octopi-filter','mask')]:
 values=octo[arr];values=values/max(float(values.max()),1) if arr=='distance_transform' else values;trace_stage(values,n+'.webp',kind)
# The selected DeepFinder component is a real 3D connected semantic region.
mask=np.zeros(volume.shape,bool);v=np.rint(old['component_xyz']-np.array([552,230,78])).astype(int);mask[v[:,2],v[:,1],v[:,0]]=True
assert mask.sum()==5696
trace_stage(mask,'deepfinder-mask.webp','mask');im=marks(overlay(base.copy(),mask,'mask'),gt,b,'#ffe08c',r=11);marks(im,old['component_raw_centers']*spacing,b,'#ff607b','x');save(im,'deepfinder-clusters.webp',b)
counts={}
for name in ['DeepFinder','DeepETPicker','OCTOPI']:
 ids=audit['methods'][name]['pred_indices'];m=case['methods'][name];pred=m['pred_points_xyz'][ids];hit=m['pred_to_gt'][ids]>=0
 save(final(base.copy(),pred,hit,gt,b),name.lower()+'-final.webp',b);counts[name]=audit['methods'][name]['matching_counts']
 # Exact replayed outputs must coincide with archived final predictions.
 if name=='DeepETPicker':replay=dep['roi_final_ferritin_candidates'][:,1:4]*spacing
 elif name=='OCTOPI':replay=octo['roi_final_points_xyz']*spacing
 else:continue
 assert len(replay)==len(pred);assert np.max(np.min(np.linalg.norm(replay[:,None]-pred[None],axis=2),axis=1))<.01
provenance={'main':{'run':'TS_89_8','gt':4,'componentCount':1,'nativeCenters':2,'DeepFinder':{'TP':0,'FP':2,'FN':4},'TomoDETR':{'TP':4,'FP':0,'FN':0}},'trace':{'run':'TS_100_4','gt':6,'counts':counts},'sourceHashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [p,source/'deepetpicker_dense_roi_intermediates.npz',source/'octopi_dense_roi_intermediates.npz']}}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps({'surfaceFaces':len(faces),'traces':counts,'files':len(list(out.iterdir()))}))
