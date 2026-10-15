"""Export the recorded Case A volume, endpoints and a 0.5 foreground isosurface.
No smoothing, inferred trajectories or synthesized particle shapes are used.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mrcfile
from scipy import ndimage
p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
r=args.workspace_root;o=args.output;o.mkdir(parents=True,exist_ok=True)
source=r/'paper/figures/qualitative_design_2026-09-13/final_delivery/mechanism_plot_arrays.npz'
a=np.load(source);fg=a['foreground_class1_crop'];bounds=a['bounds_xyz_angstrom'];shape=np.array(fg.shape[::-1]);spacing=(bounds[1]-bounds[0])/shape
assert tuple(shape)==(46,46,22) and ndimage.label(fg)[1]==1
with mrcfile.mmap(r/'phantom/test_private_test/tomograms/TS_89_8.mrc',permissive=True) as m:raw=m.data[112:134,345:391,562:608].copy()
assert raw.shape==fg.shape and np.allclose(raw.mean(0),a['raw_mean'],atol=1e-5)
lo,hi=np.percentile(raw,[1,99]);norm=np.uint8(np.clip((raw-lo)/(hi-lo),0,1)*255);(o/'density.bin').write_bytes(norm.tobytes())
# Marching tetrahedra on native binary sample centers, padded with background.
field=np.pad(fg.astype(float),1);corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
tets=[[0,5,1,6],[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6]]
verts=[];indices=[];lookup={}
def vertex(p):
 key=tuple(p)
 if key not in lookup:lookup[key]=len(verts);verts.append(p)
 return lookup[key]
# Only cubes crossing the foreground boundary require triangulation.
shapeZYX=np.array(field.shape)-1
low=np.ones(shapeZYX);high=np.zeros(shapeZYX)
for x,y,z in corners:
 v=field[z:z+shapeZYX[0],y:y+shapeZYX[1],x:x+shapeZYX[2]];low=np.minimum(low,v);high=np.maximum(high,v)
for z,y,x in np.argwhere(low!=high):
 pos=corners+[x,y,z];v=field[pos[:,2],pos[:,1],pos[:,0]]
 for tet in tets:
  inside=[i for i in tet if v[i]>.5];outside=[i for i in tet if v[i]<=.5]
  if not inside or not outside:continue
  pts=np.array([(pos[i]+pos[j])*.5-1 for i in inside for j in outside])
  normal=pos[outside].mean(0)-pos[inside].mean(0);normal=normal/np.linalg.norm(normal)
  center=pts.mean(0);u=pts[0]-center;u=u/np.linalg.norm(u);vaxis=np.cross(normal,u)
  angles=np.arctan2((pts-center)@vaxis,(pts-center)@u);pts=pts[np.argsort(angles)]
  ids=[vertex(p) for p in pts]
  for k in range(1,len(ids)-1):indices.extend([ids[0],ids[k],ids[k+1]])
np.asarray(verts,dtype='<f4').tofile(o/'foreground-vertices.bin');np.asarray(indices,dtype='<u4').tofile(o/'foreground-indices.bin')
def points(key):return ((a[key]-bounds[0])/spacing-shape/2).round(5).tolist()
data={'shape':shape.tolist(),'spacingAngstrom':spacing.tolist(),'physicalSizeNm':((bounds[1]-bounds[0])/10).tolist(),'reference':points('gt_xyz_angstrom'),'native':points('native_centers_xyz_angstrom'),'baseline':points('DeepFinder_pred_xyz_angstrom'),'local':points('Local_pred_xyz_angstrom'),'localMatched':a['Local_matched'].tolist(),'direct':points('TomoDETR_pred_xyz_angstrom'),'directMatched':a['TomoDETR_matched'].tolist(),'sourceSHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'isosurface':{'level':.5,'method':'marching tetrahedra','smoothing':False,'vertices':len(verts),'triangles':len(indices)//3},'display':'Shared physical coordinates. Marker sizes are visual guides, not particle radii. Recorded stage endpoints; no interpolated prediction trajectories.'}
(o/'scene.json').write_text(json.dumps(data,separators=(',',':'))+'\n');print(json.dumps(data['isosurface']))
