"""Export a raw-volume demo from figure provenance and recorded 3D predictions.
Usage: python tools/export_volume_demo.py --source /path/to/figure_arrays --output site/demo/assets
The source directory contains comparison_manifest.json and comparison_plot_arrays.npz.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mrcfile
from PIL import Image

def export(source, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest=json.loads((source/'comparison_manifest.json').read_text())
    arrays=np.load(source/'comparison_plot_arrays.npz',allow_pickle=False)
    cases=[]
    for i,m in enumerate(manifest):
        low,high=np.array(m['box'],int)
        with mrcfile.mmap(m['sources']['raw'],permissive=True) as f:
            spacing=np.array([float(f.voxel_size[k]) for k in ('x','y','z')])
            volume=np.array(f.data[low[2]:high[2],low[1]:high[1],low[0]:high[0]],copy=True)
        assert list(volume.shape)==list((high-low)[::-1])
        assert np.isfinite(volume).all()
        bounds=arrays[f'row{i}_bounds_xyz_angstrom']
        assert np.allclose(bounds[0],low*spacing,atol=.01)
        # A single intensity mapping for every plane and every method in this volume.
        vmin,vmax=np.percentile(volume,[1,99]); views={}
        for name,axis,wh in [('z',0,(high-low)[[0,1]]),('y',1,(high-low)[[0,2]]),('x',2,(high-low)[[1,2]])]:
            count=volume.shape[axis];tile=160;cols=10;rows=(count+cols-1)//cols
            sheet=Image.new('L',(cols*tile,rows*tile))
            for j in range(count):
                # Three real adjacent slices, clipped at the volume edge.
                slab=np.take(volume,range(max(0,j-1),min(count,j+2)),axis=axis).mean(axis=axis)
                img=Image.fromarray(np.uint8(np.clip((slab-vmin)/(vmax-vmin),0,1)*255))
                img=img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((tile,tile),Image.Resampling.BILINEAR)
                sheet.paste(img,((j%cols)*tile,(j//cols)*tile))
            file=f'volume-{i}-{name}.webp';sheet.save(output/file,quality=88)
            views[name]={'file':'assets/'+file,'count':count,'tile':tile,'columns':cols,'aspect':float(wh[0]/wh[1])}
        def coords(points):return ((points-low*spacing)/10).round(6).tolist()
        methods={}
        for method in ['DeepETPicker','DeepFinder','OCTOPI','Local','TomoDETR']:
            pred=arrays[f'row{i}_{method}_pred_xyz_angstrom'];hit=arrays[f'row{i}_{method}_matched'];fn=arrays[f'row{i}_{method}_fn_xyz_angstrom']
            methods[method]={'tp':coords(pred[hit]),'fp':coords(pred[~hit]),'fn':coords(fn),'counts':[int(hit.sum()),int((~hit).sum()),len(fn)]}
        cases.append({'run':m['run'],'dataset':m['dataset'],'sizeNm':((high-low)*spacing/10).tolist(),'spacingNm':(spacing/10).tolist(),'views':views,'gt':coords(arrays[f'row{i}_gt_xyz_angstrom']),'methods':methods,'rawCropSHA256':hashlib.sha256(volume.tobytes()).hexdigest(),'shapeZYX':list(volume.shape)})
    detail=np.load(source/'xz_detail.npz',allow_pickle=False)
    cases[2]['focusNm']=((detail['gt_xyz_A']-np.array(manifest[2]['box'][0])*detail['spacing_A'])/10).tolist()
    cases[2]['localErrorNm']=float(np.linalg.norm(detail['local_xyz_A']-detail['gt_xyz_A'])/10)
    cases[2]['fullErrorNm']=float(np.linalg.norm(detail['full_xyz_A']-detail['gt_xyz_A'])/10)
    (output/'volumes.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'rawShapes':[c['shapeZYX'] for c in cases],'assetsBytes':sum(p.stat().st_size for p in output.iterdir())}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.source,a.output)
