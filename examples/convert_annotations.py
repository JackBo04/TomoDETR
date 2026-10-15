"""Convert named x/y/z CSV coordinates to detector-voxel z/y/x CSV."""
import argparse
import csv
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-spacing',type=float,required=True)
    p.add_argument('--detector-spacing',type=float,required=True)
    a=p.parse_args()
    if not all(math.isfinite(s) and s>0 for s in (a.input_spacing,a.detector_spacing)):
        p.error('Spacings must be finite and positive')
    rows=[]
    with a.input.open(newline='') as f:
        reader=csv.DictReader(f)
        if not {'x','y','z','class_id'}.issubset(reader.fieldnames or []):
            p.error('Input requires x,y,z,class_id columns')
        for row in reader:
            point=[float(row[k])*a.input_spacing/a.detector_spacing for k in ('z','y','x')]
            label=float(row['class_id'])
            if not all(math.isfinite(x) and x>=0 for x in point) or not math.isfinite(label) or label<0 or not label.is_integer():
                p.error('Invalid point or class ID')
            rows.append([*point,int(label)])
    with a.output.open('x',newline='') as f:
        writer=csv.writer(f); writer.writerow(['z','y','x','class_id']); writer.writerows(rows)


if __name__=='__main__': main()
