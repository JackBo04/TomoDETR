"""Occupancy-based SHREC2021 localization matching, independent of predicted class."""
import numpy as np
from scipy.ndimage import generate_binary_structure, grey_dilation


def load_truth(row):
    import mrcfile
    names = ['background']
    for line in open(row['particle_locations']):
        if line.strip():
            names.append(line.split()[0].upper())
    with mrcfile.open(row['occupancy'],permissive=False) as f:
        occupancy = grey_dilation(np.asarray(f.data),footprint=generate_binary_structure(3,1))
    if occupancy.min()<0 or occupancy.max()>=len(names):
        raise ValueError('Occupancy IDs do not match particle_locations row order')
    return np.asarray(names),occupancy


def counts(prediction, names, occupancy, classes):
    keep = np.asarray([classes[int(c)].lower()!='vesicle' for c in prediction['labels']],dtype=bool)
    xyz = prediction['xyz'][keep]
    point = np.rint(np.clip(xyz,0,np.asarray(occupancy.shape)-1)).astype(np.int64)
    found = np.unique(occupancy[tuple(point.T)])
    eligible = np.flatnonzero((names!='4V94') & (names!='VESICLE') & (names!='background'))
    tp = len(np.intersect1d(found,eligible))
    # Organizer convention: exclude 4V94; retain the historical vesicle denominator.
    denominator = int(((names!='4V94') & (names!='background')).sum())
    return np.asarray([tp,len(xyz)-tp,denominator-tp],dtype=np.int64)


def write_submission(path,prediction,classes):
    with open(path,'w') as f:
        for (z,y,x),label,score in zip(prediction['xyz'],prediction['labels'],prediction['scores']):
            name = classes[int(label)]
            if name.lower()!='vesicle':
                f.write(f'{name} {x:.6f} {y:.6f} {z:.6f} {score:.8f}\n')
