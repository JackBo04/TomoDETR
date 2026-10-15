"""Strict one-to-one localization metrics (not a replacement for official scorers)."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def counts(pred, truth, radius):
    if not len(pred) or not len(truth):
        return np.asarray([0,len(pred),len(truth)],dtype=np.int64)
    distance = cdist(pred,truth)
    valid = distance <= radius
    penalty = (min(distance.shape)+1)*(radius+1)
    i,j = linear_sum_assignment(np.where(valid,distance,penalty))
    tp = int(valid[i,j].sum())
    return np.asarray([tp,len(pred)-tp,len(truth)-tp],dtype=np.int64)


def class_counts(prediction, xyz, labels, tolerances):
    return np.stack([counts(prediction['xyz'][prediction['labels']==c],
                            xyz[labels==c],r) for c,r in enumerate(tolerances)])


def fbeta(count, beta=1):
    tp,fp,fn = np.asarray(count).T
    denominator = (1+beta**2)*tp + beta**2*fn + fp
    return np.divide((1+beta**2)*tp,denominator,
                     out=np.zeros_like(denominator,dtype=float),where=denominator>0)


def summarize(count, config):
    pooled = np.asarray(count).sum(0)
    return {'counts_tp_fp_fn':np.asarray(count).tolist(),
            'class_fbeta':fbeta(count,config['beta']).tolist(),
            'weighted_macro_fbeta':float(np.average(fbeta(count,config['beta']),weights=config['metric_weights'])),
            'pooled_f1':float(fbeta(pooled,1)),
            'scorer':('CZII official reference-coverage semantics' if config['dataset']=='czii' else 'strict class-aware maximum-cardinality matching')}


def czii_counts(pred, truth, radius):
    """Official CZII reference-coverage semantics, including close-reference behavior."""
    from scipy.spatial import cKDTree
    if not len(pred) or not len(truth):
        return np.asarray([0,len(pred),len(truth)],dtype=np.int64)
    covered = cKDTree(pred).query_ball_tree(cKDTree(truth),radius)
    tp = len({i for neighbors in covered for i in neighbors})
    return np.asarray([tp,len(pred)-tp,len(truth)-tp],dtype=np.int64)


def benchmark_counts(prediction, xyz, labels, config):
    matcher = czii_counts if config['dataset']=='czii' else counts
    return np.stack([matcher(prediction['xyz'][prediction['labels']==c],
                             xyz[labels==c],r) for c,r in enumerate(config['tolerances'])])
