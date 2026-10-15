"""Explicit dataset settings; method architecture is fixed in code."""
from pathlib import Path
import yaml
import numpy as np


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    expected = {'dataset','classes','radii','nms_radii','tolerances','metric_weights',
                'beta','graph_radius','detector_epochs','schedule_epochs','seed'}
    if set(config) - {'nms_class_aware'} != expected:
        raise ValueError(f'Config keys differ: {set(config)^expected}')
    if config['dataset'] not in {'czii','shrec','deepict'}:
        raise ValueError('Unknown dataset')
    if 'nms_class_aware' in config and not isinstance(config['nms_class_aware'], bool):
        raise ValueError('nms_class_aware must be a boolean')
    if not config.get('nms_class_aware', True) and len(set(config['nms_radii'])) != 1:
        raise ValueError('Class-agnostic NMS requires identical radii')
    count = len(config['classes'])
    if count < 1 or len(set(config['classes'])) != count:
        raise ValueError('Class names must be unique and nonempty')
    for key in ('radii','nms_radii','tolerances','metric_weights'):
        values = np.asarray(config[key],dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f'{key} needs {count} finite values')
        if np.any(values < 0) or (key != 'metric_weights' and np.any(values==0)):
            raise ValueError(f'Invalid {key}')
    if sum(config['metric_weights']) <= 0 or config['graph_radius'] <= 0 or config['beta'] <= 0:
        raise ValueError('Invalid metric weights, graph radius or beta')
    if not 10 < config['detector_epochs'] <= config['schedule_epochs']:
        raise ValueError('Detector schedule must exceed warmup and cover training')
    return config
