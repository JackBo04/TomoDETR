import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tomodetr import cli
from tomodetr.config import load_config


def test_annotation_free_prediction_and_frozen_identity(tmp_path, monkeypatch):
    volume=tmp_path/'test.npy'
    volume.write_bytes(b'unique test volume')
    config={'dataset':'deepict','classes':['a','b']}
    identity=dict(config=config,detector_sha256='detector',cwqr_sha256='cwqr')
    pred=dict(xyz=np.array([[1.,2.,3.]]),labels=np.array([0]),scores=np.array([.8]))
    rows=[(dict(id='test',volume=str(volume)),pred,np.array([96]*3))]
    monkeypatch.setattr(cli,'predictions',lambda *args:(rows,identity))
    def forbidden(*args):
        raise AssertionError('Inference opened annotations')
    monkeypatch.setattr(cli,'read_annotations',forbidden)
    calibration=dict(identity,thresholds=[.5,.5],validation_volume_hashes=['different'])
    path=tmp_path/'calibration.json'
    path.write_text(json.dumps(calibration))
    args=SimpleNamespace(split='test',calibration=path,output=tmp_path,evaluate=False)
    cli.infer(args,config)
    assert (tmp_path/'test.csv').read_text().startswith('z,y,x,class_id,score')
    calibration['cwqr_sha256']='wrong'
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError,match='differs'):
        cli.infer(args,config)


def test_czii_validation_calibration_and_test_overlap(tmp_path,monkeypatch):
    config=load_config(Path(__file__).parents[1]/'configs/czii.yaml')
    volume=tmp_path/'volume.npy'; volume.write_bytes(b'validation only')
    annotation=tmp_path/'gt.csv'; annotation.write_text('z,y,x,class_id\n10,10,10,0\n')
    pred=dict(xyz=np.array([[10.,10,10],[90,90,90]]),labels=np.array([0,0]),scores=np.array([.8,.1]))
    rows=[(dict(id='val',volume=str(volume),annotations=str(annotation)),pred,np.array([96]*3))]
    identity=dict(config=config,detector_sha256='d',cwqr_sha256=None)
    def only_validation(args,c,split):
        assert split=='validation'
        return rows,identity
    monkeypatch.setattr(cli,'predictions',only_validation)
    cli.calibrate(SimpleNamespace(output=tmp_path),config)
    calibration=json.loads((tmp_path/'calibration.json').read_text())
    assert .1<calibration['thresholds'][0]<=.8
    assert calibration['validation_metrics']['counts_tp_fp_fn'][0]==[1,0,0]
    monkeypatch.setattr(cli,'predictions',lambda *args:(rows,dict(config=config,detector_sha256='d',cwqr_sha256=None)))
    with pytest.raises(ValueError,match='used for calibration'):
        cli.infer(SimpleNamespace(split='test',calibration=tmp_path/'calibration.json',output=tmp_path,evaluate=False),config)


def test_unknown_experimental_config_key_rejected(tmp_path):
    import yaml
    source=Path(__file__).parents[1]/'configs/czii.yaml'
    config=yaml.safe_load(source.read_text())
    config['roi_refinement']=True
    path=tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='Config keys differ'):
        load_config(path)
