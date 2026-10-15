import numpy as np
import pytest
import torch
from tomodetr import LocalDetector, CWQR
from tomodetr.graph import build_neighbor_graph
from tomodetr.pipeline import refiner_inputs, nms, origins, load_cache, save_cache
from tomodetr.losses import local_loss, global_loss
from tomodetr.metrics import counts, czii_counts
from tomodetr.geometry import crop_local, rotate90_z
from tomodetr.shrec import counts as shrec_counts
from tomodetr.data import manifest, read_annotations


def candidates(n=4,classes=6):
    return dict(query_feat=np.zeros((n,128),np.float32),local_logits=np.zeros((n,classes+1),np.float32),
                xyz_global=np.full((n,3),20,np.float32),xyz_local=np.full((n,3),20,np.float32),
                boundary_dist=np.full(n,20,np.float32),patch_id=np.arange(n),
                query_id=np.zeros(n,dtype=np.int64),tomo_shape=np.array([96,96,96]))


def test_paper_parameter_counts_and_grid():
    detector,refiner = LocalDetector(6),CWQR(6)
    assert sum(p.numel() for p in detector.parameters())==6_983_784
    assert sum(p.numel() for p in detector.backbone.parameters())==5_585_600
    assert sum(p.numel() for p in refiner.parameters())==468_506
    assert detector.initial_centers.shape==(64,3)
    assert torch.equal(detector.initial_centers.unique(),torch.tensor([.125,.375,.625,.875]))


def test_graph_cross_window_nearest_and_empty():
    xyz=np.array([[0,0,0],[0,0,.1],[0,0,1],[0,0,2],[0,0,3]],np.float32)
    graph=build_neighbor_graph(xyz,np.array([0,0,1,2,3]),2.5,2)
    assert graph.neighbor_idx[0].tolist()==[0,2]
    assert graph.neighbor_mask[0].all()
    np.testing.assert_allclose(graph.relative_xyz[0,1],[0,0,1])
    assert build_neighbor_graph(np.empty((0,3)),np.empty(0),10,32).neighbor_idx.shape==(0,32)


def test_cwqr_identity_bounds_and_gradient():
    arrays=candidates()
    inputs=refiner_inputs(arrays,10,'cpu')
    model=CWQR(6).eval()
    output=model(**inputs)
    torch.testing.assert_close(output['pred_logits'],inputs['local_logits'],rtol=0,atol=0)
    torch.testing.assert_close(output['pred_xyz'],inputs['xyz_global'],rtol=0,atol=0)
    loss=global_loss(output,torch.tensor([[21.,22.,23.]]),torch.tensor([0]),inputs['tomo_shape'],[10]*6)['loss']
    loss.backward()
    assert model.coordinate_delta.weight.grad.abs().sum()>0
    assert model.class_delta.weight.grad.abs().sum()>0
    with torch.no_grad():
        model.coordinate_delta.bias.fill_(100)
        inputs['xyz_global'][:]=94
        output=model(**inputs)
    assert output['pred_xyz'].max()<=95
    assert output['coordinate_delta_voxels'].abs().max()<=4


def test_empty_candidates_and_empty_truth():
    model=CWQR(6)
    inputs=refiner_inputs(candidates(0),10,'cpu')
    result=model(**inputs)
    loss=global_loss(result,torch.empty(0,3),torch.empty(0,dtype=torch.long),inputs['tomo_shape'],[10]*6)['loss']
    assert loss.item()==0
    loss.backward()


def test_local_matching_per_layer_and_loss_reduction():
    xyz=torch.zeros(2,64,3,requires_grad=True)
    logits=torch.zeros(2,64,3,requires_grad=True)
    truth=[torch.ones(1,3)*.2,torch.empty(0,3)]
    labels=[torch.tensor([1]),torch.empty(0,dtype=torch.long)]
    output={'layers':[dict(centers=xyz,class_logits=logits) for _ in range(4)]}
    row=local_loss(output,truth,labels)
    assert row['center_loss'].item()==pytest.approx(.2)
    assert row['class_loss'].item()==pytest.approx(np.log(3))
    assert row['loss'].item()==pytest.approx(1+np.log(3))
    row['loss'].backward()
    assert xyz.grad[1].abs().sum()==0
    assert logits.grad[1].abs().sum()>0


def test_nms_preserves_other_class_and_threshold_boundaries():
    xyz=np.array([[0,0,0],[0,0,1],[0,0,1]],np.float32)
    idx=nms(xyz,np.array([0,0,1]),np.array([.9,.8,.7]),[1,1])
    assert idx.tolist()==[0,2]
    assert list(origins([100,96,96]))==[(0,0,0),(4,0,0)]


def test_official_czii_and_strict_match_are_distinct():
    pred=np.array([[0.,0,0]])
    truth=np.array([[0.,0,0],[0,0,.1]])
    assert counts(pred,truth,1).tolist()==[1,0,1]
    assert czii_counts(pred,truth,1).tolist()==[2,-1,0]
    # Maximum cardinality must win over a shorter but incomplete assignment.
    pred=np.array([[0.,0,0],[0,0,.9]])
    truth=np.array([[0.,0,.8],[0,0,1.7]])
    assert counts(pred,truth,1).tolist()==[2,0,0]


def test_shrec_exclusions_and_occupancy():
    occupancy=np.array([[[0,1,2,3]]])
    names=np.array(['background','1S3X','4V94','VESICLE'])
    pred=dict(xyz=np.array([[0,0,1],[0,0,1],[0,0,3]]),labels=np.array([0,0,1]),scores=np.ones(3))
    assert shrec_counts(pred,names,occupancy,['1S3X','vesicle']).tolist()==[1,1,1]


def test_closed_window_coordinate_contract():
    points=np.array([[0,95,95],[0,95.1,3],[95,0,0]],np.float32)
    local,keep=crop_local(points,[0,0,0],96)
    assert keep.tolist()==[True,False,True]
    assert (rotate90_z(local,96,1)>=0).all()


def test_data_validation_and_cache_identity(tmp_path):
    csv=tmp_path/'truth.csv'
    csv.write_text('z,y,x,class_id\n1,2,3,0\n1,2,3,0\n')
    xyz,labels=read_annotations(csv,2,[96]*3)
    assert len(xyz)==1
    csv.write_text('z,y,x,class_id\n1,2,3,0.5\n')
    with pytest.raises(ValueError): read_annotations(csv,2)
    path=tmp_path/'bad.json'
    path.write_text('[{"id":"../bad","split":"train","volume":"v.npy"}]')
    with pytest.raises(ValueError): manifest(path)
    arrays=candidates()
    arrays['patch_id'][:]=0
    save_cache(tmp_path/'bad.npz',arrays,dict(schema=1,coordinate_order='zyx'))
    with pytest.raises(ValueError,match='Duplicate'): load_cache(tmp_path/'bad.npz',6)


def test_shrec_class_agnostic_nms_in_prediction():
    from tomodetr.pipeline import predict
    from tomodetr.config import load_config
    from pathlib import Path
    config = load_config(Path(__file__).parents[1] / 'configs/shrec.yaml')
    logits = np.full((3, 15), -10., dtype=np.float32)
    logits[0, 0], logits[1, 1], logits[2, 1] = 5., 4., 3.
    arrays = {'local_logits': logits, 'xyz_global': np.array([[0.,0.,0.], [0.,0.,5.], [0.,0.,11.]])}
    result = predict(arrays, config)
    assert result['labels'].tolist() == [0, 1]
    assert result['xyz'][:, 2].tolist() == [0., 11.]
    assert nms(np.empty((0, 3)), np.array([], dtype=int), np.array([]), [10.], False).size == 0
    with pytest.raises(ValueError, match='identical radii'):
        nms(arrays['xyz_global'], np.array([0,1,1]), np.array([.9,.8,.7]), [5.,10.], False)
