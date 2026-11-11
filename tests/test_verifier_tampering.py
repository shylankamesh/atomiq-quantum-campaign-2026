"""Corrupt genuine outputs and require the independent verifier to fail."""
from pathlib import Path
import json,shutil
import pytest
from verify_submission import check

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    for name in ['stage1_predictions.csv','stage2_selection.json','stage3_simulation.json','run_manifest.json']:
        shutil.copy2(ROOT/name,tmp_path/name)
    (tmp_path/'data').mkdir()
    shutil.copy2(ROOT/'data/bank_campaign_participant_release.xlsx',tmp_path/'data/bank_campaign_participant_release.xlsx')
    return tmp_path


def test_real_bundle_passes(bundle):
    assert check(bundle)['status']=='PASS'


@pytest.mark.parametrize('filename,field,value',[
    ('stage2_selection.json','objective_value',12345),
    ('stage2_selection.json','qaoa_measured_optimality_gap',99),
    ('stage2_selection.json','budget_used',99),
    ('stage2_selection.json','measured_feasible_fraction',.5),
    ('stage2_selection.json','objective_type','expected_conversions'),
    ('stage3_simulation.json','quantum_estimated_probability',.99),
    ('stage3_simulation.json','quantum_noisy_probability',.99),
    ('stage3_simulation.json','exact_analytical_probability',.99),
    ('stage3_simulation.json','probabilities',[.5]*6),
    ('stage3_simulation.json','threshold',4),
])
def test_detects_inconsistent_outputs_without_hash_manifest(bundle,filename,field,value):
    path=bundle/filename; obj=json.loads(path.read_text()); obj[field]=value
    path.write_text(json.dumps(obj))
    with pytest.raises(AssertionError): check(bundle)


def test_detects_false_top_ten_flags(bundle):
    import pandas as pd
    path=bundle/'stage1_predictions.csv'; frame=pd.read_csv(path)
    frame.loc[0,'is_top_10']=False; frame.loc[10,'is_top_10']=True
    frame.to_csv(path,index=False)
    with pytest.raises(AssertionError): check(bundle)


def test_detects_changed_stage2_probability(bundle):
    path=bundle/'stage2_selection.json'; obj=json.loads(path.read_text()); obj['inputs'][0]['p']+=.01
    path.write_text(json.dumps(obj))
    with pytest.raises(AssertionError): check(bundle)
