"""Portable end-to-end reference test; does not require Qiskit."""
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_demo_pipeline_and_output_audit(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'run_pipeline.py'),'--demo','--engine','numpy',
                           '--qaoa-iterations','2','--qaoa-shots','1000',
                           '--simulation-shots','1500','--mc-iterations','5000'],
                          cwd=tmp_path,capture_output=True,text=True,timeout=100)
    assert result.returncode==0, result.stdout + result.stderr
    outputs=tmp_path/'example/demo_outputs'
    assert (outputs/'stage1_predictions.csv').exists()
    verify=subprocess.run([sys.executable,str(ROOT/'verify_submission.py'),
                           '--out',str(outputs),'--allow-demo'],
                          cwd=tmp_path,capture_output=True,text=True,timeout=100)
    assert verify.returncode==0,verify.stdout+verify.stderr
    assert 'SYNTHETIC ONLY' in verify.stdout
