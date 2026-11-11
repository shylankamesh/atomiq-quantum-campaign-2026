"""Build an editable Atomiq pitch deck from validated synthetic demonstration numbers.

The deck is deliberately watermarked; replace results with OFFICIAL run before submission.
"""
from pathlib import Path
import json
import math
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'example/demo_outputs'
S1=json.loads((DEMO/'stage1_validation.json').read_text())
S2=json.loads((DEMO/'stage2_selection.json').read_text())
S3=json.loads((DEMO/'stage3_simulation.json').read_text())
OUT=ROOT/'slides/Atomiq_Quantum_Campaign_Pitch_DEMO.pptx'

prs=Presentation()
prs.slide_width=Inches(13.333)
prs.slide_height=Inches(7.5)
BLK=RGBColor(11,17,35)
PANEL=RGBColor(21,32,56)
PANEL2=RGBColor(28,43,75)
WHITE=RGBColor(238,246,254)
MUTED=RGBColor(161,176,203)
CYAN=RGBColor(64,221,222)
PURPLE=RGBColor(165,130,255)
YELLOW=RGBColor(255,212,112)
GREEN=RGBColor(109,229,166)
RED=RGBColor(251,135,140)

def add_shape(slide,x,y,w,h,color,shape=MSO_SHAPE.ROUNDED_RECTANGLE,line=None):
    sh=slide.shapes.add_shape(shape, Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=color
    sh.line.fill.background() if line is None else None
    if line:
        sh.line.color.rgb=line
    return sh

def txt(slide,text,x,y,w,h,size=20,color=WHITE,bold=False,align=None):
    sh=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame=sh.text_frame
    frame.clear(); frame.word_wrap=True
    frame.margin_left=Inches(.02); frame.margin_top=Inches(.01)
    frame.margin_right=Inches(.02); frame.margin_bottom=Inches(.01)
    for i,line in enumerate(str(text).split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        p.text=line; p.font.name='Aptos'; p.font.size=Pt(size)
        p.font.bold=bold; p.font.color.rgb=color
        p.space_after=Pt(4)
        if align: p.alignment=align
    return sh

def base(title,subtitle=None,number=None):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb=BLK
    add_shape(slide,.5,.39,.09,.55,CYAN,MSO_SHAPE.RECTANGLE)
    txt(slide,title,.78,.35,11.6,.62,29,WHITE,True)
    if subtitle: txt(slide,subtitle,.8,1.05,11.5,.45,13,MUTED)
    # Thin footer and reference-run banner.
    add_shape(slide,.65,7.04,12,.012,PANEL2,MSO_SHAPE.RECTANGLE)
    txt(slide,'ATOMIQ / QFF 2026      •      SYNTHETIC DEMO — NOT OFFICIAL RESULTS',.78,7.11,11,.25,10,MUTED)
    if number is not None: txt(slide,f'{number:02d}',12.28,7.09,.43,.28,11,CYAN,True)
    return slide

def panel(slide,x,y,w,h,title,body=None,accent=CYAN):
    add_shape(slide,x,y,w,h,PANEL)
    add_shape(slide,x,y,.06,h,accent,MSO_SHAPE.RECTANGLE)
    txt(slide,title,x+.28,y+.22,w-.5,.45,21,WHITE,True)
    if body: txt(slide,body,x+.28,y+.85,w-.55,h-.95,16,MUTED)

def number(slide,number,label,x,y,w=2.5,accent=CYAN):
    add_shape(slide,x,y,w,1.35,PANEL)
    txt(slide,number,x+.16,y+.14,w-.3,.6,34,accent,True)
    txt(slide,label,x+.17,y+.85,w-.3,.34,12,MUTED)

def bullet(slide,text,x,y,w,accent=CYAN,size=16):
    add_shape(slide,x,y+.14,.09,.09,accent,MSO_SHAPE.OVAL)
    txt(slide,text,x+.22,y,w-.22,.67,size,MUTED)

def arrow(slide,x1,y1,x2,y2,color=CYAN):
    line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
    line.line.color.rgb=color;line.line.width=Pt(2)

# 1 Intro
s=base('Quantum Campaign Optimizer','A transparent three-stage quantum–classical marketing pipeline',1)
add_shape(s,.8,1.72,7.25,4.54,PANEL)
txt(s,'PREDICT.  OPTIMIZE.  VERIFY.',1.2,2.17,6.4,1.2,32,CYAN,True)
txt(s,'Four-qubit learning  →  ten-qubit selection  →  six-qubit simulation',1.25,3.55,6.4,1.08,21,WHITE)
txt(s,'Real constraints. Exact classical certificates. Explicit quantum evidence.',1.25,5.24,6.22,.5,16,MUTED)
# Orbital art made with native shapes
for j,r in enumerate([.7,1.3,1.9,2.5]):
    sh=add_shape(s,9.55-r/2,3.78-r/2,r,r,BLK,MSO_SHAPE.OVAL,line=PURPLE if j%2 else CYAN)
    sh.fill.transparency=100
for x,y,c in [(9.1,2.46,CYAN),(10.37,4.15,PURPLE),(8.62,4.19,YELLOW),(10.48,2.62,GREEN)]:
    add_shape(s,x,y,.17,.17,c,MSO_SHAPE.OVAL)
txt(s,'Chi(r)ps & Qreativity',8.33,5.44,4,.44,19,WHITE,True,PP_ALIGN.CENTER)

# 2 Challenge
s=base('One campaign. Three scientific questions.','What the shared Atomiq problem asks us to compute',2)
panel(s,.75,1.85,3.86,3.45,'01  Who will convert?',
      'Predict probabilities for all 100 unseen customers; report quality and rank the top ten.',CYAN)
panel(s,4.75,1.85,3.84,3.45,'02  Who should we target?',
      'Select exactly six from ten under official budget and capacity limits.',PURPLE)
panel(s,8.73,1.85,3.84,3.45,'03  Will we hit the target?',
      'Estimate the probability that at least three of those six convert.',YELLOW)
txt(s,'Quantum complexity is not the goal; correctness, feasibility, traceability and honest benchmarks are.',1.13,5.82,11.1,.7,20,WHITE,True,PP_ALIGN.CENTER)

# 3 Architecture
s=base('End-to-end architecture','Unbroken customer-ID and continuous-probability traceability',3)
steps=[('INPUT','Labeled history + 100 test customers'),('STAGE 1','4Q QML kernel + classical benchmarks'),
       ('STAGE 2','10Q XY-QAOA + 210 exact checks'),('STAGE 3','6Q RY encoding, shots and noise'),('OUTPUT','Six IDs and campaign-success probability')]
for i,(head,body) in enumerate(steps):
    xx=.65+i*2.55
    add_shape(s,xx,2.0,2.33,3.24,PANEL)
    txt(s,head,xx+.14,2.3,2.03,.41,19,[CYAN,PURPLE,YELLOW,GREEN,CYAN][i],True,PP_ALIGN.CENTER)
    txt(s,body,xx+.14,3.06,2.03,1.35,16,WHITE,align=PP_ALIGN.CENTER)
    if i<4: txt(s,'›',xx+2.33,3.3,.22,.36,22,CYAN,True,PP_ALIGN.CENTER)
add_shape(s,.84,5.78,11.72,.68,PANEL2)
txt(s,'Guardrails:  training-only preprocessing    •    measured QAOA gap    •    exact analytical Stage 3 check',1.02,5.91,11.4,.33,16,WHITE)

# 4 Stage1
s=base('Stage 1  /  Four-qubit quantum learning','Qiskit ML-compatible entangled fidelity kernel + calibration-aware blend',4)
panel(s,.79,1.64,4.14,3.75,'Quantum feature map',
      'RY(πx) + RZ(πx) on each qubit\nNeighbor RZZ entanglers\nQuantum kernel: |⟨φ(x)|φ(y)⟩|²\nSVC probability model',CYAN)
number(s,f"{S1['hybrid_validation_metrics']['pr_auc']:.3f}",'Validation PR-AUC',5.26,1.66,2.1,PURPLE)
number(s,f"{S1['hybrid_validation_metrics']['brier']:.3f}",'Validation Brier',7.61,1.66,2.1,YELLOW)
number(s,f"{S1['quantum_weight']:.0%}",'Quantum blend share',9.96,1.66,2.23,GREEN)
add_shape(s,5.24,3.32,7.13,2.74,PANEL)
txt(s,'How we preserve evaluation integrity',5.55,3.59,6.6,.43,21,WHITE,True)
bullet(s,'Fit transforms and feature selection only on training fold',5.55,4.31,6.55,accent=CYAN)
bullet(s,'Choose blend weight on held-out validation, never test labels',5.55,5.08,6.55,accent=CYAN)

# 5 Stage2
s=base('Stage 2  /  Ten-qubit constrained optimization','QAOA candidate sampling + an exact classical feasibility certificate',5)
number(s,'210','Exact six-person combinations',.86,1.69,3.01,CYAN)
number(s,f"{S2['qaoa_measured_optimality_gap']:.3f}",'Observed QAOA objective gap',4.2,1.69,3.01,GREEN)
number(s,'6 / 10','Selected from candidates',7.55,1.69,3.01,PURPLE)
add_shape(s,.8,3.42,7.7,2.91,PANEL)
txt(s,'Maximize Σ pᵢ xᵢ   with   Σ xᵢ = 6',1.13,3.75,6.95,.57,23,WHITE,True)
txt(s,'Subject to budget + capacity constraints',1.13,4.43,6.95,.4,18,CYAN)
txt(s,'XY mixer preserves exactly six (not budget). Exact solver certifies final IDs; no speedup claim.',1.13,5.12,6.95,.8,16,MUTED)
add_shape(s,8.85,3.42,3.72,2.91,PANEL)
txt(s,'Synthetic selected IDs',9.17,3.78,3.05,.4,19,WHITE,True)
txt(s,'\n'.join(S2['selected_customer_ids']),9.19,4.31,3.05,1.75,15,GREEN)

# 6 Stage3
s=base('Stage 3  /  Six-qubit conversion simulation','An interpretable, independently benchmarked probability',6)
add_shape(s,.74,1.68,5.91,3.92,PANEL)
txt(s,'θᵢ = 2 asin(√pᵢ)',1.1,2.08,5.15,.51,24,CYAN,True)
txt(s,'RY(θ₀)     •     RY(θ₁)     •     …     RY(θ₅)',1.1,2.91,5.1,.49,18,WHITE)
txt(s,'Measure six qubits\nSuccess ⇢ at least three “1” outcomes',1.11,3.83,4.9,1.22,19,MUTED)
number(s,f"{S3['quantum_estimated_probability']:.5f}",'Ideal sampled tail probability',6.98,1.75,5.21,GREEN)
number(s,f"{S3['exact_analytical_probability']:.5f}",'Exact Poisson–binomial reference',6.98,3.42,5.21,CYAN)
txt(s,'Also tested: Monte Carlo, readout bit-flip noise, mitigation and sensitivity.',7.04,5.27,5.16,.88,15,MUTED)

# 7 Evidence, including negative results
s=base('Evidence over hype','We report limitations and failure cases, not just favorable scores',7)
img=DEMO/'stage3_comparison.png'
if img.exists():
    s.shapes.add_picture(str(img),Inches(.8), Inches(1.52),width=Inches(7.5))
add_shape(s,8.6,1.71,3.91,4.7,PANEL)
txt(s,'What the demo shows',8.93,2.07,3.3,.49,22,CYAN,True)
bullet(s,'Ideal shots track exact DP',8.93,2.86,3.22,GREEN,15)
bullet(s,'Readout noise shifts estimates',8.93,3.77,3.22,YELLOW,15)
bullet(s,'Mitigation can worsen error',8.93,4.67,3.22,RED,15)
txt(s,'Synthetic independent readout model ≠ device-calibrated noise.',8.95,5.69,3.28,.56,12,MUTED)

# 8 Integrity and limits
s=base('What is quantum — and what is classical?','A precise, reproducible research boundary',8)
panel(s,.86,1.71,5.7,3.71,'Quantum components',
      '4Q entangling feature-map fidelity\n10Q QAOA diagonal phase oracle\n6Q rotation-encoded shot circuit\nAer execution after Qiskit installation',CYAN)
panel(s,6.78,1.71,5.7,3.71,'Classical components',
      'Data preprocessing and model fitting\nOptimizer parameter search\nExact enumeration of 210 subsets\nExact Poisson-binomial + Monte Carlo',PURPLE)
add_shape(s,.88,5.75,11.6,.65,PANEL2)
txt(s,'No hardware speedup • no organizer results • simulated X vs XY comparison only',1.11,5.92,11.1,.33,17,YELLOW,True,PP_ALIGN.CENTER)

# 9 Next actions
s=base('From reproducible prototype to valid submission','Complete these organizer-specific actions before the 10 PM IST deadline',9)
actions=[
 ('01','Load the actual CSV/XLSX datasets and verify the objective/limits'),
 ('02','Execute --engine qiskit and rerun all tests on installed Qiskit/Aer'),
 ('03','Replace synthetic IDs/plots/numbers with actual validated outputs'),
 ('04','Publish root-level outputs + lockfile + report in public GitHub'),
 ('05','Upload view-only deck and record the mandatory 1–2 minute video'),
]
for i,(n,body) in enumerate(actions):
    y=1.6+i*1.0
    add_shape(s,.88,y,11.61,.79,PANEL)
    txt(s,n,1.15,y+.16,.55,.45,22,CYAN,True)
    txt(s,body,1.83,y+.15,10.16,.45,18,WHITE)

# 10 Mixer architecture experiment: an honestly constrained synthetic comparison
s=base('Why the XY mixer matters','Cardinality is guaranteed by the ansatz; budget/capacity still need checks',10)
comp=S2['qaoa_mixer_comparison']
if comp:
    xy=comp['xy']; x=comp['x']
    panel(s,.79,1.58,5.78,3.85,'XY-QAOA  /  10 qubits',
          f"Uniform weight-six initial state\nXY ring mixer conserves Hamming weight\nFeasible probability: {xy['feasible_probability']:.1%}\nExactly-six mass: {xy['cardinality_probability']:.1%}",GREEN)
    panel(s,6.75,1.58,5.78,3.85,'X-mixer  /  original baseline',
          f"Uniform plus state (all 1024 states)\nRX mixer may flip selection count\nFeasible probability: {x['feasible_probability']:.1%}\nExactly-six mass: {x['cardinality_probability']:.1%}",PURPLE)
    add_shape(s,.85,5.69,11.62,.76,PANEL2)
    txt(s,'Same synthetic inputs + equal optimizer budget • 22/210 feasible subsets • not a hardware test',1.07,5.89,11.15,.39,17,YELLOW,True,PP_ALIGN.CENTER)
else:
    txt(s,'Generate this experiment with --compare-mixers before presenting.',1.2,3.0,10.9,.7,21,YELLOW,True)

# 11 Close
s=base('A campaign decision you can audit.','Our goal: scientific rigor over unsupported quantum-advantage claims',11)
txt(s,'4 qubits    →    10 qubits    →    6 qubits',1.2,2.18,10.9,1.05,37,CYAN,True,PP_ALIGN.CENTER)
add_shape(s,1.35,3.56,10.62,1.69,PANEL)
txt(s,'Prediction quality  •  Feasibility  •  Probability accuracy\nClassical certificates  •  Full customer-ID traceability',1.75,3.84,9.82,1.16,22,WHITE,True,PP_ALIGN.CENTER)
txt(s,'Atomiq Quantum Campaign Optimizer / Source and reproducibility in README',1.55,6.04,10.3,.48,16,MUTED,align=PP_ALIGN.CENTER)

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(OUT)
