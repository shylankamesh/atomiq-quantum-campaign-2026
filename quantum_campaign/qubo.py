"""Quadratic slack encoding for integral costs/capacities; not the executed oracle.

E=-r.x + L[(sum(x)-6)^2+(c.x+sB-B)^2+(a.x+sA-A)^2].
Binary x obeys x^2=x. Ising uses x=(1-Z)/2. Original implementation.
"""
import numpy as np


def slack_weights(limit):
    if int(limit)!=limit or limit<0:
        raise ValueError('Slack representation requires nonnegative integer limits')
    remaining=int(limit)
    weights=[]
    power=1
    while remaining:
        w=min(power,remaining)
        weights.append(w)
        remaining-=w
        power*=2
    return weights


def build_qubo(costs, capacities, rewards, budget, capacity, k=6):
    costs, capacities, rewards = map(lambda x:np.asarray(x,dtype=float),(costs,capacities,rewards))
    if np.any(costs!=np.round(costs)) or np.any(capacities!=np.round(capacities)):
        raise ValueError('Integer units required; no silent discretization')
    n=len(rewards)
    wb,wa=slack_weights(budget),slack_weights(capacity)
    width=n+len(wb)+len(wa)
    linear=np.zeros(width); linear[:n]=-rewards
    pairs={}; constant=0.
    penalty=float(np.abs(rewards).sum()+1.)
    vectors=[(np.r_[np.ones(n),np.zeros(width-n)],k),
             (np.r_[costs,wb,np.zeros(len(wa))],budget),
             (np.r_[capacities,np.zeros(len(wb)),wa],capacity)]
    for v,b in vectors:
        constant+=penalty*b*b
        linear+=penalty*(v*v-2*b*v)
        for i in range(width):
            for j in range(i+1,width):
                pairs[i,j]=pairs.get((i,j),0.)+2*penalty*v[i]*v[j]
    h=-linear/2
    offset=constant+linear.sum()/2
    for (i,j),q in pairs.items():
        h[i]-=q/4; h[j]-=q/4; offset+=q/4
    return dict(width=width,customer_qubits=n,budget_slack_weights=wb,capacity_slack_weights=wa,
                penalty=penalty,constant=float(constant),linear=linear.tolist(),
                quadratic=[dict(i=i,j=j,value=float(q)) for (i,j),q in pairs.items() if q],
                ising_offset=float(offset),ising_z=h.tolist(),
                ising_zz=[dict(i=i,j=j,value=float(q/4)) for (i,j),q in pairs.items() if q],
                execution_note='Validated algebraic alternative, not the 10-qubit circuit executed in this run')


def qubo_energy(model,bits):
    return model['constant']+np.dot(model['linear'],bits)+sum(t['value']*bits[t['i']]*bits[t['j']] for t in model['quadratic'])


def ising_energy(model,bits):
    z=1-2*np.asarray(bits)
    return model['ising_offset']+np.dot(model['ising_z'],z)+sum(t['value']*z[t['i']]*z[t['j']] for t in model['ising_zz'])
