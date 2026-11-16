"""Independent posterior check using uniform grids and probability-space filtering.

This developer check is not called by the verifier. It independently implements
observation likelihoods and calendar filtering; it imports the predictive sampler
only to turn those independently computed posterior distributions into forecasts.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp
from reserving import history_from_rows, sample_forecasts


ROOT = Path(__file__).resolve().parents[1]


def uniform_posterior(x, history, current, step=.05):
    nodes = np.arange(-7., 7.+step/2, step)
    weights = np.full(len(nodes), step)
    weights[[0,-1]] *= .5
    normal_weights = weights*np.exp(-.5*nodes**2)/np.sqrt(2*np.pi)
    normal_weights /= normal_weights.sum()
    ue, ve = nodes[:, None], nodes[None, :]
    by_event = defaultdict(list)
    for row in history+current:
        by_event[row['event_id']].append(row)
    count = max(int(row['cohort_index']) for row in history+current)
    emissions = np.zeros((count,len(nodes)))
    event_posterior = {}
    current_ids = {row['event_id'] for row in current}
    for event,rows in by_event.items():
        ll = np.zeros((len(nodes),len(nodes)))
        for row in rows:
            components=[]
            for g,off in enumerate((1,6)):
                ms=x[off]+x[11]*ue+x[13]*ve
                mt=x[off+1]+x[12]*ue+x[14]*ve
                ss,st,rho=x[off+2:off+5]
                if row['closed']=='0':
                    logf=log_ndtr((mt-np.log(float(row['observation_age_days'])))/st)
                else:
                    duration=np.log(float(row['settlement_days']))
                    zt=(duration-mt)/st
                    if row['limit_exhausted']=='1':
                        threshold=np.log(float(row['policy_limit_usd']))
                        cut=(threshold-ms-rho*ss*zt)/(ss*np.sqrt(1-rho*rho))
                        logf=-.5*zt*zt-np.log(st*np.sqrt(2*np.pi))-duration+log_ndtr(-cut)
                    else:
                        severity=np.log(float(row['paid_indemnity_usd']))
                        zs=(severity-ms)/ss
                        logf=(-np.log(2*np.pi*ss*st*np.sqrt(1-rho*rho))
                              -(zs*zs-2*rho*zs*zt+zt*zt)/(2*(1-rho*rho))-severity-duration)
                components.append(np.log(x[0] if g==0 else 1-x[0])+logf)
            ll += np.logaddexp(components[0],components[1])
        joint=ll+np.log(normal_weights)[:,None]
        normalizer=logsumexp(joint,axis=0)
        emissions[int(rows[0]['cohort_index'])-1] += normalizer
        if event in current_ids:
            event_posterior[event]=np.exp(joint-normalizer[None,:])
    variance=1-x[15]**2
    transition=np.exp(-.5*(nodes[None,:]-x[15]*nodes[:,None])**2/variance)
    transition *= weights[None,:]
    transition /= transition.sum(axis=1)[:,None]
    density=np.exp(emissions-emissions.max(axis=1)[:,None])
    alpha=np.empty_like(density)
    scales=np.empty(count)
    for q in range(count):
        prior=normal_weights if q==0 else alpha[q-1]@transition
        alpha[q]=prior*density[q]
        scales[q]=alpha[q].sum()
        alpha[q]/=scales[q]
    beta=np.ones_like(alpha)
    posterior_transition=np.empty((count-1,len(nodes),len(nodes)))
    for q in range(count-2,-1,-1):
        matrix=transition*(density[q+1]*beta[q+1])[None,:]
        posterior_transition[q]=matrix/matrix.sum(axis=1)[:,None]
        beta[q]=matrix.sum(axis=1)/scales[q+1]
    marginal=alpha*beta
    marginal/=marginal.sum(axis=1)[:,None]
    return nodes,nodes,event_posterior,marginal,posterior_transition


def compare(reference, outputs):
    maximum_monetary=maximum_fraction=maximum_probability=0.
    for collection in ['claims','portfolios']:
        for j,expected in enumerate(reference[collection]):
            for field,value in expected.items():
                if field.endswith('_id'):
                    continue
                estimate=float(np.mean([o[collection][j][field] for o in outputs]))
                error=abs(estimate-value)
                if field=='probability_above_attachment':
                    maximum_probability=max(maximum_probability,error)
                    fraction=error/.005
                else:
                    maximum_monetary=max(maximum_monetary,error)
                    fraction=error/max(25,.01*abs(value))
                maximum_fraction=max(maximum_fraction,fraction)
    return {'max_monetary_error_usd':maximum_monetary,
            'max_probability_error':maximum_probability,
            'max_fraction_of_declared_tolerance':maximum_fraction}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--power',type=int,default=20)
    parser.add_argument('--replicates',type=int,default=3)
    parser.add_argument('--step',type=float,default=.05)
    args=parser.parse_args()
    x=np.array(json.loads((ROOT/'tests/model_reference.json').read_text())['maximum_likelihood_parameters'])
    with (ROOT/'environment/data/claim_history.csv').open(newline='') as stream:
        history=list(csv.DictReader(stream))
    with (ROOT/'environment/data/current_events.csv').open(newline='') as stream:
        current=list(csv.DictReader(stream))
    queries=json.loads((ROOT/'environment/data/open_claims.json').read_text())
    reference=json.loads((ROOT/'tests/ground_truth.json').read_text())
    context=uniform_posterior(x,history,current,args.step)
    outputs=[]
    for i in range(args.replicates):
        outputs.append(sample_forecasts(x,queries,history_from_rows(current),context,args.power,7919+65537*i))
        print('Independent posterior replicate',i+1,flush=True)
    report=compare(reference,outputs)
    print(json.dumps(report,indent=2),flush=True)
    assert report['max_fraction_of_declared_tolerance']<.5, report


if __name__=='__main__':
    main()
