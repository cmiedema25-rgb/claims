"""Censored claim likelihood with event effects and a Gaussian calendar chain.

Parameter order: pi; (mu_S, mu_T, sigma_S, sigma_T, rho) for each
population; beta_S, beta_T, gamma_S, gamma_T, phi.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri, owens_t, roots_hermitenorm
from scipy.stats import qmc
from calendar_filter import calendar_rule, smooth

BOUNDS = [(.1, .9), (8, 11), (2, 6), (.2, 1.2), (.2, 1.2), (-.8, .8),
          (11, 14), (2, 6), (.2, 1.2), (.2, 1.2), (-.8, .8),
          (.05, .9), (-.7, .7), (.05, .9), (-.7, .7), (.05, .95)]


def history_from_rows(rows):
    rows = sorted(rows, key=lambda r: (int(r['cohort_index']), r['event_id'], r['claim_id']))
    labels = np.array([r['event_id'] for r in rows])
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1]
    return {
        'rows': rows, 'events': labels[starts], 'starts': starts,
        'periods': np.array([int(rows[i]['cohort_index'])-1 for i in starts]),
        'count': max(int(r['cohort_index']) for r in rows),
        'age': np.log([float(r['observation_age_days']) for r in rows])[:, None],
        'limit': np.log([float(r['policy_limit_usd']) for r in rows])[:, None],
        'closed': np.array([str(r['closed']) == '1' for r in rows]),
        'exhausted': np.array([str(r['limit_exhausted']) == '1' for r in rows]),
        'S': np.log([float(r['paid_indemnity_usd']) if r['paid_indemnity_usd'] else 1 for r in rows])[:, None],
        'T': np.log([float(r['settlement_days']) if r['settlement_days'] else 1 for r in rows])[:, None],
    }


def load_history(path):
    with Path(path).open(newline='') as stream:
        return history_from_rows(list(csv.DictReader(stream)))


def normal_rule(n):
    nodes, weights = roots_hermitenorm(n)
    return nodes, weights / np.sqrt(2*np.pi)


def forecast_event_rule(n=201):
    nodes, weights = leggauss(n)
    nodes, weights = 7*nodes, 7*weights
    weights *= np.exp(-.5*nodes**2)/np.sqrt(2*np.pi)
    return nodes, weights/weights.sum()


def component(x, h, u, v, off, derivative=False):
    """Row likelihood conditional on both latent effects and population."""
    ms, mt, ss, st, rho = x[off:off+5]
    u, v = np.asarray(u)[None, :], np.asarray(v)[None, :]
    zs = (h['S']-ms-x[11]*u-x[13]*v)/ss
    zt = (h['T']-mt-x[12]*u-x[14]*v)/st
    variance = 1-rho*rho
    root = np.sqrt(variance)
    exact = h['closed'] & ~h['exhausted']
    capped = h['closed'] & h['exhausted']
    opened = ~h['closed']
    values = np.zeros_like(zs)
    scores = np.zeros((5, *zs.shape)) if derivative else None
    z, t = zs[exact], zt[exact]
    quadratic = z*z-2*rho*z*t+t*t
    values[exact] = (-np.log(2*np.pi*ss*st*root)-quadratic/(2*variance)
                     -h['S'][exact]-h['T'][exact])
    if derivative:
        scores[0, exact] = (z-rho*t)/(ss*variance)
        scores[1, exact] = (t-rho*z)/(st*variance)
        scores[2, exact] = (-1+(z*z-rho*z*t)/variance)/ss
        scores[3, exact] = (-1+(t*t-rho*z*t)/variance)/st
        scores[4, exact] = rho/variance+z*t/variance-rho*quadratic/variance**2
    t = zt[capped]
    b = (h['limit'][capped]-ms-x[11]*u-x[13]*v)/ss
    a = (b-rho*t)/root
    values[capped] = (-.5*t*t-np.log(st*np.sqrt(2*np.pi))-h['T'][capped]+log_ndtr(-a))
    if derivative:
        mills = np.exp(-.5*a*a-.5*np.log(2*np.pi)-log_ndtr(-a))
        scores[0, capped] = mills/(ss*root)
        scores[1, capped] = t/st-mills*rho/(st*root)
        scores[2, capped] = mills*b/(ss*root)
        scores[3, capped] = (t*t-1)/st-mills*rho*t/(st*root)
        scores[4, capped] = -mills*(rho*b-t)/(variance*root)
    a = (h['age'][opened]-mt-x[12]*u-x[14]*v)/st
    values[opened] = log_ndtr(-a)
    if derivative:
        mills = np.exp(-.5*a*a-.5*np.log(2*np.pi)-log_ndtr(-a))
        scores[1, opened] = mills/st
        scores[3, opened] = mills*a/st
    return values, scores


def event_likelihoods(x, h, u, uw, v, derivative=False):
    """Integrate each event's product of row mixtures over its single shared U."""
    uf, vf = np.repeat(u, len(v)), np.tile(v, len(u))
    parts = [component(x, h, uf, vf, off, derivative) for off in (1, 6)]
    terms = np.stack([np.log(x[0])+parts[0][0], np.log1p(-x[0])+parts[1][0]])
    mixture = logsumexp(terms, axis=0)
    shape = (len(h['events']), len(u), len(v))
    joint = np.add.reduceat(mixture, h['starts'], axis=0).reshape(shape)+np.log(uw)[None, :, None]
    log_event = logsumexp(joint, axis=1)
    posterior = np.exp(joint-log_event[:, None, :])
    if not derivative:
        return log_event, posterior
    responsibilities = np.exp(terms-mixture[None, :, :])
    gradient = np.zeros((16, len(h['events']), len(v)))

    def integrate(score):
        summed = np.add.reduceat(score, h['starts'], axis=0).reshape(shape)
        return (posterior*summed).sum(axis=1)

    gradient[0] = integrate(responsibilities[0]/x[0]-responsibilities[1]/(1-x[0]))
    severity_score = np.zeros_like(mixture)
    time_score = np.zeros_like(mixture)
    for g, off in enumerate((1, 6)):
        for j in range(5):
            gradient[off+j] = integrate(responsibilities[g]*parts[g][1][j])
        severity_score += responsibilities[g]*parts[g][1][0]
        time_score += responsibilities[g]*parts[g][1][1]
    for index, score, effect in [(11, severity_score, uf), (12, time_score, uf),
                                 (13, severity_score, vf), (14, time_score, vf)]:
        gradient[index] = integrate(score*effect[None, :])
    return log_event, gradient


def emissions(x, history, u, uw, v, derivative=False):
    result = np.zeros((history['count'], len(v)))
    gradient = np.zeros((16, history['count'], len(v))) if derivative else None
    rows_by_period = defaultdict(list)
    for row in history['rows']:
        rows_by_period[int(row['cohort_index'])-1].append(row)
    for period, rows in rows_by_period.items():
        h = history_from_rows(rows)
        likelihood, other = event_likelihoods(x, h, u, uw, v, derivative)
        result[period] = likelihood.sum(axis=0)
        if derivative:
            gradient[:, period] = other.sum(axis=1)
    return result, gradient


def marginal_objective(x, history, u, uw, v, vw, derivative=True):
    observed, scores = emissions(x, history, u, uw, v, derivative)
    likelihood, marginal, _, phi_score = smooth(observed, x[15], v, vw)
    if not derivative:
        return -likelihood
    gradient = np.sum(scores*marginal[None, :, :], axis=(1, 2))
    gradient[15] = phi_score
    return -likelihood, -gradient


def calibrate(history, final_event_order=160, final_calendar_order=121):
    starts = [
        [.55, 9.4, 3.5, .65, .6, -.3, 12, 4.8, .8, .7, .3, .5, .3, .35, -.2, .6],
        [.7, 9.8, 4, .4, .4, -.5, 12.6, 5.4, .5, .5, .5, .3, .5, .65, .3, .85],
        [.35, 8.8, 3, .85, .9, .3, 11.5, 4.5, 1, 1, -.2, .7, -.2, .2, -.4, .3],
    ]
    u, uw = normal_rule(40)
    v, vw = calendar_rule(51)
    fits = []
    for start in starts:
        fit = minimize(marginal_objective, start, args=(history, u, uw, v, vw), jac=True,
                       method='L-BFGS-B', bounds=BOUNDS,
                       options={'maxiter':500, 'ftol':1e-11, 'gtol':1e-4, 'maxls':35})
        print('Initial fit', fit.fun, fit.success, fit.nit, flush=True)
        fits.append(fit)
    best = min(fits, key=lambda fit: fit.fun)
    for eu, cv in [(80, 81), (final_event_order, final_calendar_order)]:
        u, uw = normal_rule(eu)
        v, vw = calendar_rule(cv)
        best = minimize(marginal_objective, best.x, args=(history, u, uw, v, vw), jac=True,
                        method='L-BFGS-B', bounds=BOUNDS,
                        options={'maxiter':350, 'ftol':1e-12, 'gtol':1e-4, 'maxls':35})
        print('Refined fit', eu, cv, best.fun, best.success, best.nit, flush=True)
    if not np.isfinite(best.fun):
        raise RuntimeError('Nonfinite calibration')
    return best.x


def bvn_cdf(h, k, rho):
    """Deterministic normal CDF via Owen's T."""
    h, k = np.broadcast_arrays(np.asarray(h, float), np.asarray(k, float))
    root = np.sqrt(1-rho*rho)
    ah = (np.divide(k, h, out=np.zeros_like(h), where=h != 0)-rho)/root
    ak = (np.divide(h, k, out=np.zeros_like(k), where=k != 0)-rho)/root
    result = .5*(ndtr(h)+ndtr(k))-owens_t(h, ah)-owens_t(k, ak)-.5*(h*k < 0)
    result = np.where(h == 0, .5*ndtr(k)+owens_t(k, rho/root), result)
    result = np.where(k == 0, .5*ndtr(h)+owens_t(h, rho/root), result)
    return np.clip(result, np.maximum(0, ndtr(h)+ndtr(k)-1), np.minimum(ndtr(h), ndtr(k)))


def open_moments(x, row, u, v):
    age, limit = float(row['observation_age_days']), float(row['policy_limit_usd'])
    uf, vf = u[:, None], v[None, :]
    totals = np.zeros((2, len(u), len(v)))
    survival = np.zeros((len(u), len(v)))
    for prior, off in ((x[0], 1), (1-x[0], 6)):
        ms, mt, ss, st, rho = x[off:off+5]
        ms, mt = ms+x[11]*uf+x[13]*vf, mt+x[12]*uf+x[14]*vf
        h, k = (np.log(limit)-ms)/ss, (mt-np.log(age))/st
        survival += prior*ndtr(k)
        atom = bvn_cdf(-h, k, rho)
        for index, power in enumerate((1, 2)):
            below = np.exp(power*ms+.5*power*power*ss*ss)*bvn_cdf(h-power*ss, k+power*rho*ss, -rho)
            totals[index] += prior*(below+limit**power*atom)
    return totals/np.maximum(survival[None, :, :], 1e-300)


def forecast_context(x, history, current, calendar_order=121, event_order=201):
    v, vw = calendar_rule(calendar_order)
    u, uw = forecast_event_rule(event_order)
    hu, hw = normal_rule(160)
    observed, _ = emissions(x, history, hu, hw, v)
    full = np.zeros((current['count'], len(v)))
    full[:history['count']] = observed
    posterior = {}
    by_event = defaultdict(list)
    for row in current['rows']:
        by_event[row['event_id']].append(row)
    for event, rows in by_event.items():
        h = history_from_rows(rows)
        log_event, post = event_likelihoods(x, h, u, uw, v)
        period = int(rows[0]['cohort_index'])-1
        full[period] += log_event[0]
        posterior[event] = post[0]
    _, marginal, transitions, _ = smooth(full, x[15], v, vw)
    return u, v, posterior, marginal, transitions


def conditional_draw(cdfs, labels, uniforms):
    draws = np.empty(len(uniforms), dtype=int)
    for label in np.unique(labels):
        selected = labels == label
        draws[selected] = np.searchsorted(cdfs[label], uniforms[selected], side='right')
    return np.minimum(draws, cdfs.shape[1]-1)


def sample_forecasts(x, queries, current, context, power=21, seed=1731):
    """Joint predictive draws keep both calendar and within-event dependence."""
    u, v, event_post, marginal, transitions = context
    rows = {r['claim_id']: r for r in current['rows']}
    claim_ids = sorted(r['claim_id'] for r in queries['claims'])
    events = sorted({rows[c]['event_id'] for c in claim_ids})
    periods = {e: int(next(r for r in current['rows'] if r['event_id'] == e)['cohort_index'])-1 for e in events}
    first, last = min(periods.values()), max(periods.values())
    event_cdf = {e: np.cumsum(event_post[e], axis=0).T for e in events}
    transition_cdf = np.cumsum(transitions, axis=2)
    pools = sorted(queries['portfolios'], key=lambda r:r['portfolio_id'])
    position = {cid: i for i, cid in enumerate(claim_ids)}
    pool_columns = [[position[cid] for cid in p['claim_ids']] for p in pools]
    dimensions = last-first+1+len(events)+3*len(claim_ids)
    engine = qmc.Sobol(dimensions, scramble=True, seed=seed)
    count = 2**power
    sums, squares = np.zeros(len(claim_ids)), np.zeros(len(claim_ids))
    pool_sums, pool_squares, tails = (np.zeros(len(pools)) for _ in range(3))
    for start in range(0, count, 2**14):
        n = min(2**14, count-start)
        draws = np.clip(engine.random(n), 2**-53, 1-2**-53)
        calendar = {}
        calendar[first] = np.minimum(np.searchsorted(np.cumsum(marginal[first]), draws[:, 0]), len(v)-1)
        for period in range(first+1, last+1):
            calendar[period] = conditional_draw(transition_cdf[period-1], calendar[period-1], draws[:, period-first])
        effects = {}
        offset = last-first+1
        for i, event in enumerate(events):
            effects[event] = u[conditional_draw(event_cdf[event], calendar[periods[event]], draws[:, offset+i])]
        pay = np.empty((n, len(claim_ids)))
        offset += len(events)
        for j, cid in enumerate(claim_ids):
            row = rows[cid]
            ue, ve = effects[row['event_id']], v[calendar[int(row['cohort_index'])-1]]
            age = np.log(float(row['observation_age_days']))
            limit = float(row['policy_limit_usd'])
            survival = np.stack([ndtr((x[off+1]+x[12]*ue+x[14]*ve-age)/x[off+3]) for off in (1, 6)])
            chance = x[0]*survival[0]/np.maximum(x[0]*survival[0]+(1-x[0])*survival[1], 1e-300)
            g = (draws[:, offset+3*j] >= chance).astype(int)
            off = 1+5*g
            sf = survival[g, np.arange(n)]
            zt = -ndtri(np.maximum(sf*draws[:, offset+3*j+1], 1e-300))
            zs = ndtri(draws[:, offset+3*j+2])
            rho, sigma = x[off+4], x[off+2]
            logs = x[off]+x[11]*ue+x[13]*ve+sigma*(rho*zt+np.sqrt(1-rho*rho)*zs)
            cut = np.log(limit)
            pay[:, j] = np.where(logs >= cut, limit, np.exp(np.minimum(logs, cut)))
        sums += pay.sum(axis=0)
        squares += (pay*pay).sum(axis=0)
        for j, (pool, columns) in enumerate(zip(pools, pool_columns, strict=True)):
            total = pay[:, columns].sum(axis=1)
            pool_sums[j] += total.sum()
            pool_squares[j] += total@total
            tails[j] += np.count_nonzero(total > float(pool['attachment_usd']))
    means, pool_means = sums/count, pool_sums/count
    return {
        'claims': [{'claim_id':cid, 'expected_unpaid_indemnity_usd':float(means[j]),
                    'standard_deviation_usd':float(np.sqrt(max(0, squares[j]/count-means[j]**2)))} for j,cid in enumerate(claim_ids)],
        'portfolios': [{'portfolio_id':p['portfolio_id'], 'expected_unpaid_indemnity_usd':float(pool_means[j]),
                        'standard_deviation_usd':float(np.sqrt(max(0, pool_squares[j]/count-pool_means[j]**2))),
                        'probability_above_attachment':float(tails[j]/count)} for j,p in enumerate(pools)]}


def predict(x, queries, current, history, calendar_order=121, event_order=201, power=21, replicates=4, diagnostics=None):
    context = forecast_context(x, history, current, calendar_order, event_order)
    u, v, post, marginal, transitions = context
    rows = {r['claim_id']:r for r in current['rows']}
    moments = {query['claim_id']:open_moments(x, rows[query['claim_id']], u, v) for query in queries['claims']}
    claims = []
    for cid in sorted(moments):
        row = rows[cid]
        period = int(row['cohort_index'])-1
        conditional = (moments[cid]*post[row['event_id']][None, :, :]).sum(axis=1)
        mean, second = conditional@marginal[period]
        claims.append({'claim_id':cid, 'expected_unpaid_indemnity_usd':float(mean),
                       'standard_deviation_usd':float(np.sqrt(max(0, second-mean*mean)))})
    pools = []
    for pool in sorted(queries['portfolios'], key=lambda r:r['portfolio_id']):
        by_event = defaultdict(list)
        for cid in pool['claim_ids']:
            by_event[rows[cid]['event_id']].append(cid)
        per_period = {}
        for event, members in by_event.items():
            period = int(rows[members[0]]['cohort_index'])-1
            cm = sum(moments[cid][0] for cid in members)
            cv = sum(moments[cid][1]-moments[cid][0]**2 for cid in members)
            em = (cm*post[event]).sum(axis=0)
            ev = ((cv+cm*cm)*post[event]).sum(axis=0)-em*em
            if period not in per_period:
                per_period[period] = [np.zeros(len(v)), np.zeros(len(v))]
            per_period[period][0] += em
            per_period[period][1] += np.maximum(ev, 0)
        mean = sum(pm@marginal[q] for q,(pm,pv) in per_period.items())
        second = sum((pv+pm*pm)@marginal[q] for q,(pm,pv) in per_period.items())
        periods = sorted(per_period)
        for i,q in enumerate(periods):
            for r in periods[i+1:]:
                propagated = per_period[r][0].copy()
                for k in range(r-1, q-1, -1):
                    propagated = transitions[k]@propagated
                second += 2*np.sum(marginal[q]*per_period[q][0]*propagated)
        pools.append({'portfolio_id':pool['portfolio_id'], 'expected_unpaid_indemnity_usd':float(mean),
                      'standard_deviation_usd':float(np.sqrt(max(0, second-mean*mean)))})
    samples = []
    for rep in range(replicates):
        samples.append(sample_forecasts(x, queries, current, context, power, 1731+104729*rep))
        print('Predictive replicate', rep+1, 'of', replicates, flush=True)
    for j, pool in enumerate(pools):
        pool['probability_above_attachment'] = float(np.mean([s['portfolios'][j]['probability_above_attachment'] for s in samples]))
    if diagnostics is not None:
        diagnostics['replicate_probabilities'] = [[p['probability_above_attachment'] for p in s['portfolios']] for s in samples]
        diagnostics['sample_outputs'] = samples
    return {'claims':claims, 'portfolios':pools}
