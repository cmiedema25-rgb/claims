"""Quadrature filter for the stationary Gaussian calendar process."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp


def calendar_rule(order):
    nodes, weights = leggauss(order)
    return 7 * nodes, 7 * weights


def transition(phi, nodes, weights):
    variance = 1 - phi * phi
    delta = nodes[None, :] - phi * nodes[:, None]
    raw = (-.5 * delta * delta / variance - .5 * np.log(2 * np.pi * variance)
           + np.log(weights)[None, :])
    logp = raw - logsumexp(raw, axis=1)[:, None]
    score = phi / variance + delta * nodes[:, None] / variance - phi * delta**2 / variance**2
    score -= (np.exp(logp) * score).sum(axis=1)[:, None]
    return logp, score


def smooth(emissions, phi, nodes, weights):
    """Normalized forward/backward recursion, marginals and posterior transitions."""
    logp, transition_score = transition(phi, nodes, weights)
    initial = -.5 * nodes**2 - .5 * np.log(2 * np.pi) + np.log(weights)
    initial -= logsumexp(initial)
    periods, count = emissions.shape
    forward = np.empty((periods, count))
    scales = np.empty(periods)
    for q in range(periods):
        predicted = initial if q == 0 else logsumexp(forward[q-1, :, None] + logp, axis=0)
        unnormalized = predicted + emissions[q]
        scales[q] = logsumexp(unnormalized)
        forward[q] = unnormalized - scales[q]
    backward = np.zeros_like(forward)
    posterior_transition = np.empty((periods-1, count, count))
    for q in range(periods-2, -1, -1):
        joint = logp + emissions[q+1][None, :] + backward[q+1][None, :]
        row_total = logsumexp(joint, axis=1)
        backward[q] = row_total - scales[q+1]
        posterior_transition[q] = np.exp(joint - row_total[:, None])
    log_marginal = forward + backward
    log_marginal -= logsumexp(log_marginal, axis=1)[:, None]
    marginal = np.exp(log_marginal)
    phi_score = sum(np.sum(marginal[q, :, None] * posterior_transition[q] * transition_score)
                    for q in range(periods-1))
    return float(scales.sum()), marginal, posterior_transition, float(phi_score)
