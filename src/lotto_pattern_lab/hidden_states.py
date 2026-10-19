"""Hypothetical hidden-state filtering and causal online change-point signals."""

from dataclasses import dataclass

import numpy as np

from .features import Example


@dataclass(frozen=True, slots=True)
class GaussianHMM:
    means: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    start_probability: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    iterations: int


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    result = safe_maximum + np.log(np.sum(np.exp(values - safe_maximum), axis=axis, keepdims=True))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def _log_emissions(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    differences = observations[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(
        np.log(2 * np.pi * variances)[None, :, :] + differences**2 / variances[None, :, :],
        axis=2,
    )


def fit_gaussian_hmm(
    observations: np.ndarray,
    state_count: int = 3,
    seed: int = 12345,
    max_iterations: int = 30,
) -> GaussianHMM:
    """Fit a diagonal Gaussian HMM on training observations only."""
    if observations.ndim != 2 or len(observations) < max(10, state_count * 3):
        raise ValueError(
            "HMM needs a 2D training sequence with at least three observations per state."
        )
    if not 2 <= state_count <= 5:
        raise ValueError("HMM state count must be between 2 and 5.")
    scaler_mean = observations.mean(axis=0)
    scaler_scale = observations.std(axis=0)
    scaler_scale[scaler_scale < 1e-8] = 1.0
    sequence = (observations - scaler_mean) / scaler_scale
    rng = np.random.default_rng(seed)

    # Quantile starts are deterministic and state labels have no semantic meaning.
    order = np.argsort(sequence[:, 0])
    positions = np.linspace(0, len(order) - 1, state_count).round().astype(int)
    means = sequence[order[positions]].copy()
    means += rng.normal(0.0, 0.01, size=means.shape)
    global_variance = np.maximum(sequence.var(axis=0), 0.1)
    variances = np.tile(global_variance, (state_count, 1))
    transition = np.full((state_count, state_count), 0.15 / (state_count - 1))
    np.fill_diagonal(transition, 0.85)
    start_probability = np.full(state_count, 1 / state_count)
    log_transition = np.log(transition)
    log_start = np.log(start_probability)
    previous_likelihood = -np.inf

    for iteration in range(1, max_iterations + 1):
        emissions = _log_emissions(sequence, means, variances)
        forward = np.empty_like(emissions)
        forward[0] = log_start + emissions[0]
        for time_index in range(1, len(sequence)):
            forward[time_index] = emissions[time_index] + _logsumexp(
                forward[time_index - 1, :, None] + log_transition, axis=0
            )
        likelihood = float(_logsumexp(forward[-1]).item())

        backward = np.zeros_like(emissions)
        for time_index in range(len(sequence) - 2, -1, -1):
            backward[time_index] = _logsumexp(
                log_transition
                + emissions[time_index + 1, None, :]
                + backward[time_index + 1, None, :],
                axis=1,
            )
        log_gamma = forward + backward
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)
        xi_sum = np.zeros((state_count, state_count))
        for time_index in range(len(sequence) - 1):
            log_xi = (
                forward[time_index, :, None]
                + log_transition
                + emissions[time_index + 1, None, :]
                + backward[time_index + 1, None, :]
            )
            log_xi -= _logsumexp(log_xi)
            xi_sum += np.exp(log_xi)

        start_probability = np.maximum(gamma[0], 1e-8)
        start_probability /= start_probability.sum()
        transition = xi_sum / np.maximum(gamma[:-1].sum(axis=0)[:, None], 1e-8)
        transition = np.maximum(transition, 1e-8)
        transition /= transition.sum(axis=1, keepdims=True)
        weights = gamma.sum(axis=0)[:, None]
        means = gamma.T @ sequence / np.maximum(weights, 1e-8)
        for state_index in range(state_count):
            residuals = sequence - means[state_index]
            variances[state_index] = np.maximum(
                np.sum(gamma[:, state_index, None] * residuals**2, axis=0)
                / max(float(weights[state_index, 0]), 1e-8),
                0.05,
            )
        log_transition = np.log(transition)
        log_start = np.log(start_probability)
        if abs(likelihood - previous_likelihood) < 1e-5:
            break
        previous_likelihood = likelihood

    return GaussianHMM(
        means, variances, transition, start_probability, scaler_mean, scaler_scale, iteration
    )


def causal_state_probabilities(model: GaussianHMM, examples: tuple[Example, ...]) -> np.ndarray:
    """Return a next-round state prior using only the immediately preceding draw history."""
    state_count = len(model.start_probability)
    alpha = model.start_probability.copy()
    probabilities = np.zeros((len(examples), state_count))
    log_transition = np.log(np.maximum(model.transition, 1e-12))
    for index, example in enumerate(examples):
        prior_draw = np.asarray(example.features[1:7], dtype=float)
        normalized = (prior_draw - model.scaler_mean) / model.scaler_scale
        emissions = np.exp(_log_emissions(normalized[None, :], model.means, model.variances)[0])
        posterior = alpha * emissions
        total = posterior.sum()
        posterior = posterior / total if total > 0 else model.start_probability.copy()
        alpha = np.exp(
            _logsumexp(np.log(np.maximum(posterior, 1e-12))[:, None] + log_transition, axis=0)
        )
        alpha /= alpha.sum()
        probabilities[index] = alpha
    return probabilities


def online_change_points(
    examples: tuple[Example, ...], window: int = 12, threshold: float = 2.5
) -> tuple[dict[str, int | float], ...]:
    """Causal rolling two-window mean-shift flags based on known prior-draw sums."""
    if window < 3 or threshold <= 0:
        raise ValueError("Change-point window must be >= 3 and threshold positive.")
    signals: list[dict[str, int | float]] = []
    observations: list[float] = []
    for example in examples:
        prior_sum = float(example.features[1])
        score = 0.0
        if len(observations) >= 2 * window:
            older = np.asarray(observations[-2 * window : -window])
            recent = np.asarray(observations[-window:])
            scale = max(float(np.std(older, ddof=1)), 1.0)
            score = abs(float(recent.mean() - older.mean())) / scale
        signals.append(
            {
                "round_number": example.round_number,
                "score": score,
                "change_point": int(score >= threshold),
            }
        )
        observations.append(prior_sum)
    return tuple(signals)
