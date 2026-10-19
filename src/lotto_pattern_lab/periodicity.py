"""Retrospective autocorrelation and FFT diagnostics; these do not generate forecasts."""

import numpy as np


def periodicity_report(
    values: tuple[float, ...] | list[float], max_lag: int = 100
) -> dict[str, object]:
    """Return normalized autocorrelation and strongest FFT periods for a finite sequence."""
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size < 4:
        raise ValueError("Periodicity analysis requires at least four scalar observations.")
    if not np.isfinite(series).all():
        raise ValueError("Periodicity analysis only accepts finite observations.")
    centered = series - float(series.mean())
    denominator = float(np.dot(centered, centered))
    lag_limit = min(max(int(max_lag), 1), series.size - 1)
    if denominator == 0:
        correlations = [1.0] + [0.0] * lag_limit
        peaks: list[dict[str, float]] = []
    else:
        correlations = [
            float(np.dot(centered[:-lag], centered[lag:]) / denominator)
            for lag in range(1, lag_limit + 1)
        ]
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        frequencies = np.fft.rfftfreq(series.size)
        eligible = np.flatnonzero(frequencies > 0)
        ranked = sorted(eligible, key=lambda index: spectrum[index], reverse=True)[:5]
        peaks = [
            {
                "period": float(1 / frequencies[index]),
                "power": float(spectrum[index]),
            }
            for index in ranked
        ]
        correlations.insert(0, 1.0)
    return {
        "observation_count": int(series.size),
        "autocorrelation_by_lag": correlations,
        "fft_period_peaks": peaks,
        "interpretation": "Exploratory retrospective diagnostic; not a validated prediction rule.",
    }
