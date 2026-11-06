import numpy as np

from throwtrack.signals import interp_short_gaps, rolling_nanmedian


def test_short_gaps_filled_long_gaps_and_edges_kept():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0, np.nan])
    y = interp_short_gaps(x, max_gap=2)
    assert np.isnan(y[0]) and np.isnan(y[-1])          # edges never extrapolated
    assert y[2] == 2.0                                  # 1-frame gap filled
    assert np.isnan(y[4:8]).all()                       # 4-frame gap left alone


def test_interp_handles_multichannel():
    x = np.zeros((5, 2, 2))
    x[2] = np.nan
    assert np.allclose(interp_short_gaps(x, 1), 0.0)


def test_rolling_nanmedian_ignores_nans():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    m = rolling_nanmedian(x, 3)
    assert m[1] == 2.0          # median of (1, 3), NaN skipped
    assert m[3] == 4.0
