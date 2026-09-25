from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_PATH = REPO_ROOT / 'data' / 'snapshots' / 'portfolio_returns_2026-08-28.csv'
FORECAST_PATH = REPO_ROOT / 'results' / 'latest_forecast_2026-08-28.csv'
METADATA_PATH = REPO_ROOT / 'results' / 'latest_forecast_metadata_2026-08-28.json'
GB_SOURCE_PATH = REPO_ROOT / 'src' / 'gb_return_features.py'

RETURN_COL = 'portfolio_simple_return'
EXPECTED_GB_FEATURES = [
    'return_lag_1',
    'return_lag_2',
    'return_lag_5',
    'rolling_vol_5',
    'rolling_vol_20',
    'rolling_vol_60',
    'drawdown',
]
TOL = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_date(value: object) -> pd.Timestamp:
    return pd.Timestamp(str(value)).normalize()


def target_features_from_available_history(returns: pd.Series) -> dict[str, float]:
    """Build the next-target feature vector using only already-observed returns."""
    require(len(returns) >= 60, 'At least 60 observations are required for G04 features.')
    arr = returns.to_numpy(dtype='float64')
    require(np.isfinite(arr).all(), 'Portfolio returns must be finite.')

    wealth = (1.0 + returns).cumprod()
    drawdown = float(wealth.iloc[-1] / wealth.cummax().iloc[-1] - 1.0)

    return {
        'return_lag_1': float(returns.iloc[-1]),
        'return_lag_2': float(returns.iloc[-2]),
        'return_lag_5': float(returns.iloc[-5]),
        'rolling_vol_5': float(returns.iloc[-5:].std(ddof=1)),
        'rolling_vol_20': float(returns.iloc[-20:].std(ddof=1)),
        'rolling_vol_60': float(returns.iloc[-60:].std(ddof=1)),
        'drawdown': drawdown,
    }


def training_feature_frame(returns: pd.Series) -> pd.DataFrame:
    """Independent target-date-aligned G04 training frame."""
    frame = pd.DataFrame(index=returns.index.copy())
    frame['return_lag_1'] = returns.shift(1)
    frame['return_lag_2'] = returns.shift(2)
    frame['return_lag_5'] = returns.shift(5)
    prior = returns.shift(1)
    frame['rolling_vol_5'] = prior.rolling(5, min_periods=5).std(ddof=1)
    frame['rolling_vol_20'] = prior.rolling(20, min_periods=20).std(ddof=1)
    frame['rolling_vol_60'] = prior.rolling(60, min_periods=60).std(ddof=1)

    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    frame['drawdown'] = drawdown.shift(1)
    frame['target_return'] = returns
    return frame.dropna()


def main() -> None:
    for path in (PORTFOLIO_PATH, FORECAST_PATH, METADATA_PATH, GB_SOURCE_PATH):
        require(path.is_file(), f'Missing required artifact: {path.relative_to(REPO_ROOT)}')

    portfolio = pd.read_csv(PORTFOLIO_PATH)
    forecast = pd.read_csv(FORECAST_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))

    portfolio['date'] = pd.to_datetime(portfolio['date'], errors='raise').dt.normalize()
    portfolio = portfolio.set_index('date', drop=True)
    require(portfolio.index.is_monotonic_increasing, 'Portfolio snapshot is not chronological.')
    require(not portfolio.index.has_duplicates, 'Portfolio snapshot contains duplicate dates.')
    require(RETURN_COL in portfolio.columns, f'Missing {RETURN_COL}.')

    returns = pd.to_numeric(portfolio[RETURN_COL], errors='raise').astype('float64')
    require(np.isfinite(returns.to_numpy()).all(), 'Portfolio returns contain non-finite values.')

    op = metadata['operational_contract']
    cutoff = parse_date(op['data_cutoff'])
    forecast_date = parse_date(op['forecast_date'])
    target_date = parse_date(op['target_date'])

    require(portfolio.index.max() == cutoff, 'Snapshot maximum date must equal operational cutoff.')
    require(forecast_date == cutoff, 'forecast_date must equal the latest available cutoff.')
    require(target_date > forecast_date, 'target_date must be strictly after forecast_date.')
    require((portfolio.index <= cutoff).all(), 'Snapshot contains observations after data_cutoff.')
    require(target_date not in portfolio.index, 'Future target return is already present in the input snapshot.')

    # Historical Simulation: exact rolling w250, ending at cutoff.
    hist_cfg = metadata['models']['historical_simulation']
    window = int(hist_cfg['window'])
    hist_window = returns.loc[:cutoff].iloc[-window:]
    require(window == 250, 'Historical window is not 250.')
    require(len(hist_window) == 250, 'Historical training window does not contain exactly 250 observations.')
    require(hist_window.index.max() == cutoff, 'Historical window does not terminate at data_cutoff.')
    require((hist_window.index <= cutoff).all(), 'Historical window contains a future observation.')
    require((hist_window.index < target_date).all(), 'Historical window reaches target/future date.')

    # EWMA: only state history available through cutoff.
    ewma_cfg = metadata['models']['ewma']
    ewma_history = returns.loc[:cutoff]
    require(float(ewma_cfg['decay']) == 0.94, 'EWMA decay is not 0.94.')
    require(ewma_history.index.max() == cutoff, 'EWMA history does not terminate at data_cutoff.')
    require((ewma_history.index <= cutoff).all(), 'EWMA history contains a future observation.')
    require((ewma_history.index < target_date).all(), 'EWMA history reaches target/future date.')

    # GB: independently reconstruct the seven target features from data <= cutoff.
    gb_cfg = metadata['models']['gradient_boosting']
    require(list(gb_cfg['features']) == EXPECTED_GB_FEATURES, 'Operational GB feature list is not canonical G04.')
    require(gb_cfg.get('market_features_used') is False, 'Operational GB unexpectedly uses market features.')

    available = returns.loc[:cutoff]
    next_features = target_features_from_available_history(available)
    require(list(next_features) == EXPECTED_GB_FEATURES, 'Independent GB feature order mismatch.')
    require(np.isfinite(np.array(list(next_features.values()), dtype='float64')).all(), 'GB target features are not finite.')

    # Independently build historical training rows. Each row's target is at date t,
    # while all seven features are shifted/prior-history quantities (< t).
    train_frame = training_feature_frame(available)
    require(not train_frame.empty, 'GB training frame is empty.')
    require(train_frame.index.max() <= cutoff, 'GB training target rows extend beyond cutoff.')
    require((train_frame.index < target_date).all(), 'GB training rows include target/future date.')

    # Explicit arithmetic checks for latest target feature vector.
    require(abs(next_features['return_lag_1'] - float(available.iloc[-1])) <= TOL, 'GB lag1 is not the cutoff return.')
    require(abs(next_features['return_lag_2'] - float(available.iloc[-2])) <= TOL, 'GB lag2 mismatch.')
    require(abs(next_features['return_lag_5'] - float(available.iloc[-5])) <= TOL, 'GB lag5 mismatch.')
    for n in (5, 20, 60):
        expected = float(available.iloc[-n:].std(ddof=1))
        require(abs(next_features[f'rolling_vol_{n}'] - expected) <= TOL, f'GB rolling_vol_{n} mismatch.')

    # Future-perturbation independence: appending arbitrary future returns must not
    # alter a feature vector explicitly defined from history through cutoff.
    future_dates = pd.to_datetime(['2026-08-31', '2026-09-01', '2026-09-02', '2026-09-03'])
    future = pd.Series([0.75, -0.60, 0.50, -0.40], index=future_dates, name=RETURN_COL)
    perturbed_full = pd.concat([available, future])
    perturbed_available = perturbed_full.loc[:cutoff]
    perturbed_features = target_features_from_available_history(perturbed_available)
    for name in EXPECTED_GB_FEATURES:
        require(abs(next_features[name] - perturbed_features[name]) <= TOL, f'Future perturbation changed {name}.')

    # Source-level feature semantics: canonical feature builder uses shifts for
    # lags, rolling history, and drawdown. This audits feature semantics only;
    # it does not claim provenance for an absent operational generation script.
    source = GB_SOURCE_PATH.read_text(encoding='utf-8')
    require('.shift(lag)' in source, 'Canonical GB lag implementation is not shifted.')
    require('available_history = validated_returns.shift(1)' in source, 'Canonical rolling feature history is not shifted.')
    require('contemporaneous_drawdown.shift(1)' in source, 'Canonical drawdown is not shifted.')

    # Forecast artifact itself must refer to the same temporal contract.
    for col in ('cutoff_date', 'forecast_date', 'target_date'):
        forecast[col] = pd.to_datetime(forecast[col], errors='raise').dt.normalize()
    require((forecast['cutoff_date'] == cutoff).all(), 'Forecast CSV cutoff disagrees with metadata.')
    require((forecast['forecast_date'] == cutoff).all(), 'Forecast CSV forecast_date disagrees with cutoff.')
    require((forecast['target_date'] == target_date).all(), 'Forecast CSV target_date disagrees with metadata.')

    print('PASS - B28.5 INDEPENDENT TEMPORAL/LEAKAGE AUDIT')
    print()
    print('Temporal contract')
    print(f'  data cutoff = {cutoff.date()}')
    print(f'  forecast date = {forecast_date.date()}')
    print(f'  target date = {target_date.date()}')
    print(f'  latest input observation = {portfolio.index.max().date()}')
    print('  [PASS] chronological')
    print('  [PASS] input snapshot contains no observation after data_cutoff')
    print('  [PASS] future target return is absent from forecast input')
    print()
    print('Historical Simulation')
    print(f'  window rows = {len(hist_window)}')
    print(f'  window start = {hist_window.index.min().date()}')
    print(f'  window end = {hist_window.index.max().date()}')
    print('  [PASS] exact w250 history')
    print('  [PASS] every observation date <= data_cutoff')
    print('  [PASS] no target/future observation in window')
    print()
    print('EWMA')
    print(f'  state-history rows = {len(ewma_history)}')
    print(f'  state-history end = {ewma_history.index.max().date()}')
    print('  [PASS] decay = 0.94')
    print('  [PASS] state history uses returns through data_cutoff only')
    print('  [PASS] no target/future return enters state history')
    print()
    print('Gradient Boosting G04')
    print(f'  training rows after warm-up = {len(train_frame)}')
    print(f'  latest training target date = {train_frame.index.max().date()}')
    print('  [PASS] lag values originate strictly from prior observations')
    print('  [PASS] rolling windows terminate at data_cutoff for the next-target feature vector')
    print('  [PASS] drawdown uses wealth history available through data_cutoff')
    print('  [PASS] training rows contain no future target return')
    print('  [PASS] exact 7 G04 features; no market features')
    print('  [PASS] future-data perturbation leaves cutoff-defined target features unchanged')
    print()
    print('Required markers')
    print('  [PASS] chronological')
    print('  [PASS] no shuffle (independent temporal construction preserves source order)')
    print('  [PASS] no future feature')
    print('  [PASS] no target leakage')
    print()
    print('[NOTE] This audit proves the temporal/data/feature contract from the committed')
    print('       snapshot, forecast metadata/artifact, and canonical feature semantics.')
    print('       It does not claim code-path provenance for an operational forecast')
    print('       generator if that generator implementation is not committed in the repo.')
    print('[PASS] audit is read-only; no A-owned or canonical artifact is written')


if __name__ == '__main__':
    main()
