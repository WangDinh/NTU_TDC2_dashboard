"""End-to-end experiment orchestration.

`prepare_data()` (split → subset → scale) is shared by the notebook and
`run_experiment()`, so there is one implementation of the split/scale logic.
`run_experiment(cfg)` is the one-call path used by the CLI script and dashboard:
build data → prepare → window → train → evaluate → save → plot.

Heavy imports (torch via trainer) are done lazily inside run_experiment so that
`import rack_forecast.pipeline` stays cheap for callers that only want prepare_data.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import build_dataset
from .paths import RESULTS_ROOT
from .windowing import make_supervised

# Field prefix a rack's own PM columns are renamed to when pooling multiple
# racks for shared-model training (see canonicalize_own_rack / run_shared_experiment).
OWN_PREFIX = 'OWN'


# ── prepared-data bundle passed between steps ──────────────────────────────

@dataclass
class PreparedData:
    """Everything downstream steps need after split + scale.

    `y_train`/`y_test` are the SAME arrays as `X_train`/`X_test` (full feature
    vectors, target included) — the model is trained to forecast every column,
    not just the target. `target_idx` says which column is the one scored and
    plotted downstream.
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    months: list
    feature_cols: list
    target_idx: int
    n_feat: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def prepare_data(data, cfg):
    """Split by calendar month, optionally subset days, then StandardScale.

    - Train = all months except the last; Test = the last month.
    - `train_days` / `predict_days` optionally clip each side to the first N days.
    - The target column stays IN the feature set (power is an input like any
      other sensor) so the model can also forecast it forward as a feature
      during the autoregressive rollout, instead of freezing it.
    - One scaler is fit on TRAIN only (no leakage) across all columns —
      inverse-transforming just the target later needs a zero-pad trick
      (see `evaluate._inverse_target`), but this keeps the rolled-forward
      target's scaling consistent with every other feature.
    """
    # ── month split (last month = test) ────────────────────────────────
    months = sorted(data.index.to_period('M').unique())
    train_df = data[data.index.to_period('M').isin(months[:-1])]
    test_df  = data[data.index.to_period('M').isin(months[-1:])]

    # ── optional day subsetting (fast iteration / demo) ────────────────
    if cfg.train_days is not None:
        cutoff = train_df.index.min() + pd.Timedelta(days=cfg.train_days)
        train_df = train_df[train_df.index < cutoff]
    if cfg.predict_days is not None:
        cutoff = test_df.index.min() + pd.Timedelta(days=cfg.predict_days)
        test_df = test_df[test_df.index < cutoff]

    # ── scale (target stays in the feature set) ─────────────────────────
    target_col   = cfg.target_col
    feature_cols = list(data.columns)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols]).astype('float32')
    X_test  = scaler.transform(test_df[feature_cols]).astype('float32')
    y_train, y_test = X_train, X_test          # multi-output target = full feature vector

    target_idx = feature_cols.index(target_col)

    return PreparedData(
        train_df=train_df, test_df=test_df, months=months,
        feature_cols=feature_cols, target_idx=target_idx, n_feat=X_train.shape[1],
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        scaler=scaler,
    )


# ── shared-model training (one model pooled across several racks) ──────────

def canonicalize_own_rack(data, rack):
    """Rename `rack`'s own PM columns ('<rack>__kW' etc.) to rack-agnostic
    'OWN__<field>' names, so multiple racks' data share one column layout —
    needed to pool them into a single shared-model training set. TH sensor
    columns are already rack-agnostic names and are left untouched."""
    rename = {f'{rack}__{f}': f'{OWN_PREFIX}__{f}' for f in ['kW', 'V', 'A', 'PF', 'Hz']}
    return data.rename(columns=rename)


def _prepare_pooled(cfg, racks):
    """Build + canonicalize + month-split each rack, then fit ONE StandardScaler
    on the pooled training rows across all racks (cfg.target_rack is ignored).

    Returns:
        per_rack: {rack: {'train_df', 'test_df'}} (canonical columns, unscaled)
        feature_cols: columns common to every rack (sorted, canonical names)
        scaler, target_idx, n_feat
    """
    per_rack_data = {rack: canonicalize_own_rack(build_dataset(replace(cfg, target_rack=rack)), rack)
                     for rack in racks}

    feature_cols = sorted(set.intersection(*(set(d.columns) for d in per_rack_data.values())))
    target_col = f'{OWN_PREFIX}__kW'
    if target_col not in feature_cols:
        raise ValueError(f'{target_col} missing from one or more racks — cannot pool {racks}.')

    per_rack = {}
    for rack, data in per_rack_data.items():
        data = data[feature_cols]
        months = sorted(data.index.to_period('M').unique())
        train_df = data[data.index.to_period('M').isin(months[:-1])]
        test_df  = data[data.index.to_period('M').isin(months[-1:])]
        if cfg.train_days is not None:
            cutoff = train_df.index.min() + pd.Timedelta(days=cfg.train_days)
            train_df = train_df[train_df.index < cutoff]
        if cfg.predict_days is not None:
            cutoff = test_df.index.min() + pd.Timedelta(days=cfg.predict_days)
            test_df = test_df[test_df.index < cutoff]
        per_rack[rack] = {'train_df': train_df, 'test_df': test_df}

    scaler = StandardScaler()
    pooled_train = pd.concat([v['train_df'] for v in per_rack.values()], axis=0)
    scaler.fit(pooled_train[feature_cols])

    return per_rack, feature_cols, scaler, feature_cols.index(target_col), len(feature_cols)


def _fit_ridge_accumulated(per_rack_windows, alpha=1.0):
    """Fit one Ridge regression shared across all racks by accumulating
    sufficient statistics (sum(X), sum(y), XtX, Xty) rack-by-rack instead of
    pooling every rack's windows into memory at once. Mathematically identical
    to fitting sklearn Ridge(alpha) on every rack's data concatenated together
    — exact, no learning rate to tune, no rack-order dependence (unlike the DL/
    XGBoost continuation paths, which are order-sensitive)."""
    n_total = 0
    sum_X = sum_y = XtX = Xty = None
    for rack, (X_3d, y_sup) in per_rack_windows.items():
        print(f'    accumulating {rack} ({len(X_3d):,} windows)...')
        X_flat = X_3d.reshape(len(X_3d), -1).astype(np.float64)
        y = y_sup.astype(np.float64)
        if XtX is None:
            d, k = X_flat.shape[1], y.shape[1]
            sum_X, sum_y = np.zeros(d), np.zeros(k)
            XtX, Xty = np.zeros((d, d)), np.zeros((d, k))
        n_total += len(X_flat)
        sum_X += X_flat.sum(axis=0)
        sum_y += y.sum(axis=0)
        XtX += X_flat.T @ X_flat
        Xty += X_flat.T @ y

    mean_X, mean_y = sum_X / n_total, sum_y / n_total
    Sxx = XtX - n_total * np.outer(mean_X, mean_X)
    Sxy = Xty - n_total * np.outer(mean_X, mean_y)
    beta = np.linalg.solve(Sxx + alpha * np.eye(Sxx.shape[0]), Sxy)   # (d, k)
    intercept = mean_y - mean_X @ beta

    from .models.linear import LinearModel
    m = LinearModel(alpha=alpha)
    m.model.coef_ = beta.T.astype(np.float32)          # sklearn Ridge: (n_targets, n_features)
    m.model.intercept_ = intercept.astype(np.float32)
    m.model.n_features_in_ = beta.shape[0]
    return m


def _train_rack_by_rack(name, per_rack_windows, lookback, n_feat, dl_epochs):
    """Fine-tune ONE shared model sequentially across racks — train on the
    first rack's windows, then continue training the SAME model (weights /
    booster carried over) on the next rack's windows, and so on.

    Only one rack's windowed data is ever materialized at a time, so memory
    doesn't scale with rack count the way pooling all racks at once would.

    Args:
        per_rack_windows: {rack: (X_3d, y_sup)} in the order to fine-tune on.
    """
    from .trainer import DL_MODELS, train_dl
    from .models.lstm import LSTMModel
    from .models.cnn1d import CNN1DModel
    from .models.transformer import TransformerModel
    from .models import REGISTRY

    if name in DL_MODELS:
        model = {'lstm': LSTMModel(n_feat), 'cnn1d': CNN1DModel(n_feat, lookback),
                 'transformer': TransformerModel(n_feat)}[name]
        for rack, (X_3d, y_sup) in per_rack_windows.items():
            print(f'    fine-tuning on {rack} ({len(X_3d):,} windows)...')
            model = train_dl(model, X_3d, y_sup, epochs=dl_epochs)
        return model

    if name == 'xgboost':
        m = REGISTRY[name]()
        booster = None
        for rack, (X_3d, y_sup) in per_rack_windows.items():
            print(f'    fine-tuning on {rack} ({len(X_3d):,} windows)...')
            m.fit(X_3d.reshape(len(X_3d), -1), y_sup, xgb_model=booster)
            booster = m.model.get_booster()
        return m

    if name == 'linear':
        return _fit_ridge_accumulated(per_rack_windows)

    raise ValueError(f'Rack-by-rack fine-tuning not supported for model {name!r} '
                      f'(only linear, xgboost, lstm, cnn1d, transformer).')


def run_shared_experiment(cfg, racks, save=True, make_plots=True):
    """Train ONE shared model per requested type, fine-tuned rack-by-rack
    across `racks` (see `_train_rack_by_rack`).

    `cfg.target_rack` is ignored — every rack in `racks` is built and
    canonicalized (see `canonicalize_own_rack`) so they all share one
    'OWN__<field>' column layout. Every other cfg field (lookback, horizon,
    models, dl_epochs, fast_mode, resample, train_days, predict_days, run_id)
    applies as usual.

    Evaluation is POOLED: all racks' test windows are scored together as one
    metrics table — there is no per-rack breakdown, matching how the shared
    model is meant to generalize across racks rather than specialize to one.

    Returns:
        (results, metrics_df, prep_info)
        results : {name: {'preds', 'actuals', 'metrics', 'window_racks', 'window_times'}}
        metrics_df : per-model pooled metrics table (rounded)
        prep_info : dict with racks/feature_cols/target_idx/n_feat/scaler/per_rack
    """
    from .evaluate import evaluate, compute_metrics

    print(f'Building dataset for {len(racks)} racks: {racks}')
    per_rack, feature_cols, scaler, target_idx, n_feat = _prepare_pooled(cfg, racks)

    per_rack_windows, per_rack_test = {}, {}
    for rack, d in per_rack.items():
        X_train = scaler.transform(d['train_df'][feature_cols]).astype('float32')
        X_test  = scaler.transform(d['test_df'][feature_cols]).astype('float32')
        per_rack_windows[rack] = make_supervised(X_train, X_train, cfg.lookback)
        per_rack_test[rack] = X_test

    trained, results = {}, {}
    for name in cfg.models:
        print(f'[{name}] fine-tuning rack-by-rack across {len(racks)} racks...')
        model = _train_rack_by_rack(name, per_rack_windows, cfg.lookback, n_feat, cfg.dl_epochs)

        all_preds, all_actuals, window_racks, window_times = [], [], [], []
        for rack, X_test in per_rack_test.items():
            preds, actuals = evaluate(model, X_test, X_test, cfg.lookback, cfg.horizon,
                                      target_idx, scaler, n_feat)
            n_windows = len(preds) // cfg.horizon
            stop = cfg.lookback + n_windows * cfg.horizon
            test_df = per_rack[rack]['test_df']
            all_preds.append(preds)
            all_actuals.append(actuals)
            window_racks.extend([rack] * n_windows)
            window_times.extend(test_df.index[cfg.lookback:stop:cfg.horizon])

        preds   = np.concatenate(all_preds)
        actuals = np.concatenate(all_actuals)
        metrics = compute_metrics(actuals, preds)
        trained[name] = model
        results[name] = {'preds': preds, 'actuals': actuals, 'metrics': metrics,
                         'window_racks': np.array(window_racks),
                         'window_times': pd.DatetimeIndex(window_times)}
        print(f'  MAE={metrics["MAE"]:.4f}  RMSE={metrics["RMSE"]:.4f}  R2={metrics["R2"]:.4f}')

    metrics_df = pd.DataFrame({n: r['metrics'] for n, r in results.items()}).T.round(4)

    prep_info = {'racks': racks, 'feature_cols': feature_cols, 'target_idx': target_idx,
                'n_feat': n_feat, 'scaler': scaler, 'per_rack': per_rack}

    if save:
        _save_shared_run(cfg, racks, prep_info, trained, results, metrics_df)

    return results, metrics_df, prep_info


def _save_shared_run(cfg, racks, prep_info, trained, results, metrics_df):
    """Write a shared-model run's artifacts (config, metrics, scaler, weights,
    predictions) to its own results/ folder, tagged 'shared_<phase>'."""
    from . import persistence as io

    phase = racks[0].rsplit('-', 1)[1]
    run_name = f'{cfg.run_id}_shared_{phase}_L{cfg.lookback}_H{cfg.horizon}'
    run_folder = RESULTS_ROOT / run_name
    run_folder.mkdir(parents=True, exist_ok=True)

    io.save_config(cfg, run_folder, extra={
        'shared_model': True,
        'racks': racks,
        'target_rack': racks[0],           # display-only; no single target rack
        'n_features':  prep_info['n_feat'],
        'target_col':  f'{OWN_PREFIX}__kW',
        'feature_cols': prep_info['feature_cols'],
    })
    io.save_metrics(metrics_df, run_folder)
    io.save_scalers(run_folder, prep_info['scaler'])

    for name, model in trained.items():
        io.save_model(name, model, run_folder / name)
        io.save_predictions(run_folder, name, results[name]['preds'], results[name]['actuals'],
                            results[name]['window_times'], cfg.horizon,
                            window_racks=results[name]['window_racks'])

    print(f'\nDone. Shared-model artifacts in {run_folder}')


# ── full run ───────────────────────────────────────────────────────────────

def run_experiment(cfg, data=None, save=True, make_plots=True):
    """Train + evaluate every model in `cfg.models`, save artifacts, return results.

    Returns:
        (results, metrics_df, prep)
        results : {name: {'preds', 'actuals', 'metrics'}}
        metrics_df : per-model metrics table (rounded)
        prep : the PreparedData bundle (handy for further inspection)
    """
    from .trainer import build_and_train                 # lazy: pulls in torch
    from .evaluate import evaluate, compute_metrics

    # ── data ───────────────────────────────────────────────────────────
    if data is None:
        print('Building dataset...')
        data = build_dataset(cfg)
    print(f'Dataset: {data.shape}  |  {data.index.min()} → {data.index.max()}')

    prep = prepare_data(data, cfg)
    print(f'Train: {len(prep.X_train):,}  Test: {len(prep.X_test):,}  '
          f'Features: {prep.n_feat}')

    X_3d, y_sup = make_supervised(prep.X_train, prep.y_train, cfg.lookback)

    # ── train + evaluate each model ────────────────────────────────────
    trained, results = {}, {}
    for name in cfg.models:
        print(f'[{name}] training...')
        model = build_and_train(name, X_3d, y_sup, cfg.lookback, prep.n_feat,
                                dl_epochs=cfg.dl_epochs)
        preds, actuals = evaluate(model, prep.X_test, prep.y_test,
                                  cfg.lookback, cfg.horizon,
                                  prep.target_idx, prep.scaler, prep.n_feat)
        metrics = compute_metrics(actuals, preds)
        trained[name] = model
        results[name] = {'preds': preds, 'actuals': actuals, 'metrics': metrics}
        print(f'  MAE={metrics["MAE"]:.4f}  RMSE={metrics["RMSE"]:.4f}  R2={metrics["R2"]:.4f}')

    metrics_df = pd.DataFrame({n: r['metrics'] for n, r in results.items()}).T.round(4)

    if save:
        save_results(cfg, prep, trained, results, metrics_df, make_plots=make_plots)

    print(f'\nDone. Artifacts in {cfg.run_folder}')
    return results, metrics_df, prep


# ── artifact writers ───────────────────────────────────────────────────────

def save_results(cfg, prep, trained, results, metrics_df, make_plots=True):
    """Write all run artifacts. Callable directly from a notebook after training
    models by hand (so the notebook never re-implements the save logic)."""
    _save_run(cfg, prep, trained, results, metrics_df)
    if make_plots:
        _save_figures(cfg, prep, results, metrics_df)

def _window_start_times(prep, cfg):
    """One timestamp per evaluation window (the window's first target step)."""
    n_windows = (len(prep.X_test) - cfg.lookback) // cfg.horizon
    stop = cfg.lookback + n_windows * cfg.horizon
    return prep.test_df.index[cfg.lookback:stop:cfg.horizon]


def _save_run(cfg, prep, trained, results, metrics_df):
    """Write config, metrics, scalers, per-model weights, and predictions."""
    from . import persistence as io

    run_folder = cfg.run_folder
    run_folder.mkdir(parents=True, exist_ok=True)

    io.save_config(cfg, run_folder, extra={
        'n_features':   prep.n_feat,
        'target_col':   cfg.target_col,
        'train_months': f'{prep.months[0]} → {prep.months[-2]}',
        'test_month':   str(prep.months[-1]),
        'feature_cols': prep.feature_cols,
    })
    io.save_metrics(metrics_df, run_folder)
    io.save_scalers(run_folder, prep.scaler)

    window_times = _window_start_times(prep, cfg)
    for name, model in trained.items():
        io.save_model(name, model, run_folder / name)
        io.save_predictions(run_folder, name, results[name]['preds'],
                            results[name]['actuals'], window_times, cfg.horizon)


def _save_figures(cfg, prep, results, metrics_df):
    """Render and save run-level and per-model PNGs."""
    import matplotlib.pyplot as plt
    from . import plots

    run_folder = cfg.run_folder
    plots.plot_metrics_bar(metrics_df, cfg).savefig(
        run_folder / 'metrics_bar.png', bbox_inches='tight')
    plots.plot_all_models_vs_actual(results, prep.test_df.index, cfg).savefig(
        run_folder / 'all_models_vs_actual.png', bbox_inches='tight')
    for name in results:
        (run_folder / name).mkdir(parents=True, exist_ok=True)
        plots.plot_actual_vs_pred(name, results, prep.test_df.index, cfg).savefig(
            run_folder / name / 'actual_vs_pred_1window.png', bbox_inches='tight')
        plots.plot_residuals(name, results).savefig(
            run_folder / name / 'residuals.png', bbox_inches='tight')
    plt.close('all')
