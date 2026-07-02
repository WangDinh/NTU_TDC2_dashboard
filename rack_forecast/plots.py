"""Matplotlib figures for a run.

Each function RETURNS a Figure (it does not save or show). That way the pipeline
can save the figure to PNG while a notebook can display the very same object —
no duplicated plotting code between the two.

`results` is the dict produced by the pipeline: {name: {'preds', 'actuals', 'metrics'}}
with preds/actuals as 1-D arrays laid out window-by-window.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110


def plot_metrics_bar(metrics_df, cfg):
    """Side-by-side MAE and RMSE bar charts across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ['MAE', 'RMSE']):
        metrics_df[col].sort_values().plot.barh(
            ax=ax, color='steelblue', edgecolor='k', linewidth=0.4)
        ax.set_title(col)
    fig.suptitle(f'{cfg.target_rack} — Model Comparison  '
                 f'(H={cfg.horizon} steps = {cfg.horizon // 2} min)')
    fig.tight_layout()
    return fig


def plot_all_models_vs_actual(results, test_index, cfg):
    """Overlay every model's first-window prediction against the actual curve."""
    steps = cfg.horizon
    times = test_index[cfg.lookback: cfg.lookback + steps]
    first = next(iter(results))
    actual = results[first]['actuals'][:steps]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times[:len(actual)], actual, color='black', lw=1.6, label='Actual', zorder=5)
    for name in results:
        p = results[name]['preds'][:steps]
        ax.plot(times[:len(p)], p, lw=0.9, label=name, alpha=0.85)
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'{cfg.target_rack} — All Models vs Actual '
                 f'(1 window = {cfg.horizon} steps = {cfg.horizon // 2} min)')
    ax.legend(ncol=4, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def plot_actual_vs_pred(name, results, test_index, cfg):
    """One model's first-window prediction vs the actual curve."""
    steps = cfg.horizon
    times = test_index[cfg.lookback: cfg.lookback + steps]
    actual = results[name]['actuals'][:steps]
    pred = results[name]['preds'][:steps]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times[:len(actual)], actual, color='black', lw=1.4, label='Actual', zorder=5)
    ax.plot(times[:len(pred)], pred, lw=0.9, color='steelblue', label=name, alpha=0.9)
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'{cfg.target_rack} [{name}] — Actual vs Predicted '
                 f'(1 window = {cfg.horizon} steps = {cfg.horizon // 2} min)')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d %H:%M'))
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def plot_residuals(name, results):
    """Residuals over time + their distribution for one model."""
    residuals = results[name]['actuals'] - results[name]['preds']
    m = results[name]['metrics']

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(residuals, lw=0.7, color='tomato')
    axes[0].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[0].set_title(f'Residuals over time — {name}')
    axes[0].set_ylabel('Error (kW)')
    axes[1].hist(residuals, bins=60, edgecolor='k', linewidth=0.3, color='steelblue')
    axes[1].set_title('Residual distribution')
    axes[1].set_xlabel('Error (kW)')
    fig.suptitle(f'[{name}]  MAE={m["MAE"]:.4f}  RMSE={m["RMSE"]:.4f}  R2={m["R2"]:.4f}')
    fig.tight_layout()
    return fig
