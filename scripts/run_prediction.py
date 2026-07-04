"""CLI entry point: edit the config below, then run

    conda run -n ntu_cooling python scripts/run_prediction.py

Set TARGET to a single rack (e.g. 'R0605-PA') to run one experiment, or to a
phase ('PA' / 'PB') to sweep every rack on that side. With SHARE_MODEL=False
(default) each rack in the sweep gets its own independently trained model and
its own results/ folder. With SHARE_MODEL=True, one model per requested type
is FINE-TUNED rack-by-rack across every rack in the phase (trained on rack 1,
then continues training the same weights on rack 2, and so on — only one
rack's data is ever in memory at a time), evaluated pooled across all racks,
and saved to a single results/ folder — SHARE_MODEL is ignored for a
single-rack TARGET (sharing needs more than one rack).

Everything heavy lives in the `rack_forecast` package; this file only builds
an ExperimentConfig (or configs) and calls run_experiment()/run_shared_
experiment(). Uses the non-interactive matplotlib backend so it works
headless.
"""

from rack_forecast.paths import DATA_ROOT
from rack_forecast.pipeline import run_experiment, run_shared_experiment
from rack_forecast import ExperimentConfig
import matplotlib
matplotlib.use('Agg')


# ── edit me ────────────────────────────────────────────────────────────────
TARGET = 'R0605-PA'                # a single rack, or 'PA'/'PB' to sweep that side

SHARE_MODEL = False
# SHARE_MODEL setting for TARGET='PA'/'PB' (ignored for a single rack TARGET)
# True: one model pooled across the swept racks / False: independent models per rack

LOOKBACK = 60
HORIZON = 30
MODELS = ['linear', 'xgboost', 'lstm', 'cnn1d', 'transformer']
# e.g. ['linear','rf','xgboost','lstm','cnn1d','transformer']

FAST_MODE = True
# True = only loads the target rack's own PM/TH sensor columns
# False = loads all racks' PM/TH sensor columns and SensorGW columns (182 features, process the whole facility)

TRAIN_DAYS = None                  # None = full 5 training months
PREDICT_DAYS = None                   # None = full test month
# names the output folder (rack is appended automatically)
RUN_ID = 'run_03'
# ───────────────────────────────────────────────────────────────────────────


def racks_for(target):
    """A single rack name as-is, or every rack ending in '-PA'/'-PB' for a phase."""
    if target in ('PA', 'PB'):
        return sorted(d.name for d in DATA_ROOT.iterdir()
                      if d.is_dir() and d.name.endswith(f'-{target}'))
    return [target]


if __name__ == '__main__':
    racks = racks_for(TARGET)

    if len(racks) > 1 and SHARE_MODEL:
        print(f'Training one shared model across {len(racks)} racks: {racks}')
        cfg = ExperimentConfig(
            lookback=LOOKBACK, horizon=HORIZON, models=MODELS,
            fast_mode=FAST_MODE, train_days=TRAIN_DAYS, predict_days=PREDICT_DAYS,
            run_id=RUN_ID,
        )
        _, metrics_df, _ = run_shared_experiment(cfg, racks)
        print(metrics_df)
    else:
        if len(racks) > 1:
            print(f'Sweeping {len(racks)} racks (independent models): {racks}')
        for rack in racks:
            cfg = ExperimentConfig(
                target_rack=rack,
                lookback=LOOKBACK,
                horizon=HORIZON,
                models=MODELS,
                fast_mode=FAST_MODE,
                train_days=TRAIN_DAYS,
                predict_days=PREDICT_DAYS,
                run_id=RUN_ID,
            )
            print(f'\n=== {rack} ===')
            _, metrics_df, _ = run_experiment(cfg)
            print(metrics_df)
