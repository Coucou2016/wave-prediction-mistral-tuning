from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wave_llm.evaluation.metrics_numeric import mae, rmse, skill_score  # noqa: E402
from wave_llm.models.baseline_lightgbm import train_predict_hs_residual  # noqa: E402
from wave_llm.models.baseline_persistence import add_persistence_columns  # noqa: E402
from wave_llm.physics.baseline import climatology_baseline_hourly  # noqa: E402
from wave_llm.physics.residual import residual_vs_baseline  # noqa: E402
from wave_llm.physics.wave_features import add_lag_features  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    leads = [int(x) for x in mcfg.get("target_lead_hours", [24])]
    df = pd.read_parquet(ROOT / "data" / "processed" / "panel_hourly.parquet")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = add_lag_features(df)
    df["Hs_clim"] = climatology_baseline_hourly(df)
    df["r_clim"] = residual_vs_baseline(df["Hs_m"], df["Hs_clim"])

    t_cut = df["time_utc"].quantile(0.8)
    tr = df[df["time_utc"] < t_cut].copy()
    te = df[df["time_utc"] >= t_cut].copy()

    feat = [c for c in df.columns if c.startswith("Hs_lag_") or c.startswith("Tp_lag_")]
    if not feat:
        feat = ["Hs_lag_24h"]

    results = []
    pred_frames: list[pd.DataFrame] = []
    out_dir = ensure_dir(ROOT / "data" / "processed" / "metrics")
    pred_dir = ensure_dir(ROOT / "data" / "processed" / "predictions")
    for L in leads:
        trp = add_persistence_columns(tr, L)
        tep = add_persistence_columns(te, L)
        m = tep["y_target_hs"].notna() & tep["y_pred_persist_hs"].notna()
        r_p = rmse(tep.loc[m, "y_target_hs"].to_numpy(), tep.loc[m, "y_pred_persist_hs"].to_numpy())
        a_p = mae(tep.loc[m, "y_target_hs"].to_numpy(), tep.loc[m, "y_pred_persist_hs"].to_numpy())

        tr_l = trp.dropna(subset=["y_target_hs"] + feat)
        te_l = tep.dropna(subset=["y_target_hs"] + feat)
        if te_l.empty or tr_l.empty:
            logging.warning("Empty train/test for lead=%s — skip", L)
            continue
        _, pred = train_predict_hs_residual(tr_l, te_l, feat, "y_target_hs")
        y_true = te_l["y_target_hs"].to_numpy()
        y_hat = np.asarray(pred, dtype=float)
        r_l = rmse(y_true, y_hat)
        sk = skill_score(r_l, r_p)
        results.append({"lead_h": L, "rmse_persist": r_p, "mae_persist": a_p, "rmse_lgbm": r_l, "skill_vs_persist": sk})
        logging.info("lead=%sh persist_rmse=%.4f lgbm_rmse=%.4f skill=%.4f", L, r_p, r_l, sk)

        te_out = te_l[["time_utc", "station_id", "y_target_hs", "y_pred_persist_hs"]].copy()
        te_out = te_out.rename(columns={"y_target_hs": "y_true", "y_pred_persist_hs": "y_pred_persist"})
        te_out["y_pred_lgbm"] = y_hat
        te_out["lead_h"] = int(L)
        pred_frames.append(te_out)

    with (out_dir / "numeric_baselines.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if pred_frames:
        pred_all = pd.concat(pred_frames, ignore_index=True)
        pred_path = pred_dir / "test_predictions_lgbm.parquet"
        pred_all.to_parquet(pred_path, index=False)
        logging.info("Wrote %s rows=%s", pred_path, len(pred_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
