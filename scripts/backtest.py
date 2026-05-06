"""
scripts/backtest.py — Comprehensive backtesting engine.

Measures:
  NHL: accuracy by confidence tier, Brier score, calibration,
       elite ROI at various odds, AUC, precision/recall
  MLB: MAE on H/HR/RBI/R/TB/HRR by tier, HR direction accuracy,
       contact prediction accuracy, calibration of HR prob
  NBA: MAE on pts/reb/ast by tier, DD accuracy, over/under accuracy,
       calibration of threshold predictions

Results saved to data/cache/predictions/backtest_results.json.
"""

import argparse, json, os, sys, requests, time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
for _path in [_ROOT, _ROOT/"config", _ROOT/"core"/"pipelines",
              _ROOT/"data"/"api"]:
    if str(_path) not in sys.path: sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd

ET       = ZoneInfo("America/New_York")
PRED_DIR = _ROOT / "data" / "cache" / "predictions"
HIST_DIR = PRED_DIR / "history"
OUT_FILE = PRED_DIR / "backtest_results.json"
HEADERS  = {"User-Agent": "Mozilla/5.0 (compatible; SportsPredictorBot/1.0)"}


def log(msg): print(f"[{datetime.now(ET).strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Calibration metrics ───────────────────────────────────────────────────────

def expected_calibration_error(probs, labels, n_bins=10):
    """
    ECE: measures how well predicted probabilities match actual frequencies.
    0.0 = perfect calibration. 0.10 = systematically off by 10%.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    total_error = 0; n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        avg_conf = float(probs[mask].mean())
        avg_acc  = float(labels[mask].mean())
        total_error += abs(avg_conf - avg_acc) * mask.sum()
    return total_error / max(n, 1)


def brier_score(probs, labels):
    return float(np.mean((probs - labels.astype(float))**2))


def auc_score(probs, labels):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs))
    except Exception:
        return None


def roi_estimate(probs, labels, odds_american=150):
    """
    Simulated ROI betting $1 on every pick at given American odds.
    +150 = win $1.50 per $1 bet on a win.
    """
    odds_decimal = (odds_american / 100) + 1 if odds_american > 0 else (100 / abs(odds_american)) + 1
    wins   = int(labels.sum())
    total  = len(labels)
    profit = wins * (odds_decimal - 1) - (total - wins)
    return profit / total if total > 0 else 0.0


def tier_stats(df, tier_col="conf", prob_col="pred_prob", label_col="scored"):
    """Compute per-tier accuracy and calibration stats."""
    results = {}
    for tier in ["Elite","High","Medium","Low"]:
        sub = df[df[tier_col] == tier]
        if len(sub) < 5: continue
        probs  = sub[prob_col].values.astype(float)
        labels = sub[label_col].values.astype(int)
        results[tier] = {
            "n":               int(len(sub)),
            "accuracy":        round(float(labels.mean()), 4),
            "avg_pred_prob":   round(float(probs.mean()),  4),
            "ece":             round(expected_calibration_error(probs, labels), 4),
            "brier":           round(brier_score(probs, labels), 4),
            "auc":             round(auc_score(probs, labels) or 0, 4),
            "roi_plus150":     round(roi_estimate(labels, labels, 150), 4),
            "roi_plus120":     round(roi_estimate(labels, labels, 120), 4),
        }
    return results


# ── NHL Backtest ──────────────────────────────────────────────────────────────

def backtest_nhl(days: int) -> dict:
    log("NHL backtest…")
    hist_files = sorted(HIST_DIR.glob("nhl_*.parquet")) if HIST_DIR.exists() else []
    if not hist_files:
        return {"sport":"nhl","message":"No history yet — accumulates daily.","rows":[]}

    try:
        from nhl_api import _get
    except ImportError:
        return {"sport":"nhl","error":"NHL API not importable","rows":[]}

    rows = []
    for pred_file in hist_files[-days:]:
        date_str = pred_file.stem.replace("nhl_", "")
        try:
            preds = pd.read_parquet(pred_file)
            sched = _get(f"/schedule/{date_str}")
            if not sched: continue
            for game in sched.get("games", []):
                gid = game.get("id")
                if not gid: continue
                try:
                    bs = _get(f"/gamecenter/{gid}/boxscore")
                    if not bs: continue
                    for side in ("homeTeam", "awayTeam"):
                        for player in bs.get(side, {}).get("players", []):
                            pid   = player.get("playerId")
                            goals = int(player.get("goals", 0))
                            shots = int(player.get("shots", 0))
                            if pid is None: continue
                            match = preds[preds["player_id"] == pid]
                            if match.empty: continue
                            r = match.iloc[0]
                            rows.append({
                                "date":       date_str,
                                "player_id":  pid,
                                "name":       r.get("player_name",""),
                                "team":       r.get("team",""),
                                "conf":       str(r.get("confidence","Low")),
                                "pred_prob":  float(r.get("goal_probability",0)),
                                "scored":     int(goals > 0),
                                "goals":      goals,
                                "actual_sog": shots,
                                "pred_sog":   float(r.get("projected_sog",0)),
                            })
                    time.sleep(0.1)
                except Exception as e:
                    log(f"  NHL game {gid}: {e}")
        except Exception as e:
            log(f"  NHL {date_str}: {e}")

    return {"sport":"nhl","rows":rows}


def aggregate_nhl(rows: list) -> dict:
    if not rows:
        return {}
    df  = pd.DataFrame(rows)
    out = {
        "sport":    "nhl",
        "n_games":  int(df["date"].nunique()),
        "n_rows":   int(len(df)),
    }
    # Overall metrics
    probs  = df["pred_prob"].values.astype(float)
    labels = df["scored"].values.astype(int)
    out["overall_accuracy"]   = round(float(labels.mean()), 4)
    out["overall_brier"]      = round(brier_score(probs, labels), 4)
    out["overall_ece"]        = round(expected_calibration_error(probs, labels), 4)
    out["overall_auc"]        = round(auc_score(probs, labels) or 0, 4)

    # Per-tier
    out["tiers"] = tier_stats(df)

    # SOG accuracy
    if "actual_sog" in df.columns and "pred_sog" in df.columns:
        out["sog_mae"] = round(float((df["pred_sog"] - df["actual_sog"]).abs().mean()), 3)

    # Elite ROI at multiple odds lines
    elite = df[df["conf"] == "Elite"]
    if len(elite) >= 10:
        el_labels = elite["scored"].values
        out["elite_roi_plus150"] = round(roi_estimate(el_labels, el_labels, 150), 4)
        out["elite_roi_plus130"] = round(roi_estimate(el_labels, el_labels, 130), 4)
        out["elite_roi_plus110"] = round(roi_estimate(el_labels, el_labels, 110), 4)
        out["elite_n"]           = int(len(elite))
        out["elite_accuracy"]    = round(float(el_labels.mean()), 4)

    # Calibration curve (10 bins)
    cal_bins = []
    for lo in np.arange(0, 1, 0.1):
        mask = (probs >= lo) & (probs < lo + 0.1)
        if mask.sum() >= 5:
            cal_bins.append({
                "bin_lo":     round(float(lo), 1),
                "bin_hi":     round(float(lo + 0.1), 1),
                "n":          int(mask.sum()),
                "avg_pred":   round(float(probs[mask].mean()), 3),
                "avg_actual": round(float(labels[mask].mean()), 3),
            })
    out["calibration_curve"] = cal_bins

    return out


# ── MLB Backtest ──────────────────────────────────────────────────────────────

def backtest_mlb(days: int) -> dict:
    log("MLB backtest…")
    hist_files = sorted(HIST_DIR.glob("mlb_*.parquet")) if HIST_DIR.exists() else []
    if not hist_files:
        return {"sport":"mlb","message":"No history yet.","rows":[]}

    MLB_API = "https://statsapi.mlb.com/api/v1"
    rows = []
    for pred_file in hist_files[-days:]:
        date_str = pred_file.stem.replace("mlb_", "")
        try:
            preds = pd.read_parquet(pred_file)
            r = requests.get(f"{MLB_API}/schedule",
                params={"sportId":1,"date":date_str,"hydrate":"boxscore"},
                headers=HEADERS, timeout=15)
            if r.status_code != 200: continue
            for date_data in r.json().get("dates",[]):
                for game in date_data.get("games",[]):
                    box = game.get("teams",{})
                    for side in ("away","home"):
                        for pid_str, pdata in box.get(side,{}).get("players",{}).items():
                            pid  = pdata.get("person",{}).get("id")
                            stat = pdata.get("stats",{}).get("batting",{})
                            h    = int(stat.get("hits",0))
                            hr   = int(stat.get("homeRuns",0))
                            rbi  = int(stat.get("rbi",0))
                            r2   = int(stat.get("runs",0))
                            k    = int(stat.get("strikeOuts",0))
                            tb   = int(stat.get("totalBases",0))
                            bb   = int(stat.get("baseOnBalls",0))
                            ab   = int(stat.get("atBats",0))
                            if ab == 0: continue
                            match = preds[preds["player_id"] == pid]
                            if match.empty: continue
                            pr = match.iloc[0]
                            rows.append({
                                "date":        date_str,
                                "player_id":   pid,
                                "name":        pr.get("player_name",""),
                                "team":        pr.get("team",""),
                                "conf":        str(pr.get("confidence","Low")),
                                "pred_h":      float(pr.get("proj_hits",0)),
                                "pred_hr":     float(pr.get("proj_hr",0)),
                                "pred_rbi":    float(pr.get("proj_rbi",0)),
                                "pred_runs":   float(pr.get("proj_runs",0)),
                                "pred_tb":     float(pr.get("proj_tb",0)),
                                "pred_k":      float(pr.get("proj_k",0)),
                                "pred_hrr":    float(pr.get("proj_hrr",0)),
                                "actual_h":    h,
                                "actual_hr":   hr,
                                "actual_rbi":  rbi,
                                "actual_runs": r2,
                                "actual_tb":   tb,
                                "actual_k":    k,
                                "actual_bb":   bb,
                                "actual_hrr":  h + rbi + r2,
                                # HR probability calibration
                                "hr_prob_pred": float(pr.get("proj_hr",0)),
                                "hr_scored":    int(hr > 0),
                            })
            time.sleep(0.25)
        except Exception as e:
            log(f"  MLB {date_str}: {e}")

    return {"sport":"mlb","rows":rows}


def aggregate_mlb(rows: list) -> dict:
    if not rows:
        return {}
    df  = pd.DataFrame(rows)
    out = {
        "sport":   "mlb",
        "n_games": int(df["date"].nunique()),
        "n_rows":  int(len(df)),
    }

    # Overall stat MAE
    for stat, pred_col, actual_col in [
        ("hits",    "pred_h",    "actual_h"),
        ("hr",      "pred_hr",   "actual_hr"),
        ("rbi",     "pred_rbi",  "actual_rbi"),
        ("runs",    "pred_runs", "actual_runs"),
        ("tb",      "pred_tb",   "actual_tb"),
        ("k",       "pred_k",    "actual_k"),
        ("hrr",     "pred_hrr",  "actual_hrr"),
    ]:
        if pred_col in df.columns and actual_col in df.columns:
            mae = float((df[pred_col] - df[actual_col]).abs().mean())
            # RMSE for additional context
            rmse = float(np.sqrt(((df[pred_col] - df[actual_col])**2).mean()))
            # Bias (positive = over-predicting)
            bias = float((df[pred_col] - df[actual_col]).mean())
            out[f"mae_{stat}"]  = round(mae, 3)
            out[f"rmse_{stat}"] = round(rmse, 3)
            out[f"bias_{stat}"] = round(bias, 3)

    # HR probability calibration
    if "hr_prob_pred" in df.columns and "hr_scored" in df.columns:
        hr_probs  = df["hr_prob_pred"].values.astype(float)
        hr_labels = df["hr_scored"].values.astype(int)
        out["hr_brier"] = round(brier_score(hr_probs, hr_labels), 4)
        out["hr_ece"]   = round(expected_calibration_error(hr_probs, hr_labels), 4)
        out["hr_auc"]   = round(auc_score(hr_probs, hr_labels) or 0, 4)
        # Direction accuracy: did we correctly predict HR vs no HR?
        threshold = 0.08
        out["hr_direction_acc"] = round(
            float(((hr_probs >= threshold) == (hr_labels == 1)).mean()), 4)

    # Per-tier breakdown
    tiers = {}
    for tier in ["Elite","High","Medium","Low"]:
        sub = df[df["conf"] == tier]
        if len(sub) < 5: continue
        tier_stats = {"n": int(len(sub))}
        for stat, pred_col, actual_col in [
            ("h",   "pred_h",   "actual_h"),
            ("hr",  "pred_hr",  "actual_hr"),
            ("hrr", "pred_hrr", "actual_hrr"),
        ]:
            if pred_col in sub.columns and actual_col in sub.columns:
                tier_stats[f"mae_{stat}"]       = round(float((sub[pred_col]-sub[actual_col]).abs().mean()),3)
                tier_stats[f"avg_pred_{stat}"]  = round(float(sub[pred_col].mean()),3)
                tier_stats[f"avg_actual_{stat}"]= round(float(sub[actual_col].mean()),3)
                tier_stats[f"bias_{stat}"]      = round(float((sub[pred_col]-sub[actual_col]).mean()),3)
        tiers[tier] = tier_stats
    out["tiers"] = tiers

    # Hit calibration curve
    cal_bins = []
    for lo in [0, 0.25, 0.50, 0.75]:
        mask = (df["pred_h"] >= lo) & (df["pred_h"] < lo + 0.25)
        if mask.sum() >= 5:
            cal_bins.append({
                "bin":         f"{lo:.2f}-{lo+0.25:.2f}",
                "n":           int(mask.sum()),
                "avg_pred_h":  round(float(df[mask]["pred_h"].mean()),3),
                "avg_actual_h":round(float(df[mask]["actual_h"].mean()),3),
            })
    out["hit_calibration_curve"] = cal_bins

    return out


# ── NBA Backtest ──────────────────────────────────────────────────────────────

def backtest_nba(days: int) -> dict:
    log("NBA backtest…")
    hist_files = sorted(HIST_DIR.glob("nba_*.parquet")) if HIST_DIR.exists() else []
    if not hist_files:
        return {"sport":"nba","message":"No history yet.","rows":[]}

    try:
        from nba_client import _get as nba_get, ESPN_BASE
    except ImportError:
        return {"sport":"nba","error":"NBA client not importable","rows":[]}

    rows = []
    for pred_file in hist_files[-days:]:
        date_str = pred_file.stem.replace("nba_", "")
        try:
            preds = pd.read_parquet(pred_file)
            # ESPN scoreboard for that date
            url  = f"{ESPN_BASE}/scoreboard?dates={date_str.replace('-','')}"
            resp = nba_get(url)
            if not resp: continue
            for event in resp.get("events",[]):
                for comp in event.get("competitions",[]):
                    for competitor in comp.get("competitors",[]):
                        for roster_player in competitor.get("roster",[]):
                            stats  = {s["name"]: s.get("value",0)
                                      for s in roster_player.get("stats",[])}
                            pid    = roster_player.get("athlete",{}).get("id")
                            if not pid: continue
                            pts    = int(float(stats.get("points",0)))
                            reb    = int(float(stats.get("rebounds",0)))
                            ast_   = int(float(stats.get("assists",0)))
                            fg3m   = int(float(stats.get("threePointFieldGoalsMade",0)))
                            stl    = int(float(stats.get("steals",0)))
                            blk    = int(float(stats.get("blocks",0)))
                            min_   = float(stats.get("minutes",0))
                            if min_ < 5: continue
                            match  = preds[preds["player_id"]==int(pid)]
                            if match.empty: continue
                            pr     = match.iloc[0]
                            rows.append({
                                "date":        date_str,
                                "player_id":   int(pid),
                                "name":        pr.get("player_name",""),
                                "team":        pr.get("team",""),
                                "conf":        str(pr.get("confidence","Low")),
                                "pred_pts":    float(pr.get("proj_pts",0)),
                                "pred_reb":    float(pr.get("proj_reb",0)),
                                "pred_ast":    float(pr.get("proj_ast",0)),
                                "pred_fg3m":   float(pr.get("proj_fg3m",0)),
                                "pred_stocks": float(pr.get("proj_stocks",0)),
                                "pred_dd":     float(pr.get("proj_dd",0)),
                                "actual_pts":  pts,
                                "actual_reb":  reb,
                                "actual_ast":  ast_,
                                "actual_fg3m": fg3m,
                                "actual_stocks": stl + blk,
                                "actual_dd":   int(pts>=10 and reb>=10 or
                                                   pts>=10 and ast_>=10 or
                                                   reb>=10 and ast_>=10),
                            })
            time.sleep(0.3)
        except Exception as e:
            log(f"  NBA {date_str}: {e}")

    return {"sport":"nba","rows":rows}


def aggregate_nba(rows: list) -> dict:
    if not rows:
        return {}
    df  = pd.DataFrame(rows)
    out = {
        "sport":   "nba",
        "n_games": int(df["date"].nunique()),
        "n_rows":  int(len(df)),
    }

    for stat, pred_col, actual_col in [
        ("pts",    "pred_pts",    "actual_pts"),
        ("reb",    "pred_reb",    "actual_reb"),
        ("ast",    "pred_ast",    "actual_ast"),
        ("fg3m",   "pred_fg3m",   "actual_fg3m"),
        ("stocks", "pred_stocks", "actual_stocks"),
    ]:
        if pred_col in df.columns and actual_col in df.columns:
            out[f"mae_{stat}"]  = round(float((df[pred_col]-df[actual_col]).abs().mean()),3)
            out[f"bias_{stat}"] = round(float((df[pred_col]-df[actual_col]).mean()),3)

    # DD calibration
    if "pred_dd" in df.columns and "actual_dd" in df.columns:
        dd_probs  = df["pred_dd"].values.astype(float)
        dd_labels = df["actual_dd"].values.astype(int)
        out["dd_brier"]        = round(brier_score(dd_probs, dd_labels), 4)
        out["dd_ece"]          = round(expected_calibration_error(dd_probs, dd_labels), 4)
        out["dd_direction_acc"]= round(
            float(((dd_probs>=0.35)==(dd_labels==1)).mean()), 4)

    # Per-tier
    tiers = {}
    for tier in ["Elite","High","Medium","Low"]:
        sub = df[df["conf"]==tier]
        if len(sub) < 5: continue
        t = {"n":int(len(sub))}
        for stat,pc,ac in [("pts","pred_pts","actual_pts"),
                            ("reb","pred_reb","actual_reb"),
                            ("ast","pred_ast","actual_ast")]:
            if pc in sub.columns and ac in sub.columns:
                t[f"mae_{stat}"]       = round(float((sub[pc]-sub[ac]).abs().mean()),3)
                t[f"avg_pred_{stat}"]  = round(float(sub[pc].mean()),3)
                t[f"avg_actual_{stat}"]= round(float(sub[ac].mean()),3)
        tiers[tier] = t
    out["tiers"] = tiers

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_backtest(days: int = 30) -> dict:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Backtesting last {days} days…")

    results = {}
    for sport, bt_fn, agg_fn in [
        ("nhl", backtest_nhl, aggregate_nhl),
        ("mlb", backtest_mlb, aggregate_mlb),
        ("nba", backtest_nba, aggregate_nba),
    ]:
        data = bt_fn(days)
        agg  = agg_fn(data.get("rows", []))
        results[sport] = {
            "aggregate":  agg,
            "message":    data.get("message",""),
            "error":      data.get("error",""),
            "updated_at": datetime.now(ET).isoformat(),
            "days":       days,
            "n_rows":     len(data.get("rows",[])),
        }
        log(f"  {sport.upper()}: {len(data.get('rows',[]))} player-games")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"Saved → {OUT_FILE}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    run_backtest(args.days)
