"""
scripts/backtest.py
Backtesting engine — compares past predictions to actual results.

How it works:
  1. Loads saved prediction parquet files (one per past date)
  2. Fetches actual results from the APIs for those dates
  3. Computes accuracy, calibration, ROI estimate, and confidence tier breakdown
  4. Saves results to data/cache/predictions/backtest_results.json

Run manually:  python scripts/backtest.py --days 30
"""

import argparse, json, os, sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
for _p in [_ROOT, _ROOT/"config", _ROOT/"core"/"pipelines",
           _ROOT/"core"/"models", _ROOT/"data"/"api"]:
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

import pandas as pd
import numpy as np

ET       = ZoneInfo("America/New_York")
PRED_DIR = _ROOT / "data" / "cache" / "predictions"
HIST_DIR = _ROOT / "data" / "cache" / "predictions" / "history"
OUT_FILE = PRED_DIR / "backtest_results.json"


def log(msg): print(f"[{datetime.now(ET).strftime('%H:%M:%S')}] {msg}", flush=True)


# ── NHL Backtest ───────────────────────────────────────────────────────────────

def backtest_nhl(days: int) -> dict:
    """
    For each past date where we have saved predictions:
    - Pull actual game results from NHL API
    - Check if players marked as Elite/High actually scored
    - Compute accuracy by confidence tier
    """
    log("NHL backtest…")
    try:
        from nhl_api import _get, _load_cache, _save_cache
    except ImportError:
        return {"error": "NHL API not available", "rows": []}

    results = []
    hist_files = sorted(HIST_DIR.glob("nhl_*.parquet")) if HIST_DIR.exists() else []

    if not hist_files:
        return {
            "sport":   "nhl",
            "message": "No historical prediction files found. Run the app daily to accumulate history.",
            "rows":    [],
        }

    for pred_file in hist_files[-days:]:
        date_str = pred_file.stem.replace("nhl_","")
        try:
            preds = pd.read_parquet(pred_file)
            # Get actual goals from NHL API for that date
            sched = _get(f"/schedule/{date_str}")
            if not sched: continue
            for game in sched.get("games", []):
                gid = game.get("id")
                if not gid: continue
                boxscore = _get(f"/gamecenter/{gid}/boxscore")
                if not boxscore: continue
                # Extract actual scorers
                for side in ("homeTeam","awayTeam"):
                    for p in boxscore.get(side,{}).get("players",[]):
                        pid   = p.get("playerId")
                        goals = p.get("goals",0)
                        pred_row = preds[preds["player_id"]==pid]
                        if pred_row.empty: continue
                        pred_prob = float(pred_row.iloc[0].get("goal_probability",0))
                        conf      = str(pred_row.iloc[0].get("confidence","Low"))
                        scored    = int(goals) > 0
                        results.append({
                            "date":      date_str,
                            "player_id": pid,
                            "name":      pred_row.iloc[0].get("player_name",""),
                            "team":      pred_row.iloc[0].get("team",""),
                            "conf":      conf,
                            "pred_prob": pred_prob,
                            "scored":    scored,
                            "goals":     int(goals),
                        })
        except Exception as e:
            log(f"  NHL {date_str}: {e}")

    return {"sport":"nhl","rows":results}


# ── MLB Backtest ───────────────────────────────────────────────────────────────

def backtest_mlb(days: int) -> dict:
    log("MLB backtest…")
    results = []
    hist_files = sorted(HIST_DIR.glob("mlb_*.parquet")) if HIST_DIR.exists() else []

    if not hist_files:
        return {
            "sport":   "mlb",
            "message": "No historical prediction files found. Run the app daily to accumulate history.",
            "rows":    [],
        }

    try:
        import requests
        MLB_API = "https://statsapi.mlb.com/api/v1"
        headers = {"User-Agent":"Mozilla/5.0"}

        for pred_file in hist_files[-days:]:
            date_str = pred_file.stem.replace("mlb_","")
            try:
                preds = pd.read_parquet(pred_file)
                # Fetch actual boxscores for that date
                r = requests.get(f"{MLB_API}/schedule",
                    params={"sportId":1,"date":date_str,"hydrate":"boxscore"},
                    headers=headers, timeout=15)
                if r.status_code != 200: continue
                for d in r.json().get("dates",[]):
                    for game in d.get("games",[]):
                        box = game.get("teams",{})
                        for side in ("away","home"):
                            for pid_str, pdata in box.get(side,{}).get(
                                    "players",{}).items():
                                pid  = pdata.get("person",{}).get("id")
                                stat = pdata.get("stats",{}).get("batting",{})
                                h    = int(stat.get("hits",0))
                                hr   = int(stat.get("homeRuns",0))
                                rbi  = int(stat.get("rbi",0))
                                r2   = int(stat.get("runs",0))
                                k    = int(stat.get("strikeOuts",0))
                                pred_row = preds[preds["player_id"]==pid]
                                if pred_row.empty: continue
                                pr = pred_row.iloc[0]
                                results.append({
                                    "date":        date_str,
                                    "player_id":   pid,
                                    "name":        pr.get("player_name",""),
                                    "team":        pr.get("team",""),
                                    "conf":        str(pr.get("confidence","Low")),
                                    "pred_h":      float(pr.get("proj_hits",0)),
                                    "pred_hr":     float(pr.get("proj_hr",0)),
                                    "pred_hrr":    float(pr.get("proj_hrr",0)),
                                    "actual_h":    h,
                                    "actual_hr":   hr,
                                    "actual_rbi":  rbi,
                                    "actual_runs": r2,
                                    "actual_k":    k,
                                    "actual_hrr":  h + rbi + r2,
                                })
            except Exception as e:
                log(f"  MLB {date_str}: {e}")
    except Exception as e:
        return {"sport":"mlb","error":str(e),"rows":[]}

    return {"sport":"mlb","rows":results}


# ── Aggregate stats ────────────────────────────────────────────────────────────

def aggregate(sport: str, rows: list) -> dict:
    """Compute accuracy, calibration, and confidence-tier breakdown."""
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    out = {"sport": sport, "n_games": df["date"].nunique(), "n_rows": len(df)}

    if sport == "nhl":
        tiers = {}
        for conf in ["Elite","High","Medium","Low"]:
            sub = df[df["conf"]==conf]
            if len(sub) == 0: continue
            acc = float(sub["scored"].mean())
            avg_prob = float(sub["pred_prob"].mean())
            tiers[conf] = {
                "n":         len(sub),
                "accuracy":  round(acc, 3),
                "avg_pred":  round(avg_prob, 3),
                "calibration_err": round(abs(acc - avg_prob), 3),
            }
        out["tiers"] = tiers
        # Brier score (lower = better)
        out["brier_score"] = round(float(
            ((df["pred_prob"] - df["scored"].astype(float))**2).mean()), 4)
        # Pseudo-ROI: if we bet $1 on every Elite pick at average +150 odds
        elite = df[df["conf"]=="Elite"]
        if len(elite):
            wins  = int(elite["scored"].sum())
            total = len(elite)
            roi   = (wins * 1.50 - (total - wins)) / total
            out["elite_roi_estimate"] = round(roi, 3)

    elif sport == "mlb":
        tiers = {}
        for conf in ["Elite","High","Medium","Low"]:
            sub = df[df["conf"]==conf]
            if len(sub) == 0: continue
            mae_h   = float((sub["pred_h"]   - sub["actual_h"]).abs().mean())
            mae_hrr = float((sub["pred_hrr"] - sub["actual_hrr"]).abs().mean())
            tiers[conf] = {
                "n":       len(sub),
                "mae_hits":round(mae_h,   3),
                "mae_hrr": round(mae_hrr, 3),
                "avg_actual_hits": round(float(sub["actual_h"].mean()),  2),
                "avg_pred_hits":   round(float(sub["pred_h"].mean()),   2),
                "avg_actual_hrr":  round(float(sub["actual_hrr"].mean()),2),
                "avg_pred_hrr":    round(float(sub["pred_hrr"].mean()),  2),
                "hr_pred_correct": round(float(
                    ((sub["pred_hr"]>0.08)==(sub["actual_hr"]>0)).mean()), 3),
            }
        out["tiers"] = tiers

    return out


def run_backtest(days: int = 30):
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Running backtest over last {days} days…")

    results = {}
    for sport, fn in [("nhl",backtest_nhl),("mlb",backtest_mlb)]:
        data = fn(days)
        agg  = aggregate(sport, data.get("rows",[]))
        results[sport] = {
            "aggregate": agg,
            "message":   data.get("message",""),
            "error":     data.get("error",""),
            "updated_at":datetime.now(ET).isoformat(),
            "days":      days,
        }
        log(f"  {sport.upper()}: {len(data.get('rows',[]))} rows processed")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    log(f"Backtest results saved → {OUT_FILE}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    run_backtest(args.days)
