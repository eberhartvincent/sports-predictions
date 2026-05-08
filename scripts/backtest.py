"""
scripts/backtest.py — Compare past predictions against actual results.

Run from Codespaces terminal:
    python scripts/backtest.py --days 7
"""

import argparse, json, sys, requests, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
for _p in [_ROOT, _ROOT/"config", _ROOT/"core"/"pipelines", _ROOT/"data"/"api"]:
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

ET       = ZoneInfo("America/New_York")
PRED_DIR = _ROOT / "data" / "cache" / "predictions"
HIST_DIR = PRED_DIR / "history"
OUT_FILE = PRED_DIR / "backtest_results.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "application/json",
}

MLB_API   = "https://statsapi.mlb.com/api/v1"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


def log(msg): print(f"[{datetime.now(ET).strftime('%H:%M:%S')}] {msg}", flush=True)


def _hist_files(sport: str, days: int) -> list:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if not HIST_DIR.exists():
        return []
    return sorted(
        f for f in HIST_DIR.glob(f"{sport}_*.parquet")
        if f.stem.replace(f"{sport}_", "") < today
    )[-days:]


def _valid_parquet(df: pd.DataFrame, required: set) -> bool:
    return required.issubset(df.columns)


# ── Metrics ───────────────────────────────────────────────────────────────────

def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        total += abs(float(probs[mask].mean()) - float(labels[mask].mean())) * mask.sum()
    return total / max(len(probs), 1)


def brier(probs, labels):
    return float(np.mean((probs - labels.astype(float))**2))


def auc_score(probs, labels):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs))
    except Exception:
        return None


def roi(labels, odds=150):
    dec = (odds/100)+1 if odds > 0 else (100/abs(odds))+1
    wins = int(labels.sum()); n = len(labels)
    return round((wins*(dec-1) - (n-wins)) / n, 4) if n > 0 else 0.0


def tier_stats(df, prob_col, label_col):
    out = {}
    for tier in ["Elite", "High", "Medium", "Low"]:
        sub = df[df["conf"] == tier]
        if len(sub) < 5: continue
        probs  = sub[prob_col].values.astype(float)
        labels = sub[label_col].values.astype(int)
        out[tier] = {
            "n":                 int(len(sub)),
            "accuracy":          round(float(labels.mean()), 4),
            "avg_predicted":     round(float(probs.mean()),  4),
            "calibration_error": round(ece(probs, labels),   4),
            "brier_score":       round(brier(probs, labels),  4),
        }
    return out


# ── MLB ───────────────────────────────────────────────────────────────────────

def backtest_mlb(days: int) -> dict:
    log("MLB backtest…")
    files = _hist_files("mlb", days)
    if not files:
        return {"sport":"mlb","message":"No history files yet.","rows":[],"files_found":0,"api_failures":0}

    log(f"  {len(files)} history files")
    rows = []; api_failures = 0

    for pred_file in files:
        date_str = pred_file.stem.replace("mlb_", "")
        try:
            preds = pd.read_parquet(pred_file)
            if not _valid_parquet(preds, {"player_id","proj_hits","confidence"}):
                log(f"  {date_str}: skipping — old format parquet")
                continue

            # Get game IDs for this date
            r = requests.get(f"{MLB_API}/schedule",
                params={"sportId":1, "date":date_str},
                headers=HEADERS, timeout=15)
            if r.status_code != 200:
                api_failures += 1
                log(f"  {date_str}: schedule API {r.status_code}")
                continue

            game_ids = []
            for d in r.json().get("dates", []):
                for g in d.get("games", []):
                    if g.get("gamePk"):
                        game_ids.append(g["gamePk"])

            if not game_ids:
                log(f"  {date_str}: no games found")
                continue

            # Fetch boxscore per game individually (same as pipeline does)
            matched = 0
            for gid in game_ids:
                bs = requests.get(f"{MLB_API}/game/{gid}/boxscore",
                    headers=HEADERS, timeout=15)
                if bs.status_code != 200:
                    continue
                data = bs.json()
                for side in ("away", "home"):
                    players = data.get("teams",{}).get(side,{}).get("players",{})
                    for pid_str, pdata in players.items():
                        pid  = pdata.get("person", {}).get("id")
                        stat = pdata.get("stats",  {}).get("batting", {})
                        ab   = int(stat.get("atBats", 0))
                        if ab == 0 or pid is None:
                            continue
                        match = preds[preds["player_id"] == int(pid)]
                        if match.empty:
                            continue
                        pr  = match.iloc[0]
                        h   = int(stat.get("hits",       0))
                        hr  = int(stat.get("homeRuns",   0))
                        rbi = int(stat.get("rbi",        0))
                        r2  = int(stat.get("runs",       0))
                        rows.append({
                            "date":        date_str,
                            "player_id":   int(pid),
                            "name":        pr.get("player_name", ""),
                            "conf":        str(pr.get("confidence", "Low")),
                            "pred_h":      float(pr.get("proj_hits",  0)),
                            "pred_hr":     float(pr.get("proj_hr",    0)),
                            "pred_rbi":    float(pr.get("proj_rbi",   0)),
                            "pred_runs":   float(pr.get("proj_runs",  0)),
                            "pred_hrr":    float(pr.get("proj_hrr",   0)),
                            "actual_h":    h,
                            "actual_hr":   hr,
                            "actual_rbi":  rbi,
                            "actual_runs": r2,
                            "actual_hrr":  h + rbi + r2,
                            "hr_scored":   int(hr > 0),
                        })
                        matched += 1
                time.sleep(0.05)
            log(f"  {date_str}: {matched} players from {len(game_ids)} games")
        except Exception as e:
            log(f"  MLB {date_str} error: {e}")
            import traceback; traceback.print_exc()

    log(f"  MLB total: {len(rows)} rows")
    return {"sport":"mlb","rows":rows,"message":"",
            "files_found":len(files),"api_failures":api_failures}


def aggregate_mlb(rows: list) -> dict:
    if not rows:
        return {}
    df  = pd.DataFrame(rows)
    out = {
        "n_games": int(df["date"].nunique()),
        "n_rows":  int(len(df)),
        "tiers":   {},
        "stats":   {},
        "calibration_curve": [],
    }
    for label, pc, ac in [
        ("Hits",     "pred_h",   "actual_h"),
        ("Home Runs","pred_hr",  "actual_hr"),
        ("RBI",      "pred_rbi", "actual_rbi"),
        ("Runs",     "pred_runs","actual_runs"),
        ("H+R+RBI",  "pred_hrr", "actual_hrr"),
    ]:
        if pc in df and ac in df:
            diff = df[pc] - df[ac]
            out["stats"][label] = {
                "mae":        round(float(diff.abs().mean()), 3),
                "bias":       round(float(diff.mean()),       3),
                "avg_pred":   round(float(df[pc].mean()),     3),
                "avg_actual": round(float(df[ac].mean()),     3),
            }
    if "pred_hr" in df and "hr_scored" in df:
        hp = df["pred_hr"].values.astype(float)
        hl = df["hr_scored"].values.astype(int)
        out["hr_direction_accuracy"] = round(float(((hp>=0.08)==(hl==1)).mean()), 4)
        out["hr_brier"]              = round(brier(hp, hl), 4)

    out["tiers"] = tier_stats(df, "pred_h", "actual_h")

    cal = []
    for lo in [0, 0.25, 0.50, 0.75]:
        mask = (df["pred_h"] >= lo) & (df["pred_h"] < lo+0.25)
        if mask.sum() >= 5:
            cal.append({
                "bucket":        f"{lo:.2f}–{lo+0.25:.2f} proj hits",
                "n":             int(mask.sum()),
                "avg_predicted": round(float(df[mask]["pred_h"].mean()),  3),
                "avg_actual":    round(float(df[mask]["actual_h"].mean()), 3),
            })
    out["calibration_curve"] = cal
    return out


# ── NBA ───────────────────────────────────────────────────────────────────────

def backtest_nba(days: int) -> dict:
    log("NBA backtest…")
    files = _hist_files("nba", days)
    if not files:
        return {"sport":"nba","message":"No history files yet.","rows":[],"files_found":0,"api_failures":0}

    log(f"  {len(files)} history files")
    rows = []; api_failures = 0

    for pred_file in files:
        date_str = pred_file.stem.replace("nba_", "")
        date_fmt = date_str.replace("-", "")
        try:
            preds = pd.read_parquet(pred_file)
            if not _valid_parquet(preds, {"player_id","proj_pts","confidence"}):
                log(f"  {date_str}: skipping — old format")
                continue

            r = requests.get(f"{ESPN_BASE}/scoreboard",
                params={"dates":date_fmt}, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                api_failures += 1
                log(f"  {date_str}: ESPN {r.status_code}")
                continue

            game_ids = [e.get("id") for e in r.json().get("events",[]) if e.get("id")]
            if not game_ids:
                log(f"  {date_str}: no games found")
                continue

            matched = 0
            for gid in game_ids:
                bs = requests.get(f"{ESPN_BASE}/summary",
                    params={"event":gid}, headers=HEADERS, timeout=15)
                if bs.status_code != 200:
                    continue
                for team in bs.json().get("boxscore",{}).get("players",[]):
                    for grp in team.get("statistics",[]):
                        for athlete in grp.get("athletes",[]):
                            pid   = athlete.get("athlete",{}).get("id")
                            stats = athlete.get("stats",[])
                            if not pid or not stats:
                                continue
                            try:
                                mins  = float(str(stats[0]).split(":")[0]) if stats else 0
                                pts   = int(float(stats[13])) if len(stats)>13 else 0
                                reb   = int(float(stats[6]))  if len(stats)>6  else 0
                                ast_  = int(float(stats[7]))  if len(stats)>7  else 0
                                stl   = int(float(stats[8]))  if len(stats)>8  else 0
                                blk   = int(float(stats[9]))  if len(stats)>9  else 0
                                fg3s  = str(stats[2]) if len(stats)>2 else "0-0"
                                fg3m  = int(fg3s.split("-")[0]) if "-" in fg3s else 0
                            except (ValueError, IndexError):
                                continue
                            if mins < 5:
                                continue
                            match = preds[preds["player_id"] == int(pid)]
                            if match.empty:
                                continue
                            pr = match.iloc[0]
                            rows.append({
                                "date":          date_str,
                                "player_id":     int(pid),
                                "name":          pr.get("player_name",""),
                                "conf":          str(pr.get("confidence","Low")),
                                "pred_pts":      float(pr.get("proj_pts",  0)),
                                "pred_reb":      float(pr.get("proj_reb",  0)),
                                "pred_ast":      float(pr.get("proj_ast",  0)),
                                "pred_fg3m":     float(pr.get("proj_fg3m", 0)),
                                "pred_dd":       float(pr.get("proj_dd",   0)),
                                "actual_pts":    pts,
                                "actual_reb":    reb,
                                "actual_ast":    ast_,
                                "actual_fg3m":   fg3m,
                                "actual_stocks": stl+blk,
                                "actual_dd":     int((pts>=10 and reb>=10) or
                                                     (pts>=10 and ast_>=10) or
                                                     (reb>=10 and ast_>=10)),
                            })
                            matched += 1
                time.sleep(0.1)
            log(f"  {date_str}: {matched} players from {len(game_ids)} games")
        except Exception as e:
            log(f"  NBA {date_str} error: {e}")
            import traceback; traceback.print_exc()

    log(f"  NBA total: {len(rows)} rows")
    return {"sport":"nba","rows":rows,"message":"",
            "files_found":len(files),"api_failures":api_failures}


def aggregate_nba(rows: list) -> dict:
    if not rows:
        return {}
    df  = pd.DataFrame(rows)
    out = {
        "n_games": int(df["date"].nunique()),
        "n_rows":  int(len(df)),
        "tiers":   {},
        "stats":   {},
    }
    for label, pc, ac in [
        ("Points",    "pred_pts",  "actual_pts"),
        ("Rebounds",  "pred_reb",  "actual_reb"),
        ("Assists",   "pred_ast",  "actual_ast"),
        ("3-Pointers","pred_fg3m", "actual_fg3m"),
    ]:
        if pc in df and ac in df:
            diff = df[pc] - df[ac]
            out["stats"][label] = {
                "mae":        round(float(diff.abs().mean()), 3),
                "bias":       round(float(diff.mean()),       3),
                "avg_pred":   round(float(df[pc].mean()),     3),
                "avg_actual": round(float(df[ac].mean()),     3),
            }
    if "pred_dd" in df and "actual_dd" in df:
        dp = df["pred_dd"].values.astype(float)
        dl = df["actual_dd"].values.astype(int)
        out["dd_accuracy"] = round(float(((dp>=0.35)==(dl==1)).mean()), 4)
        out["dd_brier"]    = round(brier(dp, dl), 4)
    out["tiers"] = tier_stats(df, "pred_pts", "actual_pts")
    return out


# ── NHL ───────────────────────────────────────────────────────────────────────

def backtest_nhl(days: int) -> dict:
    log("NHL backtest…")
    files = _hist_files("nhl", days)
    if not files:
        return {"sport":"nhl","message":"No history files yet.","rows":[],"files_found":0,"api_failures":0}

    log(f"  {len(files)} history files")
    rows = []; api_failures = 0

    for pred_file in files:
        date_str = pred_file.stem.replace("nhl_", "")
        try:
            preds = pd.read_parquet(pred_file)
            if not _valid_parquet(preds, {"player_id","goal_probability","confidence"}):
                log(f"  {date_str}: skipping — old format")
                continue

            results = {}
            for url in [
                f"https://api-web.nhle.com/v1/score/{date_str}",
                f"https://api-web.nhle.com/v1/schedule/{date_str}",
            ]:
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code != 200:
                    continue
                for game in r.json().get("games", []):
                    gid = game.get("id")
                    if not gid: continue
                    bs = requests.get(
                        f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore",
                        headers=HEADERS, timeout=15)
                    if bs.status_code != 200: continue
                    for side in ("homeTeam","awayTeam"):
                        for player in bs.json().get(side,{}).get("players",[]):
                            pid = player.get("playerId")
                            if pid:
                                results[int(pid)] = {
                                    "goals": int(player.get("goals",0)),
                                    "shots": int(player.get("shots",0)),
                                }
                    time.sleep(0.1)
                if results: break

            if not results:
                api_failures += 1
                log(f"  {date_str}: no API results (NHL may block this IP)")
                continue

            matched = 0
            for pid, stats in results.items():
                match = preds[preds["player_id"] == pid]
                if match.empty: continue
                pr = match.iloc[0]
                rows.append({
                    "date":       date_str,
                    "player_id":  pid,
                    "name":       pr.get("player_name",""),
                    "conf":       str(pr.get("confidence","Low")),
                    "pred_prob":  float(pr.get("goal_probability",0)),
                    "scored":     int(stats["goals"]>0),
                    "goals":      stats["goals"],
                    "pred_sog":   float(pr.get("projected_sog",0)),
                    "actual_sog": stats["shots"],
                })
                matched += 1
            log(f"  {date_str}: {matched} players matched")
        except Exception as e:
            log(f"  NHL {date_str} error: {e}")
            import traceback; traceback.print_exc()

    msg = (f"NHL API blocked {api_failures} date(s) — NHL results may be incomplete."
           if api_failures else "")
    log(f"  NHL total: {len(rows)} rows")
    return {"sport":"nhl","rows":rows,"message":msg,
            "files_found":len(files),"api_failures":api_failures}


def aggregate_nhl(rows: list) -> dict:
    if not rows:
        return {}
    df  = pd.DataFrame(rows)
    out = {
        "n_games":           int(df["date"].nunique()),
        "n_rows":            int(len(df)),
        "tiers":             {},
        "stats":             {},
        "calibration_curve": [],
    }
    probs  = df["pred_prob"].values.astype(float)
    labels = df["scored"].values.astype(int)
    a = auc_score(probs, labels)
    out["stats"]["Goal Scoring"] = {
        "overall_accuracy":  round(float(labels.mean()), 4),
        "brier_score":       round(brier(probs, labels),  4),
        "calibration_error": round(ece(probs, labels),    4),
        "auc":               round(a, 4) if a else None,
    }
    elite = df[df["conf"]=="Elite"]
    if len(elite) >= 5:
        el = elite["scored"].values
        out["elite_n"]           = int(len(elite))
        out["elite_accuracy"]    = round(float(el.mean()),4)
        out["elite_roi_plus150"] = roi(el, 150)
        out["elite_roi_plus130"] = roi(el, 130)
        out["elite_roi_plus110"] = roi(el, 110)
    if "pred_sog" in df and "actual_sog" in df:
        out["stats"]["Shots on Goal"] = {
            "mae": round(float((df["pred_sog"]-df["actual_sog"]).abs().mean()),3),
        }
    out["tiers"] = tier_stats(df, "pred_prob", "scored")
    cal = []
    for lo in np.arange(0, 1, 0.1):
        mask = (probs>=lo)&(probs<lo+0.1)
        if mask.sum() >= 5:
            cal.append({
                "bucket":        f"{lo:.0%}–{lo+0.1:.0%} predicted",
                "n":             int(mask.sum()),
                "avg_predicted": round(float(probs[mask].mean()),3),
                "avg_actual":    round(float(labels[mask].mean()),3),
            })
    out["calibration_curve"] = cal
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(days: int = 30) -> dict:
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Backtesting last {days} days…")

    existing = {}
    if OUT_FILE.exists():
        try: existing = json.loads(OUT_FILE.read_text())
        except Exception: pass

    results = dict(existing)

    for sport, bt_fn, agg_fn in [
        ("nhl", backtest_nhl, aggregate_nhl),
        ("mlb", backtest_mlb, aggregate_mlb),
        ("nba", backtest_nba, aggregate_nba),
    ]:
        try:
            data = bt_fn(days)
            agg  = agg_fn(data.get("rows", []))
            results[sport] = {
                "aggregate":    agg,
                "message":      data.get("message",      ""),
                "updated_at":   datetime.now(ET).isoformat(),
                "days":         days,
                "n_rows":       len(data.get("rows",     [])),
                "files_found":  data.get("files_found",  0),
                "api_failures": data.get("api_failures", 0),
            }
            log(f"  {sport.upper()}: {len(data.get('rows',[]))} rows")
        except Exception as e:
            log(f"  {sport.upper()} ERROR: {e}")
            import traceback; traceback.print_exc()
            if sport not in results:
                results[sport] = {
                    "aggregate":{},"message":f"Error: {e}",
                    "updated_at":datetime.now(ET).isoformat(),
                    "days":days,"n_rows":0,"files_found":0,"api_failures":0,
                }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    log(f"Saved → {OUT_FILE}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    run_backtest(args.days)
