"""
scripts/backtest.py — Compare predictions against actual results.
Tracks player stats AND game outcomes (Win/Loss, Over/Under).

Run: python scripts/backtest.py --days 7
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
                  "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
}
MLB_API   = "https://statsapi.mlb.com/api/v1"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
NHL_API   = "https://api-web.nhle.com/v1"


def log(msg): print(f"[{datetime.now(ET).strftime('%H:%M:%S')}] {msg}", flush=True)


def _hist_files(sport: str, days: int) -> list:
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if not HIST_DIR.exists(): return []
    return sorted(
        f for f in HIST_DIR.glob(f"{sport}_*.parquet")
        if f.stem.replace(f"{sport}_","") < today
    )[-days:]


def _valid(df, required): return required.issubset(df.columns)


# ── Metrics ───────────────────────────────────────────────────────────────────

def ece(probs, labels, n_bins=10):
    bins = np.linspace(0,1,n_bins+1); total=0
    for i in range(n_bins):
        mask=(probs>=bins[i])&(probs<bins[i+1])
        if mask.sum()==0: continue
        total+=abs(float(probs[mask].mean())-float(labels[mask].mean()))*mask.sum()
    return total/max(len(probs),1)

def brier(probs,labels): return float(np.mean((probs-labels.astype(float))**2))

def auc_score(probs,labels):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels,probs))
    except: return None

def roi(labels,odds=150):
    dec=(odds/100)+1 if odds>0 else (100/abs(odds))+1
    wins=int(labels.sum()); n=len(labels)
    return round((wins*(dec-1)-(n-wins))/n,4) if n>0 else 0.0

def tier_stats(df,prob_col,label_col):
    out={}
    for tier in ["Elite","High","Medium","Low"]:
        sub=df[df["conf"]==tier]
        if len(sub)<5: continue
        probs=sub[prob_col].values.astype(float)
        labels=sub[label_col].values.astype(int)
        out[tier]={"n":int(len(sub)),
                   "accuracy":round(float(labels.mean()),4),
                   "avg_predicted":round(float(probs.mean()),4),
                   "calibration_error":round(ece(probs,labels),4),
                   "brier_score":round(brier(probs,labels),4)}
    return out


# ── MLB ───────────────────────────────────────────────────────────────────────

def backtest_mlb(days: int) -> dict:
    log("MLB backtest…")
    files = _hist_files("mlb", days)
    if not files:
        return {"sport":"mlb","message":"No history files yet.",
                "rows":[],"game_rows":[],"files_found":0,"api_failures":0}

    log(f"  {len(files)} history files")
    rows=[]; game_rows=[]; api_failures=0

    # Load saved game projections for comparison
    proj_file = PRED_DIR / "mlb_game_projections.json"
    saved_projs = {}
    if proj_file.exists():
        try:
            for p in json.loads(proj_file.read_text()):
                key = f"{p.get('away_team','')}_vs_{p.get('home_team','')}"
                saved_projs[key] = p
        except: pass

    for pred_file in files:
        date_str = pred_file.stem.replace("mlb_","")
        try:
            preds = pd.read_parquet(pred_file)
            if not _valid(preds,{"player_id","proj_hits","confidence"}):
                log(f"  {date_str}: skipping — old format"); continue

            r = requests.get(f"{MLB_API}/schedule",
                params={"sportId":1,"date":date_str},headers=HEADERS,timeout=15)
            if r.status_code!=200:
                api_failures+=1; log(f"  {date_str}: API {r.status_code}"); continue

            game_ids=[]
            for d in r.json().get("dates",[]):
                for g in d.get("games",[]):
                    if g.get("gamePk"): game_ids.append(g["gamePk"])

            matched=0
            for gid in game_ids:
                bs = requests.get(f"{MLB_API}/game/{gid}/boxscore",
                    headers=HEADERS,timeout=15)
                if bs.status_code!=200: continue
                data=bs.json()

                # Game-level result
                teams = data.get("teams",{})
                away_runs = int(teams.get("away",{}).get("teamStats",{}).get("batting",{}).get("runs",0))
                home_runs = int(teams.get("home",{}).get("teamStats",{}).get("batting",{}).get("runs",0))
                away_name = teams.get("away",{}).get("team",{}).get("abbreviation","")
                home_name = teams.get("home",{}).get("team",{}).get("abbreviation","")
                total_runs = away_runs + home_runs
                winner = away_name if away_runs > home_runs else home_name

                # Match against our game projection
                key = f"{away_name}_vs_{home_name}"
                proj = saved_projs.get(key,{})
                pred_total = float(proj.get("total_proj_runs",0))
                our_ou_rec = proj.get("recommendation","")
                ou_line    = float(proj.get("best_ou_line",0))
                pred_winner= proj.get("favourite",home_name)

                game_rows.append({
                    "date":       date_str,
                    "matchup":    f"{away_name} @ {home_name}",
                    "away":       away_name,
                    "home":       home_name,
                    "away_runs":  away_runs,
                    "home_runs":  home_runs,
                    "total_runs": total_runs,
                    "winner":     winner,
                    "pred_total": pred_total,
                    "pred_winner":pred_winner,
                    "our_ou_rec": our_ou_rec,
                    "ou_line":    ou_line,
                    "winner_correct": int(winner==pred_winner) if pred_winner else None,
                    "ou_correct": int((total_runs>ou_line)==(our_ou_rec=="OVER")) if ou_line>0 else None,
                })

                # Player stats
                for side in ("away","home"):
                    for pid_str,pdata in data.get("teams",{}).get(side,{}).get("players",{}).items():
                        pid=pdata.get("person",{}).get("id")
                        stat=pdata.get("stats",{}).get("batting",{})
                        ab=int(stat.get("atBats",0))
                        if ab==0 or pid is None: continue
                        match=preds[preds["player_id"]==int(pid)]
                        if match.empty: continue
                        pr=match.iloc[0]
                        h=int(stat.get("hits",0)); hr=int(stat.get("homeRuns",0))
                        rbi=int(stat.get("rbi",0)); r2=int(stat.get("runs",0))
                        rows.append({
                            "date":date_str,"player_id":int(pid),
                            "name":pr.get("player_name",""),
                            "team":pr.get("team",""),
                            "conf":str(pr.get("confidence","Low")),
                            "pred_h":float(pr.get("proj_hits",0)),
                            "pred_hr":float(pr.get("proj_hr",0)),
                            "pred_rbi":float(pr.get("proj_rbi",0)),
                            "pred_runs":float(pr.get("proj_runs",0)),
                            "pred_hrr":float(pr.get("proj_hrr",0)),
                            "actual_h":h,"actual_hr":hr,
                            "actual_rbi":rbi,"actual_runs":r2,
                            "actual_hrr":h+rbi+r2,"hr_scored":int(hr>0),
                        })
                        matched+=1
                time.sleep(0.05)
            log(f"  {date_str}: {matched} players, {len(game_ids)} games")
        except Exception as e:
            log(f"  MLB {date_str} error: {e}")
            import traceback; traceback.print_exc()

    log(f"  MLB: {len(rows)} player rows, {len(game_rows)} game rows")
    return {"sport":"mlb","rows":rows,"game_rows":game_rows,
            "files_found":len(files),"api_failures":api_failures,"message":""}


def aggregate_mlb(rows, game_rows):
    out={"n_days":0,"n_player_rows":0,"tiers":{},"stats":{},"calibration_curve":[],
         "game_stats":{},"game_rows":game_rows}
    if not rows and not game_rows: return out

    if rows:
        df=pd.DataFrame(rows)
        out["n_days"]=int(df["date"].nunique())
        out["n_player_rows"]=int(len(df))
        for label,pc,ac in [
            ("Hits","pred_h","actual_h"),("Home Runs","pred_hr","actual_hr"),
            ("RBI","pred_rbi","actual_rbi"),("Runs","pred_runs","actual_runs"),
            ("H+R+RBI","pred_hrr","actual_hrr")]:
            if pc in df and ac in df:
                diff=df[pc]-df[ac]
                out["stats"][label]={"mae":round(float(diff.abs().mean()),3),
                    "bias":round(float(diff.mean()),3),
                    "avg_pred":round(float(df[pc].mean()),3),
                    "avg_actual":round(float(df[ac].mean()),3)}
        if "pred_hr" in df and "hr_scored" in df:
            hp=df["pred_hr"].values.astype(float); hl=df["hr_scored"].values.astype(int)
            out["hr_direction_accuracy"]=round(float(((hp>=0.08)==(hl==1)).mean()),4)
            out["hr_brier"]=round(brier(hp,hl),4)

        # Tier breakdown
        tiers={}
        for tier in ["Elite","High","Medium","Low"]:
            sub=df[df["conf"]==tier]
            if len(sub)<5: continue
            t={"n":int(len(sub))}
            for label,pc,ac in [("h","pred_h","actual_h"),
                                  ("hrr","pred_hrr","actual_hrr"),
                                  ("hr","pred_hr","actual_hr")]:
                if pc in sub and ac in sub:
                    t[f"mae_{label}"]=round(float((sub[pc]-sub[ac]).abs().mean()),3)
                    t[f"bias_{label}"]=round(float((sub[pc]-sub[ac]).mean()),3)
                    t[f"avg_pred_{label}"]=round(float(sub[pc].mean()),3)
                    t[f"avg_actual_{label}"]=round(float(sub[ac].mean()),3)
            tiers[tier]=t
        out["tiers"]=tiers

        # Top 10 by pred_hrr
        top10=df.sort_values("pred_hrr",ascending=False).head(10)
        out["top10_players"]=[{
            "name":r["name"],"team":r.get("team",""),"conf":r["conf"],
            "pred_h":round(r["pred_h"],2),"actual_h":int(r["actual_h"]),
            "pred_hr":round(r["pred_hr"],3),"actual_hr":int(r["actual_hr"]),
            "pred_hrr":round(r["pred_hrr"],2),"actual_hrr":int(r["actual_hrr"]),
        } for _,r in top10.iterrows()]

    if game_rows:
        gdf=pd.DataFrame(game_rows)
        out["n_days"]=max(out["n_days"],int(gdf["date"].nunique()))
        # Win/loss accuracy
        wc=gdf.dropna(subset=["winner_correct"])
        if len(wc)>=2:
            out["game_stats"]["Win/Loss Accuracy"]=round(float(wc["winner_correct"].mean()),4)
        # Over/under accuracy
        oc=gdf.dropna(subset=["ou_correct"])
        if len(oc)>=2:
            out["game_stats"]["Over/Under Accuracy"]=round(float(oc["ou_correct"].mean()),4)
        # Average run total error
        valid=gdf[gdf["pred_total"]>0]
        if len(valid)>=2:
            out["game_stats"]["Run Total Error"]=round(
                float((valid["pred_total"]-valid["total_runs"]).abs().mean()),2)

    return out


# ── NHL ───────────────────────────────────────────────────────────────────────

def backtest_nhl(days: int) -> dict:
    log("NHL backtest…")
    files = _hist_files("nhl", days)
    if not files:
        return {"sport":"nhl","message":"No history files yet.",
                "rows":[],"game_rows":[],"files_found":0,"api_failures":0}

    log(f"  {len(files)} history files")
    rows=[]; game_rows=[]; api_failures=0

    # Load saved game projections
    proj_file = PRED_DIR / "nhl_game_projections.json"
    saved_projs = {}
    if proj_file.exists():
        try:
            for p in json.loads(proj_file.read_text()):
                key = f"{p.get('away_team','')}_vs_{p.get('home_team','')}"
                saved_projs[key] = p
        except: pass

    for pred_file in files:
        date_str = pred_file.stem.replace("nhl_","")
        try:
            preds=pd.read_parquet(pred_file)
            if not _valid(preds,{"player_id","goal_probability","confidence"}):
                log(f"  {date_str}: skipping — old format"); continue

            results={}
            for url in [f"{NHL_API}/score/{date_str}",f"{NHL_API}/schedule/{date_str}"]:
                r=requests.get(url,headers=HEADERS,timeout=15)
                if r.status_code!=200: continue
                for game in r.json().get("games",[]):
                    gid=game.get("id")
                    if not gid: continue

                    # Game scores
                    away_team=game.get("awayTeam",{})
                    home_team=game.get("homeTeam",{})
                    away_abbr=away_team.get("abbrev","")
                    home_abbr=home_team.get("abbrev","")
                    away_score=int(away_team.get("score",0))
                    home_score=int(home_team.get("score",0))
                    total_goals=away_score+home_score
                    winner=away_abbr if away_score>home_score else home_abbr

                    key=f"{away_abbr}_vs_{home_abbr}"
                    proj=saved_projs.get(key,{})
                    ou_line=float(proj.get("best_ou_line",0))
                    ou_rec=proj.get("recommendation","")
                    pred_winner=proj.get("favourite",home_abbr)
                    pred_total=float(proj.get("proj_total",0))

                    game_rows.append({
                        "date":date_str,
                        "matchup":f"{away_abbr} @ {home_abbr}",
                        "away":away_abbr,"home":home_abbr,
                        "away_score":away_score,"home_score":home_score,
                        "total_goals":total_goals,"winner":winner,
                        "pred_total":pred_total,"pred_winner":pred_winner,
                        "our_ou_rec":ou_rec,"ou_line":ou_line,
                        "winner_correct":int(winner==pred_winner) if pred_winner else None,
                        "ou_correct":int((total_goals>ou_line)==(ou_rec=="OVER")) if ou_line>0 else None,
                    })

                    bs=requests.get(f"{NHL_API}/gamecenter/{gid}/boxscore",
                        headers=HEADERS,timeout=15)
                    if bs.status_code!=200: continue
                    for side in ("homeTeam","awayTeam"):
                        for player in bs.json().get(side,{}).get("players",[]):
                            pid=player.get("playerId")
                            if pid:
                                results[int(pid)]={
                                    "goals":int(player.get("goals",0)),
                                    "shots":int(player.get("shots",0))}
                    time.sleep(0.1)
                if results: break

            if not results:
                api_failures+=1
                log(f"  {date_str}: NHL API blocked"); continue

            matched=0
            for pid,stats in results.items():
                match=preds[preds["player_id"]==pid]
                if match.empty: continue
                pr=match.iloc[0]
                rows.append({
                    "date":date_str,"player_id":pid,
                    "name":pr.get("player_name",""),
                    "team":pr.get("team",""),
                    "conf":str(pr.get("confidence","Low")),
                    "pred_prob":float(pr.get("goal_probability",0)),
                    "scored":int(stats["goals"]>0),
                    "goals":stats["goals"],
                    "pred_sog":float(pr.get("projected_sog",0)),
                    "actual_sog":stats["shots"],
                })
                matched+=1
            log(f"  {date_str}: {matched} players, {len(game_rows)} games")
        except Exception as e:
            log(f"  NHL {date_str} error: {e}")
            import traceback; traceback.print_exc()

    msg=(f"NHL API blocked {api_failures} date(s)." if api_failures else "")
    log(f"  NHL: {len(rows)} player rows, {len(game_rows)} game rows")
    return {"sport":"nhl","rows":rows,"game_rows":game_rows,
            "files_found":len(files),"api_failures":api_failures,"message":msg}


def aggregate_nhl(rows, game_rows):
    out={"n_days":0,"n_player_rows":0,"tiers":{},"stats":{},"calibration_curve":[],
         "game_stats":{},"game_rows":game_rows}

    if rows:
        df=pd.DataFrame(rows)
        out["n_days"]=int(df["date"].nunique())
        out["n_player_rows"]=int(len(df))
        probs=df["pred_prob"].values.astype(float)
        labels=df["scored"].values.astype(int)
        a=auc_score(probs,labels)
        out["stats"]["Goal Scoring"]={"overall_accuracy":round(float(labels.mean()),4),
            "brier_score":round(brier(probs,labels),4),
            "calibration_error":round(ece(probs,labels),4),
            "auc":round(a,4) if a else None}
        if "pred_sog" in df and "actual_sog" in df:
            out["stats"]["Shots on Goal"]={"mae":round(
                float((df["pred_sog"]-df["actual_sog"]).abs().mean()),3)}
        elite=df[df["conf"]=="Elite"]
        if len(elite)>=5:
            el=elite["scored"].values
            out["elite_n"]=int(len(elite))
            out["elite_accuracy"]=round(float(el.mean()),4)
            out["elite_roi_plus150"]=roi(el,150)
            out["elite_roi_plus130"]=roi(el,130)
            out["elite_roi_plus110"]=roi(el,110)
        out["tiers"]=tier_stats(df,"pred_prob","scored")

        # Calibration curve
        cal=[]
        for lo in np.arange(0,1,0.1):
            mask=(probs>=lo)&(probs<lo+0.1)
            if mask.sum()>=5:
                cal.append({"bucket":f"{lo:.0%}–{lo+0.1:.0%}",
                    "n":int(mask.sum()),
                    "avg_predicted":round(float(probs[mask].mean()),3),
                    "avg_actual":round(float(labels[mask].mean()),3)})
        out["calibration_curve"]=cal

        # Top 10 by pred_prob
        top10=df.sort_values("pred_prob",ascending=False).head(10)
        out["top10_players"]=[{
            "name":r["name"],"team":r.get("team",""),"conf":r["conf"],
            "pred_prob":round(r["pred_prob"],3),
            "scored":bool(r["scored"]),"goals":int(r["goals"]),
            "pred_sog":round(r["pred_sog"],1),"actual_sog":int(r["actual_sog"]),
        } for _,r in top10.iterrows()]

    if game_rows:
        gdf=pd.DataFrame(game_rows)
        out["n_days"]=max(out["n_days"],int(gdf["date"].nunique()))
        wc=gdf.dropna(subset=["winner_correct"])
        if len(wc)>=2:
            out["game_stats"]["Win/Loss Accuracy"]=round(float(wc["winner_correct"].mean()),4)
        oc=gdf.dropna(subset=["ou_correct"])
        if len(oc)>=2:
            out["game_stats"]["Over/Under Accuracy"]=round(float(oc["ou_correct"].mean()),4)
        valid=gdf[gdf["pred_total"]>0]
        if len(valid)>=2:
            out["game_stats"]["Goal Total Error"]=round(
                float((valid["pred_total"]-valid["total_goals"]).abs().mean()),2)

    return out


# ── NBA ───────────────────────────────────────────────────────────────────────

def backtest_nba(days: int) -> dict:
    log("NBA backtest…")
    files=_hist_files("nba",days)
    if not files:
        return {"sport":"nba","message":"No history files yet.",
                "rows":[],"game_rows":[],"files_found":0,"api_failures":0}

    log(f"  {len(files)} history files")
    rows=[]; game_rows=[]; api_failures=0

    proj_file = PRED_DIR / "nba_game_projections.json"
    saved_projs={}
    if proj_file.exists():
        try:
            for p in json.loads(proj_file.read_text()):
                key=f"{p.get('away_team','')}_vs_{p.get('home_team','')}"
                saved_projs[key]=p
        except: pass

    for pred_file in files:
        date_str=pred_file.stem.replace("nba_","")
        date_fmt=date_str.replace("-","")
        try:
            preds=pd.read_parquet(pred_file)
            if not _valid(preds,{"player_id","proj_pts","confidence"}):
                log(f"  {date_str}: skipping"); continue

            r=requests.get(f"{ESPN_BASE}/scoreboard",
                params={"dates":date_fmt},headers=HEADERS,timeout=15)
            if r.status_code!=200:
                api_failures+=1; log(f"  {date_str}: ESPN {r.status_code}"); continue

            events=r.json().get("events",[])
            game_ids=[e.get("id") for e in events if e.get("id")]

            matched=0
            for gid in game_ids:
                bs=requests.get(f"{ESPN_BASE}/summary",
                    params={"event":gid},headers=HEADERS,timeout=15)
                if bs.status_code!=200: continue
                bs_data=bs.json()

                # Game-level result
                comps=bs_data.get("header",{}).get("competitions",[])
                if comps:
                    comp=comps[0]
                    competitors=comp.get("competitors",[])
                    scores={}
                    for c in competitors:
                        abbr=c.get("team",{}).get("abbreviation","")
                        score=int(c.get("score",0))
                        scores[abbr]=score
                    if len(scores)==2:
                        teams=list(scores.keys())
                        away,home=teams[0],teams[1]
                        away_pts,home_pts=scores[away],scores[home]
                        total_pts=away_pts+home_pts
                        winner=away if away_pts>home_pts else home
                        key=f"{away}_vs_{home}"
                        proj=saved_projs.get(key,{})
                        pred_total=float(proj.get("total_proj_pts",0))
                        pred_winner=proj.get("favourite",home)
                        game_rows.append({
                            "date":date_str,
                            "matchup":f"{away} @ {home}",
                            "away":away,"home":home,
                            "away_pts":away_pts,"home_pts":home_pts,
                            "total_pts":total_pts,"winner":winner,
                            "pred_total":pred_total,"pred_winner":pred_winner,
                            "winner_correct":int(winner==pred_winner) if pred_winner else None,
                            "ou_line":float(proj.get("total_line",0)),
                            "ou_rec":proj.get("recommendation",""),
                        })

                # Player stats
                for team_data in bs_data.get("boxscore",{}).get("players",[]):
                    for grp in team_data.get("statistics",[]):
                        for athlete in grp.get("athletes",[]):
                            pid=athlete.get("athlete",{}).get("id")
                            stats=athlete.get("stats",[])
                            if not pid or not stats: continue
                            try:
                                mins=float(str(stats[0]).split(":")[0]) if stats else 0
                                pts=int(float(stats[13])) if len(stats)>13 else 0
                                reb=int(float(stats[6])) if len(stats)>6 else 0
                                ast_=int(float(stats[7])) if len(stats)>7 else 0
                                stl=int(float(stats[8])) if len(stats)>8 else 0
                                blk=int(float(stats[9])) if len(stats)>9 else 0
                                fg3s=str(stats[2]) if len(stats)>2 else "0-0"
                                fg3m=int(fg3s.split("-")[0]) if "-" in fg3s else 0
                            except (ValueError,IndexError): continue
                            if mins<5: continue
                            match=preds[preds["player_id"]==int(pid)]
                            if match.empty: continue
                            pr=match.iloc[0]
                            rows.append({
                                "date":date_str,"player_id":int(pid),
                                "name":pr.get("player_name",""),
                                "team":pr.get("team",""),
                                "conf":str(pr.get("confidence","Low")),
                                "pred_pts":float(pr.get("proj_pts",0)),
                                "pred_reb":float(pr.get("proj_reb",0)),
                                "pred_ast":float(pr.get("proj_ast",0)),
                                "pred_fg3m":float(pr.get("proj_fg3m",0)),
                                "pred_dd":float(pr.get("proj_dd",0)),
                                "actual_pts":pts,"actual_reb":reb,
                                "actual_ast":ast_,"actual_fg3m":fg3m,
                                "actual_stocks":stl+blk,
                                "actual_dd":int((pts>=10 and reb>=10) or
                                               (pts>=10 and ast_>=10) or
                                               (reb>=10 and ast_>=10)),
                            })
                            matched+=1
                time.sleep(0.1)
            log(f"  {date_str}: {matched} players, {len(game_ids)} games")
        except Exception as e:
            log(f"  NBA {date_str} error: {e}")
            import traceback; traceback.print_exc()

    log(f"  NBA: {len(rows)} player rows, {len(game_rows)} game rows")
    return {"sport":"nba","rows":rows,"game_rows":game_rows,
            "files_found":len(files),"api_failures":api_failures,"message":""}


def aggregate_nba(rows, game_rows):
    out={"n_days":0,"n_player_rows":0,"tiers":{},"stats":{},
         "game_stats":{},"game_rows":game_rows}

    if rows:
        df=pd.DataFrame(rows)
        out["n_days"]=int(df["date"].nunique())
        out["n_player_rows"]=int(len(df))
        for label,pc,ac in [
            ("Points","pred_pts","actual_pts"),
            ("Rebounds","pred_reb","actual_reb"),
            ("Assists","pred_ast","actual_ast"),
            ("3-Pointers","pred_fg3m","actual_fg3m")]:
            if pc in df and ac in df:
                diff=df[pc]-df[ac]
                out["stats"][label]={"mae":round(float(diff.abs().mean()),3),
                    "bias":round(float(diff.mean()),3),
                    "avg_pred":round(float(df[pc].mean()),3),
                    "avg_actual":round(float(df[ac].mean()),3)}
        if "pred_dd" in df and "actual_dd" in df:
            dp=df["pred_dd"].values.astype(float)
            dl=df["actual_dd"].values.astype(int)
            out["dd_accuracy"]=round(float(((dp>=0.35)==(dl==1)).mean()),4)
            out["dd_brier"]=round(brier(dp,dl),4)
        out["tiers"]=tier_stats(df,"pred_pts","actual_pts")

        # Top 10 by pred_pts
        top10=df.sort_values("pred_pts",ascending=False).head(10)
        out["top10_players"]=[{
            "name":r["name"],"team":r.get("team",""),"conf":r["conf"],
            "pred_pts":round(r["pred_pts"],1),"actual_pts":int(r["actual_pts"]),
            "pred_reb":round(r["pred_reb"],1),"actual_reb":int(r["actual_reb"]),
            "pred_ast":round(r["pred_ast"],1),"actual_ast":int(r["actual_ast"]),
        } for _,r in top10.iterrows()]

    if game_rows:
        gdf=pd.DataFrame(game_rows)
        out["n_days"]=max(out["n_days"],int(gdf["date"].nunique()))
        wc=gdf.dropna(subset=["winner_correct"])
        if len(wc)>=2:
            out["game_stats"]["Win/Loss Accuracy"]=round(float(wc["winner_correct"].mean()),4)
        # OU for NBA
        ou_valid=gdf[(gdf["ou_line"]>0)&(gdf["ou_rec"]!="")]
        if len(ou_valid)>=2:
            ou_correct=ou_valid.apply(
                lambda row: int((row["total_pts"]>row["ou_line"])==(row["ou_rec"]=="OVER")),axis=1)
            out["game_stats"]["Over/Under Accuracy"]=round(float(ou_correct.mean()),4)
        valid=gdf[gdf["pred_total"]>0]
        if len(valid)>=2:
            out["game_stats"]["Score Total Error"]=round(
                float((valid["pred_total"]-valid["total_pts"]).abs().mean()),1)

    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def run_backtest(days: int=30) -> dict:
    HIST_DIR.mkdir(parents=True,exist_ok=True)
    log(f"Backtesting last {days} days…")

    existing={}
    if OUT_FILE.exists():
        try: existing=json.loads(OUT_FILE.read_text())
        except: pass

    results=dict(existing)

    for sport,bt_fn,agg_fn in [
        ("nhl",backtest_nhl,aggregate_nhl),
        ("mlb",backtest_mlb,aggregate_mlb),
        ("nba",backtest_nba,aggregate_nba),
    ]:
        try:
            data=bt_fn(days)
            agg=agg_fn(data.get("rows",[]),data.get("game_rows",[]))
            results[sport]={
                "aggregate":agg,
                "message":data.get("message",""),
                "updated_at":datetime.now(ET).isoformat(),
                "days":days,
                "n_rows":len(data.get("rows",[])),
                "files_found":data.get("files_found",0),
                "api_failures":data.get("api_failures",0),
            }
            log(f"  {sport.upper()}: {len(data.get('rows',[]))} player rows, "
                f"{len(data.get('game_rows',[]))} game rows")
        except Exception as e:
            log(f"  {sport.upper()} ERROR: {e}")
            import traceback; traceback.print_exc()
            if sport not in results:
                results[sport]={"aggregate":{},"message":f"Error: {e}",
                    "updated_at":datetime.now(ET).isoformat(),"days":days,
                    "n_rows":0,"files_found":0,"api_failures":0}

    OUT_FILE.parent.mkdir(parents=True,exist_ok=True)
    OUT_FILE.write_text(json.dumps(results,indent=2,default=str))
    log(f"Saved → {OUT_FILE}")
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--days",type=int,default=30)
    args=parser.parse_args()
    run_backtest(args.days)
