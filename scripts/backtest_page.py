"""app/pages/backtest_page.py — Admin backtesting report"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

ET       = ZoneInfo("America/New_York")
PRED_DIR = Path("data/cache/predictions")
HIST_DIR = PRED_DIR / "history"
BT_FILE  = PRED_DIR / "backtest_results.json"


# ── UI helpers ────────────────────────────────────────────────────────────────

def _card(title, value_str, grade, color, desc, note=""):
    st.markdown(
        f'<div style="background:#1a1f2e;border-radius:10px;padding:14px 16px;'
        f'margin-bottom:8px;border:2px solid {color}44;">'
        f'<div style="font-size:11px;color:#8892a4;text-transform:uppercase;'
        f'letter-spacing:.06em;margin-bottom:6px;">{title}</div>'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">'
        f'<span style="font-size:28px;font-weight:800;color:#e8ecf4;">{value_str}</span>'
        f'<span style="font-size:22px;font-weight:800;color:{color};">{grade}</span>'
        f'</div>'
        f'<div style="font-size:12px;color:{color};font-weight:600;">{desc}</div>'
        + (f'<div style="font-size:11px;color:#6c757d;margin-top:3px;">{note}</div>' if note else '')
        + '</div>', unsafe_allow_html=True)


def _grade_lower(v, good, ok, great=None):
    """Grade where lower is better."""
    if great and v <= great: return "A+ 🟢","#27ae60"
    if v <= good:   return "A 🟢","#27ae60"
    if v <= ok:     return "B 🟡","#f59e0b"
    if v <= ok*1.5: return "C 🟠","#e67e22"
    return "D 🔴","#e74c3c"


def _grade_higher(v, good, ok):
    """Grade where higher is better."""
    if v >= good: return "A 🟢","#27ae60"
    if v >= ok:   return "B 🟡","#f59e0b"
    if v >= ok*0.85: return "C 🟠","#e67e22"
    return "D 🔴","#e74c3c"


def _stat_row(label, mae, bias, avg_pred, avg_actual, mae_good, mae_ok):
    if mae <= mae_good:       color,badge="#27ae60","✅ On target"
    elif mae <= mae_ok:       color,badge="#f59e0b","⚠️ Close"
    else:                     color,badge="#e74c3c","❌ Off"
    if abs(bias) <= mae_good*0.4: b_label,b_color="On target","#27ae60"
    elif bias > 0:               b_label,b_color=f"Over +{bias:.2f}","#f59e0b"
    else:                        b_label,b_color=f"Under {bias:.2f}","#f59e0b"
    st.markdown(
        f'<div style="background:#111827;border-radius:8px;padding:10px 14px;'
        f'margin-bottom:6px;border-left:3px solid {color};">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-weight:700;color:#e8ecf4;font-size:14px;">{label}</span>'
        f'&nbsp;<span style="font-size:12px;color:{color};font-weight:600;">{badge}</span>'
        f'<div style="text-align:right;">'
        f'<span style="font-size:15px;font-weight:800;color:#e8ecf4;">±{mae:.3f}</span>'
        f'<span style="font-size:11px;color:#6c757d;"> avg error</span></div></div>'
        f'<div style="margin-top:4px;display:flex;gap:16px;flex-wrap:wrap;">'
        f'<span style="font-size:11px;color:{b_color};">{b_label}</span>'
        f'<span style="font-size:11px;color:#6c757d;">Predicted {avg_pred:.2f} · Actual {avg_actual:.2f}</span>'
        f'</div></div>', unsafe_allow_html=True)


def _game_table(game_rows, score_cols):
    """Render game outcomes table."""
    if not game_rows:
        st.info("No game data yet.")
        return
    rows = []
    for g in game_rows:
        row = {
            "Date":    g.get("date",""),
            "Game":    g.get("matchup",""),
            "Score":   score_cols(g),
            "Winner":  g.get("winner",""),
            "We Picked": g.get("pred_winner","—"),
            "✓ Pick":  "✅" if g.get("winner_correct") else "❌" if g.get("winner_correct")==0 else "—",
        }
        ou = g.get("ou_correct")
        our_rec = g.get("our_ou_rec") or g.get("ou_rec","")
        row["O/U Result"] = ("✅ " if ou==1 else "❌ ") + our_rec if ou is not None else "—"
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _player_table_nhl(players):
    if not players:
        st.info("No player data yet.")
        return
    rows = []
    for p in players:
        scored = p.get("scored", False)
        pred   = p.get("pred_prob", 0)
        rows.append({
            "Player":       p["name"],
            "Team":         p.get("team",""),
            "Tier":         p.get("conf",""),
            "We Predicted": f"{pred:.3f} ({pred*100:.0f}%)",
            "Scored?":      "✅ Yes" if scored else "❌ No",
            "Goals":        p.get("goals",0),
            "Pred SOG":     p.get("pred_sog",0),
            "Actual SOG":   p.get("actual_sog",0),
            "SOG Δ":        f"{p.get('actual_sog',0)-p.get('pred_sog',0):+.1f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _player_table_mlb(players):
    if not players:
        st.info("No player data yet.")
        return
    rows = []
    for p in players:
        h_diff = p.get("actual_h",0) - p.get("pred_h",0)
        rows.append({
            "Player":      p["name"],
            "Team":        p.get("team",""),
            "Tier":        p.get("conf",""),
            "Pred H":      p.get("pred_h",0),
            "Actual H":    p.get("actual_h",0),
            "Hit Δ":       f"{h_diff:+.2f}",
            "Pred HR":     p.get("pred_hr",0),
            "Actual HR":   p.get("actual_hr",0),
            "Pred HRR":    p.get("pred_hrr",0),
            "Actual HRR":  p.get("actual_hrr",0),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _player_table_nba(players):
    if not players:
        st.info("No player data yet.")
        return
    rows = []
    for p in players:
        pts_diff = p.get("actual_pts",0) - p.get("pred_pts",0)
        rows.append({
            "Player":      p["name"],
            "Team":        p.get("team",""),
            "Tier":        p.get("conf",""),
            "Pred Pts":    p.get("pred_pts",0),
            "Actual Pts":  p.get("actual_pts",0),
            "Pts Δ":       f"{pts_diff:+.1f}",
            "Pred Reb":    p.get("pred_reb",0),
            "Actual Reb":  p.get("actual_reb",0),
            "Pred Ast":    p.get("pred_ast",0),
            "Actual Ast":  p.get("actual_ast",0),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _history_counts():
    if not HIST_DIR.exists(): return {}
    today = datetime.now(ET).strftime("%Y-%m-%d")
    out = {}
    for sport in ("nhl","mlb","nba"):
        files = sorted(f for f in HIST_DIR.glob(f"{sport}_*.parquet")
                       if f.stem.replace(f"{sport}_","") < today)
        out[sport]={"n":len(files),
                    "dates":[f.stem.replace(f"{sport}_","") for f in files]}
    return out


def _status_check(sport_data):
    agg      = sport_data.get("aggregate",{})
    files    = sport_data.get("files_found",  agg.get("files_found",  0))
    failures = sport_data.get("api_failures", agg.get("api_failures", 0))
    n_rows   = sport_data.get("n_rows",       agg.get("n_rows",       0))
    msg      = sport_data.get("message",      agg.get("message",      ""))
    return agg, files, failures, n_rows, msg


# ── Main ─────────────────────────────────────────────────────────────────────

def render_backtest():
    st.markdown("## 🔬 Backtesting Report")
    st.caption("🟢 A = Great  ·  🟡 B = Good  ·  🟠 C = Okay  ·  🔴 D = Needs work")

    history = _history_counts()
    total   = max((v["n"] for v in history.values()), default=0)

    c1,c2,c3 = st.columns(3)
    for col,sport,emoji in [(c1,"nhl","🏒"),(c2,"mlb","⚾"),(c3,"nba","🏀")]:
        n=history.get(sport,{}).get("n",0)
        dates=history.get(sport,{}).get("dates",[])
        span=f"{dates[0]} → {dates[-1]}" if len(dates)>=2 else (dates[0] if dates else "none yet")
        with col:
            st.metric(f"{emoji} {sport.upper()} Days Saved",n,
                      delta=span if n else "Run workflow to start",
                      delta_color="normal" if n>=7 else "off")

    if total==0:
        st.info("**No history yet.** The workflow saves a snapshot every morning. "
                "Come back after a week of snapshots.")
        return

    st.divider()
    max_days=min(total,90)
    c1,c2=st.columns([3,1])
    with c1:
        days=(max_days if max_days<=1
              else st.slider("Days to analyse",1,max_days,min(14,max_days),1))
        if max_days<=1: st.info(f"Only {max_days} completed day(s) available.")
    with c2:
        run_btn=st.button("▶ Run Backtest",type="primary",use_container_width=True)

    if run_btn:
        with st.spinner("Fetching actual results and comparing…"):
            try:
                import sys
                sys.path.insert(0,str(Path(__file__).resolve().parent.parent.parent))
                from scripts.backtest import run_backtest
                run_backtest(days)
                st.success("Done!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}"); st.exception(e); return

    if not BT_FILE.exists():
        st.info("Click **▶ Run Backtest** to generate the report.")
        return

    try: results=json.loads(BT_FILE.read_text())
    except Exception as e: st.error(f"Could not load: {e}"); return

    for sd in results.values():
        if sd.get("updated_at"):
            dt=datetime.fromisoformat(sd["updated_at"]).astimezone(ET)
            st.caption(f"Last run: {dt.strftime('%I:%M %p ET on %b %d, %Y')}"); break

    st.divider()
    st.markdown("### 🏒 NHL")
    _render_nhl(results.get("nhl",{}))

    st.divider()
    st.markdown("### ⚾ MLB")
    _render_mlb(results.get("mlb",{}))

    st.divider()
    st.markdown("### 🏀 NBA")
    _render_nba(results.get("nba",{}))

    st.divider()
    with st.expander("📖 How to read these results"):
        st.markdown("""
        **Grade scale:** 🟢 A = Great · 🟡 B = Good · 🟠 C = Okay · 🔴 D = Needs work

        | Term | Plain English |
        |------|---------------|
        | **Win/Loss Accuracy** | Did we pick the right team to win? |
        | **Over/Under Accuracy** | Did we correctly call Over or Under on goals/runs/points? |
        | **Goal/Scoring Accuracy** | What % of predicted scorers actually scored? |
        | **Calibration** | When we say 30% chance, does it happen ~30% of the time? |
        | **Brier Score** | 0.00=perfect, 0.25=random guessing |
        | **AUC** | 0.50=coin flip, 0.60+=useful, 0.70+=strong |
        | **Avg Error** | How many hits/goals/points off on average |
        | **Bias** | "Over" = we consistently over-predicted · "Under" = we under-predicted |
        | **Elite ROI** | If you bet $1 on every Elite pick, profit or loss? |
        | **Δ column** | Difference between predicted and actual (+ = we were too high) |
        """)


# ── NHL ───────────────────────────────────────────────────────────────────────

def _render_nhl(nhl):
    agg,files,failures,n_rows,msg = _status_check(nhl)
    if files==0: st.info("No history snapshots yet."); return
    if n_rows==0:
        st.warning(f"Found {files} history file(s) but NHL API blocked {failures}/{files} dates. "
                   "Try: `python scripts/backtest.py --days 7` in the terminal."); return

    st.caption(f"{files} days · {n_rows} player-games · {failures} API failures")
    if msg: st.info(msg)

    # ── Category overview ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Category Accuracy")
    game_stats = agg.get("game_stats",{})
    gs         = agg.get("stats",{}).get("Goal Scoring",{})

    cols = st.columns(4)
    metrics = [
        ("Goals",
         gs.get("overall_accuracy"),
         lambda v: _grade_higher(v,0.20,0.15),
         f"{gs.get('overall_accuracy',0):.1%}",
         "% of predicted scorers who scored",
         "Baseline: ~15% of skaters score any night"),
        ("Win/Loss",
         game_stats.get("Win/Loss Accuracy"),
         lambda v: _grade_higher(v,0.60,0.52),
         f"{game_stats.get('Win/Loss Accuracy',0):.1%}" if game_stats.get("Win/Loss Accuracy") else "—",
         "Did we pick the right winner?",
         "50% = coin flip"),
        ("Over/Under",
         game_stats.get("Over/Under Accuracy"),
         lambda v: _grade_higher(v,0.58,0.52),
         f"{game_stats.get('Over/Under Accuracy',0):.1%}" if game_stats.get("Over/Under Accuracy") else "—",
         "Did we call O/U correctly?",
         "50% = coin flip"),
        ("Goal Total Error",
         game_stats.get("Goal Total Error"),
         lambda v: _grade_lower(v,0.8,1.5),
         f"±{game_stats.get('Goal Total Error',0):.2f}" if game_stats.get("Goal Total Error") else "—",
         "Avg goals off on game total",
         "Lower = better"),
    ]

    for col,(title,val,grade_fn,display,desc,note) in zip(cols,metrics):
        with col:
            if val is not None:
                g,c=grade_fn(val)
                _card(title,display,g,c,desc,note)
            else:
                _card(title,"—","—","#6c757d","No data yet",note)

    # ── Probability quality ────────────────────────────────────────────────────
    bri=gs.get("brier_score",0)
    cal=gs.get("calibration_error",0)
    auc_v=gs.get("auc")

    st.markdown("#### 🎯 Probability Quality")
    c1,c2,c3=st.columns(3)
    with c1:
        g,c=_grade_lower(bri,0.12,0.22,0.08)
        _card("Brier Score (lower=better)",f"{bri:.4f}",g,c,
              "Overall probability accuracy","0.00=perfect · 0.25=random guessing")
    with c2:
        g,c=_grade_lower(cal,0.03,0.10,0.02)
        _card("Calibration (lower=better)",f"{cal:.4f}",g,c,
              "When we say 30%, does it hit 30%?","0.00=perfectly calibrated")
    with c3:
        if auc_v:
            g,c=_grade_higher(auc_v,0.70,0.60)
            _card("AUC — Ranking Quality",f"{auc_v:.3f}",g,c,
                  "Can we rank players correctly?","0.50=coin flip · 0.70+=strong")

    # ── Elite ROI ──────────────────────────────────────────────────────────────
    elite_n=agg.get("elite_n")
    if elite_n:
        r150=agg.get("elite_roi_plus150",0)
        r130=agg.get("elite_roi_plus130",0)
        r110=agg.get("elite_roi_plus110",0)
        ea=agg.get("elite_accuracy",0)
        col=("#27ae60" if r150>0 else "#e74c3c")
        st.markdown("#### ⭐ Elite Picks ROI")
        st.markdown(
            f'<div style="background:#111827;border-radius:10px;padding:14px 18px;'
            f'border:1px solid {col}66;">'
            f'<div style="color:#e8ecf4;font-weight:700;margin-bottom:10px;">'
            f'{elite_n} Elite picks · {ea:.1%} scored</div>'
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
            + "".join([
                f'<div style="text-align:center;background:#1a1f2e;border-radius:8px;padding:10px;">'
                f'<div style="font-size:22px;font-weight:800;color:{col};">{r:+.1%}</div>'
                f'<div style="font-size:11px;color:#6c757d;">$1 bet at {o}</div></div>'
                for r,o in [(r150,"+150"),(r130,"+130"),(r110,"+110")]
            ])
            + '</div></div>', unsafe_allow_html=True)

    # ── Calibration chart ──────────────────────────────────────────────────────
    cal_data=agg.get("calibration_curve",[])
    if len(cal_data)>=3:
        df_c=pd.DataFrame(cal_data)
        fig=go.Figure()
        fig.add_scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect",
                        line=dict(color="#27ae60",dash="dash",width=2))
        fig.add_scatter(x=df_c["avg_predicted"],y=df_c["avg_actual"],
                        mode="lines+markers",name="Our model",
                        marker=dict(size=10,color="#e67e22"),
                        line=dict(color="#e67e22",width=2))
        fig.update_layout(title="Calibration — dots on green line = perfect",
            height=280,plot_bgcolor="#0e1117",paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"),
            xaxis=dict(title="What we predicted",tickformat=".0%"),
            yaxis=dict(title="What actually happened",tickformat=".0%"))
        st.plotly_chart(fig,use_container_width=True)

    # ── Tier chart ─────────────────────────────────────────────────────────────
    tiers=agg.get("tiers",{})
    if tiers:
        st.markdown("#### 🏷 Accuracy by Confidence Tier")
        tnames=list(tiers.keys())
        fig2=go.Figure()
        fig2.add_bar(name="Actually Scored",x=tnames,
                     y=[tiers[t]["accuracy"] for t in tnames],marker_color="#27ae60",
                     text=[f"{tiers[t]['accuracy']:.1%}" for t in tnames],textposition="outside")
        fig2.add_bar(name="We Predicted",x=tnames,
                     y=[tiers[t]["avg_predicted"] for t in tnames],marker_color="#3498db",
                     text=[f"{tiers[t]['avg_predicted']:.1%}" for t in tnames],textposition="outside")
        fig2.update_layout(title="Predicted vs Actual — closer bars = better",
            barmode="group",height=280,plot_bgcolor="#0e1117",paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"),yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig2,use_container_width=True)

    # ── Game results ───────────────────────────────────────────────────────────
    game_rows=agg.get("game_rows",[])
    if game_rows:
        st.markdown("#### 🏒 Game Results vs Our Predictions")
        _game_table(game_rows,
                    lambda g: f"{g['away_score']}–{g['home_score']}")
    else:
        st.caption("Game results: not available (NHL API may have blocked requests)")

    # ── Top 10 players ─────────────────────────────────────────────────────────
    top10=agg.get("top10_players",[])
    if top10:
        st.markdown("#### 🎯 Top 10 Predicted Players vs Actual")
        _player_table_nhl(top10)


# ── MLB ───────────────────────────────────────────────────────────────────────

def _render_mlb(mlb):
    agg,files,failures,n_rows,msg=_status_check(mlb)
    if files==0: st.info("No history snapshots yet."); return
    if n_rows==0:
        st.warning(f"Found {files} history file(s) but 0 players matched. "
                   f"Try: `python scripts/backtest.py --days 7` in terminal."); return

    st.caption(f"{files} days · {n_rows} individual player-game results")
    if msg: st.info(msg)

    # ── Category overview ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Category Accuracy")
    game_stats=agg.get("game_stats",{})
    stats=agg.get("stats",{})
    hr_dir=agg.get("hr_direction_accuracy")

    cols=st.columns(4)
    metrics=[
        ("Win/Loss",
         game_stats.get("Win/Loss Accuracy"),
         lambda v:_grade_higher(v,0.60,0.52),
         f"{game_stats.get('Win/Loss Accuracy',0):.1%}" if game_stats.get("Win/Loss Accuracy") else "—",
         "Did we pick the right winner?","50%=coin flip"),
        ("Over/Under Runs",
         game_stats.get("Over/Under Accuracy"),
         lambda v:_grade_higher(v,0.58,0.52),
         f"{game_stats.get('Over/Under Accuracy',0):.1%}" if game_stats.get("Over/Under Accuracy") else "—",
         "Did we call O/U correctly?","50%=coin flip"),
        ("HR Yes/No",
         hr_dir,
         lambda v:_grade_higher(v,0.68,0.55),
         f"{hr_dir:.1%}" if hr_dir else "—",
         "Correctly called HR vs no HR","50%=coin flip"),
        ("Run Total Error",
         game_stats.get("Run Total Error"),
         lambda v:_grade_lower(v,1.5,3.0),
         f"±{game_stats.get('Run Total Error',0):.2f}" if game_stats.get("Run Total Error") else "—",
         "Avg runs off on game total","Lower=better"),
    ]
    for col,(title,val,grade_fn,display,desc,note) in zip(cols,metrics):
        with col:
            if val is not None:
                g,c=grade_fn(val); _card(title,display,g,c,desc,note)
            else:
                _card(title,"—","—","#6c757d","No data yet","")

    # ── Player stat accuracy ───────────────────────────────────────────────────
    st.markdown("#### 📈 Player Stat Accuracy")
    thresholds={"Hits":(0.30,0.50),"Home Runs":(0.05,0.10),
                "RBI":(0.40,0.70),"Runs":(0.40,0.70),"H+R+RBI":(0.70,1.20)}
    for label,v in stats.items():
        good,ok=thresholds.get(label,(0.35,0.60))
        _stat_row(label,v.get("mae",0),v.get("bias",0),
                  v.get("avg_pred",0),v.get("avg_actual",0),good,ok)

    # ── HR quality ─────────────────────────────────────────────────────────────
    hr_bri=agg.get("hr_brier")
    if hr_dir or hr_bri:
        c1,c2=st.columns(2)
        with c1:
            if hr_dir:
                g,c=_grade_higher(hr_dir,0.70,0.55)
                _card("Home Run Yes/No Accuracy",f"{hr_dir:.1%}",g,c,
                      "% correct on HR vs no-HR call","50%=random guessing")
        with c2:
            if hr_bri:
                g,c=_grade_lower(hr_bri,0.07,0.13)
                _card("HR Probability Quality (lower=better)",f"{hr_bri:.4f}",g,c,
                      "How good are the HR probabilities?","0.00=perfect · 0.25=random")

    # ── Tier table ─────────────────────────────────────────────────────────────
    tiers=agg.get("tiers",{})
    if tiers:
        st.markdown("#### 🏷 Hit Accuracy by Confidence Tier")
        rows=[]
        for t,v in tiers.items():
            mae=v.get("mae_h",0)
            quality=("✅ Accurate" if mae<=0.30 else "⚠️ Decent" if mae<=0.50 else "❌ Off")
            bias=v.get("bias_h",0)
            tend=("Over ⬆️" if bias>0.05 else "Under ⬇️" if bias<-0.05 else "On target ✅")
            rows.append({"Tier":t,"Picks":v["n"],"Quality":quality,
                         "Avg Hit Error":f"±{mae:.3f}",
                         "We Predicted":f"{v.get('avg_pred_h',0):.2f}",
                         "Actual":f"{v.get('avg_actual_h',0):.2f}",
                         "Tendency":tend})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

    # ── Game results ───────────────────────────────────────────────────────────
    game_rows=agg.get("game_rows",[])
    if game_rows:
        st.markdown("#### ⚾ Game Results vs Our Predictions")
        _game_table(game_rows,
                    lambda g: f"{g['away_runs']}–{g['home_runs']}")

    # ── Top 10 ─────────────────────────────────────────────────────────────────
    top10=agg.get("top10_players",[])
    if top10:
        st.markdown("#### 🎯 Top 10 Predicted Batters vs Actual")
        _player_table_mlb(top10)


# ── NBA ───────────────────────────────────────────────────────────────────────

def _render_nba(nba):
    agg,files,failures,n_rows,msg=_status_check(nba)
    if files==0: st.info("No history snapshots yet."); return
    if n_rows==0:
        st.warning(f"Found {files} history file(s) but 0 players matched."); return

    st.caption(f"{files} days · {n_rows} individual player-game results")

    # ── Category overview ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Category Accuracy")
    game_stats=agg.get("game_stats",{})
    dd_acc=agg.get("dd_accuracy")

    cols=st.columns(4)
    metrics=[
        ("Win/Loss",
         game_stats.get("Win/Loss Accuracy"),
         lambda v:_grade_higher(v,0.62,0.52),
         f"{game_stats.get('Win/Loss Accuracy',0):.1%}" if game_stats.get("Win/Loss Accuracy") else "—",
         "Did we pick the right winner?","50%=coin flip"),
        ("Over/Under Pts",
         game_stats.get("Over/Under Accuracy"),
         lambda v:_grade_higher(v,0.58,0.52),
         f"{game_stats.get('Over/Under Accuracy',0):.1%}" if game_stats.get("Over/Under Accuracy") else "—",
         "Did we call O/U correctly?","50%=coin flip"),
        ("Double-Double",
         dd_acc,
         lambda v:_grade_higher(v,0.70,0.55),
         f"{dd_acc:.1%}" if dd_acc else "—",
         "Correctly predicted DD vs none","50%=coin flip"),
        ("Score Total Error",
         game_stats.get("Score Total Error"),
         lambda v:_grade_lower(v,8.0,15.0),
         f"±{game_stats.get('Score Total Error',0):.1f}" if game_stats.get("Score Total Error") else "—",
         "Avg points off on game total","Lower=better"),
    ]
    for col,(title,val,grade_fn,display,desc,note) in zip(cols,metrics):
        with col:
            if val is not None:
                g,c=grade_fn(val); _card(title,display,g,c,desc,note)
            else:
                _card(title,"—","—","#6c757d","No data yet","")

    # ── Player stat accuracy ───────────────────────────────────────────────────
    st.markdown("#### 📈 Player Stat Accuracy")
    thresholds={"Points":(3.0,6.0),"Rebounds":(2.0,4.0),
                "Assists":(1.5,3.0),"3-Pointers":(0.5,1.0)}
    for label,v in agg.get("stats",{}).items():
        good,ok=thresholds.get(label,(2.0,4.0))
        _stat_row(label,v.get("mae",0),v.get("bias",0),
                  v.get("avg_pred",0),v.get("avg_actual",0),good,ok)

    # ── Tier table ─────────────────────────────────────────────────────────────
    tiers=agg.get("tiers",{})
    if tiers:
        st.markdown("#### 🏷 Points Accuracy by Confidence Tier")
        rows=[]
        for t,v in tiers.items():
            mae=v.get("mae",0)
            quality=("✅ Accurate" if mae<=4 else "⚠️ Decent" if mae<=7 else "❌ Off")
            rows.append({"Tier":t,"Picks":v["n"],"Quality":quality,
                         "Avg Error":f"±{mae:.1f} pts",
                         "We Predicted":f"{v.get('avg_predicted',0):.1f}",
                         "Actual":f"{v.get('avg_actual',0):.1f}"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

    # ── Game results ───────────────────────────────────────────────────────────
    game_rows=agg.get("game_rows",[])
    if game_rows:
        st.markdown("#### 🏀 Game Results vs Our Predictions")
        def nba_score(g):
            return f"{g.get('away_pts',0)}–{g.get('home_pts',0)}"
        _game_table(game_rows,nba_score)

    # ── Top 10 ─────────────────────────────────────────────────────────────────
    top10=agg.get("top10_players",[])
    if top10:
        st.markdown("#### 🎯 Top 10 Predicted Players vs Actual")
        _player_table_nba(top10)
