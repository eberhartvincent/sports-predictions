"""app/pages/backtest_page.py — Admin backtesting report (all three sports)"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

ET       = ZoneInfo("America/New_York")
PRED_DIR = Path("data/cache/predictions")
HIST_DIR = PRED_DIR / "history"
BT_FILE  = PRED_DIR / "backtest_results.json"


def _history_counts():
    if not HIST_DIR.exists(): return {}
    counts = {}
    for sport in ("nhl","mlb","nba"):
        files = sorted(HIST_DIR.glob(f"{sport}_*.parquet"))
        counts[sport] = {
            "n":     len(files),
            "dates": [f.stem.replace(f"{sport}_","") for f in files],
        }
    return counts


def _metric_card(col, label, value, delta=None, delta_color="normal", help=None):
    with col:
        st.metric(label, value, delta=delta, delta_color=delta_color, help=help)


def _color_bar(val, baseline, higher_better=True):
    """Return colored value based on whether it beats baseline."""
    if val is None: return "—"
    better = (val > baseline) if higher_better else (val < baseline)
    color  = "green" if better else "red"
    return f":{color}[**{val}**]"


def render_backtest():
    st.markdown("## 🔬 Backtesting Report")
    st.caption("Compares pre-game predictions against actual results to measure model quality.")

    history = _history_counts()
    total   = max((v["n"] for v in history.values()), default=0)

    # ── History status ────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    for col, sport, emoji in [(m1,"nhl","🏒"),(m2,"mlb","⚾"),(m3,"nba","🏀")]:
        n     = history.get(sport,{}).get("n",0)
        dates = history.get(sport,{}).get("dates",[])
        rng   = f"{dates[0]} → {dates[-1]}" if len(dates)>=2 else (dates[0] if dates else "none")
        _metric_card(col, f"{emoji} {sport.upper()} Snapshots", n,
                     delta=rng if n else "No history yet",
                     delta_color="normal" if n>=7 else "inverse")

    if total == 0:
        st.info("""
        **No history yet.** The backtest engine needs daily prediction snapshots
        to compare against actual results. These are saved automatically every time
        the GitHub Actions warm_cache workflow runs.

        Come back after 7+ days of workflow runs for meaningful results.
        """)
        st.divider()
        st.markdown("### What this will show once ready")
        st.markdown("""
        | Sport | Metric | What it measures |
        |-------|--------|-----------------|
        | NHL | **Accuracy by tier** | % of Elite/High picks who actually scored |
        | NHL | **Calibration (ECE)** | Does a 30% pred hit ~30% of the time? |
        | NHL | **Brier score** | Probability quality (lower = better, 0.25 = random) |
        | NHL | **AUC** | Model discrimination ability (0.5 = random, 1.0 = perfect) |
        | NHL | **Elite ROI at +150/+130/+110** | Simulated betting return |
        | NHL | **Calibration curve** | Visual pred vs actual across probability buckets |
        | MLB | **MAE per stat** | Mean absolute error on H, HR, RBI, R, TB, H+R+RBI |
        | MLB | **Bias** | Systematic over/under-prediction per stat |
        | MLB | **HR prob calibration** | How accurate is the HR probability specifically |
        | MLB | **HR direction accuracy** | % correct on HR vs no HR |
        | NBA | **MAE per stat** | Error on Pts, Reb, Ast, 3PM, Stocks |
        | NBA | **DD calibration** | Double-double probability accuracy |
        | All | **Per-tier breakdown** | Elite/High/Medium/Low accuracy separately |
        """)
        return

    # ── Run controls ──────────────────────────────────────────────────────────
    st.divider()
    max_days = min(total, 90)
    c1, c2 = st.columns([3, 1])
    with c1:
        if max_days <= 1:
            days = max_days
            st.info(f"Only {max_days} day(s) available.")
        else:
            days = st.slider("Days to analyse", 1, max_days, min(30,max_days), 1)
    with c2:
        run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Running backtest ({days} days)…"):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
                from scripts.backtest import run_backtest
                run_backtest(days)
                st.success("Complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}"); st.exception(e); return

    if not BT_FILE.exists():
        st.info("Click **▶ Run Backtest** to generate the report.")
        return

    try:
        results = json.loads(BT_FILE.read_text())
    except Exception as e:
        st.error(f"Could not load results: {e}"); return

    # Show last run time
    for sd in results.values():
        if sd.get("updated_at"):
            dt = datetime.fromisoformat(sd["updated_at"]).astimezone(ET)
            st.caption(f"Last run: {dt.strftime('%I:%M %p ET on %b %d, %Y')}")
            break

    # ── NHL ───────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏒 NHL — Goal Probability Model")
    _render_nhl(results.get("nhl",{}))

    # ── MLB ───────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚾ MLB — Batter Projection Model")
    _render_mlb(results.get("mlb",{}))

    # ── NBA ───────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏀 NBA — Player Props Model")
    _render_nba(results.get("nba",{}))

    # ── Methodology ───────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📖 Methodology & Interpretation Guide"):
        st.markdown("""
        **How results are gathered:**
        Each day the workflow saves prediction snapshots to `data/cache/predictions/history/`.
        The backtest fetches actual game results from official APIs and matches by player ID.

        **Key metrics explained:**

        | Metric | Good | Concerning | What it means |
        |--------|------|------------|----------------|
        | **ECE** | < 0.05 | > 0.10 | Calibration error. How far off are the probabilities? |
        | **Brier** | < 0.15 | > 0.22 | Probability accuracy. 0.25 = random guessing |
        | **AUC** | > 0.60 | < 0.52 | Ranking quality. 0.50 = random, 1.0 = perfect |
        | **Elite ROI** | > 0% | < -10% | Simulated profit per $1 bet on Elite picks |
        | **MAE Hits** | < 0.35 | > 0.60 | Average miss on hit projections |
        | **Bias** | near 0 | > ±0.15 | Systematic over (+) or under (-) prediction |
        | **HR Direction** | > 60% | < 52% | Correct on HR vs no-HR binary call |
        | **DD Acc** | > 65% | < 52% | Correct on double-double vs none |

        **Calibration curve:** Ideal is a diagonal line (predicted = actual).
        Curve above diagonal = under-confident. Below = over-confident.

        **Caveats:**
        - Needs 50+ picks per tier for stable statistics
        - Sports have irreducible randomness — even perfect models miss often
        - ROI estimates use fixed illustrative odds, not actual market lines
        - Past performance is not predictive of future results
        """)


def _render_nhl(nhl: dict):
    if nhl.get("message") or nhl.get("error"):
        st.info(nhl.get("message") or nhl.get("error")); return

    agg = nhl.get("aggregate",{})
    if not agg:
        st.info("No NHL data processed yet."); return

    # Summary metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    _metric_card(c1,"Games",     agg.get("n_games","—"))
    _metric_card(c2,"Accuracy",  f"{agg.get('overall_accuracy',0):.1%}")
    _metric_card(c3,"Brier",     f"{agg.get('overall_brier',0):.4f}",
                 help="Lower = better. 0.25 = random")
    _metric_card(c4,"AUC",       f"{agg.get('overall_auc',0):.3f}",
                 help="0.50 = random, 1.0 = perfect")
    _metric_card(c5,"ECE",       f"{agg.get('overall_ece',0):.4f}",
                 help="Calibration error. 0 = perfect")

    # Elite metrics
    if agg.get("elite_n"):
        st.markdown(f"**Elite picks:** {agg['elite_n']} total, "
                    f"{agg.get('elite_accuracy',0):.1%} scored — "
                    f"ROI at +150: `{agg.get('elite_roi_plus150',0):+.1%}` | "
                    f"+130: `{agg.get('elite_roi_plus130',0):+.1%}` | "
                    f"+110: `{agg.get('elite_roi_plus110',0):+.1%}`")

    # SOG accuracy
    if agg.get("sog_mae"):
        st.caption(f"Shot projection MAE: {agg['sog_mae']:.2f} shots/game")

    # Tier table
    tiers = agg.get("tiers",{})
    if tiers:
        tier_rows = []
        for t, v in tiers.items():
            tier_rows.append({
                "Tier":        t,
                "Picks":       v["n"],
                "Accuracy":    f"{v['accuracy']:.1%}",
                "Avg Pred":    f"{v['avg_pred_prob']:.3f}",
                "ECE":         f"{v['ece']:.4f}",
                "Brier":       f"{v['brier']:.4f}",
                "AUC":         f"{v.get('auc',0):.3f}",
            })
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

        # Accuracy vs predicted chart
        if len(tiers) >= 2:
            tnames = list(tiers.keys())
            fig = go.Figure()
            fig.add_bar(name="Actual %", x=tnames,
                        y=[tiers[t]["accuracy"] for t in tnames],
                        marker_color="#27ae60",
                        text=[f"{tiers[t]['accuracy']:.1%}" for t in tnames],
                        textposition="outside")
            fig.add_bar(name="Predicted %", x=tnames,
                        y=[tiers[t]["avg_pred_prob"] for t in tnames],
                        marker_color="#3498db",
                        text=[f"{tiers[t]['avg_pred_prob']:.1%}" for t in tnames],
                        textposition="outside")
            fig.update_layout(title="NHL: Predicted vs Actual Goal Rate by Tier",
                barmode="group", height=320,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e8ecf4"),
                yaxis=dict(tickformat=".0%"))
            st.plotly_chart(fig, use_container_width=True)

    # Calibration curve
    cal = agg.get("calibration_curve",[])
    if len(cal) >= 3:
        df_cal = pd.DataFrame(cal)
        fig2 = go.Figure()
        fig2.add_scatter(x=df_cal["avg_pred"], y=df_cal["avg_actual"],
                         mode="lines+markers", name="Model",
                         marker=dict(size=8, color="#e67e22"),
                         line=dict(color="#e67e22", width=2))
        fig2.add_scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect",
                         line=dict(color="#27ae60", dash="dash"))
        fig2.update_layout(title="Calibration Curve (ideal = diagonal)",
            height=320, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"),
            xaxis=dict(title="Predicted Prob", tickformat=".0%"),
            yaxis=dict(title="Actual Rate",    tickformat=".0%"))
        st.plotly_chart(fig2, use_container_width=True)


def _render_mlb(mlb: dict):
    if mlb.get("message") or mlb.get("error"):
        st.info(mlb.get("message") or mlb.get("error")); return

    agg = mlb.get("aggregate",{})
    if not agg:
        st.info("No MLB data processed yet."); return

    c1,c2,c3 = st.columns(3)
    _metric_card(c1,"Games",   agg.get("n_games","—"))
    _metric_card(c2,"Players", agg.get("n_rows","—"))
    if agg.get("hr_auc"):
        _metric_card(c3,"HR Prob AUC", f"{agg['hr_auc']:.3f}",
                     help="HR probability model quality")

    # Overall MAE table
    stats = [("Hits","h"),("HR","hr"),("RBI","rbi"),("Runs","runs"),
             ("TB","tb"),("K","k"),("H+R+RBI","hrr")]
    mae_rows = []
    for label, key in stats:
        mae  = agg.get(f"mae_{key}")
        bias = agg.get(f"bias_{key}")
        rmse = agg.get(f"rmse_{key}")
        if mae is not None:
            mae_rows.append({
                "Stat":   label,
                "MAE":    f"{mae:.3f}",
                "Bias":   f"{bias:+.3f}" if bias is not None else "—",
                "RMSE":   f"{rmse:.3f}" if rmse is not None else "—",
            })
    if mae_rows:
        st.markdown("**Overall projection accuracy:**")
        st.dataframe(pd.DataFrame(mae_rows), use_container_width=True, hide_index=True)

    # HR calibration
    col1, col2, col3 = st.columns(3)
    if agg.get("hr_brier"):
        _metric_card(col1,"HR Brier",    f"{agg['hr_brier']:.4f}")
    if agg.get("hr_ece"):
        _metric_card(col2,"HR ECE",      f"{agg['hr_ece']:.4f}")
    if agg.get("hr_direction_acc"):
        _metric_card(col3,"HR Direction",f"{agg['hr_direction_acc']:.1%}")

    # Tier breakdown
    tiers = agg.get("tiers",{})
    if tiers:
        st.markdown("**By confidence tier:**")
        tier_rows = []
        for t, v in tiers.items():
            tier_rows.append({
                "Tier":             t,
                "Picks":            v["n"],
                "MAE H":            v.get("mae_h","—"),
                "MAE HRR":          v.get("mae_hrr","—"),
                "Avg Pred H":       v.get("avg_pred_h","—"),
                "Avg Actual H":     v.get("avg_actual_h","—"),
                "Avg Pred HRR":     v.get("avg_pred_hrr","—"),
                "Avg Actual HRR":   v.get("avg_actual_hrr","—"),
                "Bias H":           f'{v.get("bias_h",0):+.3f}' if isinstance(v.get("bias_h"),float) else "—",
            })
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

    # MAE bar chart
    if mae_rows:
        labels = [r["Stat"] for r in mae_rows]
        maes   = [float(r["MAE"]) for r in mae_rows]
        biases = [float(r["Bias"].replace("+","")) for r in mae_rows if r["Bias"]!="—"]
        fig = go.Figure()
        fig.add_bar(name="MAE", x=labels, y=maes,
                    marker_color="#3498db",
                    text=[f"{v:.3f}" for v in maes], textposition="outside")
        fig.update_layout(title="MLB: Mean Absolute Error per Stat",
            height=320, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"))
        st.plotly_chart(fig, use_container_width=True)

    # Hit calibration
    hit_cal = agg.get("hit_calibration_curve",[])
    if len(hit_cal) >= 2:
        df_cal = pd.DataFrame(hit_cal)
        fig2 = go.Figure()
        fig2.add_bar(name="Avg Predicted H", x=df_cal["bin"],
                     y=df_cal["avg_pred_h"], marker_color="#27ae60")
        fig2.add_bar(name="Avg Actual H",    x=df_cal["bin"],
                     y=df_cal["avg_actual_h"], marker_color="#e67e22")
        fig2.update_layout(title="Hit Prediction Calibration by Bucket",
            barmode="group", height=300,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"))
        st.plotly_chart(fig2, use_container_width=True)


def _render_nba(nba: dict):
    if nba.get("message") or nba.get("error"):
        st.info(nba.get("message") or nba.get("error")); return

    agg = nba.get("aggregate",{})
    if not agg:
        st.info("No NBA data processed yet."); return

    c1,c2,c3,c4 = st.columns(4)
    _metric_card(c1,"Games",   agg.get("n_games","—"))
    _metric_card(c2,"Players", agg.get("n_rows","—"))
    if agg.get("dd_ece"):
        _metric_card(c3,"DD ECE",   f"{agg['dd_ece']:.4f}")
    if agg.get("dd_direction_acc"):
        _metric_card(c4,"DD Acc",   f"{agg['dd_direction_acc']:.1%}")

    # MAE table
    stats = [("Points","pts"),("Rebounds","reb"),("Assists","ast"),
             ("3-Pointers","fg3m"),("Stk+Blk","stocks")]
    mae_rows = []
    for label, key in stats:
        mae  = agg.get(f"mae_{key}")
        bias = agg.get(f"bias_{key}")
        if mae is not None:
            mae_rows.append({
                "Stat": label,
                "MAE":  f"{mae:.3f}",
                "Bias": f"{bias:+.3f}" if bias is not None else "—",
            })
    if mae_rows:
        st.dataframe(pd.DataFrame(mae_rows), use_container_width=True, hide_index=True)

    # Tier breakdown
    tiers = agg.get("tiers",{})
    if tiers:
        tier_rows = []
        for t, v in tiers.items():
            tier_rows.append({
                "Tier":          t,
                "Picks":         v["n"],
                "MAE Pts":       v.get("mae_pts","—"),
                "Avg Pred Pts":  v.get("avg_pred_pts","—"),
                "Avg Actual Pts":v.get("avg_actual_pts","—"),
                "MAE Reb":       v.get("mae_reb","—"),
                "MAE Ast":       v.get("mae_ast","—"),
            })
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

    if mae_rows:
        labels = [r["Stat"] for r in mae_rows]
        maes   = [float(r["MAE"]) for r in mae_rows]
        fig = go.Figure()
        fig.add_bar(name="MAE", x=labels, y=maes, marker_color="#9b59b6",
                    text=[f"{v:.2f}" for v in maes], textposition="outside")
        fig.update_layout(title="NBA: Mean Absolute Error per Stat",
            height=300, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"))
        st.plotly_chart(fig, use_container_width=True)
