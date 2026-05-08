"""app/pages/backtest_page.py — Admin backtesting report"""

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

CONF_COLORS = {
    "Elite": "#c0392b",
    "High":  "#e67e22",
    "Medium":"#2980b9",
    "Low":   "#6c757d",
}


def _history_counts():
    if not HIST_DIR.exists():
        return {}
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    counts = {}
    for sport in ("nhl","mlb","nba"):
        files = sorted(
            f for f in HIST_DIR.glob(f"{sport}_*.parquet")
            if f.stem.replace(f"{sport}_","") < today
        )
        counts[sport] = {
            "n":     len(files),
            "dates": [f.stem.replace(f"{sport}_","") for f in files],
        }
    return counts


def _metric(col, label, value, help=None, good=None, bad=None):
    """Render a metric with colour coding when thresholds provided."""
    delta = None; delta_color = "off"
    if good is not None and isinstance(value, float):
        if value >= good:
            delta = "good"; delta_color = "normal"
        elif bad is not None and value <= bad:
            delta = "needs improvement"; delta_color = "inverse"
    with col:
        st.metric(label, value, delta=delta, delta_color=delta_color, help=help)


def render_backtest():
    st.markdown("## 🔬 Backtesting Report")
    st.caption("How accurate were yesterday's predictions? Compares pre-game picks against actual results.")

    history = _history_counts()
    total   = max((v["n"] for v in history.values()), default=0)

    # History status
    c1, c2, c3 = st.columns(3)
    for col, sport, emoji in [(c1,"nhl","🏒"),(c2,"mlb","⚾"),(c3,"nba","🏀")]:
        n     = history.get(sport,{}).get("n", 0)
        dates = history.get(sport,{}).get("dates",[])
        span  = f"{dates[0]} → {dates[-1]}" if len(dates)>=2 else (dates[0] if dates else "none yet")
        with col:
            st.metric(f"{emoji} {sport.upper()} Days Saved", n,
                      delta=span if n else "Run workflow to start",
                      delta_color="normal" if n >= 7 else "off")

    if total == 0:
        st.info("""
        **No history yet — totally normal if you just set this up.**

        Every morning at 11 AM the GitHub Actions workflow runs and saves that day's
        predictions as a snapshot. After a week of snapshots, come back here and
        click **Run Backtest** to see how the model is performing.
        """)
        st.markdown("### What you'll see once you have data")
        st.markdown("""
        | Metric | Plain English |
        |--------|---------------|
        | **Accuracy** | Out of every 10 Elite picks, how many actually happened? |
        | **Calibration error** | When we say 30% chance, does it happen ~30% of the time? |
        | **Brier score** | Overall probability quality — lower is better, 0.25 = random guessing |
        | **AUC** | Can the model rank players correctly? 0.5 = coin flip, 1.0 = perfect |
        | **Elite ROI** | If you bet $1 on every Elite pick, did you make money? |
        | **MAE** | On average, how many hits/goals/points off were we? |
        | **Bias** | Do we consistently over-predict or under-predict? |
        """)
        return

    # Run controls
    st.divider()
    max_days = min(total, 90)
    c1, c2 = st.columns([3,1])
    with c1:
        if max_days <= 1:
            days = max_days
            st.info(f"Only {max_days} completed day(s) of history available.")
        else:
            days = st.slider("How many past days to analyse", 1, max_days,
                             min(14, max_days), 1)
    with c2:
        run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Fetching actual results and comparing against predictions…"):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
                from scripts.backtest import run_backtest
                run_backtest(days)
                st.success("Done!")
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

    for sd in results.values():
        if sd.get("updated_at"):
            dt = datetime.fromisoformat(sd["updated_at"]).astimezone(ET)
            st.caption(f"Last run: {dt.strftime('%I:%M %p ET on %b %d, %Y')}")
            break

    # ── NHL ───────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏒 NHL — Goal Predictions")
    _render_nhl(results.get("nhl", {}))

    # ── MLB ───────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚾ MLB — Batter Predictions")
    _render_mlb(results.get("mlb", {}))

    # ── NBA ───────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏀 NBA — Player Predictions")
    _render_nba(results.get("nba", {}))

    st.divider()
    with st.expander("📖 How to read these results"):
        st.markdown("""
        **Accuracy** — The % of predictions that were correct.
        Elite picks should be right more often than the baseline (NHL ~15%, MLB ~25%, NBA varies).

        **Calibration error** — If the model says "30% chance", does it happen about 30% of the time?
        A value near 0 means yes. A value near 0.10 means the probabilities are off by ~10 percentage points.

        **Brier score** — A single number for overall probability quality.
        - 0.00 = perfect predictions  
        - 0.25 = pure random guessing  
        - Lower is always better

        **AUC** — Can the model correctly rank higher-probability players above lower ones?
        - 0.50 = no better than random  
        - 0.60+ = genuinely useful  
        - 0.70+ = strong

        **Elite ROI at +150** — If you bet $1 on every Elite pick at +150 American odds,
        what would your return be? Positive = profit, negative = loss.

        **MAE (Mean Absolute Error)** — How many hits/goals/points off was the projection on average.
        Smaller is better. For MLB hits, < 0.35 is good.

        **Bias** — Positive bias = we over-predicted. Negative = under-predicted.
        Close to 0 is ideal.

        **Calibration curve** — The diagonal chart. If dots fall on the diagonal line,
        predictions are perfectly calibrated. Above = under-confident. Below = over-confident.
        """)


def _status_check(sport_data: dict) -> tuple:
    """Returns (agg, files, failures, msg, ok) — ok=True if we have data to show."""
    agg      = sport_data.get("aggregate", {})
    files    = sport_data.get("files_found",  agg.get("files_found",  0))
    failures = sport_data.get("api_failures", agg.get("api_failures", 0))
    n_rows   = sport_data.get("n_rows",       agg.get("n_rows",       0))
    msg      = sport_data.get("message",      agg.get("message",      ""))
    return agg, files, failures, n_rows, msg


def _render_nhl(nhl: dict):
    agg, files, failures, n_rows, msg = _status_check(nhl)

    if files == 0:
        st.info("No history snapshots yet."); return
    if n_rows == 0:
        st.warning(
            f"Found {files} history file(s) but the NHL API blocked requests "
            f"({failures}/{files} dates). The NHL API (api-web.nhle.com) sometimes "
            "blocks cloud IP ranges. Run `python scripts/backtest.py --days 7` "
            "from the Codespaces terminal for better results."
        )
        if msg: st.caption(msg)
        return

    st.caption(f"Based on {files} days · {n_rows} player-games · "
               f"{failures} API failures")
    if msg: st.info(msg)

    stats = agg.get("stats", {}).get("Goal Scoring", {})
    c1,c2,c3,c4 = st.columns(4)
    _metric(c1, "Overall Accuracy",
            f"{stats.get('overall_accuracy',0):.1%}",
            help="What % of all predicted scorers actually scored")
    _metric(c2, "Brier Score",
            f"{stats.get('brier_score',0):.4f}",
            help="Probability quality. 0.00=perfect, 0.25=random. Lower=better")
    _metric(c3, "Calibration Error",
            f"{stats.get('calibration_error',0):.4f}",
            help="How far off are the probabilities on average. Closer to 0=better")
    if stats.get("auc"):
        _metric(c4, "AUC",
                f"{stats.get('auc',0):.3f}",
                help="Ranking quality. 0.50=random, 0.60+=useful, 0.70+=strong")

    elite = agg.get("elite_n")
    if elite:
        r150 = agg.get("elite_roi_plus150", 0)
        rc   = "normal" if r150 > 0 else "inverse"
        st.markdown(
            f"**Elite picks:** {elite} total · "
            f"Accuracy: **{agg.get('elite_accuracy',0):.1%}** · "
            f"ROI at +150 odds: **{r150:+.1%}** · "
            f"+130: **{agg.get('elite_roi_plus130',0):+.1%}** · "
            f"+110: **{agg.get('elite_roi_plus110',0):+.1%}**"
        )

    if agg.get("stats",{}).get("Shots on Goal"):
        sog_mae = agg["stats"]["Shots on Goal"]["mae"]
        st.caption(f"Shot projection error: ±{sog_mae:.2f} shots/game on average")

    tiers = agg.get("tiers", {})
    if tiers:
        st.markdown("**Accuracy by confidence tier:**")
        rows = []
        for t, v in tiers.items():
            rows.append({
                "Tier":             t,
                "Picks":            v["n"],
                "Scored %":         f"{v['accuracy']:.1%}",
                "We Predicted %":   f"{v['avg_predicted']:.1%}",
                "Calibration Off":  f"{v['calibration_error']:.4f}",
                "Brier Score":      f"{v['brier_score']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        tnames = list(tiers.keys())
        fig = go.Figure()
        fig.add_bar(name="Actually Scored %", x=tnames,
                    y=[tiers[t]["accuracy"] for t in tnames],
                    marker_color="#27ae60",
                    text=[f"{tiers[t]['accuracy']:.1%}" for t in tnames],
                    textposition="outside")
        fig.add_bar(name="We Predicted %", x=tnames,
                    y=[tiers[t]["avg_predicted"] for t in tnames],
                    marker_color="#3498db",
                    text=[f"{tiers[t]['avg_predicted']:.1%}" for t in tnames],
                    textposition="outside")
        fig.update_layout(
            title="Predicted vs Actual Goal Rate by Tier",
            barmode="group", height=320,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"),
            yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)

    cal = agg.get("calibration_curve", [])
    if len(cal) >= 3:
        df_cal = pd.DataFrame(cal)
        fig2 = go.Figure()
        fig2.add_scatter(x=df_cal["avg_predicted"], y=df_cal["avg_actual"],
                         mode="lines+markers", name="Model",
                         marker=dict(size=8, color="#e67e22"),
                         text=[f"n={r['n']}<br>{r['bucket']}" for _,r in df_cal.iterrows()],
                         line=dict(color="#e67e22", width=2))
        fig2.add_scatter(x=[0,1], y=[0,1], mode="lines",
                         name="Perfect (predicted = actual)",
                         line=dict(color="#27ae60", dash="dash"))
        fig2.update_layout(
            title="Calibration Curve — dots on the line = perfectly calibrated",
            height=320, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"),
            xaxis=dict(title="What we predicted", tickformat=".0%"),
            yaxis=dict(title="What actually happened", tickformat=".0%"))
        st.plotly_chart(fig2, use_container_width=True)


def _render_mlb(mlb: dict):
    agg, files, failures, n_rows, msg = _status_check(mlb)

    if files == 0:
        st.info("No history snapshots yet."); return
    if n_rows == 0:
        st.warning(f"Found {files} history file(s) but 0 players matched. "
                   f"API failures: {failures}/{files}. "
                   "Try: `python scripts/backtest.py --days 7` in the terminal.")
        if msg: st.caption(msg)
        return

    st.caption(f"Based on {files} days · {n_rows} player-games")
    if msg: st.info(msg)

    # Stat accuracy table
    stats = agg.get("stats", {})
    if stats:
        st.markdown("**Projection accuracy (how many off on average):**")
        rows = []
        for label, v in stats.items():
            mae  = v.get("mae",  0)
            bias = v.get("bias", 0)
            rows.append({
                "Stat":             label,
                "Avg Error (MAE)":  f"±{mae:.3f}",
                "Bias":             f"{'Over' if bias>0.05 else 'Under' if bias<-0.05 else 'On target'} ({bias:+.3f})",
                "Avg Predicted":    f"{v.get('avg_pred',0):.3f}",
                "Avg Actual":       f"{v.get('avg_actual',0):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    if agg.get("hr_direction_accuracy"):
        _metric(c1, "HR Direction Accuracy",
                f"{agg['hr_direction_accuracy']:.1%}",
                help="% of time we correctly predicted HR vs no HR")
    if agg.get("hr_brier"):
        _metric(c2, "HR Probability Quality",
                f"{agg['hr_brier']:.4f}",
                help="Brier score for HR probability. Lower=better")

    tiers = agg.get("tiers", {})
    if tiers:
        st.markdown("**By confidence tier:**")
        rows = []
        for t, v in tiers.items():
            rows.append({
                "Tier":              t,
                "Picks":             v["n"],
                "Avg Hit Error":     f"±{v.get('mae_h',   0):.3f}",
                "Avg H+R+RBI Error": f"±{v.get('mae_hrr', 0):.3f}",
                "Avg HR Error":      f"±{v.get('mae_hr',  0):.3f}",
                "Pred Hits":         f"{v.get('avg_pred_h',   0):.3f}",
                "Actual Hits":       f"{v.get('avg_actual_h', 0):.3f}",
                "Hit Bias":          f"{'Over' if v.get('bias_h',0)>0.05 else 'Under' if v.get('bias_h',0)<-0.05 else 'On target'}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    cal = agg.get("calibration_curve", [])
    if len(cal) >= 2:
        df_cal = pd.DataFrame(cal)
        fig = go.Figure()
        fig.add_bar(name="Predicted Hits", x=df_cal["bucket"],
                    y=df_cal["avg_predicted"], marker_color="#27ae60")
        fig.add_bar(name="Actual Hits", x=df_cal["bucket"],
                    y=df_cal["avg_actual"], marker_color="#e67e22")
        fig.update_layout(
            title="Hit Projection Accuracy by Bucket",
            barmode="group", height=300,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8ecf4"))
        st.plotly_chart(fig, use_container_width=True)


def _render_nba(nba: dict):
    agg, files, failures, n_rows, msg = _status_check(nba)

    if files == 0:
        st.info("No history snapshots yet."); return
    if n_rows == 0:
        st.warning(f"Found {files} history file(s) but 0 players matched. "
                   f"API failures: {failures}/{files}.")
        if msg: st.caption(msg)
        return

    st.caption(f"Based on {files} days · {n_rows} player-games")

    stats = agg.get("stats", {})
    if stats:
        st.markdown("**Projection accuracy (how many off on average):**")
        rows = []
        for label, v in stats.items():
            mae  = v.get("mae",  0)
            bias = v.get("bias", 0)
            rows.append({
                "Stat":          label,
                "Avg Error":     f"±{mae:.2f}",
                "Bias":          f"{'Over' if bias>1 else 'Under' if bias<-1 else 'On target'} ({bias:+.2f})",
                "Avg Predicted": f"{v.get('avg_pred',0):.1f}",
                "Avg Actual":    f"{v.get('avg_actual',0):.1f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    c1,c2 = st.columns(2)
    if agg.get("dd_accuracy"):
        _metric(c1, "Double-Double Accuracy",
                f"{agg['dd_accuracy']:.1%}",
                help="% of time we correctly predicted DD vs no DD")
    if agg.get("dd_brier"):
        _metric(c2, "DD Probability Quality",
                f"{agg['dd_brier']:.4f}",
                help="Lower=better, 0.25=random guessing")

    tiers = agg.get("tiers", {})
    if tiers:
        st.markdown("**By confidence tier (points predictions):**")
        rows = []
        for t, v in tiers.items():
            rows.append({
                "Tier":          t,
                "Picks":         v["n"],
                "Avg Pts Error": f"±{v.get('mae',0):.1f}",
                "Avg Pred Pts":  f"{v.get('avg_predicted',0):.1f}",
                "Avg Actual Pts":f"{v.get('avg_actual',0):.1f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
