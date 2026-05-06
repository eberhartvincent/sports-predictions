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


def _count_history():
    """Count how many daily snapshot files exist per sport."""
    counts = {}
    if not HIST_DIR.exists():
        return counts
    for sport in ("nhl", "mlb", "nba"):
        files = sorted(HIST_DIR.glob(f"{sport}_*.parquet"))
        counts[sport] = {"n": len(files), "dates": [f.stem.replace(f"{sport}_","") for f in files]}
    return counts


def render_backtest():
    st.markdown("## 🔬 Backtesting Report")
    st.caption("Compares past predictions against actual results to measure model accuracy.")

    # ── History status ─────────────────────────────────────────────────────────
    history = _count_history()
    total_days = max((v["n"] for v in history.values()), default=0)

    if total_days == 0:
        st.info("""
        **No history yet — this is normal if you just set up the app.**

        Every day the GitHub Actions workflow runs, it saves a snapshot of
        that day's predictions to `data/cache/predictions/history/`.
        After 7+ days of snapshots, the backtest will have enough data
        to measure accuracy.

        Come back in a week and run it then.
        """)

        st.markdown("### 📅 Current Status")
        c1, c2, c3 = st.columns(3)
        for col, sport, emoji in [(c1,"nhl","🏒"),(c2,"mlb","⚾"),(c3,"nba","🏀")]:
            n = history.get(sport, {}).get("n", 0)
            with col:
                st.metric(f"{emoji} {sport.upper()} Snapshots", n,
                          delta="Need 7+ days" if n < 7 else f"{n} days ready",
                          delta_color="inverse" if n < 7 else "normal")

        st.markdown("### ℹ️ What the backtest will show (once ready)")
        st.markdown("""
        | Metric | Sport | Description |
        |--------|-------|-------------|
        | **Accuracy by tier** | NHL | % of Elite/High picks who actually scored |
        | **Calibration** | NHL | Does a 30% prediction hit ~30% of the time? |
        | **Elite ROI estimate** | NHL | Simulated return at +150 odds |
        | **MAE on Hits** | MLB | Mean absolute error on projected hits |
        | **MAE on H+R+RBI** | MLB | Mean absolute error on the combined stat |
        | **HR direction accuracy** | MLB | % correct on HR vs no HR prediction |
        | **Brier Score** | NHL/MLB | Probability calibration quality (lower = better) |
        """)
        return

    # ── Run controls ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Run Backtest")
    max_days = min(total_days, 90)
    c1, c2 = st.columns([3, 1])
    with c1:
        if max_days <= 1:
            days = max_days
            st.info(f"Only {max_days} day(s) of history available — running over all of it.")
        else:
            days = st.slider("Days to analyse", min_value=1,
                             max_value=max_days, value=min(30, max_days), step=1)
    with c2:
        run_btn = st.button("▶ Run Backtest", type="primary",
                            use_container_width=True)

    # History availability
    st.markdown("**Available snapshots:**")
    mc1, mc2, mc3 = st.columns(3)
    for col, sport, emoji in [(mc1,"nhl","🏒"),(mc2,"mlb","⚾"),(mc3,"nba","🏀")]:
        n = history.get(sport, {}).get("n", 0)
        dates = history.get(sport, {}).get("dates", [])
        first = dates[0] if dates else "—"
        last  = dates[-1] if dates else "—"
        with col:
            st.metric(f"{emoji} {sport.upper()}", f"{n} days",
                      delta=f"{first} → {last}")

    if run_btn:
        with st.spinner(f"Running backtest over last {days} days…"):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
                from scripts.backtest import run_backtest
                run_backtest(days)
                st.success("Backtest complete!")
            except Exception as e:
                st.error(f"Backtest error: {e}")
                st.exception(e)
                return

    # ── Load saved results ─────────────────────────────────────────────────────
    if not BT_FILE.exists():
        st.info("Click **▶ Run Backtest** above to generate the report.")
        return

    try:
        results = json.loads(BT_FILE.read_text())
    except Exception as e:
        st.error(f"Could not load results: {e}")
        return

    # Show when last run
    for sport_data in results.values():
        if sport_data.get("updated_at"):
            dt = datetime.fromisoformat(sport_data["updated_at"]).astimezone(ET)
            st.caption(f"Last run: {dt.strftime('%I:%M %p ET on %b %d, %Y')}")
            break

    # ── NHL Results ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏒 NHL — Goalscorer Accuracy")
    nhl = results.get("nhl", {})

    if nhl.get("message"):
        st.info(nhl["message"])
    elif nhl.get("error"):
        st.error(nhl["error"])
    else:
        agg   = nhl.get("aggregate", {})
        tiers = agg.get("tiers", {})

        if not tiers:
            st.info("Not enough NHL data yet for tier breakdown.")
        else:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Games Analysed", agg.get("n_games","—"))
            with m2:
                bs = agg.get("brier_score")
                st.metric("Brier Score", f"{bs:.4f}" if bs else "—",
                          help="Lower = better. Random guessing = 0.25")
            with m3:
                roi = agg.get("elite_roi_estimate")
                if roi is not None:
                    st.metric("Elite ROI (est.)", f"{roi:+.1%}",
                              delta="profit" if roi > 0 else "loss",
                              delta_color="normal" if roi > 0 else "inverse")

            tier_rows = []
            for conf, t in tiers.items():
                tier_rows.append({
                    "Tier":              conf,
                    "Picks":             t["n"],
                    "Accuracy":          f"{t['accuracy']:.1%}",
                    "Avg Predicted":     f"{t['avg_pred']:.3f}",
                    "Calibration Error": f"{t['calibration_err']:.3f}",
                })
            st.dataframe(pd.DataFrame(tier_rows), use_container_width=True,
                         hide_index=True)

            if len(tiers) >= 2:
                tnames   = list(tiers.keys())
                actual   = [tiers[t]["accuracy"]  for t in tnames]
                predicted= [tiers[t]["avg_pred"]  for t in tnames]
                fig = go.Figure()
                fig.add_bar(name="Actual Accuracy", x=tnames, y=actual,
                            marker_color="#27ae60",
                            text=[f"{v:.1%}" for v in actual], textposition="outside")
                fig.add_bar(name="Avg Predicted",   x=tnames, y=predicted,
                            marker_color="#3498db",
                            text=[f"{v:.1%}" for v in predicted], textposition="outside")
                fig.update_layout(title="Predicted vs Actual — NHL Goal Probability",
                    barmode="group", height=340,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    yaxis=dict(tickformat=".0%"))
                st.plotly_chart(fig, use_container_width=True)

    # ── MLB Results ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚾ MLB — Batter Projection Accuracy")
    mlb = results.get("mlb", {})

    if mlb.get("message"):
        st.info(mlb["message"])
    elif mlb.get("error"):
        st.error(mlb["error"])
    else:
        agg   = mlb.get("aggregate", {})
        tiers = agg.get("tiers", {})

        if not tiers:
            st.info("Not enough MLB data yet for tier breakdown.")
        else:
            st.metric("Games Analysed", agg.get("n_games","—"))
            tier_rows = []
            for conf, t in tiers.items():
                tier_rows.append({
                    "Tier":             conf,
                    "Picks":            t["n"],
                    "MAE Hits":         t["mae_hits"],
                    "MAE H+R+RBI":      t["mae_hrr"],
                    "Avg Pred Hits":    t["avg_pred_hits"],
                    "Avg Actual Hits":  t["avg_actual_hits"],
                    "Avg Pred HRR":     t["avg_pred_hrr"],
                    "Avg Actual HRR":   t["avg_actual_hrr"],
                    "HR Direction Acc": f"{t['hr_pred_correct']:.1%}",
                })
            st.dataframe(pd.DataFrame(tier_rows), use_container_width=True,
                         hide_index=True)

            if len(tiers) >= 2:
                tnames  = list(tiers.keys())
                pred_h  = [tiers[t]["avg_pred_hits"]   for t in tnames]
                actual_h= [tiers[t]["avg_actual_hits"] for t in tnames]
                fig2 = go.Figure()
                fig2.add_bar(name="Predicted Hits", x=tnames, y=pred_h,
                             marker_color="#27ae60",
                             text=[f"{v:.2f}" for v in pred_h], textposition="outside")
                fig2.add_bar(name="Actual Hits",    x=tnames, y=actual_h,
                             marker_color="#e67e22",
                             text=[f"{v:.2f}" for v in actual_h], textposition="outside")
                fig2.update_layout(title="Predicted vs Actual Hits by Tier",
                    barmode="group", height=340,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"))
                st.plotly_chart(fig2, use_container_width=True)

    # ── Methodology ────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📖 Methodology & Caveats"):
        st.markdown("""
        **How it works:**
        Each day the warm_cache workflow saves a prediction snapshot to
        `data/cache/predictions/history/`. The backtest fetches actual results
        from the NHL and MLB APIs and matches them by player ID.

        **Metrics:**
        - **Accuracy** — what % of predictions in each tier were correct
        - **Calibration error** — |actual % − avg predicted probability|. Zero = perfect
        - **Brier score** — mean squared error of all probability predictions (0.25 = random)
        - **Elite ROI estimate** — simulated return betting $1 on every Elite pick at +150 odds
        - **MAE** — mean absolute error between projected and actual stat value
        - **HR direction accuracy** — % of time the model correctly predicted HR vs no HR

        **Caveats:**
        - Needs 7+ days of history for meaningful results
        - Small sample sizes (< 50 picks per tier) should be treated cautiously
        - ROI estimate uses illustrative +150 odds, not real market lines
        - Past accuracy does not guarantee future performance
        """)
