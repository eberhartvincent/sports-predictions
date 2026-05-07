"""
app/pages/backtest_page.py
Admin-only backtesting report rendered inside the main app.
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

ET       = ZoneInfo("America/New_York")
PRED_DIR = Path("data/cache/predictions")
BT_FILE  = PRED_DIR / "backtest_results.json"


def render_backtest():
    st.markdown("## 🔬 Backtesting Report")
    st.caption("Compares past predictions against actual results to measure model accuracy.")

    # ── Run / refresh ─────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    with c1:
        days = st.slider("Days to analyse", 7, 90, 30, 7)
    with c2:
        run_btn = st.button("▶ Run Backtest", type="primary",
                            use_container_width=True)

    if run_btn:
        with st.spinner(f"Running backtest over last {days} days…"):
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
                from scripts.backtest import run_backtest
                results = run_backtest(days)
                st.success("Backtest complete!")
            except Exception as e:
                st.error(f"Backtest error: {e}")
                st.exception(e)
                return

    # ── Load saved results ─────────────────────────────────────────────────────
    if not BT_FILE.exists():
        st.info("No backtest results yet. Click **Run Backtest** above to generate them.")
        st.markdown("""
        **What the backtest measures:**
        - **NHL:** For each confidence tier, what % of predicted scorers actually scored
        - **MLB:** Mean absolute error on H, H+R+RBI projections by confidence tier
        - **Calibration:** Whether a 30% prediction actually hits ~30% of the time
        - **Elite ROI estimate:** Simulated return if betting $1 on every Elite pick at +150 odds
        
        **Note:** History accumulates daily as the workflow runs. 
        You need at least 7 days of saved predictions for meaningful results.
        """)
        return

    try:
        results = json.loads(BT_FILE.read_text())
    except Exception as e:
        st.error(f"Could not load results: {e}")
        return

    updated = None
    for sport_data in results.values():
        if sport_data.get("updated_at"):
            updated = sport_data["updated_at"]
            break
    if updated:
        dt = datetime.fromisoformat(updated).astimezone(ET)
        st.caption(f"Last run: {dt.strftime('%I:%M %p ET on %b %d, %Y')}")

    # ── NHL Results ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🏒 NHL — Goal Probability Accuracy")

    nhl = results.get("nhl", {})
    if nhl.get("message") or nhl.get("error"):
        st.info(nhl.get("message") or nhl.get("error"))
    else:
        agg = nhl.get("aggregate", {})
        tiers = agg.get("tiers", {})

        if tiers:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Games Analysed", agg.get("n_games","—"))
            with m2:
                bs = agg.get("brier_score")
                st.metric("Brier Score", f"{bs:.4f}" if bs else "—",
                          help="Lower = better calibrated. Random = 0.25")
            with m3:
                roi = agg.get("elite_roi_estimate")
                if roi is not None:
                    color = "normal" if roi > 0 else "inverse"
                    st.metric("Elite Pick ROI", f"{roi:+.1%}",
                              delta=f"{'profit' if roi>0 else 'loss'} per $1 bet",
                              delta_color=color)

            # Tier breakdown table
            tier_rows = []
            for conf, t in tiers.items():
                tier_rows.append({
                    "Tier":         conf,
                    "Picks":        t["n"],
                    "Accuracy":     f"{t['accuracy']:.1%}",
                    "Avg Pred":     f"{t['avg_pred']:.3f}",
                    "Calibration Err": f"{t['calibration_err']:.3f}",
                })
            st.dataframe(pd.DataFrame(tier_rows), use_container_width=True,
                         hide_index=True)

            # Accuracy vs predicted bar chart
            if len(tiers) >= 2:
                tier_names = list(tiers.keys())
                actual_acc = [tiers[t]["accuracy"]  for t in tier_names]
                pred_acc   = [tiers[t]["avg_pred"]  for t in tier_names]

                fig = go.Figure()
                fig.add_bar(name="Actual Accuracy", x=tier_names, y=actual_acc,
                            marker_color="#27ae60",
                            text=[f"{v:.1%}" for v in actual_acc],
                            textposition="outside")
                fig.add_bar(name="Avg Predicted Prob", x=tier_names, y=pred_acc,
                            marker_color="#3498db",
                            text=[f"{v:.1%}" for v in pred_acc],
                            textposition="outside")
                fig.update_layout(
                    title="Predicted vs Actual Goal Probability by Tier",
                    barmode="group", height=350,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    yaxis=dict(tickformat=".0%"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data yet for NHL tier breakdown.")

    # ── MLB Results ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚾ MLB — Batter Projection Accuracy")

    mlb = results.get("mlb", {})
    if mlb.get("message") or mlb.get("error"):
        st.info(mlb.get("message") or mlb.get("error"))
    else:
        agg = mlb.get("aggregate", {})
        tiers = agg.get("tiers", {})

        if tiers:
            st.metric("Games Analysed", agg.get("n_games","—"))

            tier_rows = []
            for conf, t in tiers.items():
                tier_rows.append({
                    "Tier":          conf,
                    "Picks":         t["n"],
                    "MAE Hits":      t["mae_hits"],
                    "MAE H+R+RBI":   t["mae_hrr"],
                    "Avg Pred H":    t["avg_pred_hits"],
                    "Avg Actual H":  t["avg_actual_hits"],
                    "Avg Pred HRR":  t["avg_pred_hrr"],
                    "Avg Actual HRR":t["avg_actual_hrr"],
                    "HR Dir. Acc":   f"{t['hr_pred_correct']:.1%}",
                })
            st.dataframe(pd.DataFrame(tier_rows), use_container_width=True,
                         hide_index=True)

            # Pred vs actual hits chart
            tier_names = list(tiers.keys())
            pred_h   = [tiers[t]["avg_pred_hits"]   for t in tier_names]
            actual_h = [tiers[t]["avg_actual_hits"]  for t in tier_names]

            fig2 = go.Figure()
            fig2.add_bar(name="Predicted Hits", x=tier_names, y=pred_h,
                         marker_color="#27ae60",
                         text=[f"{v:.2f}" for v in pred_h],
                         textposition="outside")
            fig2.add_bar(name="Actual Hits", x=tier_names, y=actual_h,
                         marker_color="#e67e22",
                         text=[f"{v:.2f}" for v in actual_h],
                         textposition="outside")
            fig2.update_layout(
                title="Predicted vs Actual Hits by Tier",
                barmode="group", height=350,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#e8ecf4"),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data yet for MLB tier breakdown.")

    # ── Methodology note ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("📖 Methodology"):
        st.markdown("""
        **How results are computed:**
        
        - Each day the warm_cache workflow saves a snapshot of predictions to `data/cache/predictions/history/`
        - The backtest fetches actual results from the NHL and MLB APIs for those dates
        - Results are matched on player ID and grouped by confidence tier
        
        **NHL metrics:**
        - **Accuracy:** % of players in each tier who actually scored a goal
        - **Calibration error:** |actual accuracy − average predicted probability| (lower = better)
        - **Brier score:** Mean squared error of probability predictions (lower = better, 0.25 = random)
        - **Elite ROI:** Simulated return betting $1 on every Elite pick at +150 odds
        
        **MLB metrics:**
        - **MAE Hits:** Mean absolute error on projected hits
        - **MAE H+R+RBI:** Mean absolute error on the H+R+RBI combined stat
        - **HR Direction Accuracy:** % of time the model correctly predicted HR vs no HR
        
        **Important caveats:**
        - Results accumulate over time — more days = more reliable estimates
        - Small samples (< 50 picks per tier) should be treated cautiously
        - The ROI estimate assumes +150 odds which is illustrative, not exact
        - Past accuracy does not guarantee future performance
        """)
