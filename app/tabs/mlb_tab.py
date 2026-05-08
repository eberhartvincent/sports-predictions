"""mlb_tab.py — MLB tab UI renderer"""

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.auth import is_admin
from app.prediction_store import load_predictions, last_updated, predictions_mtime
from config import MLB_TEAM_NAMES
from mlb_pipeline import MLBPipeline

ET = ZoneInfo("America/New_York")


# ── Helpers ───────────────────────────────────────────────────────────────────

def badge(conf):
    m = {"Elite": "badge-elite", "High": "badge-high",
         "Medium": "badge-medium", "Low": "badge-low"}
    return f'<span class="{m.get(str(conf), "badge-low")}">{conf}</span>'


def bar(val, max_val, colour, fmt=".2f"):
    pct = min(val / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="flex:1;background:#1e2535;border-radius:5px;height:10px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{colour};border-radius:5px;"></div>'
        f'</div>'
        f'<span style="font-weight:700;color:#e8ecf4;min-width:36px;font-size:.88rem;">'
        f'{val:{fmt}}</span></div>'
    )


# ── Session state init ────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "mlb_preds":    pd.DataFrame(),
        "mlb_pipeline": None,
        "mlb_last_run": None,
        "_mlb_mtime":   None,
        "_mlb_date":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_mlb(date_str: str):
    _init_state()

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    is_today  = (date_str is None or date_str == today_str)
    selected  = date_str or today_str

    # ── Auto-load predictions ─────────────────────────────────────────────────
    # For a past date: load from history parquet if it exists.
    # For today: load from the live prediction store, reloading if the file
    # on disk is newer than what's in session state.

    hist_file = (
        Path("data/cache/predictions/history") / f"mlb_{selected}.parquet"
        if not is_today else None
    )

    disk_mtime    = predictions_mtime("mlb")
    session_mtime = st.session_state.get("_mlb_mtime")
    session_date  = st.session_state.get("_mlb_date")

    need_reload = (
        st.session_state.mlb_preds.empty
        or session_date != selected
        or (is_today and disk_mtime and disk_mtime != session_mtime)
    )

    if need_reload:
        if not is_today and hist_file and hist_file.exists():
            # Load historical predictions from parquet
            try:
                st.session_state.mlb_preds = pd.read_parquet(hist_file)
                st.session_state["_mlb_date"]  = selected
                st.session_state["_mlb_mtime"] = None
            except Exception as e:
                st.warning(f"Could not load history for {selected}: {e}")
                st.session_state.mlb_preds = pd.DataFrame()
        elif is_today:
            # Load today's predictions from prediction store
            df = load_predictions("mlb")
            if df is not None and not df.empty:
                st.session_state.mlb_preds    = df
                st.session_state["_mlb_mtime"] = disk_mtime
                st.session_state["_mlb_date"]  = selected
        else:
            # Past date with no history file
            st.session_state.mlb_preds = pd.DataFrame()
            st.session_state["_mlb_date"] = selected

    preds    = st.session_state.mlb_preds
    pipeline = st.session_state.mlb_pipeline

    # ── Header ────────────────────────────────────────────────────────────────
    lu = last_updated("mlb")
    col_h, col_meta = st.columns([3, 1])
    with col_h:
        date_label = "Today" if is_today else selected
        st.markdown(
            f'<div class="section-header">⚾ MLB — {date_label}</div>',
            unsafe_allow_html=True,
        )
    with col_meta:
        if lu:
            st.caption(f"Updated: {lu}")
        if not is_today and (hist_file is None or not hist_file.exists()):
            st.caption("No history available for this date.")

    # ── Admin: run pipeline ───────────────────────────────────────────────────
    if is_admin() and is_today:
        with st.expander("⚙️ Admin — Run Pipeline", expanded=False):
            if st.button("▶ Run MLB Pipeline", key="mlb_run_btn"):
                with st.spinner("Running MLB pipeline…"):
                    try:
                        t0 = time.time()
                        p  = MLBPipeline(selected)
                        p.run()
                        st.session_state.mlb_pipeline = p
                        df2 = load_predictions("mlb")
                        if df2 is not None and not df2.empty:
                            st.session_state.mlb_preds    = df2
                            st.session_state["_mlb_mtime"] = predictions_mtime("mlb")
                            st.session_state["_mlb_date"]  = selected
                        st.success(f"Done in {time.time()-t0:.1f}s")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

    # ── No data state ─────────────────────────────────────────────────────────
    if preds.empty:
        if is_today:
            st.info("No MLB predictions loaded yet. "
                    + ("Run the pipeline above." if is_admin() else
                       "Predictions are generated automatically each morning."))
        else:
            st.info(f"No saved predictions found for {selected}.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    teams_in_preds = sorted(preds["team"].dropna().unique().tolist()) if "team" in preds.columns else []
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
        sel_team = st.selectbox("Filter by team", ["All"] + teams_in_preds, key="mlb_team_filter")
    with fc2:
        conf_opts = ["All", "Elite", "High", "Medium", "Low"]
        sel_conf  = st.selectbox("Min confidence", conf_opts, key="mlb_conf_filter")
    with fc3:
        top_n = st.selectbox("Show top", [25, 50, 100, 200], key="mlb_topn")

    disp = preds.copy()
    if sel_team != "All":
        disp = disp[disp["team"] == sel_team]
    if sel_conf != "All":
        conf_order = {"Elite": 4, "High": 3, "Medium": 2, "Low": 1}
        min_rank   = conf_order.get(sel_conf, 1)
        disp = disp[disp["confidence"].map(lambda c: conf_order.get(str(c), 0)) >= min_rank]

    disp = disp.head(top_n)

    if disp.empty:
        st.warning("No players match the selected filters.")
        return

    # ── Player table ──────────────────────────────────────────────────────────
    st.markdown(f"**{len(disp)} players** shown")

    grid   = "36px 1fr 55px 55px 85px 105px 80px 80px 80px 80px 70px"
    hdrs   = ["#", "Player", "Team", "Opp", "Conf",
              "Proj Hits", "Proj HR", "Proj TB", "Proj RBI", "Proj Runs", "Season"]
    hstyle = (
        f"display:grid;grid-template-columns:{grid};gap:6px;padding:7px 12px;"
        "background:#1a1f2e;border-radius:8px;font-size:.68rem;color:#8892a4;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;"
        "border:1px solid #2d3550;"
    )
    st.markdown(
        f'<div style="{hstyle}">' + "".join(f"<div>{h}</div>" for h in hdrs) + "</div>",
        unsafe_allow_html=True,
    )

    for rank, (_, row) in enumerate(disp.iterrows(), 1):
        name  = row.get("player_name", "")
        team  = row.get("team", "")
        opp   = row.get("opponent", "")
        conf  = str(row.get("confidence", "Low"))
        gp    = int(row.get("gp", 0))
        savg  = float(row.get("season_avg", 0))
        shr   = int(row.get("season_hr", 0))
        ph    = float(row.get("proj_hits", 0))
        phr   = float(row.get("proj_hr", 0))
        ptb   = float(row.get("proj_tb", 0))
        prbi  = float(row.get("proj_rbi", 0))
        pruns = float(row.get("proj_runs", 0))

        row_bg = "#0f1320" if rank % 2 == 0 else "#111827"
        rc2    = "#e74c3c" if rank <= 3 else "#8892a4"
        hc2    = "#27ae60" if ph >= 0.9 else "#f39c12" if ph >= 0.7 else "#2980b9"

        st.markdown(
            f'<div style="display:grid;grid-template-columns:{grid};gap:6px;'
            f'padding:8px 12px;background:{row_bg};border-radius:8px;'
            f'margin-bottom:2px;align-items:center;border:1px solid #1e2535;">'
            f'<div style="font-weight:700;color:{rc2};">#{rank}</div>'
            f'<div>'
            f'  <div style="font-weight:600;color:#e8ecf4;font-size:.9rem;">{name}</div>'
            f'  <div style="font-size:.66rem;color:#5a7fc4;">{row.get("game_label","")}</div>'
            f'</div>'
            f'<div style="font-weight:600;color:#7eb3ff;font-size:.85rem;">{team}</div>'
            f'<div style="color:#8892a4;font-size:.85rem;">{opp}</div>'
            f'<div>{badge(conf)}</div>'
            f'<div>{bar(ph,   2.0,  hc2)}</div>'
            f'<div>{bar(phr,  0.4,  "#c0392b")}</div>'
            f'<div>{bar(ptb,  4.0,  "#8e44ad", ".1f")}</div>'
            f'<div>{bar(prbi, 2.0,  "#e67e22", ".1f")}</div>'
            f'<div>{bar(pruns,2.0,  "#16a085", ".1f")}</div>'
            f'<div style="font-size:.68rem;color:#8892a4;line-height:1.3;">'
            f'  .{int(savg*1000):03d}<br/>{shr}HR {gp}G'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    t1, t2 = st.tabs(["📊 Charts", "🔬 Feature Importance"])
    with t1:
        cdf = disp.head(20)
        if not cdf.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=cdf["player_name"], y=cdf["proj_hits"].round(2),
                    marker_color="#27ae60",
                    text=[f"{v:.2f}" for v in cdf["proj_hits"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Projected Hits", xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"),
                    yaxis=dict(gridcolor="#1e2535"),
                    height=370, margin=dict(t=45, b=110),
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure(go.Bar(
                    x=cdf["player_name"], y=cdf["proj_hr"].round(3),
                    marker_color="#c0392b",
                    text=[f"{v:.3f}" for v in cdf["proj_hr"]],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="HR Probability", xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"),
                    yaxis=dict(gridcolor="#1e2535"),
                    height=370, margin=dict(t=45, b=110),
                )
                st.plotly_chart(fig2, use_container_width=True)

    with t2:
        if pipeline:
            m = pipeline.models.get("hits")
            if m and hasattr(m, "feature_importances_"):
                fi = pd.Series(
                    m.feature_importances_,
                    index=pipeline.feature_names,
                ).sort_values(ascending=True).tail(20)
                fig3 = go.Figure(go.Bar(
                    x=fi.values, y=fi.index, orientation="h",
                    marker_color="#3498db",
                ))
                fig3.update_layout(
                    title="Top 20 Feature Importances (Hits model)",
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"),
                    yaxis=dict(gridcolor="#1e2535"),
                    height=500, margin=dict(l=160, t=45),
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Feature importances not available.")
        else:
            st.info("Run the pipeline to see model metrics.")

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:.75rem;'>"
        "Data from MLB Stats API · For entertainment only · "
        "Predictions are probabilistic estimates"
        "</div>",
        unsafe_allow_html=True,
    )
