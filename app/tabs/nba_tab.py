"""nba_tab.py — NBA tab UI renderer"""

import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.auth import is_admin
from app.prediction_store import load_predictions, last_updated, predictions_mtime
from config import NBA_TEAM_NAMES
from nba_pipeline import NBAPipeline

ET = ZoneInfo("America/New_York")

# Maps sort column → per-category confidence column
SORT_TO_CONF = {
    "proj_pts":    "conf_pts",
    "proj_reb":    "conf_reb",
    "proj_ast":    "conf_ast",
    "proj_fg3m":   "conf_fg3m",
    "proj_stocks": "conf_stocks",
    "proj_dd":     "conf_dd",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def badge(conf):
    m = {"Elite": "badge-elite", "High": "badge-high",
         "Medium": "badge-medium", "Low": "badge-low"}
    return f'<span class="{m.get(str(conf), "badge-low")}">{conf}</span>'


def bar(val, max_val, colour, fmt=".1f"):
    pct = min(val / max_val * 100, 100) if max_val > 0 else 0
    return (
        f'<div style="display:flex;align-items:center;gap:5px;">'
        f'<div style="flex:1;background:#1e2535;border-radius:5px;height:10px;overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;background:{colour};border-radius:5px;"></div>'
        f'</div>'
        f'<span style="font-weight:700;color:#e8ecf4;min-width:34px;font-size:.88rem;">'
        f'{val:{fmt}}</span></div>'
    )


# ── Session state init ────────────────────────────────────────────────────────

def _init_state():
    for k, v in {
        "nba_preds":    pd.DataFrame(),
        "nba_pipeline": None,
        "nba_last_run": None,
        "_nba_mtime":   None,
        "_nba_date":    None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_nba(date_str: str):
    _init_state()

    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    is_today  = (date_str is None or date_str == today_str)
    selected  = date_str or today_str

    # ── Auto-load ─────────────────────────────────────────────────────────────
    hist_file = (
        Path("data/cache/predictions/history") / f"nba_{selected}.parquet"
        if not is_today else None
    )
    disk_mtime    = predictions_mtime("nba")
    session_mtime = st.session_state.get("_nba_mtime")
    session_date  = st.session_state.get("_nba_date")

    if (st.session_state.nba_preds.empty
            or session_date != selected
            or (is_today and disk_mtime and disk_mtime != session_mtime)):
        if not is_today and hist_file and hist_file.exists():
            try:
                st.session_state.nba_preds      = pd.read_parquet(hist_file)
                st.session_state["_nba_date"]   = selected
                st.session_state["_nba_mtime"]  = None
            except Exception as e:
                st.warning(f"Could not load history for {selected}: {e}")
                st.session_state.nba_preds = pd.DataFrame()
        elif is_today:
            df = load_predictions("nba")
            if df is not None and not df.empty:
                st.session_state.nba_preds     = df
                st.session_state["_nba_mtime"] = disk_mtime
                st.session_state["_nba_date"]  = selected
        else:
            st.session_state.nba_preds = pd.DataFrame()
            st.session_state["_nba_date"] = selected

    preds    = st.session_state.nba_preds
    pipeline = st.session_state.nba_pipeline

    # ── Header ────────────────────────────────────────────────────────────────
    lu = last_updated("nba")
    col_h, col_meta = st.columns([3, 1])
    with col_h:
        st.markdown(
            f'<div class="section-header">🏀 NBA — {"Today" if is_today else selected}</div>',
            unsafe_allow_html=True,
        )
    with col_meta:
        if lu:
            st.caption(f"Updated: {lu}")
        if not is_today and (hist_file is None or not hist_file.exists()):
            st.caption("No history available for this date.")

    # ── Admin pipeline ────────────────────────────────────────────────────────
    if is_admin() and is_today:
        with st.expander("⚙️ Admin — Run Pipeline", expanded=False):
            if st.button("▶ Run NBA Pipeline", key="nba_run_btn"):
                with st.spinner("Running NBA pipeline…"):
                    try:
                        t0 = time.time()
                        p  = NBAPipeline(selected)
                        p.run()
                        st.session_state.nba_pipeline = p
                        df2 = load_predictions("nba")
                        if df2 is not None and not df2.empty:
                            st.session_state.nba_preds     = df2
                            st.session_state["_nba_mtime"] = predictions_mtime("nba")
                            st.session_state["_nba_date"]  = selected
                        st.success(f"Done in {time.time()-t0:.1f}s")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

    # ── No data ───────────────────────────────────────────────────────────────
    if preds.empty:
        if is_today:
            st.info("No NBA predictions loaded yet. "
                    + ("Run the pipeline above." if is_admin()
                       else "Predictions are generated automatically each morning."))
        else:
            st.info(f"No saved predictions found for {selected}.")
        return

    # ── Build filter option lists ─────────────────────────────────────────────
    teams       = sorted(preds["team"].dropna().unique().tolist()) if "team" in preds.columns else []
    game_labels = sorted(preds["game_label"].dropna().unique().tolist()) if "game_label" in preds.columns else []

    sort_map = {
        "Proj Pts":    "proj_pts",
        "Proj Reb":    "proj_reb",
        "Proj Ast":    "proj_ast",
        "Proj 3PM":    "proj_fg3m",
        "Proj Stocks": "proj_stocks",
    }

    # ── Filter row ────────────────────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([2, 2, 1.5, 1.8, 1])

    with f1:
        st.markdown('<div class="filter-label">Team</div>', unsafe_allow_html=True)
        team_opts = ["🏀 All Teams"] + [f"{t} — {NBA_TEAM_NAMES.get(t, t)}" for t in teams]
        sl = st.selectbox("Team", team_opts, index=0,
                          label_visibility="collapsed", key="nba_team")
        ft = None if sl == "🏀 All Teams" else sl.split(" — ")[0]

    with f2:
        st.markdown('<div class="filter-label">Game</div>', unsafe_allow_html=True)
        game_opts = ["🏀 All Games"] + game_labels
        sg = st.selectbox("Game", game_opts, index=0,
                          label_visibility="collapsed", key="nba_game")
        fg = None if sg == "🏀 All Games" else sg

    with f3:
        st.markdown('<div class="filter-label">Sort By</div>', unsafe_allow_html=True)
        ss = st.selectbox("Sort", list(sort_map.keys()), index=0,
                          label_visibility="collapsed", key="nba_sort")
        sc = sort_map[ss]

    with f4:
        st.markdown('<div class="filter-label">Confidence</div>', unsafe_allow_html=True)
        sconf = st.selectbox("Conf", ["All", "Elite", "High", "Medium", "Low"],
                             index=0, label_visibility="collapsed", key="nba_conf")

    with f5:
        st.markdown('<div class="filter-label">Show</div>', unsafe_allow_html=True)
        top_n = st.number_input("N", 5, 200, 25, 5,
                                label_visibility="collapsed", key="nba_n")

    # Active confidence column for current sort
    active_conf_col = SORT_TO_CONF.get(sc, "confidence")

    # ── Apply filters ─────────────────────────────────────────────────────────
    filt = preds.copy()
    if ft:
        filt = filt[filt["team"] == ft]
    if fg:
        filt = filt[filt["game_label"] == fg]
    if sconf != "All":
        conf_col_to_filter = active_conf_col if active_conf_col in filt.columns else "confidence"
        filt = filt[filt[conf_col_to_filter].astype(str) == sconf]
    if sc in filt.columns:
        filt = filt.sort_values(sc, ascending=False)
    disp = filt.head(int(top_n))

    st.caption(f"Showing {len(disp)} of {len(filt)} players · sorted by {ss}")

    if disp.empty:
        st.info("No players match filters.")
        return

    # ── Player table ──────────────────────────────────────────────────────────
    grid   = "36px 1fr 60px 60px 90px 105px 95px 95px 95px 75px"
    hdrs   = ["#", "Player", "Team", "Opp", "Conf",
              "Proj Pts", "Proj Reb", "Proj Ast", "Proj 3PM", "Season"]
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
        name = row.get("player_name", "")
        team = row.get("team", "")
        opp  = row.get("opponent", "")
        # Show confidence for the active sort category
        conf = str(row.get(active_conf_col, row.get("confidence", "Low")))
        gp   = int(row.get("gp", 0))
        spts = float(row.get("season_pts", 0))
        sreb = float(row.get("season_reb", 0))
        sast = float(row.get("season_ast", 0))
        pp   = float(row.get("proj_pts",   0))
        pr   = float(row.get("proj_reb",   0))
        pa   = float(row.get("proj_ast",   0))
        p3   = float(row.get("proj_fg3m",  0))

        row_bg = "#0f1320" if rank % 2 == 0 else "#111827"
        rc2    = "#e74c3c" if rank <= 3 else "#8892a4"
        pc     = "#c0392b" if pp >= 25 else "#e67e22" if pp >= 18 else "#2980b9"

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
            f'<div>{bar(pp, 45.0, pc)}</div>'
            f'<div>{bar(pr, 15.0, "#16a085")}</div>'
            f'<div>{bar(pa, 12.0, "#8e44ad")}</div>'
            f'<div>{bar(p3,  5.0, "#e67e22")}</div>'
            f'<div style="font-size:.68rem;color:#8892a4;line-height:1.3;">'
            f'  {spts:.1f}pts<br/>{sreb:.1f}r {sast:.1f}a {gp}G'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Charts", "🔬 Feature Importance"])

    with tab1:
        cdf = disp.head(20)
        if not cdf.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(
                    x=cdf["player_name"], y=cdf["proj_pts"].round(1),
                    marker_color="#e74c3c",
                    text=[f"{v:.1f}" for v in cdf["proj_pts"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="Projected Points", xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"), yaxis=dict(gridcolor="#1e2535"),
                    height=370, margin=dict(t=45, b=110),
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure(go.Bar(
                    x=cdf["player_name"], y=cdf["proj_reb"].round(1),
                    marker_color="#16a085",
                    text=[f"{v:.1f}" for v in cdf["proj_reb"]],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title="Projected Rebounds", xaxis_tickangle=-40,
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8ecf4"),
                    xaxis=dict(gridcolor="#1e2535"), yaxis=dict(gridcolor="#1e2535"),
                    height=370, margin=dict(t=45, b=110),
                )
                st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if pipeline:
            m = pipeline.models.get("pts")
            if m and hasattr(m, "feature_importance"):
                fi = m.feature_importance()
                if not fi.empty:
                    top = fi.head(15).sort_values("importance")
                    fig3 = go.Figure(go.Bar(
                        y=top["feature"], x=top["pct"], orientation="h",
                        marker_color="#2980b9",
                        text=[f"{v:.1f}%" for v in top["pct"]],
                        textposition="outside",
                    ))
                    fig3.update_layout(
                        title="Points Model — Feature Importance",
                        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font=dict(color="#e8ecf4"),
                        xaxis=dict(gridcolor="#1e2535"), yaxis=dict(gridcolor="#1e2535"),
                        height=450, margin=dict(l=180, t=45),
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.info("Feature importances not available.")
        else:
            st.info("Run the pipeline to see feature importances.")

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:.75rem;'>"
        "Data from ESPN API · For entertainment only · "
        "Predictions are probabilistic estimates"
        "</div>",
        unsafe_allow_html=True,
    )
