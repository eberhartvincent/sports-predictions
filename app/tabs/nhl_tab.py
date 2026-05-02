"""app/tabs/nhl_tab.py — NHL predictions tab"""
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def badge(conf):
    m = {"Elite":"badge-elite","High":"badge-high","Medium":"badge-medium","Low":"badge-low"}
    return f'<span class="{m.get(str(conf),"badge-low")}">{conf}</span>'


def bar(val, max_val, colour, fmt=".2f"):
    pct = min(val/max_val*100, 100) if max_val > 0 else 0
    return (f'<div style="display:flex;align-items:center;gap:5px;">' +
            f'<div style="flex:1;background:#1e2535;border-radius:5px;height:10px;overflow:hidden;">' +
            f'<div style="width:{pct:.0f}%;height:100%;background:{colour};border-radius:5px;"></div></div>' +
            f'<span style="font-weight:700;color:#e8ecf4;min-width:36px;font-size:.88rem;">{val:{fmt}}</span></div>')


def render_nhl(selected_date_str, force_retrain):
    from config.settings import NHL_TEAMS, CURRENT_SEASON
    from core.pipelines.nhl_pipeline import NHLPipeline

    for k, v in {
        "pipeline": None, "predictions": pd.DataFrame(),
        "last_run": None, "running": False,
        "teams_playing": [], "games_today": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    ET = ZoneInfo("America/New_York")

