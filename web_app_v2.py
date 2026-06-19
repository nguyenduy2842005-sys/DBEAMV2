"""
web_app_v2.py — Beam Analysis Suite (mở rộng từ web_app.py gốc)
Tabs:
  1. Single Beam     — giữ nguyên logic gốc
  2. Continuous Beam — FEM Euler-Bernoulli
  3. Plane Frame     — FEM 2D khung phẳng

Yêu cầu:
  pip install streamlit plotly numpy pandas python-docx
"""
from __future__ import annotations

import io
import math
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from beam_core import BeamInput, BeamResult, solve_beam
from fem_core import (
    SpanDef, SupportDef, ContinuousBeamInput, ContinuousBeamResult,
    solve_continuous_beam,
    FrameNode, FrameElement, FrameSupport, FramePointLoad,
    PlaneFrameInput, PlaneFrameResult, FrameElementResult,
    solve_plane_frame,
    build_docx_bytes,
)

# ══════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════
PLOT_HEIGHT = 315
PLOT_BG = "#ffffff"
GRID     = "#d9e0e8"
AXIS     = "#263238"

COLOR_SFD   = "#0b5fff"
COLOR_BMD   = "#ff2b2b"
COLOR_ELAST = "#ff8800"
COLOR_AXIAL = "#9b27af"
COLOR_BEAM  = "#7a7f85"
COLOR_SUP   = "#ef1d14"

# ══════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Beam Analysis Suite",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown("""
    <style>
    .stApp { background: #f3f5f8; }
    #MainMenu, footer, header, [data-testid="stToolbar"] { display:none !important; }
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #d8dee8; }
    .block-container { padding-top:1.1rem; padding-bottom:1.5rem; max-width:1680px; }
    .metric-strip {
        display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
        gap:10px; margin:0.2rem 0 0.8rem;
    }
    .metric-card {
        background:#ffffff; border:1px solid #d8dee8;
        border-radius:6px; padding:10px 12px;
    }
    .metric-label { color:#5f6b7a; font-size:0.78rem; margin-bottom:3px; }
    .metric-value { color:#101828; font-weight:700; font-size:1.08rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#ffffff; border-color:#d8dee8;
    }
    .stPlotlyChart {
        background:#ffffff; border:1px solid #d8dee8;
        border-radius:6px; padding:6px;
    }
    textarea { font-family:Consolas,"Courier New",monospace !important; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════

def clean_rows(df: pd.DataFrame, columns: Iterable[str]) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for _, row in df.iterrows():
        values: list[float] = []
        skip = False
        for col in columns:
            v = row.get(col)
            if v is None or pd.isna(v) or v == "":
                skip = True; break
            try:
                n = float(v)
            except (TypeError, ValueError):
                skip = True; break
            if not math.isfinite(n):
                skip = True; break
            values.append(n)
        if not skip:
            rows.append(tuple(values))
    return rows


def base_figure(title: str, x_range: float, y_title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center",
                   y=0.98, yanchor="top", font=dict(size=15, color=AXIS)),
        height=PLOT_HEIGHT,
        margin=dict(l=55, r=20, t=60, b=45),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        showlegend=False,
        xaxis=dict(title="x (m)", range=[-x_range/20, 1.05*x_range],
                   gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                   mirror=True, ticks="outside", title_font=dict(size=13)),
        yaxis=dict(title=y_title, gridcolor=GRID, zerolinecolor=AXIS,
                   linecolor=AXIS, mirror=True, ticks="outside", title_font=dict(size=13)),
    )
    return fig


def metric_html(values: list[tuple[str, str]]) -> None:
    cards = "".join(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{lbl}</div>"
        f"<div class='metric-value'>{val}</div></div>"
        for lbl, val in values
    )
    st.markdown(f"<div class='metric-strip'>{cards}</div>", unsafe_allow_html=True)


def export_buttons(report_text: str, report_title: str, key_prefix: str) -> None:
    """Render two download buttons: .txt and .docx"""
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Xuất .txt",
            data=report_text.encode("utf-8"),
            file_name=f"{key_prefix}_report.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"{key_prefix}_dl_txt",
        )
    with c2:
        try:
            docx_bytes = build_docx_bytes(report_text, report_title)
            st.download_button(
                "⬇️ Xuất .docx",
                data=docx_bytes,
                file_name=f"{key_prefix}_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key=f"{key_prefix}_dl_docx",
            )
        except ImportError:
            st.info("Cài python-docx để xuất .docx: `pip install python-docx`")


# ══════════════════════════════════════════════════════
#  ── TAB 1: SINGLE BEAM (giữ nguyên logic gốc) ──────
# ══════════════════════════════════════════════════════

def validate_single(data: BeamInput) -> list[str]:
    errors: list[str] = []
    if data.length <= 0:
        errors.append("Chiều dài dầm phải lớn hơn 0.")
    for name, rows in [("Point Load", [(x,) for _, x in data.point_loads]),
                        ("Point Moment", [(x,) for _, x in data.point_moments])]:
        for i, (xp,) in enumerate(rows, 1):
            if xp < 0 or xp > data.length:
                errors.append(f"{name} dòng {i}: vị trí x phải nằm trong [0, L].")
    for name, rows in [("UDL", data.udls), ("UVL", data.uvls)]:
        for i, (_, x1, x2) in enumerate(rows, 1):
            if x1 < 0 or x2 < 0 or x1 > data.length or x2 > data.length or x2 <= x1:
                errors.append(f"{name} dòng {i}: cần 0 ≤ x1 < x2 ≤ L.")
    return errors


def draw_supports_single(fig: go.Figure, data: BeamInput) -> None:
    l = data.length
    if data.beam_type == "simple":
        for x0 in [0, l]:
            fig.add_trace(go.Scatter(
                x=[x0, x0+l/34, x0-l/34, x0], y=[0, -0.37, -0.37, 0],
                fill="toself", mode="lines",
                line={"color": COLOR_SUP, "width": 1}, fillcolor=COLOR_SUP,
                hoverinfo="skip"))
    else:
        fig.add_shape(type="rect", x0=l, x1=l+l/42, y0=-0.42, y1=0.42,
                      fillcolor=COLOR_SUP, line={"color": COLOR_SUP})


def plot_load_diagram_single(data: BeamInput) -> go.Figure:
    l = data.length
    fig = base_figure("Load diagram", l)
    fig.update_yaxes(range=[-1.05, 1.05], showticklabels=False, title="")
    fig.add_trace(go.Scatter(x=[0, l], y=[0, 0], mode="lines",
                             line={"color": COLOR_BEAM, "width": 8}, hoverinfo="skip"))
    draw_supports_single(fig, data)
    for load, xp in data.point_loads:
        yp, yt = (-0.06, -0.74) if load > 0 else (0.06, 0.74)
        fig.add_annotation(x=xp, y=yp, ax=xp, ay=yt, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.1, arrowwidth=2, arrowcolor=COLOR_SFD, text="")
        fig.add_annotation(x=xp, y=yt, text=f"{load:g} kN", showarrow=False,
                           font={"size": 11, "color": COLOR_SFD})
    for q, x1, x2 in data.udls:
        yb = -0.58 if q > 0 else 0.58
        fig.add_trace(go.Scatter(x=[x1, x2, x2, x1, x1], y=[0, 0, yb, yb, 0],
                                 fill="toself", mode="lines",
                                 line={"color": "#168f2c", "width": 1},
                                 fillcolor="rgba(22,143,44,0.16)",
                                 hovertemplate=f"UDL: {q:g} kN/m<extra></extra>"))
        for xv in np.linspace(x1, x2, max(2, int(np.ceil((x2-x1)/0.5))+1)):
            fig.add_annotation(x=xv, y=yb, ax=xv, ay=0, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowsize=0.9, arrowwidth=1.5, arrowcolor="#168f2c", text="")
        fig.add_annotation(x=(x1+x2)/2, y=yb*1.12, text=f"{q:g} kN/m", showarrow=False,
                           font={"size": 11, "color": "#168f2c"})
    for q, x1, x2 in data.uvls:
        yb = -0.58 if q > 0 else 0.58
        ys, ye = (0, yb) if data.uvl_type == "increase" else (yb, 0)
        fig.add_trace(go.Scatter(x=[x1, x2, x2, x1, x1], y=[0, 0, ye, ys, 0],
                                 fill="toself", mode="lines",
                                 line={"color": COLOR_SFD, "width": 1},
                                 fillcolor="rgba(11,95,255,0.14)",
                                 hovertemplate=f"UVL: {q:g} kN/m<extra></extra>"))
        fig.add_annotation(x=(x1+x2)/2, y=yb*1.12, text=f"{q:g} kN/m", showarrow=False,
                           font={"size": 11, "color": COLOR_SFD})
    return fig


def plot_sfd_single(result: BeamResult) -> go.Figure:
    fig = base_figure("Shear Force Diagram", float(result.x[-1]), "Shear (kN)")
    fig.add_trace(go.Scatter(x=result.x, y=result.shear, mode="lines",
                             fill="tozeroy", line={"color": COLOR_SFD, "width": 2},
                             fillcolor="rgba(11,95,255,0.20)",
                             hovertemplate="x=%{x:.2f}m  V=%{y:.2f}kN<extra></extra>"))
    return fig


def plot_bmd_single(result: BeamResult) -> go.Figure:
    fig = base_figure("Bending Moment Diagram", float(result.x[-1]), "Moment (kNm)")
    fig.add_trace(go.Scatter(x=result.x, y=result.moment, mode="lines",
                             fill="tozeroy", line={"color": COLOR_BMD, "width": 2},
                             fillcolor="rgba(255,43,43,0.22)",
                             hovertemplate="x=%{x:.2f}m  M=%{y:.2f}kNm<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_elastic_single(data: BeamInput, result: BeamResult | None) -> go.Figure:
    fig = plot_load_diagram_single(data)
    fig.update_layout(title={"text": "<b>Elastic Curve</b>", "x": 0.5, "font": {"size": 15}})
    fig.data = tuple(fig.data[:1])
    fig.layout.annotations = tuple()
    draw_supports_single(fig, data)
    if result is not None:
        mw = float(np.max(np.abs(result.deflection)))
        y = -result.deflection * (0.72 / mw) if mw > 0 else result.deflection
        fig.add_trace(go.Scatter(x=result.x, y=y, mode="lines",
                                 line={"color": COLOR_ELAST, "width": 4},
                                 hovertemplate="x=%{x:.2f}m  w/EI=%{customdata:.4f}<extra></extra>",
                                 customdata=result.deflection))
    fig.update_yaxes(range=[-1.05, 1.05], title="Deflection (visual)")
    return fig


def metric_strip_single(result: BeamResult | None, data: BeamInput) -> None:
    if result is None:
        values = [("Span", f"{data.length:.2f} m"),
                  ("Point loads", str(len(data.point_loads))),
                  ("UDL / UVL", f"{len(data.udls)} / {len(data.uvls)}"),
                  ("Status", "Ready")]
    elif data.beam_type == "simple":
        idx_v = int(np.argmax(np.abs(result.shear)))
        idx_m = int(np.argmax(np.abs(result.moment)))
        idx_w = int(np.argmax(np.abs(result.deflection)))
        values = [("R1 / R2", f"{result.r1:.2f} / {result.r2:.2f} kN"),
                  ("Vmax", f"{result.shear[idx_v]:.2f} kN"),
                  ("Mmax", f"{result.moment[idx_m]:.2f} kNm"),
                  ("wmax/EI", f"{result.deflection[idx_w]:.4f}")]
    else:
        idx_v = int(np.argmax(np.abs(result.shear)))
        idx_w = int(np.argmax(np.abs(result.deflection)))
        values = [("RV", f"{result.rv_fixed:.2f} kN"),
                  ("MR", f"{result.mr_fixed:.2f} kNm"),
                  ("Vmax", f"{result.shear[idx_v]:.2f} kN"),
                  ("wmax/EI", f"{result.deflection[idx_w]:.4f}")]
    metric_html(values)


def render_single_beam() -> None:
    """Full UI for single beam analysis (ported from original web_app.py)."""
    with st.sidebar:
        st.header("⚙️ Single Beam — Input")
        if st.button("🆕 New Model", type="primary", use_container_width=True, key="sb_new"):
            for k in ["sb_pl","sb_pm","sb_udl","sb_uvl","sb_result","sb_input"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()
        length = st.number_input("Chiều dài L (m)", min_value=0.01, value=10.0, step=0.5,
                                 format="%.2f", key="sb_L")
        bt = st.radio("Type of Beam", ["Simply Supported", "Cantilever"],
                      horizontal=True, key="sb_bt")
        ut = st.radio("UVL Type", ["Increase", "Decrease"],
                      horizontal=True, key="sb_ut")
        st.divider()
        st.caption("Nhập tải trong các bảng ở vùng làm việc chính.")

    data = BeamInput(
        length=float(length),
        beam_type="simple" if bt == "Simply Supported" else "cantilever",
        uvl_type="increase" if ut == "Increase" else "decrease",
    )

    # Load tables
    def_pl  = pd.DataFrame(columns=["P (kN)", "x (m)"])
    def_pm  = pd.DataFrame(columns=["M (kNm)", "x (m)"])
    def_udl = pd.DataFrame(columns=["q (kN/m)", "x1 (m)", "x2 (m)"])
    def_uvl = pd.DataFrame(columns=["qmax (kN/m)", "x1 (m)", "x2 (m)"])
    for k, v in [("sb_pl",def_pl),("sb_pm",def_pm),("sb_udl",def_udl),("sb_uvl",def_uvl)]:
        st.session_state.setdefault(k, v)

    cfg = {"width": "stretch", "num_rows": "dynamic", "hide_index": True}
    t1, t2, t3, t4 = st.tabs(["Point Load", "Point Moment", "UDL", "UVL"])
    with t1: pl  = st.data_editor(st.session_state.sb_pl,  key="sb_pl_ed",  **cfg)
    with t2: pm  = st.data_editor(st.session_state.sb_pm,  key="sb_pm_ed",  **cfg)
    with t3: udl = st.data_editor(st.session_state.sb_udl, key="sb_udl_ed", **cfg)
    with t4: uvl = st.data_editor(st.session_state.sb_uvl, key="sb_uvl_ed", **cfg)

    data.point_loads    = clean_rows(pl,  ["P (kN)", "x (m)"])
    data.point_moments  = clean_rows(pm,  ["M (kNm)", "x (m)"])
    data.udls           = clean_rows(udl, ["q (kN/m)", "x1 (m)", "x2 (m)"])
    data.uvls           = clean_rows(uvl, ["qmax (kN/m)", "x1 (m)", "x2 (m)"])

    result: BeamResult | None = st.session_state.get("sb_result")

    if st.button("▶ Solve", type="primary", use_container_width=True, key="sb_solve"):
        errs = validate_single(data)
        if errs:
            for e in errs: st.error(e)
            result = None
        else:
            try:
                result = solve_beam(data)
                st.session_state.sb_result = result
                st.session_state.sb_input  = data
            except Exception as e:
                st.error(str(e)); result = None

    metric_strip_single(result, data)

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        a, b = st.columns(2)
        with a: st.plotly_chart(plot_load_diagram_single(data), use_container_width=True)
        with b: st.plotly_chart(
            plot_sfd_single(result) if result else base_figure("Shear Force Diagram", data.length, "kN"),
            use_container_width=True)
        c, d = st.columns(2)
        with c: st.plotly_chart(
            plot_bmd_single(result) if result else base_figure("Moment Diagram", data.length, "kNm"),
            use_container_width=True)
        with d: st.plotly_chart(plot_elastic_single(data, result), use_container_width=True)
    with right:
        st.subheader("📋 Thuyết minh tính toán")
        rpt = result.report if result else "Chưa có kết quả. Nhấn Solve để tính."
        st.text_area("Report", rpt, height=620, key="sb_rpt_area")
        if result:
            export_buttons(rpt, "Thuyết Minh — Dầm Đơn", "single_beam")


# ══════════════════════════════════════════════════════
#  ── TAB 2: CONTINUOUS BEAM ──────────────────────────
# ══════════════════════════════════════════════════════

def render_continuous_beam() -> None:
    with st.sidebar:
        st.header("⚙️ Continuous Beam — Input")
        if st.button("🆕 New Model", type="primary", use_container_width=True, key="cb_new"):
            for k in list(st.session_state.keys()):
                if k.startswith("cb_"):
                    st.session_state.pop(k, None)
            st.rerun()
        st.divider()

        n_spans = st.number_input("Số nhịp", min_value=1, max_value=20, value=2, step=1, key="cb_nspans")
        st.divider()

        # Per-span geometry
        st.markdown("**Thông số từng nhịp**")
        span_lengths, span_EIs = [], []
        for i in range(int(n_spans)):
            c1, c2 = st.columns(2)
            with c1:
                L_i = st.number_input(f"L{i+1} (m)", min_value=0.01, value=5.0, step=0.5,
                                      format="%.2f", key=f"cb_L{i}")
            with c2:
                EI_i = st.number_input(f"EI{i+1}", min_value=1e-6, value=1.0, step=100.0,
                                       format="%.4g", key=f"cb_EI{i}")
            span_lengths.append(float(L_i))
            span_EIs.append(float(EI_i))

        st.divider()
        # Supports — user picks type per node
        n_nodes_boundary = int(n_spans) + 1
        st.markdown("**Gối đỡ**")
        support_kinds = []
        for i in range(n_nodes_boundary):
            xpos = sum(span_lengths[:i])
            kind = st.selectbox(
                f"Node {i} (x={xpos:.2f}m)",
                ["pin", "roller", "fixed", "free"],
                key=f"cb_sup{i}",
                index=0 if i == 0 else (0 if i == n_nodes_boundary-1 else 0),
            )
            support_kinds.append(kind)

        st.divider()
        st.caption("Nhập tải trọng trong bảng ở vùng làm việc chính.")

    # ── Load tables per span ──────────────────────────
    st.markdown("#### Tải trọng từng nhịp")
    span_pl  = []   # point loads per span
    span_udl = []
    span_pm  = []

    cfg = {"width": "stretch", "num_rows": "dynamic", "hide_index": True}

    for i in range(int(n_spans)):
        with st.expander(f"Nhịp {i+1}  (L = {span_lengths[i]:.2f} m)", expanded=(i == 0)):
            t1, t2, t3 = st.tabs(["Point Load", "UDL", "Point Moment"])
            with t1:
                df_pl = st.session_state.setdefault(
                    f"cb_pl_{i}", pd.DataFrame(columns=["P (kN)", "x_local (m)"]))
                st.data_editor(
                    st.session_state[f"cb_pl_{i}"],
                    key=f"cb_pl_ed_{i}",
                    **cfg
                )

                df_pl = st.session_state[f"cb_pl_{i}"]
            with t2:
                df_udl = st.session_state.setdefault(
                    f"cb_udl_{i}", pd.DataFrame(columns=["q (kN/m)", "x1_local (m)", "x2_local (m)"]))
                st.data_editor(
                    st.session_state[f"cb_udl_{i}"],
                    key=f"cb_udl_ed_{i}",
                    **cfg
                )

                df_udl = st.session_state[f"cb_udl_{i}"]
            with t3:
                df_pm = st.session_state.setdefault(
                    f"cb_pm_{i}", pd.DataFrame(columns=["M (kNm)", "x_local (m)"]))
                st.data_editor(
                    st.session_state[f"cb_pm_{i}"],
                    key=f"cb_pm_ed_{i}",
                    **cfg
                )

                df_pm = st.session_state[f"cb_pm_{i}"]

            span_pl.append(clean_rows(df_pl,  ["P (kN)", "x_local (m)"]))
            span_udl.append(clean_rows(df_udl, ["q (kN/m)", "x1_local (m)", "x2_local (m)"]))
            span_pm.append(clean_rows(df_pm,  ["M (kNm)", "x_local (m)"]))

    result_cb: ContinuousBeamResult | None = st.session_state.get("cb_result")

    if st.button("▶ Solve", type="primary", use_container_width=True, key="cb_solve"):
        try:
            spans_def = []
            for i in range(int(n_spans)):
                spans_def.append(SpanDef(
                    length=span_lengths[i],
                    EI=span_EIs[i],
                    point_loads=[(P, x) for P, x in span_pl[i]],
                    udls=[(q, x1, x2) for q, x1, x2 in span_udl[i]],
                    point_moments=[(M, x) for M, x in span_pm[i]],
                ))
            supports_def = [SupportDef(node=i, kind=support_kinds[i])
                            for i in range(n_nodes_boundary)
                            if support_kinds[i] != "free"]
            cb_input = ContinuousBeamInput(spans=spans_def, supports=supports_def)
            result_cb = solve_continuous_beam(cb_input)
            st.session_state.cb_result = result_cb
            st.session_state.cb_input  = cb_input
        except Exception as e:
            st.error(f"Lỗi tính toán: {e}")
            result_cb = None

    # ── Metrics ──
    if result_cb is None:
        total_L = sum(span_lengths)
        metric_html([("Tổng L", f"{total_L:.2f} m"),
                     ("Số nhịp", str(n_spans)),
                     ("Số gối", str(sum(1 for k in support_kinds if k != "free"))),
                     ("Status", "Ready")])
    else:
        xv = result_cb.x_global
        V, M, w = result_cb.shear, result_cb.moment, result_cb.deflection
        iv, im, iw = int(np.argmax(np.abs(V))), int(np.argmax(np.abs(M))), int(np.argmax(np.abs(w)))
        metric_html([("Vmax", f"{V[iv]:.3f} kN  @x={xv[iv]:.2f}m"),
                     ("Mmax", f"{M[im]:.3f} kNm @x={xv[im]:.2f}m"),
                     ("wmax/EI", f"{w[iw]:.5f} m  @x={xv[iw]:.2f}m"),
                     ("Gối", f"{len(result_cb.reactions)} phản lực")])

    # ── Plots ──
    total_L_plot = sum(span_lengths)
    left, right = st.columns([1.7, 1], gap="large")

    with left:
        # Load diagram
        fig_load = _cb_load_diagram(span_lengths, span_EIs, span_pl, span_udl, support_kinds)
        a, b = st.columns(2)
        with a:
            st.plotly_chart(fig_load, use_container_width=True)
        with b:
            if result_cb:
                fig_sfd = base_figure("Shear Force Diagram", total_L_plot, "V (kN)")
                fig_sfd.add_trace(go.Scatter(
                    x=result_cb.x_global, y=result_cb.shear, mode="lines",
                    fill="tozeroy", line={"color": COLOR_SFD, "width": 2},
                    fillcolor="rgba(11,95,255,0.20)",
                    hovertemplate="x=%{x:.3f}m  V=%{y:.3f}kN<extra></extra>"))
                st.plotly_chart(fig_sfd, use_container_width=True)
            else:
                st.plotly_chart(base_figure("Shear Force Diagram", total_L_plot, "V (kN)"),
                                use_container_width=True)

        c, d = st.columns(2)
        with c:
            if result_cb:
                fig_bmd = base_figure("Bending Moment Diagram", total_L_plot, "M (kNm)")
                fig_bmd.add_trace(go.Scatter(
                    x=result_cb.x_global, y=result_cb.moment, mode="lines",
                    fill="tozeroy", line={"color": COLOR_BMD, "width": 2},
                    fillcolor="rgba(255,43,43,0.22)",
                    hovertemplate="x=%{x:.3f}m  M=%{y:.3f}kNm<extra></extra>"))
                fig_bmd.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_bmd, use_container_width=True)
            else:
                st.plotly_chart(base_figure("Bending Moment Diagram", total_L_plot, "M (kNm)"),
                                use_container_width=True)
        with d:
            if result_cb:
                fig_el = base_figure("Elastic Curve", total_L_plot, "Deflection (visual)")
                mw = float(np.max(np.abs(result_cb.deflection))) + 1e-30
                y_vis = -result_cb.deflection * (0.72 / mw)
                fig_el.add_trace(go.Scatter(
                    x=result_cb.x_global, y=y_vis, mode="lines",
                    line={"color": COLOR_ELAST, "width": 3},
                    hovertemplate="x=%{x:.3f}m  w/EI=%{customdata:.5f}<extra></extra>",
                    customdata=result_cb.deflection))
                fig_el.update_yaxes(range=[-1.05, 1.05])
                st.plotly_chart(fig_el, use_container_width=True)
            else:
                st.plotly_chart(base_figure("Elastic Curve", total_L_plot, "Deflection"),
                                use_container_width=True)

    with right:
        st.subheader("📋 Thuyết minh tính toán")
        rpt = result_cb.report if result_cb else "Chưa có kết quả. Nhấn Solve để tính."
        st.text_area("Report", rpt, height=620, key="cb_rpt_area")
        if result_cb:
            export_buttons(rpt, "Thuyết Minh — Dầm Liên Tục", "cont_beam")


def _cb_load_diagram(span_lengths, span_EIs, span_pl, span_udl, support_kinds) -> go.Figure:
    total_L = sum(span_lengths)
    fig = base_figure("Load Diagram — Dầm liên tục", total_L)
    fig.update_yaxes(range=[-1.05, 1.05], showticklabels=False, title="")

    # Beam line
    fig.add_trace(go.Scatter(x=[0, total_L], y=[0, 0], mode="lines",
                             line={"color": COLOR_BEAM, "width": 8}, hoverinfo="skip"))

    # Span labels
    x_acc = 0.0
    for i, Ls in enumerate(span_lengths):
        mid = x_acc + Ls / 2
        fig.add_annotation(x=mid, y=0.15, text=f"L{i+1}={Ls:.1f}m", showarrow=False,
                           font={"size": 10, "color": "#5f6b7a"})
        x_acc += Ls

    # Supports
    x_acc = 0.0
    node_xs = [0.0] + list(np.cumsum(span_lengths))
    for i, kind in enumerate(support_kinds):
        xp = node_xs[i]
        if kind in ("pin", "roller"):
            fig.add_trace(go.Scatter(
                x=[xp, xp + total_L/34, xp - total_L/34, xp],
                y=[0, -0.37, -0.37, 0],
                fill="toself", mode="lines",
                line={"color": COLOR_SUP, "width": 1}, fillcolor=COLOR_SUP,
                hoverinfo="skip"))
        elif kind == "fixed":
            fig.add_shape(type="rect", x0=xp-total_L/80, x1=xp+total_L/80,
                          y0=-0.42, y1=0.42, fillcolor=COLOR_SUP, line={"color": COLOR_SUP})
        fig.add_annotation(x=xp, y=-0.55, text=f"N{i}", showarrow=False,
                           font={"size": 9, "color": COLOR_SUP})

    # Loads
    x_acc = 0.0
    for s_idx, Ls in enumerate(span_lengths):
        for P, xl in span_pl[s_idx]:
            xp = x_acc + xl
            yp, yt = (-0.06, -0.74) if P > 0 else (0.06, 0.74)
            fig.add_annotation(x=xp, y=yp, ax=xp, ay=yt, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True,
                               arrowhead=3, arrowsize=1.0, arrowwidth=2, arrowcolor=COLOR_SFD, text="")
            fig.add_annotation(x=xp, y=yt*1.05, text=f"{P:g}kN", showarrow=False,
                               font={"size": 10, "color": COLOR_SFD})
        for q, x1, x2 in span_udl[s_idx]:
            xg1, xg2 = x_acc + x1, x_acc + x2
            yb = -0.58 if q > 0 else 0.58
            fig.add_trace(go.Scatter(x=[xg1, xg2, xg2, xg1, xg1], y=[0, 0, yb, yb, 0],
                                     fill="toself", mode="lines",
                                     line={"color": "#168f2c", "width": 1},
                                     fillcolor="rgba(22,143,44,0.16)",
                                     hovertemplate=f"UDL: {q:g}kN/m<extra></extra>"))
            fig.add_annotation(x=(xg1+xg2)/2, y=yb*1.12, text=f"{q:g}kN/m", showarrow=False,
                               font={"size": 10, "color": "#168f2c"})
        x_acc += Ls

    return fig


# ══════════════════════════════════════════════════════
#  ── TAB 3: PLANE FRAME ──────────────────────────────
# ══════════════════════════════════════════════════════

def render_plane_frame() -> None:
    with st.sidebar:
        st.header("⚙️ Plane Frame — Input")
        if st.button("🆕 New Model", type="primary", use_container_width=True, key="pf_new"):
            for k in list(st.session_state.keys()):
                if k.startswith("pf_"):
                    st.session_state.pop(k, None)
            st.rerun()
        st.divider()
        st.info(
            "**Hướng dẫn:**\n"
            "1. Nhập tọa độ nút (Node)\n"
            "2. Nhập phần tử (Element)\n"
            "3. Nhập gối đỡ (Support)\n"
            "4. Nhập tải trọng\n"
            "5. Nhấn Solve"
        )
        st.divider()
        st.markdown("**Đơn vị:** kN, m, kNm")

    # ── Input tables ──────────────────────────────────
    cfg = {"width": "stretch", "num_rows": "dynamic", "hide_index": True}

    tab_nd, tab_el, tab_sup, tab_pl_nd, tab_udl_el = st.tabs([
        "🔵 Nodes", "📐 Elements", "🔒 Supports", "⬇️ Node Loads", "📏 Element UDL"])

    with tab_nd:
        st.caption("Tọa độ nút (x, y) — đơn vị m")
        df_nodes_def = pd.DataFrame({"x (m)": [0.0, 0.0, 5.0, 5.0],
                                      "y (m)": [0.0, 4.0, 4.0, 0.0]})
        df_nodes = st.session_state.setdefault("pf_nodes", df_nodes_def)
        df_nodes = st.data_editor(df_nodes, key="pf_nd_ed", **cfg)
        st.session_state["pf_nodes"] = df_nodes

    with tab_el:
        st.caption("Phần tử: nút đầu, nút cuối, E (kN/m²), A (m²), I (m⁴), UDL_local (kN/m)")
        df_el_def = pd.DataFrame({
            "i": [0, 1, 3], "j": [1, 2, 2],
            "E": [200e6]*3, "A": [0.01]*3, "I": [1e-4]*3, "udl_local": [0.0]*3
        })
        df_el = st.session_state.setdefault("pf_elems", df_el_def)
        df_el = st.data_editor(df_el, key="pf_el_ed", **cfg)
        st.session_state["pf_elems"] = df_el

    with tab_sup:
        st.caption("Gối: nút, ux_fixed, uy_fixed, rz_fixed (True/False)")
        df_sup_def = pd.DataFrame({"node": [0, 3],
                                    "ux": [True, True],
                                    "uy": [True, True],
                                    "rz": [True, True]})
        df_sup = st.session_state.setdefault("pf_sups", df_sup_def)
        df_sup = st.data_editor(df_sup, key="pf_sup_ed", **cfg)
        st.session_state["pf_sups"] = df_sup

    with tab_pl_nd:
        st.caption("Tải tập trung tại nút: Fx (kN), Fy (kN), Mz (kNm)")
        df_nload_def = pd.DataFrame({"node": pd.Series(dtype=int),
                                      "Fx (kN)": pd.Series(dtype=float),
                                      "Fy (kN)": pd.Series(dtype=float),
                                      "Mz (kNm)": pd.Series(dtype=float)})
        df_nload = st.session_state.setdefault("pf_nloads", df_nload_def)
        st.data_editor(
            st.session_state["pf_nloads"],
            key="pf_nl_ed",
            **cfg
        )

        df_nload = st.session_state["pf_nloads"]

    with tab_udl_el:
        st.caption("UDL phân bố trên phần tử (nhập vào cột 'udl_local' trong bảng Elements)")
        st.info("UDL được nhập trực tiếp ở bảng **Elements** (cột `udl_local`), "
                "tính theo phương vuông góc trục cục bộ của phần tử, đơn vị kN/m.")

    result_pf: PlaneFrameResult | None = st.session_state.get("pf_result")

    if st.button("▶ Solve", type="primary", use_container_width=True, key="pf_solve"):
        try:
            nodes = [FrameNode(x=float(r["x (m)"]), y=float(r["y (m)"]))
                     for _, r in df_nodes.iterrows()
                     if pd.notna(r.get("x (m)")) and pd.notna(r.get("y (m)"))]
            elems = []
            for _, r in df_el.iterrows():
                if pd.isna(r.get("i")) or pd.isna(r.get("j")): continue
                elems.append(FrameElement(
                    i_node=int(r["i"]), j_node=int(r["j"]),
                    E=float(r["E"]), A=float(r["A"]), I=float(r["I"]),
                    udl_local=float(r.get("udl_local", 0) or 0),
                ))
            sups = []
            for _, r in df_sup.iterrows():
                if pd.isna(r.get("node")): continue
                sups.append(FrameSupport(
                    node=int(r["node"]),
                    ux_fixed=bool(r.get("ux", True)),
                    uy_fixed=bool(r.get("uy", True)),
                    rz_fixed=bool(r.get("rz", True)),
                ))
            pls = []
            for _, r in df_nload.iterrows():
                if pd.isna(r.get("node")): continue
                FramePointLoad(
                    node=int(row["node"]),
                    Fx=float(row["Fx (kN)"]),
                    Fy=float(row["Fy (kN)"]),
                    Mz=float(row["Mz (kNm)"])
                )
            pf_input = PlaneFrameInput(nodes=nodes, elements=elems, supports=sups, point_loads=pls)
            result_pf = solve_plane_frame(pf_input)
            st.session_state.pf_result = result_pf
            st.session_state.pf_input  = pf_input
        except Exception as e:
            st.error(f"Lỗi: {e}")
            result_pf = None

    # ── Metrics ──
    if result_pf is None:
        metric_html([("Nodes", str(len(df_nodes))),
                     ("Elements", str(len(df_el))),
                     ("Supports", str(len(df_sup))),
                     ("Status", "Ready")])
    else:
        all_V = np.concatenate([er.shear  for er in result_pf.element_results])
        all_M = np.concatenate([er.moment for er in result_pf.element_results])
        all_N = np.concatenate([er.axial  for er in result_pf.element_results])
        metric_html([
            ("Nmax/Nmin", f"{np.max(all_N):.2f}/{np.min(all_N):.2f} kN"),
            ("Vmax", f"{np.max(np.abs(all_V)):.3f} kN"),
            ("Mmax", f"{np.max(np.abs(all_M)):.3f} kNm"),
            ("Reactions", f"{len(result_pf.reactions)} gối"),
        ])

    # ── Plots ──
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        a, b = st.columns(2)
        with a:
            st.plotly_chart(_pf_geometry_plot(df_nodes, df_el, df_sup, result_pf),
                            use_container_width=True)
        with b:
            st.plotly_chart(_pf_diagram_plot(result_pf, "moment", "BMD"),
                            use_container_width=True)
        c, d = st.columns(2)
        with c:
            st.plotly_chart(_pf_diagram_plot(result_pf, "shear", "SFD"),
                            use_container_width=True)
        with d:
            st.plotly_chart(_pf_diagram_plot(result_pf, "axial", "AFD"),
                            use_container_width=True)
    with right:
        st.subheader("📋 Thuyết minh tính toán")
        rpt = result_pf.report if result_pf else "Chưa có kết quả. Nhấn Solve để tính."
        st.text_area("Report", rpt, height=620, key="pf_rpt_area")
        if result_pf:
            export_buttons(rpt, "Thuyết Minh — Khung Phẳng", "plane_frame")


def _pf_geometry_plot(df_nodes, df_el, df_sup, result_pf: PlaneFrameResult | None) -> go.Figure:
    """Geometry + deformed shape overlay."""
    try:
        nodes_xy = [(float(r["x (m)"]), float(r["y (m)"])) for _, r in df_nodes.iterrows()
                    if pd.notna(r.get("x (m)")) and pd.notna(r.get("y (m)"))]
    except Exception:
        nodes_xy = []

    all_x = [p[0] for p in nodes_xy] or [0]
    all_y = [p[1] for p in nodes_xy] or [0]
    x_range = max(max(all_x) - min(all_x), 1.0)
    y_range = max(max(all_y) - min(all_y), 1.0)

    fig = go.Figure()
    fig.update_layout(
        title=dict(text="<b>Geometry & Deformed Shape</b>", x=0.5, font=dict(size=14, color=AXIS)),
        height=PLOT_HEIGHT,
        margin=dict(l=40, r=20, t=55, b=40),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        showlegend=True,
        xaxis=dict(title="x (m)", gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                   mirror=True, scaleanchor="y"),
        yaxis=dict(title="y (m)", gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, mirror=True),
    )

    # Original elements
    for _, r in df_el.iterrows():
        try:
            i, j = int(r["i"]), int(r["j"])
            ni, nj = nodes_xy[i], nodes_xy[j]
            fig.add_trace(go.Scatter(x=[ni[0], nj[0]], y=[ni[1], nj[1]],
                                     mode="lines+markers",
                                     line={"color": COLOR_BEAM, "width": 5},
                                     marker={"size": 8, "color": "#263238"},
                                     showlegend=False, hoverinfo="skip"))
        except Exception:
            pass

    # Node labels
    for i, (xn, yn) in enumerate(nodes_xy):
        fig.add_annotation(x=xn, y=yn, text=f" N{i}", showarrow=False,
                           font={"size": 11, "color": "#0b5fff"},
                           xshift=8)

    # Supports
    for _, r in df_sup.iterrows():
        try:
            ni_idx = int(r["node"])
            xp, yp = nodes_xy[ni_idx]
            ux, uy = bool(r.get("ux", True)), bool(r.get("uy", True))
            color_s = "#ef1d14"
            fig.add_trace(go.Scatter(x=[xp], y=[yp], mode="markers",
                                     marker=dict(symbol="triangle-down", size=14, color=color_s),
                                     showlegend=False,
                                     hovertemplate=f"Gối N{ni_idx}<extra></extra>"))
        except Exception:
            pass

    # Deformed shape (scaled)
    if result_pf is not None:
        disp = result_pf.node_displacements
        max_disp = float(np.max(np.abs(disp[:, :2]))) + 1e-30
        scale = 0.1 * min(x_range, y_range) / max_disp

        for er in result_pf.element_results:
            xd = er.x_coords + disp[..., 0][..., np.newaxis].flatten()[0]  # simple midpoint
            # use element coords directly
            xi_plot = er.x_coords
            yi_plot = er.y_coords
            # rough scaled displacement
            n_pts = len(xi_plot)
            fig.add_trace(go.Scatter(
                x=xi_plot, y=yi_plot, mode="lines",
                line={"color": COLOR_ELAST, "width": 2, "dash": "dash"},
                name="Deformed" if er.elem_idx == 0 else None,
                showlegend=(er.elem_idx == 0),
                hoverinfo="skip",
            ))

    return fig


def _pf_diagram_plot(result_pf: PlaneFrameResult | None, field: str, title: str) -> go.Figure:
    """Plot axial / shear / moment diagram for all frame elements along their global coords."""
    color_map = {"shear": COLOR_SFD, "moment": COLOR_BMD, "axial": COLOR_AXIAL}
    unit_map  = {"shear": "kN", "moment": "kNm", "axial": "kN"}
    color = color_map.get(field, "#333")
    unit  = unit_map.get(field, "")

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14, color=AXIS)),
        height=PLOT_HEIGHT,
        margin=dict(l=40, r=20, t=55, b=40),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        showlegend=False,
        xaxis=dict(title="x (m)", gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                   mirror=True, scaleanchor="y"),
        yaxis=dict(title=f"{unit}", gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, mirror=True),
    )

    if result_pf is None:
        fig.add_annotation(text="Chưa có kết quả", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font={"size": 14, "color": "#aaa"})
        return fig

    # For frame diagrams: plot value perpendicular to element axis
    for er in result_pf.element_results:
        arr = getattr(er, field)
        # Draw along global x of element (simplified: project onto x-axis)
        fig.add_trace(go.Scatter(
            x=er.x_coords, y=arr,
            mode="lines",
            line={"color": color, "width": 2},
            fill="tozeroy",
            fillcolor=f"rgba({_hex_to_rgb(color)},0.18)",
            hovertemplate=f"Phần tử {er.elem_idx}<br>x=%{{x:.2f}}m  {field}=%{{y:.3f}}{unit}<extra></extra>",
            name=f"Elem {er.elem_idx}",
        ))

    return fig


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    st.title("🏗️ Beam Analysis Suite")

    # Top-level tab selector
    TAB_SINGLE = "📏 Single Beam"
    TAB_CB     = "🔗 Continuous Beam"
    TAB_PF     = "🏛️ Plane Frame"

    tab1, tab2, tab3 = st.tabs([TAB_SINGLE, TAB_CB, TAB_PF])

    with tab1:
        render_single_beam()

    with tab2:
        render_continuous_beam()

    with tab3:
        render_plane_frame()


if __name__ == "__main__":
    main()