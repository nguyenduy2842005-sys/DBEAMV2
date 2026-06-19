from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeamInput:
    length: float
    beam_type: str = "simple"
    uvl_type: str = "increase"
    point_loads: list[tuple[float, float]] = field(default_factory=list)
    point_moments: list[tuple[float, float]] = field(default_factory=list)
    udls: list[tuple[float, float, float]] = field(default_factory=list)
    uvls: list[tuple[float, float, float]] = field(default_factory=list)



@dataclass
class BeamResult:
    x: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    theta: np.ndarray
    deflection: np.ndarray

    r1: float
    r2: float

    rv_fixed: float
    mr_fixed: float

    report: str

def solve_beam(data: BeamInput, step: float = 0.01) -> BeamResult:
    """Port of the MATLAB GUIDE BeamAnalysis solve_Callback logic."""
    rv_fixed = 0.0
    mr_fixed = 0.0
    l = float(data.length)
    if l <= 0:
        raise ValueError("Span phải là số dương.")

    p = np.array(data.point_loads, dtype=float).reshape(-1, 2) if data.point_loads else np.empty((0, 2))
    m = np.array(data.point_moments, dtype=float).reshape(-1, 2) if data.point_moments else np.empty((0, 2))
    udl = np.array(data.udls, dtype=float).reshape(-1, 3) if data.udls else np.empty((0, 3))
    uvl = np.array(data.uvls, dtype=float).reshape(-1, 3) if data.uvls else np.empty((0, 3))

    cp, cm, cudl, cuvl = len(p), len(m), len(udl), len(uvl)
    x = np.arange(0.0, l + step / 2, step)
    shear = np.zeros_like(x)
    moment = np.zeros_like(x)
    theta_int = np.zeros_like(x)
    w_int = np.zeros_like(x)

    sum_p_mom = sum_udl_mom = sum_uvl_mom = sum_m_mom = 0.0
    sum_p_val = sum_udl_val = sum_uvl_val = 0.0
    is_simple = data.beam_type == "simple"

    # ==========================================================
    # TÍNH PHẢN LỰC GỐI / PHẢN LỰC NGÀM
    # ==========================================================

    if is_simple:

        # -------------------------
        # Point loads
        # -------------------------
        if cp > 0:
            sum_p_mom = float(np.sum(p[:, 0] * p[:, 1]))
            sum_p_val = float(np.sum(p[:, 0]))

        # -------------------------
        # UDL
        # -------------------------
        if cudl > 0:
            len_udl = udl[:, 2] - udl[:, 1]
            val_udl = udl[:, 0] * len_udl
            pos_udl = udl[:, 1] + len_udl / 2

            sum_udl_mom = float(np.sum(val_udl * pos_udl))
            sum_udl_val = float(np.sum(val_udl))

        # -------------------------
        # UVL
        # -------------------------
        if cuvl > 0:

            for q_max, a, b in uvl:

                span = b - a

                val = 0.5 * q_max * span

                if data.uvl_type == "increase":
                    pos = a + 2 * span / 3
                else:
                    pos = a + span / 3

                sum_uvl_mom += val * pos
                sum_uvl_val += val

        # -------------------------
        # Point moments
        # -------------------------
        if cm > 0:
            sum_m_mom = float(np.sum(m[:, 0]))

        # -------------------------
        # Reactions
        # -------------------------
        r2 = -(
                sum_p_mom
                + sum_udl_mom
                + sum_uvl_mom
                + sum_m_mom
        ) / l

        r1 = -(
                sum_p_val
                + sum_udl_val
                + sum_uvl_val
                + r2
        )

        shear += r1
        moment += r1 * x

    else:

        # ======================================================
        # DẦM CONSOLE
        # ======================================================

        r1 = 0.0
        r2 = 0.0

        # Tổng tải đứng
        total_vertical_load = (
                sum(load for load, _ in data.point_loads)
                + sum(q * (b - a) for q, a, b in data.udls)
                + sum(0.5 * q * (b - a) for q, a, b in data.uvls)
        )

        rv_fixed = -total_vertical_load

        # -------------------------
        # Moment do tải tập trung
        # -------------------------
        moment_p = sum(
            load * (l - pos)
            for load, pos in data.point_loads
        )

        # -------------------------
        # Moment do UDL
        # -------------------------
        moment_udl = sum(
            q * (b - a) *
            (l - (a + (b - a) / 2))
            for q, a, b in data.udls
        )

        # -------------------------
        # Moment do UVL
        # -------------------------
        moment_uvl = 0.0

        for q, a, b in data.uvls:

            span = b - a

            if data.uvl_type == "increase":
                xr = a + 2 * span / 3
            else:
                xr = a + span / 3

            moment_uvl += (
                    0.5 * q * span *
                    (l - xr)
            )
        # -------------------------
        # Moment tập trung
        # -------------------------
        moment_m = sum(
            mi
            for mi, _ in data.point_moments
        )
        mr_fixed = -(
                moment_p
                + moment_udl
                + moment_uvl
                + moment_m
        )
    theta_int += (r1 * x**2) / 2
    w_int += (r1 * x**3) / 6

    for load, a in p:
        mask = x >= a
        shear[mask] += load
        moment[mask] += load * (x[mask] - a)
        theta_int[mask] += load * (x[mask] - a) ** 2 / 2
        w_int[mask] += load * (x[mask] - a) ** 3 / 6

    for q, a, b in udl:
        mask_in = (x > a) & (x <= b)
        mask_after = x > b
        shear[mask_in] += q * (x[mask_in] - a)
        moment[mask_in] += 0.5 * q * (x[mask_in] - a) ** 2
        shear[mask_after] += q * (b - a)
        moment[mask_after] += q * (b - a) * (x[mask_after] - (a + b) / 2)

        m_a = x > a
        m_b = x > b
        theta_int[m_a] += (q / 6) * (x[m_a] - a) ** 3
        w_int[m_a] += (q / 24) * (x[m_a] - a) ** 4
        theta_int[m_b] -= (q / 6) * (x[m_b] - b) ** 3
        w_int[m_b] -= (q / 24) * (x[m_b] - b) ** 4

    for q_max, a, b in uvl:
        span = b - a
        if span <= 0:
            raise ValueError("Chiều dài UVL không hợp lệ.")
        xx = x - a
        xx[xx < 0] = 0
        xb = x - b
        xb[xb < 0] = 0

        if data.uvl_type == "increase":
            shear += (q_max / (2 * span)) * (xx**2 - xb**2) - (0.5 * q_max * span) * (x > b)
            moment += (q_max / (6 * span)) * (xx**3 - xb**3) - (0.5 * q_max * span) * xb - (
                q_max * span**2 / 6
            ) * (x > b)

            m_a = x > a
            m_b = x > b
            theta_int[m_a] += (q_max / (24 * span)) * (x[m_a] - a) ** 4
            w_int[m_a] += (q_max / (120 * span)) * (x[m_a] - a) ** 5
            theta_int[m_b] -= (q_max / (24 * span)) * (x[m_b] - b) ** 4 + (q_max * span / 6) * (
                x[m_b] - b
            ) ** 3
            w_int[m_b] -= (q_max / (120 * span)) * (x[m_b] - b) ** 5 + (q_max * span / 24) * (
                x[m_b] - b
            ) ** 4
        else:
            shear += q_max * xx - (q_max / (2 * span)) * xx**2 - (0.5 * q_max * span) * (x > b)
            moment += (q_max / 2) * xx**2 - (q_max / (6 * span)) * xx**3 - (
                0.5 * q_max * span * xb + q_max * span**2 / 3
            ) * (x > b)
            # MATLAB source left the decreasing-UVL deflection integral unimplemented.

    for mi, a in m:
        mask = x >= a
        moment[mask] -= mi
        theta_int[mask] -= mi * (x[mask] - a)
        w_int[mask] -= mi * (x[mask] - a) ** 2 / 2

    if is_simple:
        c2 = 0.0
        c1 = -w_int[-1] / l
    else:
        c1 = -theta_int[-1]
        c2 = -(w_int[-1] + c1 * l)

    deflection = -(w_int + c1 * x + c2)
    theta = theta_int + c1
    shear[np.abs(shear) < 1e-10] = 0
    moment[np.abs(moment) < 1e-10] = 0

    report = build_report(
        data,
        x,
        shear,
        moment,
        deflection,
        r1,
        r2,
        sum_p_mom,
        sum_udl_mom,
        sum_uvl_mom,
        sum_m_mom,
    )
    return BeamResult(
        x,
        shear,
        moment,
        theta,
        deflection,
        float(r1),
        float(r2),
        float(rv_fixed),
        float(mr_fixed),
        report,
    )


def build_report(
    data: BeamInput,
    x: np.ndarray,
    shear: np.ndarray,
    moment: np.ndarray,
    deflection: np.ndarray,
    r1: float,
    r2: float,
    sum_p_mom: float,
    sum_udl_mom: float,
    sum_uvl_mom: float,
    sum_m_mom: float,
) -> str:
    lines: list[str] = []
    lines.append("=========== THUYẾT MINH TÍNH TOÁN ===========")
    lines.append("")
    lines.append(f"Chiều dài dầm: {data.length:.2f} m")
    beam_name = "Dầm tựa đơn (Simply Supported)" if data.beam_type == "simple" else "Dầm console (Cantilever)"
    lines.append(f"Loại dầm: {beam_name}")
    lines.append("")
    lines.append("1. Phương trình cân bằng:")
    lines.append("  - ΣFx = 0 ; ΣFy = 0 ; ΣM = 0")
    lines.append("  - Cắt dầm tại các vị trí đặc trưng")
    lines.append("  - Xác định V(x), M(x)")
    lines.append("")
    lines.append("2. Phương pháp đồ thị:")
    lines.append("  dV/dx = q(x)")
    lines.append("  dM/dx = V(x)")
    lines.append("  => V là diện tích biểu đồ tải trọng")
    lines.append("  => M là diện tích biểu đồ lực cắt")

    lines.append("")
    lines.append("--- TẢI TẬP TRUNG ---")
    lines.append(f"Số lượng: {len(data.point_loads)}")
    for i, (load, pos) in enumerate(data.point_loads, 1):
        lines.append(f"  P{i} = {load:.2f} kN tại x = {pos:.2f} m")

    lines.append("")
    lines.append("--- TẢI PHÂN BỐ ĐỀU ---")
    lines.append(f"Số lượng: {len(data.udls)}")
    for i, (q, a, b) in enumerate(data.udls, 1):
        span = b - a
        resultant = q * span
        x_resultant = a + span / 2
        lines.append(f"  UDL{i}: q = {q:.2f} kN/m ({a:.2f} -> {b:.2f} m)")
        lines.append(f"     -> Lực tương đương: {resultant:.2f} kN tại x = {x_resultant:.2f} m")

    lines.append("")
    lines.append("--- TẢI PHÂN BỐ TUYẾN TÍNH ---")
    lines.append(f"Số lượng: {len(data.uvls)}")
    for i, (q, a, b) in enumerate(data.uvls, 1):
        span = b - a
        resultant = 0.5 * q * span
        if data.uvl_type == "increase":
            x_resultant = a + 2 * span / 3
            kind = "tăng"
        else:
            x_resultant = a + span / 3
            kind = "giảm"
        lines.append(f"  UVL{i} ({kind}): qmax = {q:.2f} kN/m")
        lines.append(f"     -> Lực tương đương: {resultant:.2f} kN tại x = {x_resultant:.2f} m")

    lines.append("")
    lines.append("--- MOMENT TẬP TRUNG ---")
    lines.append(f"Số lượng: {len(data.point_moments)}")
    for i, (mi, pos) in enumerate(data.point_moments, 1):
        lines.append(f"  M{i} = {mi:.2f} kNm tại x = {pos:.2f} m")

    t_p = sum(load for load, _ in data.point_loads)
    t_udl = sum(q * (b - a) for q, a, b in data.udls)
    t_uvl = sum(0.5 * abs(q) * (b - a) for q, a, b in data.uvls)
    total_vertical_load = t_p + t_udl + t_uvl

    lines.append("")
    if data.beam_type == "simple":
        total_moment_a = sum_p_mom + sum_udl_mom + sum_uvl_mom + sum_m_mom
        lines.append("--- PHÂN TÍCH PHẢN LỰC GỐI (Dầm tựa đơn) ---")
        lines.append("1. Phương trình cân bằng mô-men tại gối trái (x=0):")
        lines.append("   ΣMA = 0  =>  R2*L + ΣMi = 0")
        lines.append(f"   => R2 = -({total_moment_a:.2f}) / {data.length:.2f} = {r2:.2f} kN")
        lines.append("2. Phương trình cân bằng lực theo phương đứng:")
        lines.append("   ΣFy = 0  =>  R1 + R2 + ΣPi = 0")
        lines.append(f"   => R1 = -({total_vertical_load:.2f} + {r2:.2f}) = {r1:.2f} kN")
        if abs(r1 + r2 + total_vertical_load) < 1e-5:
            lines.append("   >> Kiểm tra: ΣFy = 0 (Thỏa mãn)")
    else:
        rv_fixed = -total_vertical_load
        moment_p_l = sum(load * (data.length - pos) for load, pos in data.point_loads)
        moment_udl_l = sum(q * (b - a) * (data.length - (a + (b - a) / 2)) for q, a, b in data.udls)
        moment_uvl_l = 0.0
        for q, a, b in data.uvls:
            span = b - a
            x_resultant = a + (2 * span / 3 if data.uvl_type == "increase" else span / 3)
            moment_uvl_l += 0.5 * q * span * (data.length - x_resultant)
        moment_m_l = sum(mi for mi, _ in data.point_moments)
        total_moment_l = moment_p_l + moment_udl_l + moment_uvl_l + moment_m_l
        mr_fixed = -total_moment_l

        lines.append("--- PHÂN TÍCH PHẢN LỰC TẠI NGÀM (Dầm console) ---")
        lines.append(f"Dầm được ngàm cứng tại vị trí B (x = {data.length:.2f} m)")
        lines.append("1. Phương trình cân bằng lực đứng ΣFy = 0:")
        lines.append(f"   RV + ΣPi = 0  =>  RV = -({total_vertical_load:.2f})")
        lines.append(f"   => Phản lực đứng tại ngàm: RV = {rv_fixed:.2f} kN")
        lines.append("2. Phương trình cân bằng mô-men tại ngàm (x=L):")
        lines.append("   ΣML = 0  =>  MR + Σ(Mi_tải_trọng) = 0")
        lines.append(f"   => MR = -({total_moment_l:.2f})")
        lines.append(f"   => Mô-men phản lực tại ngàm: MR = {mr_fixed:.2f} kNm")
        lines.append("")
        lines.append("* Ghi chú: Biểu đồ tính từ đầu tự do (x=0) nên nội lực bằng 0 tại x=0.")

    idx_v = int(np.argmax(np.abs(shear)))
    idx_m = int(np.argmax(np.abs(moment)))
    idx_w = int(np.argmax(np.abs(deflection)))
    lines.append("")
    lines.append("--- KẾT QUẢ NỘI LỰC ---")
    lines.append("  Lực cắt lớn nhất:")
    lines.append(f"    Vmax = {shear[idx_v]:.2f} kN tại x = {x[idx_v]:.2f} m")
    lines.append("  Mô-men lớn nhất:")
    lines.append(f"    Mmax = {moment[idx_m]:.2f} kNm tại x = {x[idx_m]:.2f} m")
    lines.append("")
    lines.append("  Chuyển vị lớn nhất (Deflection):")
    lines.append(f"    w_max = {deflection[idx_w]:.2f} / EI tại x = {x[idx_w]:.2f} m")
    if data.uvl_type == "decrease" and data.uvls:
        lines.append("")
        lines.append("Ghi chú: giữ nguyên logic MATLAB gốc: phần tích phân chuyển vị cho UVL giảm chưa được bổ sung.")

    return "\n".join(lines)
