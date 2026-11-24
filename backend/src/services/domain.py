"""Paired heat-loss acceptance for the registered insulation-pad pilot."""

from datetime import datetime, timedelta
from decimal import Decimal
from math import atan2, hypot, isfinite, pi, sqrt
from statistics import mean, stdev


PROTOCOL_ID = "HR-PAD-20261110-V1"
REGISTERED_AT = datetime(2026, 11, 10, 9)
WINDOW_FORMAT = "%Y-%m-%d %H:%M"
MARGIN_KW = 0.50
CONFIDENCE = 0.95
MIN_PAIRS = 12
MAX_REVIEW_MINIMUM = 1000


def sample_rows() -> list[dict]:
    differences = (
        -18, 12, 48, -33, 25, 7, -42, 31, -12, 58, -5, 23,
        9, -27, 40, -16, 35, -8, 14, 51, -36, 17, -22, 4,
        29, -45, 11, 46, -13, 5, 33, -20, 54, -30, 20, -2,
    )
    first_window = datetime(2026, 11, 12, 8)
    rows = []
    for index, difference in enumerate(differences):
        station_number = index + 1
        window = first_window + timedelta(days=index // 3, hours=3 * (index % 3))
        baseline_cents = 812 + (index * 47) % 361
        rows.append({
            "protocol_id": PROTOCOL_ID,
            "pair_id": f"PAD-{station_number:03d}",
            "station_id": f"HX-{station_number:02d}",
            "window_start": window.strftime(WINDOW_FORMAT),
            "baseline_loss_kw": baseline_cents / 100,
            "trial_loss_kw": (baseline_cents + difference) / 100,
        })
    return rows


def _central_t_probability(value: float, df: int) -> float:
    # With t = sqrt(df) * tan(theta), the density is proportional to
    # cos(theta)**(df - 1). Integrate by parts from 0 to theta and pi/2.
    root_df = sqrt(df)
    scale = hypot(root_df, value)
    sine, cosine = value / scale, root_df / scale
    parity = (df - 1) % 2
    integral, total = (sine, 1.0) if parity else (atan2(value, root_df), pi / 2)
    boundary = sine * cosine ** (parity + 1)
    cosine_squared = cosine * cosine
    for exponent in range(parity + 2, df, 2):
        integral = (boundary + (exponent - 1) * integral) / exponent
        total *= (exponent - 1) / exponent
        boundary *= cosine_squared
    return integral / total


def _critical_t(df: int) -> float:
    target = 2 * CONFIDENCE - 1
    # The df=1 (Cauchy) 95th percentile is below 8 and is the largest.
    low, high = 0.0, 8.0
    for _ in range(54):
        middle = (low + high) / 2
        if _central_t_probability(middle, df) < target:
            low = middle
        else:
            high = middle
    return high


def _read_identifier(row: dict, key: str, label: str, prefix: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}{label}为空，请补回试装原始记录中的编号。")
    return value.strip()


def _read_loss(row: dict, key: str, label: str, prefix: str) -> float:
    value = row.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"{prefix}{label}必须是数值，请补录以 kW 为单位的热损。")
    try:
        loss = float(value)
    except OverflowError as error:
        raise ValueError(f"{prefix}{label}超出数值范围，请核对读数和 kW 单位。") from error
    if not isfinite(loss) or loss < 0:
        raise ValueError(f"{prefix}{label}必须是有限且非负的 kW 值，请核对原始读数。")
    return loss


def _validate_pairs(rows: list[dict]) -> tuple[list[dict], list[float]]:
    pair_ids, station_ids = set(), set()
    details, differences = [], []
    for line_number, row in enumerate(rows, start=2):
        prefix = f"CSV 第 {line_number} 行，"
        if row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{prefix}方案编号有误，请只导入已登记方案 {PROTOCOL_ID} 的配对记录。")
        pair = _read_identifier(row, "pair_id", "配对编号", prefix)
        station = _read_identifier(row, "station_id", "换热站编号", prefix)
        prefix = f"CSV 第 {line_number} 行（{pair} / {station}），"
        if pair.casefold() in pair_ids:
            raise ValueError(f"{prefix}配对编号重复，请核对台账并移除重复记录。")
        if station.casefold() in station_ids:
            raise ValueError(f"{prefix}站点重复，请按登记方案每座换热站只保留预定的一个配对窗口。")

        window_text = row.get("window_start")
        try:
            window = datetime.strptime(window_text, WINDOW_FORMAT)
            if window.strftime(WINDOW_FORMAT) != window_text:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}测量窗口格式有误，请使用北京时间 YYYY-MM-DD HH:MM。") from error
        if window <= REGISTERED_AT:
            raise ValueError(f"{prefix}测量窗口须晚于方案登记时间，请核对原始时间并剔除登记前记录。")

        baseline = _read_loss(row, "baseline_loss_kw", "原垫热损", prefix)
        trial = _read_loss(row, "trial_loss_kw", "试装热损", prefix)
        # Subtract decimal readings before converting the difference to float.
        # This preserves identical differences across different meter readings.
        difference = float(Decimal(str(trial)) - Decimal(str(baseline)))
        pair_ids.add(pair.casefold())
        station_ids.add(station.casefold())
        differences.append(difference)
        details.append({
            "登记方案": PROTOCOL_ID,
            "配对编号": pair,
            "换热站编号": station,
            "测量窗口（北京时间）": window_text,
            "原垫热损（kW）": baseline,
            "试装热损（kW）": trial,
            "配对热损差（kW）": difference,
        })
    return details, differences


def compute(rows: list[dict], options: dict) -> dict:
    if options.keys() - {"min_pairs"}:
        raise ValueError("请只配置复核最少配对数。界值和置信水平由试装前登记的方案确定。")
    min_pairs = options.get("min_pairs", MIN_PAIRS)
    if type(min_pairs) is not int or not MIN_PAIRS <= min_pairs <= MAX_REVIEW_MINIMUM:
        raise ValueError(f"请将复核最少配对数设为 {MIN_PAIRS} 至 {MAX_REVIEW_MINIMUM} 之间的整数。")
    if len(rows) < 2:
        raise ValueError("请导入至少两座换热站的完整配对数据，才能计算配对差的样本标准差。")

    details, differences = _validate_pairs(rows)
    n = len(differences)
    average = mean(differences)
    try:
        spread = stdev(differences)
    except OverflowError as error:
        raise ValueError("配对差超出可计算范围，请核对热损单位和异常读数。") from error
    standard_error = spread / sqrt(n)
    metrics = {
        "有效配对数": n,
        "复核最少配对数": min_pairs,
        "自由度": n - 1,
        "单侧置信水平（%）": CONFIDENCE * 100,
        "预登记非劣效界值（kW）": MARGIN_KW,
        "平均配对热损差（kW）": average,
        "配对差样本标准差（kW）": spread,
        "平均差标准误（kW）": standard_error,
    }
    points = [{"label": "平均配对热损差", "value": average}]
    notes = [
        f"方案 {PROTOCOL_ID} 于 {REGISTERED_AT.strftime(WINDOW_FORMAT)}（北京时间）登记。",
        "热损差＝试装热损−原垫热损。正值表示试装后热损增加。每座站点只计入预定负荷窗口的一组配对。",
        "原假设为总体平均热损差 ≥ 登记界值。单侧上界＝平均差＋t(0.95, n−1)×标准误，使用未舍入的上界判定。",
        "配对 t 推断要求各站点配对差相互独立，小样本下差值应近似正态。界值或试验设计变更需在下一轮采集前登记。",
    ]
    upper = None
    if standard_error > 0:
        critical = _critical_t(n - 1)
        upper = average + critical * standard_error
        if not isfinite(upper):
            raise ValueError("置信上界超出可计算范围，请核对热损单位和异常读数。")
        metrics["单侧 t 临界值"] = critical
        metrics["单侧95%置信上界（kW）"] = upper
        points.append({"label": "单侧95%置信上界", "value": upper})
    points.append({"label": "预登记非劣效界值", "value": MARGIN_KW})

    if n < min_pairs:
        notes.append(f"当前 {n} 组配对，还需补采 {min_pairs - n} 座独立换热站的有效配对。")
    if upper is None:
        notes.append("配对差标准误为零，无法进行有效的 t 推断。请核对仪表精度和配对记录，并补采独立站点。")

    can_publish = False
    if n < min_pairs:
        headline = "配对数未达复核门槛，暂缓验收"
    elif upper is None:
        headline = "配对差无法进行 t 推断，暂缓验收"
    elif upper >= MARGIN_KW:
        headline = "置信上界达到或超过登记界值，非劣效未获证实"
        notes.append("请复核配对负荷与仪表记录，再安排新的试装确认。")
    else:
        headline = "达到配对非劣效验收条件，可提交复核"
        can_publish = True

    return {
        "headline": headline,
        "metrics": metrics,
        "rows": details,
        "notes": notes,
        "chart": {"label": "试装热损差与非劣效界值", "unit": "kW", "points": points},
        "can_publish": can_publish,
    }
