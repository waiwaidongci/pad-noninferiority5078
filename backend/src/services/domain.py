"""Paired heat-loss acceptance for the registered insulation-pad pilot.

The analysis keeps the registered one-sided Student t interval and adds an
exact sensitivity check: every subset of at most four calibration batches is
withdrawn, the interval is recomputed from the remaining stations, and the
largest upper bound across all of those subsets decides the review outcome.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from heapq import heappush, heapreplace
from itertools import combinations
from math import comb, exp, isfinite, isqrt, lgamma, log, log1p, sqrt
from statistics import NormalDist


PROTOCOL_ID = "HR-PAD-20261110-V1"
REGISTERED_AT = datetime(2026, 11, 10, 9)
WINDOW_FORMAT = "%Y-%m-%d %H:%M"
MARGIN_KW = 0.50
CONFIDENCE = 0.95
MIN_PAIRS = 12
MAX_REVIEW_MINIMUM = 1000
MAX_WITHDRAWN_BATCHES = 4
# 60 batches (523,686 subsets) enumerate in about one second; the cap leaves
# headroom while keeping the worst case comfortably inside one minute.
MAX_ENUMERABLE_BATCHES = 100
MAX_FLAGGED_COMBINATIONS = 100

REASON_STATIONS = "有效站点不足"
REASON_BOUND = "上界算不出"
REASON_MARGIN = "上界达到或超过界值"
REASONS = (REASON_STATIONS, REASON_BOUND, REASON_MARGIN)


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


def sample_batch_rows() -> list[dict]:
    """Six calibration batches of six stations for the demo ledger."""
    return [
        {
            "station_id": f"HX-{number:02d}",
            "calibration_batch": f"CAL-{(number - 1) // 6 + 1:02d}",
        }
        for number in range(1, 37)
    ]


def _regularized_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) via Lentz's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = lgamma(a + b) - lgamma(a) - lgamma(b)
    front = exp(log_beta + a * log(x) + b * log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def _beta_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    epsilon = 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for step in range(1, 1000):
        doubled = 2 * step
        factor = step * (b - step) * x / ((qam + doubled) * (a + doubled))
        d = 1.0 + factor * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + factor / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        factor = -(a + step) * (qab + step) * x / ((a + doubled) * (qap + doubled))
        d = 1.0 + factor * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + factor / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < epsilon:
            return result
    raise ArithmeticError("t 分位数迭代未收敛。")


def _t_tail(value: float, df: int) -> float:
    """P(T > value) for value >= 0 with df degrees of freedom."""
    return 0.5 * _regularized_beta(df / 2.0, 0.5, df / (df + value * value))


def _t_density(value: float, df: int) -> float:
    return exp(
        lgamma((df + 1) / 2)
        - lgamma(df / 2)
        - 0.5 * log(df * 3.141592653589793)
        - (df + 1) / 2 * log1p(value * value / df)
    )


_NORMAL_QUANTILE = NormalDist().inv_cdf(CONFIDENCE)


@lru_cache(maxsize=None)
def _critical_t(df: int) -> float:
    """One-sided 95% Student t quantile, solved with safeguarded Newton.

    The tail probability is evaluated to about 1e-15; Newton's step therefore
    bottoms out on a small plateau. The last iterate is returned instead of a
    stale bisection midpoint so the plateau cannot leak into the bracket.
    """
    target = 1.0 - CONFIDENCE
    z = _NORMAL_QUANTILE
    z2, z3 = z * z, z ** 3
    z5, z7, z9 = z ** 5, z ** 7, z ** 9
    estimate = (
        z
        + (z3 + z) / (4 * df)
        + (5 * z5 + 16 * z3 + 3 * z) / (96 * df ** 2)
        + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * df ** 3)
        + (79 * z9 + 776 * z7 + 1482 * z5 - 1920 * z3 - 945 * z) / (92160 * df ** 4)
    )
    low = 0.0
    high = max(estimate, 1.0)
    while _t_tail(high, df) > target:
        high *= 2
    if not low < estimate < high:
        estimate = (low + high) / 2
    t = estimate
    for _ in range(96):
        tail = _t_tail(t, df)
        if tail > target:
            low = t
        else:
            high = t
        if high - low <= 1e-15 * max(1.0, high):
            break
        density = _t_density(t, df)
        candidate = t + (tail - target) / density if density > 0 else (low + high) / 2
        if not (low < candidate < high) or not isfinite(candidate):
            candidate = (low + high) / 2
        if candidate == t:
            return t
        t = candidate
    return t


def _read_identifier(row: dict, key: str, label: str, prefix: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}{label}为空，请补回原始记录中的编号。")
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


def _validate_pairs(rows: list[dict]) -> tuple[list[dict], list[float], list[str]]:
    pair_ids, station_ids = set(), set()
    details, differences, stations = [], [], []
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
        stations.append(station)
        details.append({
            "登记方案": PROTOCOL_ID,
            "配对编号": pair,
            "换热站编号": station,
            "测量窗口（北京时间）": window_text,
            "原垫热损（kW）": baseline,
            "试装热损（kW）": trial,
            "配对热损差（kW）": difference,
        })
    return details, differences, stations


def station_batch_map(batches) -> dict:
    """Normalize the station → calibration batch correspondence.

    Accepts a CSV-like list of ``{"station_id", "calibration_batch"}`` records
    or a plain ``{station_id: calibration_batch}`` mapping. Returns
    ``{casefolded station: (station label, batch key, batch label)}``.
    """
    if isinstance(batches, dict):
        pairs = [{"station_id": station, "calibration_batch": batch} for station, batch in batches.items()]
    elif isinstance(batches, (list, tuple)):
        pairs = list(batches)
    else:
        raise ValueError("请导入站点与校准批次的完整对应表。")
    if not pairs:
        raise ValueError("站点与校准批次对应表为空，请导入完整的对应表。")

    mapping: dict = {}
    batch_labels: dict = {}
    for line_number, row in enumerate(pairs, start=2):
        prefix = f"对应表第 {line_number} 行，"
        if not isinstance(row, dict):
            raise ValueError(f"{prefix}记录格式有误，请使用 station_id、calibration_batch 两列。")
        station = _read_identifier(row, "station_id", "换热站编号", prefix)
        batch = _read_identifier(row, "calibration_batch", "校准批次编号", prefix)
        station_key = station.casefold()
        batch_key = batch.casefold()
        prefix = f"对应表第 {line_number} 行（{station} / {batch}），"
        if station_key in mapping:
            raise ValueError(f"{prefix}换热站编号重复，请确保每座换热站只归属一个校准批次。")
        label = batch_labels.setdefault(batch_key, batch)
        mapping[station_key] = (station, batch_key, label)
    if not batch_labels:
        raise ValueError("站点与校准批次对应表为空，请导入完整的对应表。")
    return mapping


def check_correspondence(stations: list[str], mapping: dict) -> None:
    """Reject a batch table that misses or adds stations relative to the data."""
    data_keys = {}
    for station in stations:
        data_keys.setdefault(station.casefold(), station)
    missing = [data_keys[key] for key in data_keys if key not in mapping]
    extra = [mapping[key][0] for key in mapping if key not in data_keys]
    if not missing and not extra:
        return
    parts = []
    if missing:
        sample = "、".join(missing[:6]) + (" 等" if len(missing) > 6 else "")
        parts.append(f"少配 {len(missing)} 座换热站（{sample}）")
    if extra:
        sample = "、".join(extra[:6]) + (" 等" if len(extra) > 6 else "")
        parts.append(f"多配 {len(extra)} 座换热站（{sample}）")
    raise ValueError(
        "站点与校准批次对应表与本次测量记录不一致：" + "；".join(parts)
        + "。请补齐或移除对应记录后重新导入对应表。"
    )


def _scaled_differences(differences: list[float]) -> tuple[list[int], int]:
    """Represent the float differences as integers over a common power-of-two scale."""
    ratios = [value.as_integer_ratio() for value in differences]
    scale = max(denominator for _, denominator in ratios)
    return [numerator * (scale // denominator) for numerator, denominator in ratios], scale


def _batch_totals(differences: list[float], stations: list[str], mapping: dict):
    batch_order: list[str] = []
    batch_labels: dict[str, str] = {}
    station_counts: dict[str, int] = {}
    batch_of_station: dict[str, str] = {}
    for station in stations:
        key = station.casefold()
        if key not in mapping:
            raise ValueError(f"换热站 {station} 未在校准批次对应表中，请补齐后重新导入。")
        _, batch_key, batch_label = mapping[key]
        if batch_key not in batch_labels:
            batch_labels[batch_key] = batch_label
            batch_order.append(batch_key)
            station_counts[batch_key] = 0
        station_counts[batch_key] += 1
        batch_of_station[key] = batch_key

    scaled, scale = _scaled_differences(differences)
    sums = {key: 0 for key in batch_order}
    squares = {key: 0 for key in batch_order}
    for station, value in zip(stations, scaled):
        key = batch_of_station[station.casefold()]
        sums[key] += value
        squares[key] += value * value
    return batch_order, batch_labels, station_counts, sums, squares, scale


def _sqrt_fraction(numerator: int, denominator: int) -> float:
    """Correctly rounded square root of a positive fraction.

    Mirrors CPython's ``statistics._float_sqrt_of_frac`` (Hoyt's round-to-odd
    integer square root) so the variance comes out with the same float as the
    original ``statistics.stdev`` pipeline, but from our already-scaled sums.
    """
    quotient_shift = (numerator.bit_length() - denominator.bit_length() - 109) // 2
    if quotient_shift >= 0:
        scaled, divisor = numerator, denominator << 2 * quotient_shift
        root = isqrt(scaled // divisor)
        root |= root * root * divisor != scaled
        return (root << quotient_shift) / 1
    scaled, divisor = numerator << -2 * quotient_shift, denominator
    root = isqrt(scaled // divisor)
    root |= root * root * divisor != scaled
    return root / (1 << -quotient_shift)


def _describe_batch(batch_order, batch_labels, station_counts) -> list[dict]:
    return [
        {"校准批次": batch_labels[key], "站点数": station_counts[key]}
        for key in batch_order
    ]


def _interval(count: int, total: int, total_of_squares: int, scale: int):
    """Mean, spread, standard error and one-sided upper bound from integer sums.

    All arithmetic is done on the exactly scaled integer readings, so the
    base run and every withdrawal subset use the same unrounded pipeline.
    Returns ``(average, spread, standard_error, upper)`` with ``None`` fields
    when the t interval cannot be computed.
    """
    if count < 2:
        return None, None, None, None
    numerator = count * total_of_squares - total * total
    if numerator < 0:
        return None, None, None, None
    try:
        denominator = count * (count - 1) * scale * scale
        spread = _sqrt_fraction(numerator, denominator)
        average = total / (count * scale)
        standard_error = spread / sqrt(count)
        upper = average + _critical_t(count - 1) * standard_error if standard_error > 0 else None
    except (OverflowError, ZeroDivisionError):
        return None, None, None, None
    if not (isfinite(average) and isfinite(spread) and isfinite(standard_error)):
        return None, None, None, None
    if upper is not None and not isfinite(upper):
        upper = None
    return average, spread, standard_error, upper


def _flagged_row(batches, stations, upper, reasons) -> dict:
    return {
        "撤回批次": "、".join(batches) if batches else "不撤回",
        "撤回批次数": len(batches),
        "剩余站点数": stations,
        "单侧95%置信上界（kW）": upper if upper is not None else "算不出",
        "触发原因": "；".join(reasons),
    }


def _flagged_rank(item: dict) -> tuple:
    """Sort ascending so the most severe cases come first.

    Uncountable bounds rank ahead of countable ones; within each group the
    highest bound comes first, then the smaller withdrawal set, then the
    batch labels for a stable order.
    """
    upper = item["upper"]
    return (
        0 if upper is None else 1,
        -(upper if upper is not None else 0.0),
        len(item["batches"]),
        item["batches"],
    )


def _withdrawal_scan(differences, stations, mapping, min_pairs) -> dict:
    """Exactly enumerate every withdrawal of 0–4 calibration batches."""
    batch_order, batch_labels, station_counts, sums, squares, scale = _batch_totals(
        differences, stations, mapping
    )
    batch_count = len(batch_order)
    if batch_count > MAX_ENUMERABLE_BATCHES:
        raise ValueError(
            f"校准批次共 {batch_count} 批，超过可在一分钟内精确枚举的 {MAX_ENUMERABLE_BATCHES} 批上限，"
            "请合并批次或拆分测量台账后重新提交。"
        )

    total_n = sum(station_counts.values())
    total_sum = sum(sums.values())
    total_squares = sum(squares.values())
    limit = min(MAX_WITHDRAWN_BATCHES, batch_count)
    combination_count = sum(comb(batch_count, size) for size in range(limit + 1))
    counts = [station_counts[key] for key in batch_order]
    offsets = [sums[key] for key in batch_order]
    offsets_sq = [squares[key] for key in batch_order]
    labels = [batch_labels[key] for key in batch_order]

    by_reason = {reason: 0 for reason in REASONS}
    flagged_total = 0
    kept: list[tuple] = []
    computable = 0
    highest_upper = None
    highest_batches: list[str] = []
    highest_stations = 0
    depth_rows: dict[int, dict] = {
        size: {"撤批次": size, "组合数": 0, "可计算上界": 0, "暂缓组合": 0, "最高上界（kW）": None}
        for size in range(limit + 1)
    }
    for size in range(limit + 1):
        depth = depth_rows[size]
        for combination in combinations(range(batch_count), size):
            count = total_n
            total = total_sum
            total_of_squares = total_squares
            for index in combination:
                count -= counts[index]
                total -= offsets[index]
                total_of_squares -= offsets_sq[index]
            withdrawn = [labels[index] for index in combination]
            depth["组合数"] += 1

            _, _, _, upper = _interval(count, total, total_of_squares, scale)
            reasons = []
            if count < min_pairs:
                reasons.append(REASON_STATIONS)
            if upper is None:
                reasons.append(REASON_BOUND)
            elif upper >= MARGIN_KW:
                reasons.append(REASON_MARGIN)

            if upper is not None:
                computable += 1
                depth["可计算上界"] += 1
                # Strict ">" keeps the earliest enumerated withdrawal on ties,
                # matching the overall highest-bound rule.
                if depth["最高上界（kW）"] is None or upper > depth["最高上界（kW）"]:
                    depth["最高上界（kW）"] = upper
                    depth["最高上界撤回批次"] = "、".join(withdrawn) if withdrawn else "不撤回"
                if highest_upper is None or upper > highest_upper:
                    highest_upper = upper
                    highest_batches = withdrawn
                    highest_stations = count
            if reasons:
                flagged_total += 1
                depth["暂缓组合"] += 1
                for reason in reasons:
                    by_reason[reason] += 1
                item = {"batches": withdrawn, "stations": count, "upper": upper, "reasons": reasons}
                rank = _flagged_rank(item)
                if len(kept) < MAX_FLAGGED_COMBINATIONS:
                    heappush(kept, (rank, item))
                elif rank < kept[0][0]:
                    heapreplace(kept, (rank, item))

    flagged = [item for _, item in sorted(kept, key=lambda pair: pair[0])]
    highest = None
    if highest_upper is not None:
        highest = {
            "upper": highest_upper,
            "batches": highest_batches,
            "stations": highest_stations,
        }
    return {
        "max_batches": MAX_WITHDRAWN_BATCHES,
        "batch_count": batch_count,
        "combination_count": combination_count,
        "computable_count": computable,
        "highest": highest,
        "flagged_count": flagged_total,
        "flagged_truncated": flagged_total > len(flagged),
        "flagged_by_reason": dict(by_reason),
        "flagged": flagged,
        "by_depth": [depth_rows[size] for size in range(limit + 1)],
        "batches": _describe_batch(batch_order, batch_labels, station_counts),
    }


def _panels(withdrawal: dict) -> list[dict]:
    highest = withdrawal["highest"]
    if highest is None:
        summary = "没有任何撤回组合能算出上界。"
        highest_row = {
            "撤回批次": "—",
            "撤回批次数": "—",
            "剩余站点数": "—",
            "最高单侧95%置信上界（kW）": "算不出",
        }
    else:
        withdrawn = highest["batches"]
        summary = (
            "撤回 " + ("、".join(withdrawn) if withdrawn else "不撤回")
            + f"（{len(withdrawn)} 批）后剩余 {highest['stations']} 座站点时上界最高。"
        )
        highest_row = {
            "撤回批次": "、".join(withdrawn) if withdrawn else "不撤回",
            "撤回批次数": len(withdrawn),
            "剩余站点数": highest["stations"],
            "最高单侧95%置信上界（kW）": highest["upper"],
        }
    flagged_rows = [_flagged_row(item["batches"], item["stations"], item["upper"], item["reasons"])
                    for item in withdrawal["flagged"]]
    return [
        {
            "title": "撤回校准批次后的最高上界",
            "summary": summary,
            "columns": [
                {"key": "撤回批次", "label": "撤回批次"},
                {"key": "撤回批次数", "label": "撤回批次数"},
                {"key": "剩余站点数", "label": "剩余站点数"},
                {"key": "最高单侧95%置信上界（kW）", "label": "最高单侧95%置信上界（kW）"},
            ],
            "rows": [highest_row],
        },
        {
            "title": "各撤回批次数下的上界",
            "summary": "每一档都逐一重算全部组合，最高上界取未舍入数值。",
            "columns": [
                {"key": "撤批次", "label": "撤回批次数"},
                {"key": "组合数", "label": "组合数"},
                {"key": "可计算上界", "label": "可计算上界的组合"},
                {"key": "暂缓组合", "label": "触发暂缓的组合"},
                {"key": "最高上界（kW）", "label": "最高上界（kW）"},
                {"key": "最高上界撤回批次", "label": "最高上界对应撤回批次"},
            ],
            "rows": [
                {**row, "最高上界（kW）": row["最高上界（kW）"] if row["最高上界（kW）"] is not None else "算不出"}
                for row in withdrawal["by_depth"]
            ],
        },
        {
            "title": "触发暂缓的组合",
            "summary": (
                f"共 {withdrawal['flagged_count']} 个撤回组合触发暂缓；"
                f"站数不足 {withdrawal['flagged_by_reason'][REASON_STATIONS]} 个、"
                f"上界算不出 {withdrawal['flagged_by_reason'][REASON_BOUND]} 个、"
                f"达到或超过界值 {withdrawal['flagged_by_reason'][REASON_MARGIN]} 个。"
                + (f"下表列出前 {len(flagged_rows)} 个。" if withdrawal["flagged_truncated"] else "")
            ) if withdrawal["flagged_count"] else "没有撤回组合触发暂缓。",
            "columns": [
                {"key": "撤回批次", "label": "撤回批次"},
                {"key": "撤回批次数", "label": "撤回批次数"},
                {"key": "剩余站点数", "label": "剩余站点数"},
                {"key": "单侧95%置信上界（kW）", "label": "单侧95%置信上界（kW）"},
                {"key": "触发原因", "label": "触发原因"},
            ],
            "rows": flagged_rows,
        },
        {
            "title": "校准批次站点数",
            "summary": "同一批次的仪表可能一起失准，撤回以整批为单位。",
            "columns": [
                {"key": "校准批次", "label": "校准批次"},
                {"key": "站点数", "label": "站点数"},
            ],
            "rows": withdrawal["batches"],
        },
    ]


def compute(rows: list[dict], options: dict, batches) -> dict:
    if options.keys() - {"min_pairs"}:
        raise ValueError("请只配置复核最少配对数。界值和置信水平由试装前登记的方案确定。")
    min_pairs = options.get("min_pairs", MIN_PAIRS)
    if type(min_pairs) is not int or not MIN_PAIRS <= min_pairs <= MAX_REVIEW_MINIMUM:
        raise ValueError(f"请将复核最少配对数设为 {MIN_PAIRS} 至 {MAX_REVIEW_MINIMUM} 之间的整数。")
    if len(rows) < 2:
        raise ValueError("请导入至少两座换热站的完整配对数据，才能计算配对差的样本标准差。")

    mapping = station_batch_map(batches)
    details, differences, stations = _validate_pairs(rows)
    check_correspondence(stations, mapping)

    n = len(differences)
    scaled, scale = _scaled_differences(differences)
    total = sum(scaled)
    total_of_squares = sum(value * value for value in scaled)
    try:
        average, spread, standard_error, upper = _interval(n, total, total_of_squares, scale)
    except OverflowError as error:
        raise ValueError("配对差超出可计算范围，请核对热损单位和异常读数。") from error
    if average is None:
        raise ValueError("配对差超出可计算范围，请核对热损单位和异常读数。")
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
    if standard_error > 0 and upper is not None:
        metrics["单侧 t 临界值"] = _critical_t(n - 1)
        metrics["单侧95%置信上界（kW）"] = upper
        points.append({"label": "单侧95%置信上界", "value": upper})
    else:
        upper = None
    points.append({"label": "预登记非劣效界值", "value": MARGIN_KW})

    withdrawal = _withdrawal_scan(differences, stations, mapping, min_pairs)
    highest = withdrawal["highest"]
    metrics["校准批次数"] = withdrawal["batch_count"]
    metrics["撤回批次上限"] = withdrawal["max_batches"]
    metrics["撤回组合总数"] = withdrawal["combination_count"]
    metrics["可计算上界的组合数"] = withdrawal["computable_count"]
    metrics["触发暂缓的组合数"] = withdrawal["flagged_count"]
    if highest is not None:
        metrics["最高单侧95%置信上界（kW）"] = highest["upper"]
        withdrawn = highest["batches"]
        metrics["最高上界对应撤回批次"] = "、".join(withdrawn) if withdrawn else "不撤回"
        metrics["最高上界剩余站点数"] = highest["stations"]
        points.append({"label": "最高单侧95%置信上界", "value": highest["upper"]})
    notes.extend([
        f"同一校准批次的仪表可能一起失准：复核对撤回 0 至 {withdrawal['max_batches']} 个校准批次后的剩余站点逐一按原口径重算单侧上界，"
        f"共 {withdrawal['combination_count']} 个组合；任一组合站点不足、上界算不出或达到界值即暂缓验收。",
    ])
    if highest is not None:
        withdrawn = highest["batches"]
        notes.append(
            "最高上界为 " + (f"{highest['upper']:.6f} kW" if isfinite(highest["upper"]) else "无法计算")
            + "，对应撤回 " + ("、".join(withdrawn) if withdrawn else "不撤回")
            + f"（{len(withdrawn)} 批），剩余 {highest['stations']} 座站点；"
            + ("低于" if highest["upper"] < MARGIN_KW else "未低于")
            + f" {MARGIN_KW:.2f} kW 界值。最高上界并列时取撤回合集字典序最前者。"
        )
    else:
        notes.append("没有任何撤回组合能算出上界，无法完成撤回复核。")
    if withdrawal["flagged_count"]:
        notes.append(
            f"共 {withdrawal['flagged_count']} 个撤回组合触发暂缓："
            f"有效站点不足 {withdrawal['flagged_by_reason'][REASON_STATIONS]} 个、"
            f"上界算不出 {withdrawal['flagged_by_reason'][REASON_BOUND]} 个、"
            f"达到或超过界值 {withdrawal['flagged_by_reason'][REASON_MARGIN]} 个。"
            + ("分析页面仅列出前 100 个组合。" if withdrawal["flagged_truncated"] else "")
        )
    else:
        notes.append("全部撤回组合均达到复核门槛且上界严格低于界值。")

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
    elif withdrawal["flagged_count"]:
        headline = "撤回校准批次后复核未通过，暂缓验收"
    else:
        headline = "达到配对非劣效验收条件，可提交复核"
        can_publish = True

    return {
        "headline": headline,
        "metrics": metrics,
        "rows": details,
        "notes": notes,
        "chart": {"label": "试装热损差与非劣效界值", "unit": "kW", "points": points},
        "panels": _panels(withdrawal),
        "withdrawal": withdrawal,
        "can_publish": can_publish,
    }
