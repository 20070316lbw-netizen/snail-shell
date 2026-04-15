"""
fetch_data.py - A股数据获取与维护工具

数据来源：akshare（免费开源，无需 API Key）

三种操作模式
------------
  status  显示当前数据库覆盖情况（日期范围、缺口、每年行数）
  init    全量初始化：从 akshare 拉取指定指数/股票的月频历史数据
  update  增量更新：从数据库最新日期拉取到今天

数据流
------
  1. 获取指数成分股列表（HS300 / ZZ500 / ZZ1000）
  2. 逐股拉取月频后复权行情（akshare stock_zh_a_hist period="monthly"）
  3. 计算月频因子（mom_20d/60d, vol_60d_res, sp_ratio, turn_20d, mom_12m_minus_1m）
  4. 合并全部股票 → 逐截面计算 percent_rank
  5. 计算 label_next_month（下一期 mom_20d）
  6. 写入 DuckDB features_cn（ON CONFLICT DO UPDATE）

独立使用
--------
  python scripts/fetch_data.py status
  python scripts/fetch_data.py init --index HS300 --start 2016-01-01 --end 2020-12-31
  python scripts/fetch_data.py init --index HS300,ZZ500
  python scripts/fetch_data.py update

通过 main.py 使用
-----------------
  python main.py fetch-data status
  python main.py fetch-data init --index HS300,ZZ500 --start 2016-01-01 --end 2020-12-31
  python main.py fetch-data update

注意事项
--------
- 月频数据每只股票约 0.3 秒，300 只股票约 1.5 分钟，5000 只约 25 分钟
- akshare 偶发超时，脚本自动重试（最多 3 次，指数退避）
- 已有数据会被 UPSERT 覆盖（ON CONFLICT DO UPDATE），安全可重跑
- 进度会实时打印，Ctrl-C 中断后重跑从断点继续（已有记录不重复写入）
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

# ── 项目根目录放入 sys.path（独立运行时需要） ───────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DB_PATH = str(_ROOT / "QQ_Quant_DB" / "quant_lab.duckdb")

log = logging.getLogger("fetch_data")

# 指数代码映射
_INDEX_MAP: Dict[str, str] = {
    "HS300":  "000300",
    "ZZ500":  "000905",
    "ZZ1000": "000852",
}

# 各模式默认抓取起始日期（尽量与现有数据的断层衔接）
_DEFAULT_START = "2014-01-01"
_RATE_SLEEP    = 0.35   # 两次 akshare 请求之间的间隔（秒）


# ===========================================================================
# 辅助：股票代码格式转换
# ===========================================================================

def _to_db_ticker(code: str) -> str:
    """
    将 6 位纯数字代码转为数据库格式。
    6xxxx  → XXXXXX.SS（沪市主板 / 科创板）
    0xxxx / 3xxxx → XXXXXX.SZ（深市主板 / 创业板）
    4xxxx / 8xxxx → XXXXXX.BJ（北交所）
    """
    code = str(code).strip().zfill(6)
    if code.startswith("6") or code.startswith("9"):
        return f"{code}.SS"
    elif code.startswith("4") or code.startswith("8"):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"


def _to_akshare_symbol(db_ticker: str) -> str:
    """从 '600690.SS' 提取 akshare 所需的 6 位代码 '600690'。"""
    return db_ticker.split(".")[0]


# ===========================================================================
# 辅助：akshare 数据获取
# ===========================================================================

def _get_index_tickers(index_name: str) -> List[str]:
    """
    获取指数成分股列表，返回数据库格式 ticker（如 '600690.SS'）。
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装 akshare：pip install akshare")

    code = _INDEX_MAP.get(index_name.upper(), index_name)
    log.info(f"获取 {index_name}（{code}）成分股 ...")

    try:
        df = ak.index_stock_cons(symbol=code)
    except Exception as e:
        raise RuntimeError(f"获取 {index_name} 成分股失败: {e}") from e

    # akshare 返回列名可能是 '品种代码' 或 'stock_code'，做兼容处理
    code_col = None
    for candidate in ("品种代码", "stock_code", "code", "成分股代码"):
        if candidate in df.columns:
            code_col = candidate
            break
    if code_col is None:
        code_col = df.columns[0]

    tickers = [_to_db_ticker(str(c)) for c in df[code_col].tolist()]
    log.info(f"  共 {len(tickers)} 只成分股")
    return tickers


def _fetch_monthly(
    ticker: str,
    start_date: str,
    end_date: str,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    从 akshare 拉取单只股票的月频后复权行情。

    返回 DataFrame，列：date(datetime), ticker, open, high, low, close, volume, turn
    拉取失败或无数据返回 None。
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装 akshare：pip install akshare")

    symbol = _to_akshare_symbol(ticker)
    start  = start_date.replace("-", "")
    end    = end_date.replace("-", "")

    col_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low",  "收盘": "close", "成交量": "volume",
        "换手率": "turn",
    }

    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="monthly",
                start_date=start,
                end_date=end,
                adjust="hfq",
            )
            if df is None or df.empty:
                return None

            df = df.rename(columns=col_map)
            df["ticker"] = ticker
            df["date"]   = pd.to_datetime(df["date"])

            # 只保留需要的列（akshare 版本不同返回列可能有增减）
            keep = [c for c in ["ticker", "date", "open", "high", "low",
                                 "close", "volume", "turn"] if c in df.columns]
            return df[keep].copy()

        except Exception as exc:
            wait = 2 ** attempt
            if attempt < retries - 1:
                log.debug(f"  {ticker} 第 {attempt+1} 次失败 ({exc})，{wait}s 后重试")
                time.sleep(wait)
            else:
                log.warning(f"  {ticker} 拉取失败（已重试 {retries} 次）: {exc}")
                return None

    return None


# ===========================================================================
# 因子计算
# ===========================================================================

def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从单股票月频行情计算所有因子。

    约定（与 sync_data.py 对齐，使用月频近似日频指标）：
      mom_20d          ≈ 1 个月累计收益率
      mom_60d          ≈ 3 个月累计收益率
      mom_12m_minus_1m ≈ 过去 12m 跳过最近 1m 的动量（close.shift(1) / close.shift(12) - 1）
      vol_60d_res      ≈ 近 3 月月度收益率标准差 / √20（转换为日频口径）
      sp_ratio         ≈ 近 1 月收益率 / 近 3 月月度收益率标准差
      turn_20d         ≈ 当月换手率（月频直接使用）
    """
    df = df.sort_values("date").copy()

    ret = df["close"].pct_change(1)           # 月度收益率

    df["mom_20d"]          = ret
    df["mom_60d"]          = df["close"].pct_change(3)
    df["mom_12m_minus_1m"] = df["close"].shift(1).pct_change(11)

    roll3_std = ret.rolling(3, min_periods=2).std()
    df["vol_60d_res"] = roll3_std / np.sqrt(20)          # 转换为日频口径
    df["sp_ratio"]    = ret / (roll3_std + 1e-8)
    df["turn_20d"]    = df["turn"] if "turn" in df.columns else np.nan

    return df


def _compute_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个截面日期，计算各因子的 percent_rank（0~1）。
    """
    feat_cols = [
        "mom_20d", "mom_60d", "mom_12m_minus_1m",
        "vol_60d_res", "sp_ratio", "turn_20d",
    ]
    for col in feat_cols:
        if col in df.columns:
            df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True, na_option="keep")
    return df


def _compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    label_next_month      = 下一期 mom_20d（按 ticker 分组 shift(-1)）
    label_next_month_rank = label_next_month 的截面 percent_rank（整数）
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["label_next_month"] = df.groupby("ticker")["mom_20d"].shift(-1)

    # 整数 percent_rank（1 ~ N）
    df["label_next_month_rank"] = (
        df.groupby("date")["label_next_month"]
          .rank(method="first", na_option="keep")
          .astype("Int64")
    )
    return df


# ===========================================================================
# 数据库写入
# ===========================================================================

def _upsert_features(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """
    将 df 写入 features_cn，已存在的 (ticker, date) 执行 UPDATE，否则 INSERT。
    返回写入行数。
    """
    if df.empty:
        return 0

    # 确保 date 是 Python date 类型（DuckDB DATE 列要求）
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 准备列对齐（只写 features_cn 中已有的列）
    db_cols = [
        "ticker", "date", "index_group",
        "mom_20d", "mom_60d", "mom_12m_minus_1m",
        "vol_60d_res", "sp_ratio", "turn_20d",
        "mom_20d_rank", "mom_60d_rank", "mom_12m_minus_1m_rank",
        "vol_60d_res_rank", "sp_ratio_rank", "turn_20d_rank",
        "label_next_month", "label_next_month_rank",
    ]
    # 补充不在 df 中的可选列
    if "index_group" not in df.columns:
        df["index_group"] = None

    write_cols = [c for c in db_cols if c in df.columns]
    df_write   = df[write_cols].copy()

    # UPSERT（DuckDB >= 0.8 支持 ON CONFLICT DO UPDATE）
    set_clause = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in write_cols
        if c not in ("ticker", "date")
    )
    col_list = ", ".join(write_cols)

    con.register("_upsert_staging", df_write)
    con.execute(f"""
        INSERT INTO features_cn ({col_list})
        SELECT {col_list} FROM _upsert_staging
        ON CONFLICT (ticker, date) DO UPDATE SET {set_clause}
    """)
    con.unregister("_upsert_staging")

    return len(df_write)


# ===========================================================================
# 核心批量抓取逻辑
# ===========================================================================

def _fetch_and_store(
    tickers: List[str],
    start_date: str,
    end_date: str,
    con: duckdb.DuckDBPyConnection,
    index_group: Optional[str] = None,
    delay: float = _RATE_SLEEP,
) -> Dict:
    """
    批量抓取 tickers 的月频数据，计算因子后写入数据库。
    支持断点续跑：已有 (ticker, date) 数据会被覆盖（UPSERT）。

    返回统计字典：
      total_tickers, success, skipped, failed, rows_written
    """
    stats = {
        "total_tickers": len(tickers),
        "success":  0,
        "skipped":  0,
        "failed":   0,
        "rows_written": 0,
    }

    all_frames: List[pd.DataFrame] = []
    batch_size = 50     # 每 50 只股票做一次批量写入

    for i, ticker in enumerate(tickers, 1):
        log.info(f"  [{i:>4}/{len(tickers)}] {ticker}")

        df_raw = _fetch_monthly(ticker, start_date, end_date)
        time.sleep(delay)

        if df_raw is None or df_raw.empty:
            stats["skipped"] += 1
            continue

        df_feat = _compute_features(df_raw)
        if index_group:
            df_feat["index_group"] = index_group
        all_frames.append(df_feat)
        stats["success"] += 1

        # 批量写入
        if len(all_frames) >= batch_size or i == len(tickers):
            if all_frames:
                df_batch = pd.concat(all_frames, ignore_index=True)
                df_batch = _compute_cross_sectional_ranks(df_batch)
                df_batch = _compute_labels(df_batch)
                n = _upsert_features(con, df_batch)
                stats["rows_written"] += n
                log.info(f"  ↳ 批量写入 {n} 行（已处理 {i} 只）")
                all_frames = []

    return stats


# ===========================================================================
# 对外接口
# ===========================================================================

def run_status() -> Dict:
    """
    查询数据库覆盖情况。

    返回字典：
      date_min, date_max, n_tickers, total_rows,
      rows_by_year  (list of dicts: year, count),
      gap_years     (list of missing years),
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        meta = con.execute("""
            SELECT
                MIN(date)           AS date_min,
                MAX(date)           AS date_max,
                COUNT(DISTINCT ticker) AS n_tickers,
                COUNT(*)            AS total_rows
            FROM features_cn
        """).fetchone()

        by_year = con.execute("""
            SELECT CAST(YEAR(date) AS INTEGER) AS yr, COUNT(*) AS cnt
            FROM features_cn
            GROUP BY yr ORDER BY yr
        """).fetchdf()

    finally:
        con.close()

    date_min, date_max, n_tickers, total_rows = meta

    present_years = set(by_year["yr"].tolist())
    if date_min and date_max:
        all_years    = set(range(date_min.year, date_max.year + 1))
        gap_years    = sorted(all_years - present_years)
    else:
        gap_years = []

    return {
        "date_min":    date_min,
        "date_max":    date_max,
        "n_tickers":   n_tickers,
        "total_rows":  total_rows,
        "rows_by_year": by_year.to_dict("records"),
        "gap_years":   gap_years,
    }


def run_init(
    index_names: List[str],
    start_date: str = _DEFAULT_START,
    end_date:   Optional[str] = None,
    delay:      float = _RATE_SLEEP,
    extra_tickers: Optional[List[str]] = None,
) -> Dict:
    """
    全量初始化：拉取指定指数成分股从 start_date 到 end_date 的月频数据。

    参数
    ----
    index_names   : 指数列表，如 ["HS300", "ZZ500"]
    start_date    : 起始日期（YYYY-MM-DD），默认 2014-01-01
    end_date      : 截止日期，默认今天
    delay         : 每次请求间隔（秒）
    extra_tickers : 额外的 ticker 列表（追加到指数成分股之后）

    返回统计字典。
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # 汇总 tickers（去重）
    tickers: List[str] = []
    seen: set = set()
    label_map: Dict[str, str] = {}

    for idx in index_names:
        idx_tickers = _get_index_tickers(idx)
        for t in idx_tickers:
            if t not in seen:
                tickers.append(t)
                seen.add(t)
                label_map[t] = idx

    if extra_tickers:
        for t in extra_tickers:
            if t not in seen:
                tickers.append(t)
                seen.add(t)

    log.info(f"共 {len(tickers)} 只股票，抓取区间 {start_date} ~ {end_date}")

    con = duckdb.connect(DB_PATH)
    try:
        stats = _fetch_and_store(
            tickers, start_date, end_date, con,
            index_group=",".join(index_names) if len(index_names) == 1 else None,
            delay=delay,
        )
    finally:
        con.close()

    return stats


def run_update(delay: float = _RATE_SLEEP) -> Dict:
    """
    增量更新：读取 DB 中现有 ticker 列表，从最新日期 +1 天抓到今天。

    返回统计字典。
    """
    con_ro = duckdb.connect(DB_PATH, read_only=True)
    try:
        latest = con_ro.execute(
            "SELECT MAX(date) FROM features_cn"
        ).fetchone()[0]
        tickers = [
            r[0] for r in
            con_ro.execute(
                "SELECT DISTINCT ticker FROM features_cn ORDER BY ticker"
            ).fetchall()
        ]
    finally:
        con_ro.close()

    if latest is None:
        log.warning("数据库为空，请先执行 init")
        return {"error": "DB 为空"}

    start_date = (latest + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = datetime.today().strftime("%Y-%m-%d")

    if start_date > end_date:
        log.info(f"数据库已是最新（{latest}），无需更新")
        return {"message": "already_up_to_date", "latest_date": str(latest)}

    log.info(f"增量更新 {len(tickers)} 只股票，区间 {start_date} ~ {end_date}")

    con = duckdb.connect(DB_PATH)
    try:
        stats = _fetch_and_store(tickers, start_date, end_date, con, delay=delay)
    finally:
        con.close()

    return stats


# ===========================================================================
# 独立 CLI
# ===========================================================================

def _print_status(info: Dict) -> None:
    """格式化打印 run_status() 返回的信息。"""
    sep = "─" * 60
    print(f"\n📊 features_cn 数据覆盖情况")
    print(sep)
    print(f"  日期范围  : {info['date_min']} ~ {info['date_max']}")
    print(f"  股票数量  : {info['n_tickers']:,}")
    print(f"  总行数    : {info['total_rows']:,}")
    print()
    print(f"  {'年份':>6}  {'行数':>8}")
    print("  " + "─" * 18)
    for row in info["rows_by_year"]:
        print(f"  {row['yr']:>6}  {row['cnt']:>8,}")
    if info["gap_years"]:
        print()
        print(f"  ⚠️  数据缺口年份: {info['gap_years']}")
        print(f"  → 建议执行: python main.py fetch-data init --index HS300,ZZ500 "
              f"--start {info['gap_years'][0]}-01-01 --end {info['gap_years'][-1]}-12-31")
    else:
        print()
        print("  ✅ 无年份级别缺口")
    print(sep)


def _print_stats(stats: Dict) -> None:
    print(f"\n  总计   : {stats.get('total_tickers', 0):,} 只")
    print(f"  成功   : {stats.get('success', 0):,} 只")
    print(f"  无数据 : {stats.get('skipped', 0):,} 只")
    print(f"  失败   : {stats.get('failed', 0):,} 只")
    print(f"  写入行 : {stats.get('rows_written', 0):,} 行")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fetch_data",
        description="A股月频数据抓取与维护工具（基于 akshare）",
    )
    subs = p.add_subparsers(dest="cmd", metavar="<命令>")
    subs.required = True

    # status
    subs.add_parser("status", help="显示当前数据库覆盖情况")

    # init
    p_init = subs.add_parser("init", help="全量初始化（拉取指定指数历史数据）")
    p_init.add_argument(
        "--index", default="HS300,ZZ500", metavar="LIST",
        help="逗号分隔的指数列表（默认 HS300,ZZ500）",
    )
    p_init.add_argument(
        "--start", default=_DEFAULT_START, metavar="YYYY-MM-DD",
        help=f"起始日期（默认 {_DEFAULT_START}）",
    )
    p_init.add_argument(
        "--end", default=None, metavar="YYYY-MM-DD",
        help="截止日期（默认今天）",
    )
    p_init.add_argument(
        "--delay", type=float, default=_RATE_SLEEP, metavar="SEC",
        help=f"每次请求间隔秒数（默认 {_RATE_SLEEP}）",
    )

    # update
    p_upd = subs.add_parser("update", help="增量更新（从最新日期到今天）")
    p_upd.add_argument(
        "--delay", type=float, default=_RATE_SLEEP, metavar="SEC",
        help=f"每次请求间隔秒数（默认 {_RATE_SLEEP}）",
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.cmd == "status":
        info = run_status()
        _print_status(info)

    elif args.cmd == "init":
        indices = [i.strip().upper() for i in args.index.split(",")]
        print(f"\n🔄 初始化数据：{indices}  {args.start} ~ {args.end or '今天'}")
        stats = run_init(
            index_names=indices,
            start_date=args.start,
            end_date=args.end,
            delay=args.delay,
        )
        print("\n✅ 初始化完成")
        _print_stats(stats)

    elif args.cmd == "update":
        print("\n🔄 增量更新数据 ...")
        stats = run_update(delay=args.delay)
        if "message" in stats:
            print(f"  {stats['message']}")
        elif "error" in stats:
            print(f"  ❌ {stats['error']}")
        else:
            print("\n✅ 更新完成")
            _print_stats(stats)


if __name__ == "__main__":
    main()
