"""
INVESTMENT STRATEGY DASHBOARD
========================================================================================
Institutional Quantitative Portfolio Strategy & Backtest Dashboard
Framework 4 Giai đoạn:
  Stage 1 — Data & Stock Universe (Sàng lọc từ 99 mã -> 30 mã -> Top 20 Eligible)
  Stage 2 — Stock Selection (4-Factor Multi-Factor Ranking & Correlation Filter)
  Stage 3 — Portfolio Strategy:
            + Phần 1: Cách tiếp cận bằng Shrinkage (Ledoit-Wolf Min-Vol)
            + Phần 2: Cách tiếp cận bằng Market Timing (VNINDEX SMA200)
            + Phần 3: Mô hình kết hợp Shrinkage + Market Timing
  Stage 4 — Backtest & Strategy Decision (Out-of-Sample 2022 Verification)
  Stage 5 — Tổng Quan & Khuyến Nghị (Final Investment Strategy Recommendation)
========================================================================================
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# PyPortfolioOpt for Ledoit-Wolf Shrinkage
try:
    from pypfopt import risk_models, EfficientFrontier
except ImportError:
    st.error("PyPortfolioOpt is required. Please install it using: pip install pyportfolioopt")

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & INSTITUTIONAL THEME
# ==============================================================================
st.set_page_config(
    page_title="Investment Strategy Dashboard | Quản Lý Danh Mục Đầu Tư",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS styling - Refined Typography & Layout
st.markdown("""
<style>
    /* Global font & styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 14px;
        line-height: 1.6;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Top Header Banner */
    .top-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: #F8FAFC;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }
    .top-banner h1 {
        color: #F8FAFC !important;
        font-size: 24px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.4px;
    }
    .top-banner p {
        color: #94A3B8;
        font-size: 13.5px;
        margin: 0;
        line-height: 1.55;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 12px;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
        word-break: break-word;
    }
    .kpi-sub {
        font-size: 11.5px;
        color: #94A3B8;
        line-height: 1.4;
    }
    .kpi-positive {
        color: #059669 !important;
    }
    .kpi-negative {
        color: #DC2626 !important;
    }
    .kpi-highlight {
        color: #2563EB !important;
    }

    /* Executive Callout Box */
    .exec-box {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 14px 0;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        font-size: 13.5px;
        line-height: 1.6;
    }
    .exec-box-title {
        font-weight: 700;
        font-size: 14.5px;
        color: #0F172A;
        margin-bottom: 6px;
    }
    
    /* Strategy recommendation alert */
    .rec-box {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #059669;
        border-radius: 6px;
        padding: 16px 20px;
        margin: 14px 0;
        color: #064E3B;
        font-size: 13.5px;
        line-height: 1.6;
    }

    /* Filter Step Funnel Card */
    .funnel-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .funnel-step {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .funnel-step-blue {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .funnel-step-amber {
        background: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .funnel-step-green {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .funnel-title {
        font-size: 16px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .funnel-desc {
        font-size: 13px;
        color: #475569;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Table styles */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CONSTANTS & QUANT PARAMETERS
# ==============================================================================
DEFAULT_TRADING_DAYS = 252
DEFAULT_RF_ANNUAL = 0.03
DEFAULT_INITIAL_CAPITAL = 100_000_000
DEFAULT_COST = 0.0015
DEFAULT_SMA_WINDOW = 200
DEFAULT_MAX_CORR = 0.70
SCORE_WEIGHTS = {"trend": 0.30, "momentum": 0.30, "risk": 0.20, "liquidity": 0.20}


# ==============================================================================
# DATA LOADING & COMPUTATION PIPELINE (CACHED)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_path="HOSE_2020_2023_in.csv"):
    """
    Đọc dữ liệu CSV từ sàn HOSE, kiểm tra tính toàn vẹn, chuẩn hóa cột,
    tách VNINDEX và phân chia tập TRAIN (2020-2021) & TEST (2022).
    """
    if not os.path.exists(file_path):
        return None, "File không tồn tại: " + str(file_path)

    df = pd.read_csv(file_path)
    required_cols = ["date", "ticker", "close", "adj_close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        return None, f"Dữ liệu thiếu các cột: {missing}"

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)

    benchmark_data = df.loc[df["ticker"] == "VNINDEX"].copy()
    stock_data = df.loc[df["ticker"] != "VNINDEX"].copy()

    train_data = stock_data.loc[(stock_data["date"] >= "2020-01-01") & (stock_data["date"] < "2022-01-01")].copy()
    test_data = stock_data.loc[(stock_data["date"] >= "2022-01-01") & (stock_data["date"] < "2023-01-01")].copy()

    return {
        "raw_df": df,
        "benchmark_data": benchmark_data,
        "stock_data": stock_data,
        "train_data": train_data,
        "test_data": test_data,
    }, None


@st.cache_data(show_spinner=False)
def run_stage1_pipeline(train_data):
    """
    Stage 1: Tính các chỉ báo kỹ thuật trên tập Train và lọc Eligible Universe.
    """
    px_all = train_data.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    val_all = (train_data.assign(v=train_data["close"] * train_data["volume"])
               .pivot(index="date", columns="ticker", values="v").sort_index())

    # Lọc cổ phiếu có dữ liệu >= 90% phiên
    enough = px_all.notna().sum() >= 0.9 * len(px_all)
    px_all = px_all.loc[:, enough].ffill()
    val_all = val_all[px_all.columns].fillna(0)
    ret_all = px_all.pct_change(fill_method=None)

    # Hàm RSI Wilder
    def rsi_wilder(p, n=14):
        d = p.diff()
        g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + g / (l + 1e-9))

    # Tính toán chỉ báo tại phiên cuối của Train
    sma50 = px_all.rolling(50).mean()
    sma200 = px_all.rolling(200).mean()

    ind_df = pd.DataFrame({
        "px_vs_sma50": px_all.iloc[-1] / sma50.iloc[-1] - 1,
        "sma50_vs_sma200": sma50.iloc[-1] / sma200.iloc[-1] - 1,
        "mom_6m": px_all.iloc[-21] / px_all.iloc[-126] - 1,
        "rsi14": rsi_wilder(px_all).iloc[-1],
        "vol_daily": ret_all.std(),
        "liq_value": val_all.tail(60).median(),
    }).dropna()

    # Lọc Top 30 thanh khoản -> Loại RSI > 75 hoặc px < SMA50
    pool = ind_df.nlargest(30, "liq_value")
    ok_mask = (pool["rsi14"] <= 75) & (pool["px_vs_sma50"] >= 0)
    eligible = pool[ok_mask].copy()
    removed = pool[~ok_mask].copy()

    return px_all, val_all, ret_all, ind_df, pool, eligible, removed


@st.cache_data(show_spinner=False)
def run_stage2_pipeline(eligible, ret_all, weights=SCORE_WEIGHTS, max_corr=DEFAULT_MAX_CORR):
    """
    Stage 2: Chấm điểm 4 nhân tố, xếp hạng và chọn TOP 5 qua Correlation Filter.
    """
    r = lambda s: s.rank(pct=True)
    sc = pd.DataFrame(index=eligible.index)
    sc["trend"] = (r(eligible["px_vs_sma50"]) + r(eligible["sma50_vs_sma200"])) / 2
    sc["momentum"] = (r(eligible["mom_6m"]) + r(-(eligible["rsi14"] - 60).abs())) / 2
    sc["risk"] = r(-eligible["vol_daily"])
    sc["liquidity"] = r(eligible["liq_value"])
    sc["score"] = sum(sc[k] * v for k, v in weights.items())
    ranked = sc.sort_values("score", ascending=False)

    # Correlation Filter
    corr_matrix = ret_all[ranked.index].corr()
    chosen = []
    for t in ranked.index:
        if all(corr_matrix.loc[t, c] <= max_corr for c in chosen):
            chosen.append(t)
        if len(chosen) == 5:
            break
    if len(chosen) < 5:
        chosen += [t for t in ranked.index if t not in chosen][: 5 - len(chosen)]

    return ranked, chosen, corr_matrix


@st.cache_data(show_spinner=False)
def run_sensitivity_and_stability(eligible, px_all, val_all, ret_all, top5):
    """
    Kiểm tra độ ổn định theo trọng số và theo từng quý trong Train.
    """
    scenarios = {
        "Cơ sở (30/30/20/20)": SCORE_WEIGHTS,
        "Đều (25/25/25/25)": {"trend": 0.25, "momentum": 0.25, "risk": 0.25, "liquidity": 0.25},
        "Thiên rủi ro thấp": {"trend": 0.20, "momentum": 0.20, "risk": 0.45, "liquidity": 0.15},
        "Thiên xu hướng": {"trend": 0.45, "momentum": 0.30, "risk": 0.10, "liquidity": 0.15},
    }

    sens_results = {}
    r = lambda s: s.rank(pct=True)
    for name, w in scenarios.items():
        sc = pd.DataFrame(index=eligible.index)
        sc["trend"] = (r(eligible["px_vs_sma50"]) + r(eligible["sma50_vs_sma200"])) / 2
        sc["momentum"] = (r(eligible["mom_6m"]) + r(-(eligible["rsi14"] - 60).abs())) / 2
        sc["risk"] = r(-eligible["vol_daily"])
        sc["liquidity"] = r(eligible["liq_value"])
        sc["score"] = sum(sc[k] * v for k, v in w.items())
        rk = sc.sort_values("score", ascending=False)

        corr = ret_all[rk.index].corr()
        ch = []
        for t in rk.index:
            if all(corr.loc[t, c] <= DEFAULT_MAX_CORR for c in ch):
                ch.append(t)
            if len(ch) == 5:
                break
        sens_results[name] = {
            "Top 5": ", ".join(ch),
            "Trùng với Cơ sở": f"{len(set(ch) & set(top5))}/5",
        }
    sens_df = pd.DataFrame(sens_results).T

    # Ổn định theo quý
    def rsi_wilder(p, n=14):
        d = p.diff()
        g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + g / (l + 1e-9))

    quarterly_picks = {}
    for cut in pd.date_range("2020-09-30", "2021-12-31", freq="QE"):
        cut_ts = px_all.index[px_all.index <= cut]
        if len(cut_ts) == 0:
            continue
        last_dt = cut_ts[-1]
        px_c, val_c = px_all.loc[:last_dt], val_all.loc[:last_dt]
        if len(px_c) < 200:
            continue
        sma50_c, sma200_c = px_c.rolling(50).mean(), px_c.rolling(200).mean()
        ind_c = pd.DataFrame({
            "px_vs_sma50": px_c.iloc[-1] / sma50_c.iloc[-1] - 1,
            "sma50_vs_sma200": sma50_c.iloc[-1] / sma200_c.iloc[-1] - 1,
            "mom_6m": px_c.iloc[-21] / px_c.iloc[-126] - 1,
            "rsi14": rsi_wilder(px_c).iloc[-1],
            "vol_daily": px_c.pct_change(fill_method=None).std(),
            "liq_value": val_c.tail(60).median(),
        }).dropna()

        pool_c = ind_c.nlargest(30, "liq_value")
        ok_c = (pool_c["rsi14"] <= 75) & (pool_c["px_vs_sma50"] >= 0)
        el_c = pool_c[ok_c]
        if len(el_c) < 5:
            continue

        sc_c = pd.DataFrame(index=el_c.index)
        sc_c["trend"] = (r(el_c["px_vs_sma50"]) + r(el_c["sma50_vs_sma200"])) / 2
        sc_c["momentum"] = (r(el_c["mom_6m"]) + r(-(el_c["rsi14"] - 60).abs())) / 2
        sc_c["risk"] = r(-el_c["vol_daily"])
        sc_c["liquidity"] = r(el_c["liq_value"])
        sc_c["score"] = sum(sc_c[k] * v for k, v in SCORE_WEIGHTS.items())
        rk_c = sc_c.sort_values("score", ascending=False)

        ret_c = px_c.pct_change(fill_method=None)
        corr_c = ret_c[rk_c.index].corr()
        ch_c = []
        for t in rk_c.index:
            if all(corr_c.loc[t, c] <= DEFAULT_MAX_CORR for c in ch_c):
                ch_c.append(t)
            if len(ch_c) == 5:
                break
        quarterly_picks[last_dt.strftime("%Y-%m-%d")] = ch_c

    q_df = pd.DataFrame(quarterly_picks).T
    if not q_df.empty:
        q_df.columns = [f"Top {i+1}" for i in range(q_df.shape[1])]

    return sens_df, q_df


@st.cache_data(show_spinner=False)
def run_stage3_and_4_backtest(
    train_data, test_data, benchmark_data, top5,
    initial_capital=DEFAULT_INITIAL_CAPITAL,
    cost=DEFAULT_COST,
    rf_annual=DEFAULT_RF_ANNUAL,
    sma_window=DEFAULT_SMA_WINDOW
):
    """
    Stage 3 & 4: Tính toán tỷ trọng tối ưu, sinh tín hiệu Market Timing và backtest đầy đủ
    trên cả TRAIN (In-Sample) và TEST (Out-of-Sample).
    """
    def get_px(df, tickers):
        px = df[df["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values="adj_close").sort_index().ffill()
        return px.loc[:, px.iloc[0].notna()][tickers]

    px_train = get_px(train_data, top5)
    px_test = get_px(test_data, top5)

    # 1. Ledoit-Wolf Covariance Shrinkage + Min Volatility (Chỉ dùng px_train)
    S_shrink = risk_models.CovarianceShrinkage(px_train).ledoit_wolf()
    ef = EfficientFrontier(None, S_shrink, weight_bounds=(0.0, 0.40))
    ef.min_volatility()
    cleaned_w = ef.clean_weights()
    w_shrink = np.array([cleaned_w[t] for t in top5])

    # 2. Equal Weight (20% mỗi mã)
    w_equal = np.full(len(top5), 1.0 / len(top5))

    # 3. Market Timing Signal từ VNINDEX SMA200 (Lagged 1 day to strictly prevent look-ahead bias)
    vnindex = benchmark_data.set_index("date")["close"].sort_index()
    vnindex_sma = vnindex.rolling(window=sma_window).mean()
    raw_signal = (vnindex > vnindex_sma).astype(int)
    market_signal = raw_signal.shift(1).fillna(0)

    # Hàm tính equity curves
    def calc_bh(px, weights):
        rel = px.div(px.iloc[0])
        return initial_capital * (1 - cost) * rel.dot(weights)

    def calc_timed(equity_base, sig):
        sig_aligned = sig.reindex(equity_base.index).fillna(0)
        ret_base = equity_base.pct_change(fill_method=None).fillna(0)
        timed_ret = ret_base * sig_aligned
        return initial_capital * (1 + timed_ret).cumprod(), sig_aligned

    # Tạo đường equity
    eq_bh_train = calc_bh(px_train, w_equal)
    eq_bh_test = calc_bh(px_test, w_equal)

    eq_sh_train = calc_bh(px_train, w_shrink)
    eq_sh_test = calc_bh(px_test, w_shrink)

    eq_mt_train, sig_train_mt = calc_timed(eq_bh_train, market_signal)
    eq_mt_test, sig_test_mt = calc_timed(eq_bh_test, market_signal)

    eq_comb_train, sig_train_comb = calc_timed(eq_sh_train, market_signal)
    eq_comb_test, sig_test_comb = calc_timed(eq_sh_test, market_signal)

    # Equity của VNINDEX Benchmark
    def calc_vnindex_eq(start_dt, end_dt):
        sub = vnindex.loc[start_dt:end_dt]
        return initial_capital * (sub / sub.iloc[0])

    eq_vni_train = calc_vnindex_eq(px_train.index[0], px_train.index[-1])
    eq_vni_test = calc_vnindex_eq(px_test.index[0], px_test.index[-1])

    # Hàm tính toán đầy đủ 8 chỉ số hiệu suất
    def compute_all_metrics(eq, sig=None):
        eq = eq.dropna()
        r = eq.pct_change(fill_method=None).dropna()
        n = len(r)
        
        # 1. Total Return & Annualized CAGR
        tot_ret = eq.iloc[-1] / eq.iloc[0] - 1
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (DEFAULT_TRADING_DAYS / n) - 1 if n > 0 else np.nan
        
        # 4. Volatility (Annualized)
        vol = r.std(ddof=1) * np.sqrt(DEFAULT_TRADING_DAYS) if n > 1 else np.nan
        
        # 5. Maximum Drawdown
        dd = eq / eq.cummax() - 1
        mdd = dd.min()
        
        # 6. Sharpe Ratio
        sharpe = (r.mean() * DEFAULT_TRADING_DAYS - rf_annual) / vol if vol > 0 else np.nan
        
        # 7. Sortino Ratio
        downside = r[r < 0]
        downside_std = np.sqrt((r.clip(upper=0)**2).mean()) * np.sqrt(DEFAULT_TRADING_DAYS)
        sortino = (r.mean() * DEFAULT_TRADING_DAYS - rf_annual) / downside_std if downside_std > 0 else np.nan
        
        # 8. Calmar Ratio
        calmar = cagr / abs(mdd) if (mdd < 0 and not np.isnan(mdd)) else np.nan
        
        # 3. Winning Rate (Tỷ lệ phiên sinh lời)
        active_r = r[r != 0]
        win_rate_active = (active_r > 0).sum() / len(active_r) if len(active_r) > 0 else 0.0
        win_rate_total = (r > 0).sum() / len(r) if len(r) > 0 else 0.0
        
        # Market Exposure
        exposure = sig.mean() if sig is not None else 1.0

        return {
            "Total Return": tot_ret,
            "CAGR": cagr,
            "Volatility": vol,
            "Max Drawdown": mdd,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "Win Rate (Active)": win_rate_active,
            "Win Rate (Total)": win_rate_total,
            "Market Exposure": exposure,
            "Ending Equity": eq.iloc[-1],
            "Drawdown Series": dd,
            "Equity Series": eq,
            "Daily Return": r
        }

    # Tổng hợp metrics cho cả 4 chiến lược
    strategies_data = {
        "Baseline (Equal Weight)": {
            "train": compute_all_metrics(eq_bh_train),
            "test": compute_all_metrics(eq_bh_test),
            "weights": w_equal,
            "color": "#64748B",
            "desc": "Top 5 chia đều 20% mỗi mã, Buy & Hold thụ động"
        },
        "Shrinkage Only": {
            "train": compute_all_metrics(eq_sh_train),
            "test": compute_all_metrics(eq_sh_test),
            "weights": w_shrink,
            "color": "#0284C7",
            "desc": "Ledoit-Wolf Min Volatility (Max 40%/mã), không market timing"
        },
        "Market Timing Only": {
            "train": compute_all_metrics(eq_mt_train, sig_train_mt),
            "test": compute_all_metrics(eq_mt_test, sig_test_mt),
            "weights": w_equal,
            "color": "#D97706",
            "desc": "Top 5 Equal Weight kết hợp tín hiệu VNINDEX SMA200 (Lagged 1D)"
        },
        "Combined Strategy (Shrinkage + MT)": {
            "train": compute_all_metrics(eq_comb_train, sig_train_comb),
            "test": compute_all_metrics(eq_comb_test, sig_test_comb),
            "weights": w_shrink,
            "color": "#059669",
            "desc": "Ledoit-Wolf Min-Vol khi Uptrend, chuyển 100% Cash khi VNINDEX <= SMA200"
        }
    }

    vni_metrics = {
        "train": compute_all_metrics(eq_vni_train),
        "test": compute_all_metrics(eq_vni_test),
    }

    return (
        px_train, px_test, vnindex, vnindex_sma, market_signal,
        w_shrink, cleaned_w, strategies_data, vni_metrics
    )


# ==============================================================================
# LOAD PIPELINE DATA
# ==============================================================================
data_bundle, err = load_and_preprocess_data("HOSE_2020_2023_in.csv")
if err:
    st.error(f"Lỗi tải dữ liệu: {err}")
    st.stop()

train_data = data_bundle["train_data"]
test_data = data_bundle["test_data"]
benchmark_data = data_bundle["benchmark_data"]

# Run Stages
px_all, val_all, ret_all, ind_all, pool, eligible, removed = run_stage1_pipeline(train_data)
ranked, top5, corr_matrix = run_stage2_pipeline(eligible, ret_all)
sens_df, q_df = run_sensitivity_and_stability(eligible, px_all, val_all, ret_all, top5)

(
    px_train, px_test, vnindex, vnindex_sma, market_signal,
    w_shrink, cleaned_w, strategies_data, vni_metrics
) = run_stage3_and_4_backtest(train_data, test_data, benchmark_data, top5)


# ==============================================================================
# SIDEBAR NAVIGATION & PARAMETERS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 14px 0;">
        <span style="font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 1px; text-transform: uppercase;">Quantitative Investing</span>
        <h2 style="margin: 3px 0 2px 0; font-size: 19px; font-weight: 700; color: #0F172A;">PORTFOLIO DASHBOARD</h2>
        <p style="font-size: 12px; color: #64748B; margin: 0;">Đại học Mở TP.HCM | MFB025A</p>
    </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "ĐIỀU HƯỚNG QUY TRÌNH (WORKFLOW)",
        [
            "01. Data & Universe",
            "02. Stock Selection",
            "03. Portfolio Strategy",
            "04. Backtest & Decision",
            "05. Tổng Quan & Khuyến Nghị",
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ THÔNG SỐ ĐẦU TƯ")
    st.markdown(f"""
    - **Vốn ban đầu:** `{DEFAULT_INITIAL_CAPITAL:,.0f} VND`
    - **Lãi suất phi rủi ro ($R_f$):** `{DEFAULT_RF_ANNUAL:.1%}/năm`
    - **Phí giao dịch:** `{DEFAULT_COST:.2%}`
    - **Định thời điểm:** `VNINDEX SMA200 (T-1)`
    - **Ngưỡng tương quan:** `r <= {DEFAULT_MAX_CORR}`
    - **Giới hạn tỷ trọng:** `Tối đa 40%/cổ phiếu`
    """)

    st.markdown("---")
    st.markdown("### 📊 THÔNG TIN DỮ LIỆU")
    st.markdown(f"""
    - **Tổng số mã ban đầu:** `{px_all.shape[1]} mã sàn HOSE`
    - **Tập Train (In-Sample):** `2020-01-02` ➔ `2021-12-31` (`{len(px_train)}` phiên)
    - **Tập Test (Out-of-Sample):** `2022-01-04` ➔ `2022-12-30` (`{len(px_test)}` phiên)
    """)


# ==============================================================================
# VIEW 1: STAGE 1 — DATA & STOCK UNIVERSE
# ==============================================================================
if nav_choice == "01. Data & Universe":
    st.markdown("""
    <div class="top-banner">
        <h1>STAGE 1 — DATA & STOCK UNIVERSE</h1>
        <p>Quy trình thu thập, làm sạch dữ liệu, phân tách Train/Test nghiêm ngặt và phễu sàng lọc 2 tầng.<br>
        Mục tiêu: Từ 99 cổ phiếu HOSE ➔ Sàng lọc Top 30 thanh khoản ➔ Chọn lọc Top 20 Eligible Universe đủ điều kiện.</p>
    </div>
    """, unsafe_allow_html=True)

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Vũ Trụ Ban Đầu (HOSE)", f"{px_all.shape[1]} cổ phiếu", help="Cổ phiếu có dữ liệu giao dịch >= 90% số phiên trên sàn HOSE")
    with m2:
        st.metric("Giai đoạn Train (In-Sample)", "2020 - 2021 (502 phiên)", help="Dữ liệu dùng để tính chỉ báo, chấm điểm và tối ưu hóa danh mục")
    with m3:
        st.metric("Tầng 1: Lọc Thanh Khoản", "Top 30 Cổ Phiếu", help="Top 30 mã có giá trị giao dịch trung vị 60 phiên cao nhất")
    with m4:
        st.metric("Tầng 2: Eligible Universe", f"{len(eligible)} Cổ Phiếu Đủ Chuẩn", help="Top 20 mã thỏa mãn RSI <= 75 và Giá >= SMA50 (Loại 10 mã)")

    st.markdown("---")
    st.markdown("### 1. Logic Chi Tiết: Phễu Sàng Lọc 2 Tầng (Funnel Screening Logic)")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom: 8px;">
                <span class="funnel-step funnel-step-blue">TẦNG 1: TỪ 99 MÃ ➔ TOP 30 MÃ THANH KHOẢN</span>
            </div>
            <div class="funnel-title">💧 Bộ Lọc Thanh Khoản (Liquidity Screening)</div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **1. Cơ sở định lượng & quy mô quỹ:**  
            Trong quản trị danh mục định lượng của các định chế tài chính, thanh khoản là điều kiện tiên quyết (*Gating Factor*) nhằm:
            - Đảm bảo khả năng giải ngân và thoái vốn nhanh chóng mà không gây trượt giá lớn (*Slippage*) hay tác động giá bất lợi (*Market Impact Cost*).
            - Loại trừ triệt để các cổ phiếu vốn hóa siêu nhỏ (*Penny*), cổ phiếu "bo cung", thao túng giá thiếu thanh khoản thực.

            **2. Chỉ báo sử dụng:** `60D Median Trading Value` — Giá trị giao dịch trung vị trong 60 phiên gần nhất tính đến cuối tập Train (31/12/2021):
            """)
            st.latex(r"\text{Liq Value} = \text{Median}_{t \in [T-59, T]}\left(\text{Close}_t \times \text{Volume}_t\right)")
            st.markdown("""
            **3. Tại sao chọn Trung vị (Median) thay vì Trung bình (Mean)?**  
            Trung vị triệt tiêu hoàn toàn độ nhiễu của các phiên "quay tay thanh khoản" hoặc phiên đột biến khối lượng ngắn hạn, phản ánh đúng dung lượng khớp lệnh tự nhiên hàng ngày.

            ➔ **Kết quả:** Chọn ra đúng **Top 30 cổ phiếu thanh khoản lớn nhất** toàn sàn HOSE.
            """)

    with f_col2:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom: 8px;">
                <span class="funnel-step funnel-step-amber">TẦNG 2: TỪ TOP 30 MÃ ➔ TOP 20 MÃ ELIGIBLE</span>
            </div>
            <div class="funnel-title">🛡️ Bộ Lọc Sức Khỏe Kỹ Thuật & Xu Hướng (Technical Filter)</div>
            """, unsafe_allow_html=True)

            st.markdown("""
            Từ 30 mã thanh khoản tốt nhất, áp dụng 2 tiêu chí loại trừ nhằm tránh mua đỉnh ngắn hạn và tránh cổ phiếu mất xu hướng tăng:

            **Tiêu chí 1: $RSI_{14} \le 75$ (Loại bỏ Quá Mua cực đoan):**  
            Wilder's RSI 14 phiên đo lường độ căng của động lượng giá. Nếu $RSI > 75$, cổ phiếu đang ở vùng quá nóng (*Extreme Overbought*), đối mặt rủi ro điều chỉnh phân kỳ âm cực lớn.

            **Tiêu chí 2: Giá đóng cửa $\ge SMA_{50}$ (Bảo toàn Xu hướng Tăng):**
            """)
            st.latex(r"\text{px\_vs\_sma50} = \frac{P_t}{SMA_{50}} - 1 \ge 0")
            st.markdown("""
            Cổ phiếu bắt buộc phải giao dịch trên đường trung bình động 50 ngày ($P_t \ge SMA_{50}$). Nếu giá nằm dưới $SMA_{50}$, xung lực tăng đã bị phá vỡ, cổ phiếu bước vào pha phân phối/downtrend ngắn-trung hạn.

            ➔ **Kết quả:** Loại chính xác **10 cổ phiếu** (gồm các bluechips bị gãy trend), giữ lại **Top 20 cổ phiếu Đủ Điều Kiện (Eligible Universe)**.
            """)

    st.markdown("---")
    st.markdown("### 2. Chi Tiết Các Chỉ Số Phân Tích Kỹ Thuật Dùng Cho Bộ Lọc & Chấm Điểm")
    st.markdown("""
    Toàn bộ hệ thống định lượng được vận hành dựa trên **6 chỉ báo kỹ thuật cốt lõi**, được tính toán thuần túy trên tập In-Sample (Train 2020–2021) nhằm đảm bảo nguyên tắc không rò rỉ dữ liệu (*No Look-ahead Bias*):
    """)

    c_ind1, c_ind2, c_ind3 = st.columns(3)
    with c_ind1:
        with st.container(border=True):
            st.markdown("##### 💧 1. 60D Median Trading Value")
            st.latex(r"\text{Liq Value} = \text{Median}_{60}(\text{Close} \times \text{Vol})")
            st.markdown("""
            - **Ý nghĩa:** Đo lường quy mô thanh khoản thực, triệt tiêu giao dịch đột biến bất thường.
            - **Ứng dụng:** Bộ lọc Gating Tầng 1 (lấy Top 30) & chiếm 20% điểm Composite ở Stage 2.
            """)
        with st.container(border=True):
            st.markdown("##### 📈 2. Giá vs. SMA50 (Short/Mid Trend)")
            st.latex(r"\text{px\_vs\_sma50} = \frac{P_t}{SMA_{50}(P_t)} - 1")
            st.markdown("""
            - **Ý nghĩa:** Xác định vị thế giá so với xu hướng bình quân 50 ngày gần nhất.
            - **Ứng dụng:** Điều kiện Tầng 2 ($\ge 0$) & chiếm 50% điểm thành phần Trend ở Stage 2.
            """)

    with c_ind2:
        with st.container(border=True):
            st.markdown("##### ⚡ 3. Wilder's RSI 14 (Động Lượng)")
            st.latex(r"RSI_{14} = 100 - \frac{100}{1 + \frac{\text{EMA}_{14}(\text{Gain})}{\text{EMA}_{14}(\text{Loss})}}")
            st.markdown("""
            - **Ý nghĩa:** Đo lường vận tốc và mức độ biến thiên giá theo công thức chuẩn J. Welles Wilder.
            - **Ứng dụng:** Điều kiện Tầng 2 ($\le 75$) & Factor Momentum (ưu tiên xung lực quanh mốc 60).
            """)
        with st.container(border=True):
            st.markdown("##### 🌟 4. SMA50 vs. SMA200 (Long-term Trend)")
            st.latex(r"\text{sma50\_vs\_sma200} = \frac{SMA_{50}(P_t)}{SMA_{200}(P_t)} - 1")
            st.markdown("""
            - **Ý nghĩa:** Cấu trúc Golden Cross kinh điển, xác nhận xu hướng tăng dài hạn bền vững.
            - **Ứng dụng:** Chiếm 50% điểm thành phần Trend trong mô hình Multi-Factor ở Stage 2.
            """)

    with c_ind3:
        with st.container(border=True):
            st.markdown("##### 🚀 5. Momentum 6 Tháng (Lag 1 Tháng)")
            st.latex(r"\text{mom\_6m} = \frac{P_{T-21}}{P_{T-126}} - 1")
            st.markdown("""
            - **Ý nghĩa:** Tỷ suất sinh lợi từ tháng $T-6$ đến $T-1$, loại trừ tháng $T$ gần nhất.
            - **Ứng dụng:** Triệt tiêu hiện tượng đảo chiều ngắn hạn (*Short-term reversal*); Factor Momentum.
            """)
        with st.container(border=True):
            st.markdown("##### 🛡️ 6. Độ Biến Động Ngày (Volatility)")
            st.latex(r"\sigma_{\text{daily}} = \sqrt{\frac{1}{N-1}\sum_{t=1}^N (R_t - \bar{R})^2}")
            st.markdown("""
            - **Ý nghĩa:** Độ lệch chuẩn mẫu của lợi suất ngày, phản ánh rủi ro dao động giá.
            - **Ứng dụng:** Factor Risk ở Stage 2 (ưu tiên cổ phiếu có độ biến động thấp hơn).
            """)

    st.markdown("---")
    st.markdown("### 3. Danh Sách Chi Tiết 10 Cổ Phiếu Bị Loại Khỏi Top 30")
    st.markdown("""
    Bảng dưới đây minh chứng tính khách quan của thuật toán: Dù đều là những doanh nghiệp hàng đầu thị trường với thanh khoản hàng trăm tỷ đồng mỗi phiên, **cả 10 mã đều bị loại thẳng tay vì vi phạm điều kiện $P < SMA_{50}$** (Giá gãy xuống dưới đường trung bình 50 ngày tại phiên cuối Train):
    """)

    # Process removed table for display
    disp_rem = removed.copy().reset_index()
    disp_rem["px_vs_sma50_pct"] = disp_rem["px_vs_sma50"].map("{:.1%}".format)
    disp_rem["rsi_val"] = disp_rem["rsi14"].map("{:.1f}".format)
    disp_rem["liq_bil"] = (disp_rem["liq_value"] / 1e9).map("{:,.1f} Tỷ VNĐ".format)
    disp_rem["violation"] = np.where(
        disp_rem["px_vs_sma50"] < 0,
        "Giá < SMA50 (Gãy xu hướng ngắn-trung hạn)",
        "RSI > 75 (Quá mua cực đoan)"
    )

    st.dataframe(
        disp_rem[["ticker", "liq_bil", "px_vs_sma50_pct", "rsi_val", "violation"]],
        column_config={
            "ticker": st.column_config.TextColumn("Mã CP", help="Mã chứng khoán sàn HOSE", width="small"),
            "liq_bil": st.column_config.TextColumn("GTGD Trung vị 60D", help="Giá trị giao dịch trung vị 60 phiên", width="medium"),
            "px_vs_sma50_pct": st.column_config.TextColumn("Giá so với SMA50", help="Khoảng cách phần trăm so với SMA50 (Yêu cầu >= 0%)", width="medium"),
            "rsi_val": st.column_config.TextColumn("RSI (14)", help="Chỉ số sức mạnh tương đối Wilder", width="small"),
            "violation": st.column_config.TextColumn("Lý Do Loại Bỏ Cụ Thể", help="Lý do không đạt điều kiện đưa vào rổ Eligible", width="large"),
        },
        use_container_width=True
    )

    st.markdown("---")
    st.markdown(f"### 4. Danh Sách 20 Cổ Phiếu Đủ Điều Kiện (Eligible Universe: {len(eligible)} Mã)")
    st.markdown("Đây là 20 mã vượt qua 2 tầng sàng lọc khắt khe, kết hợp hài hòa giữa **Thanh khoản vượt trội** và **Động lượng xu hướng lành mạnh**:")

    disp_el = eligible.sort_values("liq_value", ascending=False).copy().reset_index()
    disp_el["liq_bil"] = (disp_el["liq_value"] / 1e9).map("{:,.1f} Tỷ VNĐ".format)
    disp_el["px_sma50_fmt"] = disp_el["px_vs_sma50"].map("{:.1%}".format)
    disp_el["sma50_sma200_fmt"] = disp_el["sma50_vs_sma200"].map("{:.1%}".format)
    disp_el["mom_6m_fmt"] = disp_el["mom_6m"].map("{:.1%}".format)
    disp_el["vol_fmt"] = disp_el["vol_daily"].map("{:.2%}".format)
    disp_el["rsi_fmt"] = disp_el["rsi14"].map("{:.1f}".format)

    st.dataframe(
        disp_el[["ticker", "liq_bil", "px_sma50_fmt", "sma50_sma200_fmt", "mom_6m_fmt", "rsi_fmt", "vol_fmt"]],
        column_config={
            "ticker": st.column_config.TextColumn("Mã CP", width="small"),
            "liq_bil": st.column_config.TextColumn("GTGD Trung Vị 60D", width="medium"),
            "px_sma50_fmt": st.column_config.TextColumn("Giá vs SMA50 (Trend 1)", width="medium"),
            "sma50_sma200_fmt": st.column_config.TextColumn("SMA50 vs SMA200 (Trend 2)", width="medium"),
            "mom_6m_fmt": st.column_config.TextColumn("Momentum 6M", width="medium"),
            "rsi_fmt": st.column_config.TextColumn("RSI14", width="small"),
            "vol_fmt": st.column_config.TextColumn("Biến Động Ngày (Vol)", width="medium"),
        },
        use_container_width=True
    )

    # Interactive Bubble / Scatter chart of Universe
    st.markdown("### 5. Bản Đồ Trực Quan: Thanh Khoản vs. Biến Động Top 30 Cổ Phiếu")
    scatter_df = pool.copy().reset_index()
    scatter_df["Status"] = np.where(scatter_df["ticker"].isin(eligible.index), "Đủ điều kiện (Eligible Universe: 20 mã)", "Bị loại (Filtered Out: 10 mã)")
    scatter_df["liq_bil"] = scatter_df["liq_value"] / 1e9
    
    fig_scatter = px.scatter(
        scatter_df,
        x="vol_daily",
        y="liq_bil",
        size="rsi14",
        color="Status",
        text="ticker",
        color_discrete_map={"Đủ điều kiện (Eligible Universe: 20 mã)": "#059669", "Bị loại (Filtered Out: 10 mã)": "#DC2626"},
        labels={"vol_daily": "Độ biến động ngày (Daily Volatility)", "liq_bil": "GTGD Trung vị 60 phiên (Tỷ VNĐ)", "rsi14": "RSI 14"},
        title="Phân Bố Thanh Khoản và Biến Động Top 30 Mã Giao Dịch Lớn Nhất HOSE"
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(yaxis_type="log", height=450, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# VIEW 2: STAGE 2 — STOCK SELECTION (GIỮ NGUYÊN)
# ==============================================================================
elif nav_choice == "02. Stock Selection":
    st.markdown("""
    <div class="top-banner">
        <h1>STAGE 2 — STOCK SELECTION</h1>
        <p>Mô hình chấm điểm đa nhân tố (Multi-Factor Scoring) và Bộ lọc tương quan (Correlation Filter).<br>
        Mục tiêu: Chọn lọc TOP 5 cổ phiếu có xung lực tăng trưởng mạnh mẽ nhất nhưng vẫn đảm bảo tính đa dạng hóa danh mục.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1. Công Thức Chấm Điểm Đa Nhân Tố (Composite Scoring)")
    st.latex(r"""
    \text{Score} = 30\% \times \text{Trend} + 30\% \times \text{Momentum} + 20\% \times \text{Risk} + 20\% \times \text{Liquidity}
    """)

    st.markdown("""
    - **Trend (30%):** $\\frac{\\text{Percentile}(px\\_vs\\_sma50) + \\text{Percentile}(sma50\\_vs\\_sma200)}{2}$
    - **Momentum (30%):** $\\frac{\\text{Percentile}(mom\\_6m) + \\text{Percentile}(-|RSI_{14} - 60|)}{2}$ *(Ưu tiên RSI quanh vùng tối ưu 60)*
    - **Risk (20%):** $\\text{Percentile}(-\\text{vol}\\_daily)$ *(Ưu tiên biến động thấp hơn)*
    - **Liquidity (20%):** $\\text{Percentile}(\\text{liq}\\_value)$ *(Ưu tiên thanh khoản dồi dào)*
    """)

    # Top Ranked Table
    st.markdown("### 2. Bảng Xếp Hạng Điểm Tổng Hợp & Lựa Chọn TOP 5")
    c_rk1, c_rk2 = st.columns([3, 2])

    with c_rk1:
        st.markdown("#### Bảng Điểm Đầy Đủ 20 Cổ Phiếu Đủ Điều Kiện")
        disp_rk = ranked.copy()
        disp_rk["Selected"] = disp_rk.index.isin(top5)
        st.dataframe(
            disp_rk,
            column_config={
                "trend": st.column_config.NumberColumn("Trend (30%)", format="%.3f"),
                "momentum": st.column_config.NumberColumn("Momentum (30%)", format="%.3f"),
                "risk": st.column_config.NumberColumn("Risk (20%)", format="%.3f"),
                "liquidity": st.column_config.NumberColumn("Liquidity (20%)", format="%.3f"),
                "score": st.column_config.ProgressColumn(
                    "Composite Score",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
                "Selected": st.column_config.CheckboxColumn("Top 5"),
            },
            use_container_width=True
        )

    with c_rk2:
        st.markdown(f"#### 🏆 TOP 5 CỔ PHIẾU ĐƯỢC CHỌN (MAX CORR $\le$ 0.70)")
        for rank_idx, tk in enumerate(top5):
            sc_val = ranked.loc[tk, "score"]
            t_val = ranked.loc[tk, "trend"]
            m_val = ranked.loc[tk, "momentum"]
            st.markdown(f"""
            <div style="background: #F1F5F9; border-left: 4px solid #0284C7; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                <div style="font-weight: 700; font-size: 15px; color: #0F172A;">#{rank_idx+1}. {tk} — Điểm: {sc_val:.3f}</div>
                <div style="font-size: 12px; color: #64748B;">Xu hướng: {t_val:.2f} | Động lượng: {m_val:.2f} | Rủi ro: {ranked.loc[tk, 'risk']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.info("💡 **Quy tắc Correlation Filter:** Thuật toán duyệt từ mã có điểm cao nhất xuống. Các mã có tương quan lợi suất với bất kỳ mã nào đã chọn $> 0.70$ sẽ bị bỏ qua nhằm đảm bảo đa dạng hóa thực chất.")

    # Correlation Heatmap & Factor contribution
    st.markdown("---")
    st.markdown("### 3. Ma Trận Tương Quan Lợi Suất & Đóng Góp Nhân Tố Của Top 5")
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        # Heatmap
        corr_top5 = ret_all[top5].corr()
        fig_heat = px.imshow(
            corr_top5,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=0, zmax=1,
            title="Ma Trận Tương Quan Lợi Suất Ngày (Train Period)",
        )
        fig_heat.update_layout(height=380, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    with ch_col2:
        # Factor contribution bar chart
        contrib_df = pd.DataFrame({
            k: ranked.loc[top5, k] * w for k, w in SCORE_WEIGHTS.items()
        }).loc[top5]
        contrib_df.columns = ["Xu hướng (30%)", "Động lượng (30%)", "Rủi ro thấp (20%)", "Thanh khoản (20%)"]
        
        fig_contrib = px.bar(
            contrib_df,
            barmode="stack",
            title="Điểm Đóng Góp Của Từng Nhóm Nhân Tố Vào Điểm Tổng Hợp",
            color_discrete_sequence=["#2563EB", "#0284C7", "#059669", "#D97706"]
        )
        fig_contrib.update_layout(height=380, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_contrib, use_container_width=True)

    # Sensitivity & Robustness
    st.markdown("---")
    st.markdown("### 4. Kiểm Tra Độ Ổn Định & Nhạy Cảm (Robustness & Stability)")
    tab_sens, tab_stab = st.tabs(["Nhạy cảm với Trọng số Chấm điểm", "Ổn định theo từng Quý trong Train"])
    
    with tab_sens:
        st.write("Đánh giá xem Top 5 có bị đảo lộn nếu thay đổi kịch bản trọng số chấm điểm:")
        st.dataframe(sens_df, use_container_width=True)
        st.caption("Khảo sát cho thấy nhóm cổ phiếu nòng cốt như DIG, GEX, NLG luôn duy trì xuất hiện ở hầu hết các kịch bản chấm điểm.")

    with tab_stab:
        st.write("Đánh giá Top 5 được chọn tại các điểm mốc cuối mỗi quý trong giai đoạn Train:")
        st.dataframe(q_df, use_container_width=True)


# ==============================================================================
# VIEW 3: STAGE 3 — PORTFOLIO STRATEGY (BỔ SUNG 3 PHẦN TIẾP CẬN CHUYÊN SÂU)
# ==============================================================================
elif nav_choice == "03. Portfolio Strategy":
    st.markdown("""
    <div class="top-banner">
        <h1>STAGE 3 — PORTFOLIO STRATEGY</h1>
        <p>Xây dựng phương án phân bổ vốn (Capital Allocation) và xác định thời điểm tham gia thị trường (Market Timing).<br>
        Giải quyết toàn diện: <b>Phần 1: Shrinkage</b> | <b>Phần 2: Market Timing</b> | <b>Phần 3: Mô hình Kết Hợp</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    # 4 Dedicated Tabs for Stage 3
    strat_tab1, strat_tab2, strat_tab3, strat_tab4 = st.tabs([
        "📌 Tổng Quan Phân Bổ & Thời Điểm",
        "🔹 Phần 1: Tiếp Cận Bằng Shrinkage",
        "🔹 Phần 2: Tiếp Cận Bằng Market Timing",
        "🔹 Phần 3: Mô Hình Kết Hợp (Shrinkage + MT)"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: TỔNG QUAN
    # --------------------------------------------------------------------------
    with strat_tab1:
        q1_col, q2_col = st.columns(2)

        with q1_col:
            st.markdown("""
            <div class="exec-box">
                <div class="exec-box-title">CÂU HỎI 1: HOW SHOULD CAPITAL BE ALLOCATED?</div>
                <p style="margin: 0; color: #334155;">
                <b>Phương án 1 (Equal Weight 1/N):</b> Chia đều 20% cho mỗi cổ phiếu trong Top 5.<br>
                <b>Phương án 2 (Ledoit-Wolf Shrinkage Min-Vol):</b> Tối ưu hóa ma trận hiệp phương sai nhằm cực tiểu hóa độ biến động danh mục, đặt giới hạn trần 40%/cổ phiếu để đảm bảo đa dạng hóa.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with q2_col:
            st.markdown("""
            <div class="exec-box" style="border-left-color: #059669;">
                <div class="exec-box-title">CÂU HỎI 2: WHEN SHOULD WE INVEST?</div>
                <p style="margin: 0; color: #334155;">
                <b>Tín hiệu định thời điểm (VNINDEX SMA200):</b><br>
                - Khi $VNINDEX > SMA200$ ➔ <b>INVEST (Tham gia thị trường)</b>.<br>
                - Khi $VNINDEX <= SMA200$ ➔ <b>CASH (Đứng ngoài giữ 100% tiền mặt)</b>.<br>
                - Tín hiệu trễ 1 ngày (t-1) triệt tiêu hoàn toàn Look-ahead bias.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Bảng So Sánh 4 Chiến Lược Được Xây Dựng")
        method_table = pd.DataFrame({
            "Chiến Lược": [
                "1. Buy & Hold (Benchmark)",
                "2. Shrinkage Only",
                "3. Market Timing Only",
                "4. Combined Strategy"
            ],
            "Phân Bổ Vốn (How to allocate)": [
                "Top 5 Equal Weight (20% mỗi mã)",
                "Ledoit-Wolf Min Volatility (Max 40%/mã)",
                "Top 5 Equal Weight (20% mỗi mã)",
                "Ledoit-Wolf Min Volatility (Max 40%/mã)"
            ],
            "Định Thời Điểm (When to invest)": [
                "Thụ động 100% thời gian (Full exposure)",
                "Thụ động 100% thời gian (Full exposure)",
                "VNINDEX > SMA200 (Lagged 1D)",
                "VNINDEX > SMA200 (Lagged 1D)"
            ],
            "Hành Động Khi Thị Trường Xấu (Bear)": [
                "Tiếp tục ôm cổ phiếu chịu lỗ",
                "Tiếp tục ôm cổ phiếu chịu lỗ",
                "Chuyển 100% về tiền mặt bảo toàn vốn",
                "Chuyển 100% về tiền mặt bảo toàn vốn"
            ]
        }).set_index("Chiến Lược")
        st.table(method_table)

    # --------------------------------------------------------------------------
    # TAB 2: PHẦN 1 — SHRINKAGE APPROACH
    # --------------------------------------------------------------------------
    with strat_tab2:
        st.markdown("### 🔹 PHẦN 1: CÁCH TIẾP CẬN BẰNG SHRINKAGE (LEDOIT-WOLF MIN-VOL)")
        
        st.markdown("""
        **1. Khắc phục khiếm khuyết của Lý thuyết Danh mục Hiện đại (Markowitz MPT):**
        - MPT truyền thống dựa vào việc ước lượng cả lợi nhuận kỳ vọng ($\mu$) và ma trận hiệp phương sai mẫu ($S$).
        - Sai số ước lượng lợi nhuận kỳ vọng $\mu$ lớn gấp 10 lần sai số ma trận phương sai. Khi đưa $\mu$ vào bộ tối ưu, thuật toán sẽ khuếch đại sai số (*error maximization*), dồn vốn vào các cổ phiếu "ăn may" trong quá khứ, dẫn tới sụp đổ ngoài mẫu (Overfitting).
        - **Giải pháp:** Bỏ qua hoàn toàn vector $\mu$, chuyển bài toán sang tìm danh mục có **Độ Biến Động Nhỏ Nhất (Minimum Volatility Portfolio)**.
        """)

        st.markdown("""
        **2. Thuật toán khử nhiễu Ledoit-Wolf Covariance Shrinkage:**
        - Ma trận hiệp phương sai mẫu $S$ thường bị nhiễu do số ngày quan sát ngắn so với số lượng tài sản.
        - Phương pháp Ledoit-Wolf thực hiện "co" (*shrink*) ma trận mẫu $S$ về ma trận mục tiêu có cấu trúc chặt chẽ $F$ (Constant Correlation Target):
        $$\hat{\Sigma}_{LW} = (1 - \delta) S + \delta F$$
        trong đó $\delta \in [0, 1]$ là cường độ co tối ưu (*optimal shrinkage intensity*), giúp ma trận hiệp phương sai ổn định và có tính dự báo tốt hơn hẳn ngoài mẫu.
        """)

        st.latex(r"""
        \min_{w} w^T \hat{\Sigma}_{LW} w \quad \text{thỏa mãn} \quad \sum_{i=1}^5 w_i = 1, \quad 0 \le w_i \le 0.40
        """)

        sh_col1, sh_col2 = st.columns([3, 2])
        with sh_col1:
            st.markdown("#### Bộ Tỷ Trọng Tối Ưu Shrinkage Min-Vol (Ràng buộc trần 40%)")
            w_comp = pd.DataFrame({
                "Cổ phiếu": top5,
                "Equal Weight (1/N)": ["20.00%"] * 5,
                "Shrinkage Min-Vol": [f"{cleaned_w[t]:.2%}" for t in top5]
            }).set_index("Cổ phiếu")
            st.dataframe(w_comp, use_container_width=True)

        with sh_col2:
            fig_sh_w = px.bar(
                pd.DataFrame({"Ticker": top5, "Tỷ trọng": [cleaned_w[t] for t in top5]}),
                x="Ticker", y="Tỷ trọng",
                color="Tỷ trọng", color_continuous_scale="Blues",
                title="Tỷ Trọng Tối Ưu (NLG kịch trần 40% phòng thủ)"
            )
            fig_sh_w.update_layout(height=260, showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_sh_w, use_container_width=True)

        st.markdown("#### So Sánh Hiệu Quả: Chia Đều (1/N) vs. Chỉ Dùng Shrinkage")
        bh_m_tr, bh_m_te = strategies_data["Baseline (Equal Weight)"]["train"], strategies_data["Baseline (Equal Weight)"]["test"]
        sh_m_tr, sh_m_te = strategies_data["Shrinkage Only"]["train"], strategies_data["Shrinkage Only"]["test"]

        df_sh_compare = pd.DataFrame({
            "Chỉ số": ["Lợi suất (Total Return)", "Biến động năm (Volatility)", "Chỉ số Sharpe (Rf=3%)", "Sụt giảm tối đa (Max Drawdown)"],
            "Equal Weight (Train)": [f"{bh_m_tr['Total Return']:.2%}", f"{bh_m_tr['Volatility']:.2%}", f"{bh_m_tr['Sharpe']:.2f}", f"{bh_m_tr['Max Drawdown']:.2%}"],
            "Shrinkage Only (Train)": [f"{sh_m_tr['Total Return']:.2%}", f"{sh_m_tr['Volatility']:.2%}", f"{sh_m_tr['Sharpe']:.2f}", f"{sh_m_tr['Max Drawdown']:.2%}"],
            "Equal Weight (Test)": [f"{bh_m_te['Total Return']:.2%}", f"{bh_m_te['Volatility']:.2%}", f"{bh_m_te['Sharpe']:.2f}", f"{bh_m_te['Max Drawdown']:.2%}"],
            "Shrinkage Only (Test)": [f"{sh_m_te['Total Return']:.2%}", f"{sh_m_te['Volatility']:.2%}", f"{sh_m_te['Sharpe']:.2f}", f"{sh_m_te['Max Drawdown']:.2%}"],
        }).set_index("Chỉ số")
        st.dataframe(df_sh_compare, use_container_width=True)

        # Plot Test curve comparison
        fig_sh_test = go.Figure()
        fig_sh_test.add_trace(go.Scatter(x=bh_m_te["Equity Series"].index, y=bh_m_te["Equity Series"]/1e6, name="Chia Đều (Equal Weight)", line=dict(color="#DC2626", dash="dash", width=1.5)))
        fig_sh_test.add_trace(go.Scatter(x=sh_m_te["Equity Series"].index, y=sh_m_te["Equity Series"]/1e6, name="Chỉ Dùng Shrinkage (Phòng Thủ)", line=dict(color="#0284C7", width=2.5)))
        fig_sh_test.add_hline(y=100, line_dash="dot", line_color="gray")
        fig_sh_test.update_layout(title="Sức Chống Chịu Của Kỹ Thuật Shrinkage Trên Tập TEST (Downtrend 2022)", yaxis_title="Triệu VND", height=380, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_sh_test, use_container_width=True)

        st.warning("""
        **⚠️ Kết Luận Định Lượng Về Shrinkage:** Tối ưu hóa ma trận hiệp phương sai Min-Vol giúp danh mục hạ độ biến động (từ 52.53% xuống 50.84%) và giảm lỗ nhẹ (-62.25% so với -68.04%). Tuy nhiên, **Shrinkage một mình hoàn toàn bất lực trong việc bảo vệ vốn** khi thị trường bước vào Downtrend lớn (Max Drawdown vẫn lên tới -75.70%). Cần phải có bộ lọc định thời điểm để chuyển về tiền mặt!
        """)

    # --------------------------------------------------------------------------
    # TAB 3: PHẦN 2 — MARKET TIMING APPROACH
    # --------------------------------------------------------------------------
    with strat_tab3:
        st.markdown("### 🔹 PHẦN 2: CÁCH TIẾP CẬN BẰNG MARKET TIMING (VNINDEX SMA200)")

        st.markdown("""
        **1. Cơ sở lý thuyết về rủi ro hệ thống (Systemic Risk):**
        - Trong các cuộc khủng hoảng hoặc thị trường gấu, mối tương quan giữa tất cả các cổ phiếu đều tiến gần về +1.0 (*correlation breakdown*). Mọi nỗ lực phân bổ tỷ trọng cổ phiếu đều không thể ngăn chặn đà sụt giảm.
        - Phương pháp tự vệ duy nhất là **Đứng ngoài thị trường và nắm giữ 100% Tiền Mặt (Cash)**.
        - **Chỉ báo VNINDEX SMA200:** Đường trung bình 200 ngày của chỉ số VNINDEX là ranh giới kinh điển giữa thị trường Bullish và Bearish.
        """)

        st.markdown("""
        **2. Nguyên tắc tín hiệu trễ (Lagged Signal — Không có Look-ahead Bias):**
        $$Signal_t = \\mathbb{I}(VNINDEX_{t-1} > SMA200_{t-1})$$
        - $Signal_t = 1 \\implies$ **INVEST**: Đầu tư 100% vào danh mục cổ phiếu cơ sở.
        - $Signal_t = 0 \\implies$ **CASH**: Đứng ngoài giữ 100% tiền mặt, lợi suất phiên $r_t = 0\\%$.
        - Vì giá đóng cửa ngày $t-1$ được xác định trước giờ mở cửa ngày $t$, chiến lược này **khả thi thực chiến 100%**, loại bỏ hoàn toàn thiên lệch nhìn trước tương lai.
        """)

        # Market exposure stats
        sig_train = market_signal.reindex(px_train.index).fillna(0)
        sig_test = market_signal.reindex(px_test.index).fillna(0)
        
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.info(f"**TRAIN (Uptrend 2020–2021):** Thời gian tham gia thị trường: **{(sig_train == 1).sum()} / {len(sig_train)} phiên** ({sig_train.mean():.2%})")
        with c_exp2:
            st.success(f"**TEST (Downtrend 2022):** Thời gian tham gia: **{(sig_test == 1).sum()} / {len(sig_test)} phiên** ({sig_test.mean():.2%}) ➔ **Đứng ngoài tiền mặt {1 - sig_test.mean():.2%} thời gian (179 phiên)!**")

        st.markdown("#### So Sánh Hiệu Quả: Chia Đều (1/N) vs. Market Timing (VNINDEX SMA200)")
        mt_m_tr, mt_m_te = strategies_data["Market Timing Only"]["train"], strategies_data["Market Timing Only"]["test"]

        df_mt_compare = pd.DataFrame({
            "Chỉ số": ["Lợi suất (Total Return)", "Biến động năm (Volatility)", "Chỉ số Sharpe (Rf=3%)", "Sụt giảm tối đa (Max Drawdown)"],
            "Equal Weight (Train)": [f"{bh_m_tr['Total Return']:.2%}", f"{bh_m_tr['Volatility']:.2%}", f"{bh_m_tr['Sharpe']:.2f}", f"{bh_m_tr['Max Drawdown']:.2%}"],
            "Market Timing (Train)": [f"{mt_m_tr['Total Return']:.2%}", f"{mt_m_tr['Volatility']:.2%}", f"{mt_m_tr['Sharpe']:.2f}", f"{mt_m_tr['Max Drawdown']:.2%}"],
            "Equal Weight (Test)": [f"{bh_m_te['Total Return']:.2%}", f"{bh_m_te['Volatility']:.2%}", f"{bh_m_te['Sharpe']:.2f}", f"{bh_m_te['Max Drawdown']:.2%}"],
            "Market Timing (Test)": [f"{mt_m_te['Total Return']:.2%}", f"{mt_m_te['Volatility']:.2%}", f"{mt_m_te['Sharpe']:.2f}", f"{mt_m_te['Max Drawdown']:.2%}"],
        }).set_index("Chỉ số")
        st.dataframe(df_mt_compare, use_container_width=True)

        # Plot Test curve with cash area
        fig_mt_test = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
        fig_mt_test.add_trace(go.Scatter(x=bh_m_te["Equity Series"].index, y=bh_m_te["Equity Series"]/1e6, name="Equal Weight", line=dict(color="#DC2626", dash="dash", width=1.5)), row=1, col=1)
        fig_mt_test.add_trace(go.Scatter(x=mt_m_te["Equity Series"].index, y=mt_m_te["Equity Series"]/1e6, name="Market Timing SMA200", line=dict(color="#D97706", width=2.5)), row=1, col=1)
        fig_mt_test.add_hline(y=100, line_dash="dot", line_color="gray", row=1, col=1)

        # Cash zone
        fig_mt_test.add_trace(go.Scatter(
            x=sig_test.index, y=1 - sig_test,
            name="Vùng Cầm Tiền Mặt (100% Cash)",
            fill="tozeroy", fillcolor="rgba(148, 163, 184, 0.25)",
            line=dict(color="#64748B", width=1)
        ), row=2, col=1)

        fig_mt_test.update_layout(title="Hiệu Quả Market Timing Trong Downtrend 2022 (Vùng Xám = Cầm Tiền Mặt)", height=450, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_mt_test, use_container_width=True)

        st.success("""
        **🎯 Kết Luận Định Lượng Về Market Timing:** Nhờ việc chuyển về 100% tiền mặt trong 71.89% số phiên năm 2022, chiến lược Market Timing đã **giảm sụt giảm tối đa từ -78.87% xuống chỉ còn -27.76%**, đồng thời giảm một nửa độ biến động danh mục (từ 52.53% xuống 24.44%). Đây chính là yếu tố phòng thủ quan trọng nhất!
        """)

    # --------------------------------------------------------------------------
    # TAB 4: PHẦN 3 — COMBINED STRATEGY APPROACH
    # --------------------------------------------------------------------------
    with strat_tab4:
        st.markdown("### 🔹 PHẦN 3: MÔ HÌNH KẾT HỢP (SHRINKAGE MIN-VOL + MARKET TIMING SMA200)")

        st.markdown("""
        **1. Triết lý tích hợp hiệp đồng (Synergistic Power):**
        - **Market Timing trả lời: WHEN?** Khi nào thị trường rủi ro ➔ Chuyển 100% Tiền Mặt.
        - **Shrinkage Min-Vol trả lời: HOW?** Khi thị trường thuận lợi ➔ Phân bổ vốn an toàn nhất, dồn 40% vào NLG và hạ tỷ trọng các mã biến động cao.
        - **Mô hình Kết Hợp (Combined):**
          - Khi $VNINDEX_{t-1} > SMA200_{t-1}$ (Bull): Đầu tư theo bộ tỷ trọng **Shrinkage Min-Vol**.
          - Khi $VNINDEX_{t-1} \le SMA200_{t-1}$ (Bear): Chuyển toàn bộ danh mục sang **100% Tiền Mặt**.
        """)

        comb_m_tr, comb_m_te = strategies_data["Combined Strategy (Shrinkage + MT)"]["train"], strategies_data["Combined Strategy (Shrinkage + MT)"]["test"]

        st.markdown("#### Bảng So Sánh Đối Đầu 3 Phương Pháp (Train & Test)")
        df_3_compare = pd.DataFrame({
            "Chỉ số": ["Lợi suất", "Biến động năm", "Sharpe (Rf=3%)", "Max Drawdown"],
            "Shrinkage Only (Train)": [f"{sh_m_tr['Total Return']:.2%}", f"{sh_m_tr['Volatility']:.2%}", f"{sh_m_tr['Sharpe']:.2f}", f"{sh_m_tr['Max Drawdown']:.2%}"],
            "Market Timing Only (Train)": [f"{mt_m_tr['Total Return']:.2%}", f"{mt_m_tr['Volatility']:.2%}", f"{mt_m_tr['Sharpe']:.2f}", f"{mt_m_tr['Max Drawdown']:.2%}"],
            "Shrinkage + MT (Train)": [f"{comb_m_tr['Total Return']:.2%}", f"{comb_m_tr['Volatility']:.2%}", f"{comb_m_tr['Sharpe']:.2f}", f"{comb_m_tr['Max Drawdown']:.2%}"],
            "Shrinkage Only (Test)": [f"{sh_m_te['Total Return']:.2%}", f"{sh_m_te['Volatility']:.2%}", f"{sh_m_te['Sharpe']:.2f}", f"{sh_m_te['Max Drawdown']:.2%}"],
            "Market Timing Only (Test)": [f"{mt_m_te['Total Return']:.2%}", f"{mt_m_te['Volatility']:.2%}", f"{mt_m_te['Sharpe']:.2f}", f"{mt_m_te['Max Drawdown']:.2%}"],
            "Shrinkage + MT (Test)": [f"{comb_m_te['Total Return']:.2%}", f"{comb_m_te['Volatility']:.2%}", f"{comb_m_te['Sharpe']:.2f}", f"{comb_m_te['Max Drawdown']:.2%}"],
        }).set_index("Chỉ số")
        st.dataframe(df_3_compare, use_container_width=True)

        st.markdown("#### Mức Cải Thiện Của Phương Pháp Kết Hợp Trên Tập Test 2022")
        df_impr = pd.DataFrame({
            "Metric": ["Lợi suất (Total Return)", "Độ biến động (Volatility)", "Sharpe Ratio", "Sụt giảm tối đa (Max Drawdown)"],
            "Shrinkage Only": [f"{sh_m_te['Total Return']:.2%}", f"{sh_m_te['Volatility']:.2%}", f"{sh_m_te['Sharpe']:.2f}", f"{sh_m_te['Max Drawdown']:.2%}"],
            "Market Timing Only": [f"{mt_m_te['Total Return']:.2%}", f"{mt_m_te['Volatility']:.2%}", f"{mt_m_te['Sharpe']:.2f}", f"{mt_m_te['Max Drawdown']:.2%}"],
            "Combined Strategy": [f"{comb_m_te['Total Return']:.2%}", f"{comb_m_te['Volatility']:.2%}", f"{comb_m_te['Sharpe']:.2f}", f"{comb_m_te['Max Drawdown']:.2%}"],
            "Combined - Shrinkage": [f"{(comb_m_te['Total Return'] - sh_m_te['Total Return']):+.2%}", f"{(comb_m_te['Volatility'] - sh_m_te['Volatility']):+.2%}", f"{(comb_m_te['Sharpe'] - sh_m_te['Sharpe']):+.2f}", f"{(comb_m_te['Max Drawdown'] - sh_m_te['Max Drawdown']):+.2%}"],
            "Combined - Market Timing": [f"{(comb_m_te['Total Return'] - mt_m_te['Total Return']):+.2%}", f"{(comb_m_te['Volatility'] - mt_m_te['Volatility']):+.2%}", f"{(comb_m_te['Sharpe'] - mt_m_te['Sharpe']):+.2f}", f"{(comb_m_te['Max Drawdown'] - mt_m_te['Max Drawdown']):+.2%}"],
        }).set_index("Metric")
        st.dataframe(df_impr, use_container_width=True)

        # Plot 3-way showdown
        fig_comb_test = go.Figure()
        fig_comb_test.add_trace(go.Scatter(x=bh_m_te["Equity Series"].index, y=bh_m_te["Equity Series"]/1e6, name="Equal Weight Benchmark", line=dict(color="#94A3B8", dash="dot", width=1.5)))
        fig_comb_test.add_trace(go.Scatter(x=sh_m_te["Equity Series"].index, y=sh_m_te["Equity Series"]/1e6, name="1. Shrinkage Only", line=dict(color="#0284C7", dash="dash", width=1.8)))
        fig_comb_test.add_trace(go.Scatter(x=mt_m_te["Equity Series"].index, y=mt_m_te["Equity Series"]/1e6, name="2. Market Timing Only", line=dict(color="#D97706", dash="dash", width=1.8)))
        fig_comb_test.add_trace(go.Scatter(x=comb_m_te["Equity Series"].index, y=comb_m_te["Equity Series"]/1e6, name="3. Combined (Shrinkage + MT)", line=dict(color="#059669", width=3.0)))
        fig_comb_test.add_hline(y=100, line_dash="dot", line_color="gray")
        fig_comb_test.update_layout(title="Đối Đầu 3 Phương Pháp Trong Downtrend 2022 (Combined Vượt Trội)", yaxis_title="Triệu VND", height=420, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_comb_test, use_container_width=True)


# ==============================================================================
# VIEW 4: STAGE 4 — BACKTEST & DECISION
# ==============================================================================
elif nav_choice == "04. Backtest & Decision":
    st.markdown("""
    <div class="top-banner">
        <h1>STAGE 4 — BACKTEST & STRATEGY DECISION</h1>
        <p>Kiểm định hiệu suất Out-of-Sample (năm 2022) và xác định phương án đầu tư cuối cùng.<br>
        Bắt buộc đối chiếu 8 chỉ số giảng viên yêu cầu và so sánh trực diện với Benchmark Buy & Hold.</p>
    </div>
    """, unsafe_allow_html=True)

    period_selector = st.radio(
        "LỰA CHỌN GIAI ĐOẠN ĐÁNH GIÁ:",
        ["TẬP TEST (2022 — Out-of-Sample: Cơ sở quyết định chính)", "TẬP TRAIN (2020–2021 — In-Sample: Huấn luyện)"],
        horizontal=True
    )
    is_test = "TEST" in period_selector
    period_key = "test" if is_test else "train"

    # KPI Summary Cards for selected period
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Return (Combined)</div>
            <div class="kpi-value {'kpi-positive' if strategies_data['Combined Strategy (Shrinkage + MT)'][period_key]['Total Return'] > 0 else 'kpi-negative'}">
                {strategies_data['Combined Strategy (Shrinkage + MT)'][period_key]['Total Return']:.2%}
            </div>
            <div class="kpi-sub">vs B&H: {strategies_data['Baseline (Equal Weight)'][period_key]['Total Return']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Maximum Drawdown</div>
            <div class="kpi-value kpi-highlight">
                {strategies_data['Combined Strategy (Shrinkage + MT)'][period_key]['Max Drawdown']:.2%}
            </div>
            <div class="kpi-sub">vs B&H: {strategies_data['Baseline (Equal Weight)'][period_key]['Max Drawdown']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Sharpe Ratio (Rf=3%)</div>
            <div class="kpi-value kpi-highlight">
                {strategies_data['Combined Strategy (Shrinkage + MT)'][period_key]['Sharpe']:.2f}
            </div>
            <div class="kpi-sub">Sortino: {strategies_data['Combined Strategy (Shrinkage + MT)'][period_key]['Sortino']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Winning Rate</div>
            <div class="kpi-value kpi-positive">
                {strategies_data['Combined Strategy (Shrinkage + MT)'][period_key]['Win Rate (Active)']:.2%}
            </div>
            <div class="kpi-sub">Độ biến động: {strategies_data['Combined Strategy (Shrinkage + MT)'][period_key]['Volatility']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Required Metrics Table
    st.markdown(f"### 1. Bảng So Sánh Chi Tiết 8 Chỉ Số Hiệu Suất ({'TEST 2022' if is_test else 'TRAIN 2020-2021'})")
    
    comp_rows = []
    for sname, sdata in strategies_data.items():
        m = sdata[period_key]
        comp_rows.append({
            "Chiến Lược": sname,
            "1. Return (Tổng)": f"{m['Total Return']:.2%}",
            "2. CAGR (Năm)": f"{m['CAGR']:.2%}",
            "3. Winning Rate (Active)": f"{m['Win Rate (Active)']:.2%}",
            "4. Volatility (Năm)": f"{m['Volatility']:.2%}",
            "5. Max Drawdown": f"{m['Max Drawdown']:.2%}",
            "6. Sharpe (Rf=3%)": f"{m['Sharpe']:.2f}",
            "7. Sortino": f"{m['Sortino']:.2f}",
            "8. Calmar": f"{m['Calmar']:.2f}",
            "Tỷ lệ tham gia": f"{m['Market Exposure']:.1%}"
        })
    comp_df = pd.DataFrame(comp_rows).set_index("Chiến Lược")
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")

    # Interactive Equity & Drawdown curves
    st.markdown("### 2. Biểu Đồ Tăng Trưởng Tài Sản (Equity Curve) & Sụt Giảm (Drawdown)")
    
    fig_curves = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=[
            f"Đường Tăng Trưởng Vốn (Equity Curve — Khởi điểm 100 Triệu VND) [{'TEST 2022' if is_test else 'TRAIN 2020-2021'}]",
            "Mức Độ Sụt Giảm Từ Đỉnh (Underwater Drawdown Curve)"
        ]
    )

    for sname, sdata in strategies_data.items():
        eq_ser = sdata[period_key]["Equity Series"]
        dd_ser = sdata[period_key]["Drawdown Series"]
        col = sdata["color"]
        
        # Row 1: Equity
        fig_curves.add_trace(
            go.Scatter(
                x=eq_ser.index, y=eq_ser / 1_000_000,
                name=sname,
                line=dict(color=col, width=2.5 if "Combined" in sname else 1.8)
            ),
            row=1, col=1
        )
        # Row 2: Drawdown
        fig_curves.add_trace(
            go.Scatter(
                x=dd_ser.index, y=dd_ser,
                name=f"{sname} DD",
                showlegend=False,
                line=dict(color=col, width=1.5)
            ),
            row=2, col=1
        )

    # Add 100M initial line
    fig_curves.add_hline(y=100, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)
    fig_curves.update_yaxes(title_text="Giá trị danh mục (Triệu VND)", row=1, col=1)
    fig_curves.update_yaxes(title_text="Drawdown (%)", tickformat=".0%", row=2, col=1)
    fig_curves.update_layout(height=650, margin=dict(t=40, b=20, l=20, r=20), hovermode="x unified")

    st.plotly_chart(fig_curves, use_container_width=True)

    # In-Sample vs Out-of-Sample Full Comparative Matrix
    st.markdown("---")
    st.markdown("### 3. Ma Trận Đối Chiếu In-Sample (Train) vs. Out-of-Sample (Test)")

    matrix_rows = []
    for sname, sdata in strategies_data.items():
        tr_m = sdata["train"]
        te_m = sdata["test"]
        matrix_rows.append({
            "Chiến lược": sname,
            "TRAIN Return": f"{tr_m['Total Return']:.2%}",
            "TEST Return": f"{te_m['Total Return']:.2%}",
            "TRAIN Max DD": f"{tr_m['Max Drawdown']:.2%}",
            "TEST Max DD": f"{te_m['Max Drawdown']:.2%}",
            "TRAIN Vol": f"{tr_m['Volatility']:.2%}",
            "TEST Vol": f"{te_m['Volatility']:.2%}",
            "TRAIN Sharpe": f"{tr_m['Sharpe']:.2f}",
            "TEST Sharpe": f"{te_m['Sharpe']:.2f}",
        })
    matrix_df = pd.DataFrame(matrix_rows).set_index("Chiến lược")
    st.dataframe(matrix_df, use_container_width=True)


# ==============================================================================
# VIEW 5: STAGE 5 — TỔNG QUAN & KHUYẾN NGHỊ (CHỐT HẠ CHIẾN LƯỢC ĐẦU TƯ CUỐI CÙNG)
# ==============================================================================
elif nav_choice == "05. Tổng Quan & Khuyến Nghị":
    st.markdown("""
    <div class="top-banner">
        <h1>TỔNG QUAN & CHIẾN LƯỢC ĐẦU TƯ KHUYẾN NGHỊ (FINAL INVESTMENT STRATEGY)</h1>
        <p>Báo cáo quyết định phân bổ vốn định lượng dựa trên kết quả kiểm định Out-of-Sample (năm 2022).<br>
        Tổng hợp kết luận cho Hội đồng Đầu tư: Khẳng định chiến lược tối ưu có cơ sở toán học và thực nghiệm vững chắc.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Executive Summary Cards
    comb_test = strategies_data["Combined Strategy (Shrinkage + MT)"]["test"]
    bh_test = strategies_data["Baseline (Equal Weight)"]["test"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Chiến Lược Khuyến Nghị</div>
            <div class="kpi-value kpi-highlight" style="font-size: 19px;">COMBINED STRATEGY</div>
            <div class="kpi-sub">Shrinkage Min-Vol + VNINDEX SMA200</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Alpha Vượt Trội (vs Buy & Hold)</div>
            <div class="kpi-value kpi-positive">+{(comb_test['Total Return'] - bh_test['Total Return']):.2%}</div>
            <div class="kpi-sub">-22.77% vs -68.04% trong Downtrend 2022</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Kiểm Soát Rủi Ro (Max DD)</div>
            <div class="kpi-value kpi-positive">{comb_test['Max Drawdown']:.2%}</div>
            <div class="kpi-sub">Giảm hơn 52.6% sụt giảm so với B&H (-78.87%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Phòng Vệ Tiền Mặt 2022</div>
            <div class="kpi-value kpi-highlight">{1 - comb_test['Market Exposure']:.1%}</div>
            <div class="kpi-sub">179/249 phiên đứng ngoài tiền mặt an toàn</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Recommended Decision Tree & Strategy Framework
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎯 BỘ NGUYÊN TẮC THI HÀNH CHIẾN LƯỢC (EXECUTION RULES)")
        
        st.markdown(f"""
        <div class="rec-box">
            <h4 style="margin: 0 0 8px 0; color: #065F46;">✅ KHUYẾN NGHỊ CHÍNH THỨC: CHIẾN LƯỢC KẾT HỢP (COMBINED STRATEGY)</h4>
            <p style="margin: 0; line-height: 1.6;">
            Dựa trên toàn bộ kết quả kiểm định Out-of-Sample năm 2022, <b>Combined Strategy (Shrinkage Min-Vol + VNINDEX SMA200)</b> được lựa chọn là chiến lược quản trị danh mục tối ưu nhất. Chiến lược giải quyết triệt để vấn đề "bốc hơi tài sản" của trường phái thụ động Buy & Hold khi bước vào pha thị trường gấu (Bear Market).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        | Câu Hỏi Cốt Lõi | Quyết Định & Nguyên Tắc Thực Thi Cụ Thể |
        | :--- | :--- |
        | **1. Nên chọn những cổ phiếu nào?** | Chọn **TOP 5**: `DIG`, `GEX`, `NLG`, `VND`, `ITA`. Được tuyển chọn từ Top 30 thanh khoản cao nhất, thỏa mãn bộ lọc kỹ thuật và có tương quan chéo thấp ($r \le 0.70$). |
        | **2. Phân bổ vốn như thế nào?** | Phân bổ vốn theo mô hình **Ledoit-Wolf Covariance Shrinkage (Minimum Volatility)** với ràng buộc tỷ trọng trần 40%: **NLG (40.00%)**, **GEX (23.49%)**, **VND (17.97%)**, **ITA (9.86%)**, **DIG (8.68%)**. |
        | **3. Khi nào INVEST (Tham gia)?** | **Khi $VNINDEX_{t-1} > SMA200_{t-1}$**: Thị trường xác nhận xu hướng Uptrend dài hạn. Giải ngân 100% tài sản theo đúng bộ tỷ trọng Shrinkage Min-Vol. |
        | **4. Khi nào chuyển CASH (Tiền mặt)?** | **Khi $VNINDEX_{t-1} \le SMA200_{t-1}$**: Xu hướng tăng bị gãy, rủi ro sụt giảm hệ thống gia tăng. Lập tức rút 100% danh mục về tiền mặt / tài sản phi rủi ro. |
        | **5. Đảm bảo tính trung thực (No Look-ahead Bias)?** | Tín hiệu chuyển pha sử dụng **Lagged Signal (shift 1 phiên)**. Giá đóng cửa ngày $t-1$ quyết định hành động tại ngày $t$, đảm bảo khả thi thực chiến 100%. |
        """)

    with col_right:
        st.markdown("### 🍰 TỶ TRỌNG PHÂN BỔ TỐI ƯU")
        # Donut Chart for Portfolio Weights
        weights_df = pd.DataFrame({
            "Ticker": top5,
            "Tỷ trọng": [cleaned_w[t] for t in top5]
        }).sort_values("Tỷ trọng", ascending=False)
        
        fig_pie = px.pie(
            weights_df,
            names="Ticker",
            values="Tỷ trọng",
            hole=0.45,
            color_discrete_sequence=["#059669", "#0284C7", "#3B82F6", "#D97706", "#64748B"],
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Tỷ trọng: %{percent:.2%}<extra></extra>'
        )
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
            height=260
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.caption("📌 **Ghi chú phân bổ:** NLG được cấp tỷ trọng tối đa kịch trần (40%) nhờ biến động thấp nhất và đóng góp rủi ro an toàn nhất, trong khi các mã có beta cao như DIG chỉ nhận 8.68%.")

    st.markdown("---")

    # 3. Quick Out-of-Sample Performance Comparison
    st.markdown("### 🏆 BẢNG TỔNG KẾT SO SÁNH OUT-OF-SAMPLE (TEST 2022)")
    
    summary_rows = []
    for sname, sdata in strategies_data.items():
        m = sdata["test"]
        summary_rows.append({
            "Chiến lược": sname,
            "Lợi suất (Total Return)": f"{m['Total Return']:.2%}",
            "Lợi suất/năm (CAGR)": f"{m['CAGR']:.2%}",
            "Tỷ lệ thắng (Active Win Rate)": f"{m['Win Rate (Active)']:.2%}",
            "Độ biến động (Volatility)": f"{m['Volatility']:.2%}",
            "Sụt giảm tối đa (Max Drawdown)": f"{m['Max Drawdown']:.2%}",
            "Sharpe (Rf=3%)": f"{m['Sharpe']:.2f}",
            "Sortino": f"{m['Sortino']:.2f}",
            "Calmar": f"{m['Calmar']:.2f}",
            "Thời gian đầu tư": f"{m['Market Exposure']:.1%}"
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Chiến lược")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("""
    > **💡 TẠI SAO COMBINED STRATEGY CHIẾN THẮNG?**
    > 1. **Khắc chế triệt để thiên nga đen 2022:** Trong năm 2022 khi VN-Index lao dốc từ 1,500 điểm xuống 1,000 điểm, chiến lược thụ động Buy & Hold lỗ thảm khốc **-68.04%** và Max Drawdown lên đến **-78.87%**.
    > 2. **Sức mạnh phòng vệ của Tiền Mặt (Market Timing):** Nhờ bộ lọc VNINDEX SMA200, danh mục chuyển sang **100% Cash trong 71.89% số phiên**, cắt đứt hoàn toàn các đợt bán tháo lớn nhất.
    > 3. **Tối ưu hóa hiệp phương sai (Ledoit-Wolf Min-Vol):** Trong các giai đoạn thị trường ngắn hạn hồi phục, danh mục được dồn 40% vốn vào cổ phiếu phòng thủ nhất (NLG), giúp danh mục đạt biến động thấp nhất hệ thống (**23.28%**) và mức sụt giảm tối đa thấp nhất (**-26.20%**).
    """)


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 12px; padding-bottom: 20px;">
    Investment Strategy Dashboard | Master of Finance & Banking (MFB025A) — Open University<br>
    Hệ thống phân tích và kiểm định chiến lược quản lý danh mục định lượng | Nghiêm cấm mọi hành vi Look-ahead Bias.
</div>
""", unsafe_allow_html=True)
