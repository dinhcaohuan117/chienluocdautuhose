"""
INVESTMENT STRATEGY DASHBOARD
========================================================================================
Institutional Quantitative Portfolio Strategy & Backtest Dashboard
Framework 4 Giai đoạn:
  Stage 1 — Data & Stock Universe
  Stage 2 — Stock Selection (4-Factor Multi-Factor Ranking & Correlation Filter)
  Stage 3 — Portfolio Strategy (Buy & Hold, Ledoit-Wolf Shrinkage Min-Vol, Market Timing)
  Stage 4 — Backtest & Strategy Decision (Out-of-Sample 2022 Verification & Recommendation)
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

# Custom Institutional CSS styling
st.markdown("""
<style>
    /* Global font & styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Top Header Banner */
    .top-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: #F8FAFC;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    .top-banner h1 {
        color: #F8FAFC !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .top-banner p {
        color: #94A3B8;
        font-size: 13.5px;
        margin: 0;
        line-height: 1.5;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .kpi-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #94A3B8;
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
        padding: 18px 22px;
        margin: 18px 0;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .exec-box-title {
        font-weight: 700;
        font-size: 15px;
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
        margin: 16px 0;
        color: #064E3B;
    }
    
    /* Table headers styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
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
        # Downside Deviation considering target 0 return
        downside = r[r < 0]
        downside_std = np.sqrt((r.clip(upper=0)**2).mean()) * np.sqrt(DEFAULT_TRADING_DAYS)
        sortino = (r.mean() * DEFAULT_TRADING_DAYS - rf_annual) / downside_std if downside_std > 0 else np.nan
        
        # 8. Calmar Ratio
        calmar = cagr / abs(mdd) if (mdd < 0 and not np.isnan(mdd)) else np.nan
        
        # 3. Winning Rate (Tỷ lệ phiên sinh lời)
        # Winning rate trên các phiên có giao dịch / active (r != 0)
        active_r = r[r != 0]
        win_rate_active = (active_r > 0).sum() / len(active_r) if len(active_r) > 0 else 0.0
        # Winning rate trên toàn bộ số phiên
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
    <div style="padding: 10px 0 16px 0;">
        <span style="font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 1px; text-transform: uppercase;">Quantitative Investing</span>
        <h2 style="margin: 4px 0 2px 0; font-size: 20px; font-weight: 700; color: #0F172A;">PORTFOLIO DASHBOARD</h2>
        <p style="font-size: 12px; color: #64748B; margin: 0;">Đại học Mở TP.HCM | MFB025A</p>
    </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "ĐIỀU HƯỚNG DASHBOARD",
        [
            "📌 Tổng Quan & Khuyến Nghị",
            "01. Data & Universe",
            "02. Stock Selection",
            "03. Portfolio Strategy",
            "04. Backtest & Decision",
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
    st.markdown("### 📊 DỮ LIỆU THỰC TẾ")
    st.markdown(f"""
    - **Tổng số mã:** `{px_all.shape[1]} mã sàn HOSE`
    - **Tập Train (In-Sample):** `2020-01-02` ➔ `2021-12-31` (`{len(px_train)}` phiên)
    - **Tập Test (Out-of-Sample):** `2022-01-04` ➔ `2022-12-30` (`{len(px_test)}` phiên)
    """)


# ==============================================================================
# VIEW 0: HOME — INVESTMENT STRATEGY OVERVIEW & FINAL RECOMMENDATION
# ==============================================================================
if nav_choice == "📌 Tổng Quan & Khuyến Nghị":
    st.markdown("""
    <div class="top-banner">
        <h1>CHIẾN LƯỢC ĐẦU TƯ ĐƯỢC KHUYẾN NGHỊ (RECOMMENDED INVESTMENT STRATEGY)</h1>
        <p>Báo cáo quyết định phân bổ vốn định lượng dựa trên kết quả kiểm định Out-of-Sample (năm 2022).<br>
        Quy trình kỷ luật 4 giai đoạn: Sàng lọc Vũ trụ đầu tư ➔ Đa nhân tố & Đa dạng hóa ➔ Tối ưu Ledoit-Wolf ➔ Định thời điểm VNINDEX SMA200.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Executive Summary Cards
    comb_test = strategies_data["Combined Strategy (Shrinkage + MT)"]["test"]
    bh_test = strategies_data["Baseline (Equal Weight)"]["test"]
    mt_test = strategies_data["Market Timing Only"]["test"]
    sh_test = strategies_data["Shrinkage Only"]["test"]

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
            <div class="kpi-sub">179/249 phiên đứng ngoài tiền mặt</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Recommended Decision Tree & Strategy Framework
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎯 BỘ NGUYÊN TẮC THI HÀNH CHIẾN LƯỢC (EXECUTION RULES)")
        
        st.markdown(f"""
        <div class="rec-box">
            <h4 style="margin: 0 0 8px 0; color: #065F46;">✅ KHUYẾN NGHỊ CUỐI CÙNG: CHIẾN LƯỢC KẾT HỢP (COMBINED STRATEGY)</h4>
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
# VIEW 1: STAGE 1 — DATA & STOCK UNIVERSE
# ==============================================================================
elif nav_choice == "01. Data & Universe":
    st.markdown("""
    <div class="top-banner">
        <h1>STAGE 1 — DATA & STOCK UNIVERSE</h1>
        <p>Quy trình thu thập, làm sạch dữ liệu, phân tách Train/Test nghiêm ngặt và sàng lọc thanh khoản.<br>
        Mục tiêu: Loại trừ cổ phiếu rác, đảm bảo thanh khoản và tạo ra Eligible Universe đáng tin cậy.</p>
    </div>
    """, unsafe_allow_html=True)

    # Diagram workflow
    st.markdown("""
    ```mermaid
    flowchart LR
        A[Market Data HOSE] --> B[Data Cleaning & Non-missing >= 90%]
        B --> C[Train / Test Split 2020-21 vs 2022]
        C --> D[Technical Indicators Computation]
        D --> E[Liquidity Screening Top 30]
        E --> F[Filter: RSI <= 75 & Px >= SMA50]
        F --> G[Eligible Universe: 20 Stocks]
    ```
    """)

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Tổng cổ phiếu HOSE đủ chuẩn", f"{px_all.shape[1]} mã")
    with m2:
        st.metric("Giai đoạn Train (In-Sample)", "2020 - 2021 (502 phiên)")
    with m3:
        st.metric("Giai đoạn Test (Out-of-Sample)", "2022 (249 phiên)")
    with m4:
        st.metric("Eligible Universe Sau Lọc", f"{len(eligible)} / 30 mã")

    st.markdown("### 1. Phương Pháp Tính Các Chỉ Báo Kỹ Thuật (Train Period)")
    st.markdown("""
    - **Trend 1 (Giá so với SMA50):** $px\_vs\_sma50 = \\frac{P_t}{SMA_{50}} - 1$. Đảm bảo cổ phiếu đang trong xu hướng tăng ngắn hạn.
    - **Trend 2 (SMA50 so với SMA200):** $sma50\_vs\_sma200 = \\frac{SMA_{50}}{SMA_{200}} - 1$. Xác nhận xu hướng tăng trung-dài hạn (Golden Cross).
    - **Momentum 1 (Lợi suất 6 tháng bỏ 1 tháng cuối):** $mom\_6m = \\frac{P_{t-21}}{P_{t-126}} - 1$. Loại bỏ hiệu ứng đảo chiều ngắn hạn (1-month reversal effect).
    - **Momentum 2 (Wilder's RSI 14):** Tính toán theo công thức hàm mũ chuẩn Wilder's RMA.
    - **Risk (Độ biến động ngày):** Độ lệch chuẩn của tỷ suất sinh lời hàng ngày trên tập Train.
    - **Liquidity (Giá trị GD trung vị 60 phiên):** $Median(Close_t \\times Volume_t)$ trong 60 phiên gần nhất của Train.
    """)

    # Screening details
    st.markdown("### 2. Kết Quả Sàng Lọc Thanh Khoản & Bộ Lọc Kỹ Thuật")
    col_el, col_rem = st.columns([3, 2])

    with col_el:
        st.markdown(f"#### ✅ Danh Sách Đủ Điều Kiện (Eligible Universe: {len(eligible)} mã)")
        disp_el = eligible.sort_values("liq_value", ascending=False).copy()
        disp_el_fmt = disp_el.assign(
            px_vs_sma50=disp_el["px_vs_sma50"].map("{:.1%}".format),
            sma50_vs_sma200=disp_el["sma50_vs_sma200"].map("{:.1%}".format),
            mom_6m=disp_el["mom_6m"].map("{:.1%}".format),
            rsi14=disp_el["rsi14"].map("{:.1f}".format),
            vol_daily=disp_el["vol_daily"].map("{:.2%}".format),
            liq_value=disp_el["liq_value"].map("{:,.0f} VND".format),
        )
        st.dataframe(disp_el_fmt, use_container_width=True)

    with col_rem:
        st.markdown(f"#### ❌ Cổ Phiếu Bị Loại Khỏi Top 30 ({len(removed)} mã)")
        st.markdown("Nguyên nhân loại: **RSI14 > 75 (Quá mua cực đoan)** hoặc **Giá nằm dưới đường SMA50 (Gãy xu hướng ngắn hạn)**.")
        disp_rem = removed.copy()
        disp_rem["Lý do loại"] = np.where(
            disp_rem["rsi14"] > 75, "RSI > 75 (Quá mua)", "Giá < SMA50 (Mất Trend)"
        )
        disp_rem_fmt = disp_rem[["rsi14", "px_vs_sma50", "Lý do loại"]].assign(
            rsi14=disp_rem["rsi14"].map("{:.1f}".format),
            px_vs_sma50=disp_rem["px_vs_sma50"].map("{:.1%}".format)
        )
        st.dataframe(disp_rem_fmt, use_container_width=True)

    # Interactive Bubble / Scatter chart of Universe
    st.markdown("### 3. Biểu Đồ Thanh Khoản vs. Biến Động Rủi Ro")
    scatter_df = pool.copy().reset_index()
    scatter_df["Status"] = np.where(scatter_df["ticker"].isin(eligible.index), "Đủ điều kiện (Eligible)", "Bị loại (Filtered Out)")
    
    fig_scatter = px.scatter(
        scatter_df,
        x="vol_daily",
        y="liq_value",
        size="rsi14",
        color="Status",
        text="ticker",
        color_discrete_map={"Đủ điều kiện (Eligible)": "#059669", "Bị loại (Filtered Out)": "#DC2626"},
        labels={"vol_daily": "Độ biến động ngày (Daily Volatility)", "liq_value": "Giá trị GD Trung vị 60 phiên (VND)", "rsi14": "RSI 14"},
        title="Bản Đồ Phân Phối Cổ Phiếu Top 30 Thanh Khoản"
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(yaxis_type="log", height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ==============================================================================
# VIEW 2: STAGE 2 — STOCK SELECTION
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
# VIEW 3: STAGE 3 — PORTFOLIO STRATEGY
# ==============================================================================
elif nav_choice == "03. Portfolio Strategy":
    st.markdown("""
    <div class="top-banner">
        <h1>STAGE 3 — PORTFOLIO STRATEGY</h1>
        <p>Xây dựng phương án phân bổ vốn (Capital Allocation) và xác định thời điểm tham gia (Market Timing).<br>
        Giải quyết hai câu hỏi then chốt: "How should capital be allocated?" và "When should we invest?"</p>
    </div>
    """, unsafe_allow_html=True)

    # Conceptual layout
    q1_col, q2_col = st.columns(2)

    with q1_col:
        st.markdown("""
        <div class="exec-box">
            <div class="exec-box-title">Câu hỏi 1: HOW SHOULD CAPITAL BE ALLOCATED?</div>
            <p style="font-size: 13.5px; color: #334155; margin: 0;">
            <b>Phương pháp 1 (Buy & Hold Equal Weight):</b> Chia đều 20% vốn cho mỗi cổ phiếu.<br>
            <b>Phương pháp 2 (Ledoit-Wolf Covariance Shrinkage):</b> Tối thiểu hóa phương sai danh mục (Minimum Volatility) không ước lượng lợi nhuận kỳ vọng, giới hạn tỷ trọng tối đa 40%/cổ phiếu để tránh dồn vốn quá mức.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with q2_col:
        st.markdown("""
        <div class="exec-box" style="border-left-color: #059669;">
            <div class="exec-box-title">Câu hỏi 2: WHEN SHOULD WE INVEST?</div>
            <p style="font-size: 13.5px; color: #334155; margin: 0;">
            <b>Tín hiệu định thời điểm (VNINDEX SMA200):</b><br>
            - Khi $VNINDEX > SMA200$ ➔ <b>INVEST (Tham gia thị trường)</b>.<br>
            - Khi $VNINDEX \le SMA200$ ➔ <b>CASH (Đứng ngoài giữ 100% tiền mặt)</b>.<br>
            - Tín hiệu trễ 1 ngày ($t-1$) triệt tiêu hoàn toàn Look-ahead bias.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 1. So Sánh 4 Phương Pháp Tiếp Cận")
    method_table = pd.DataFrame({
        "Phương Pháp": [
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
    }).set_index("Phương Pháp")
    st.table(method_table)

    st.markdown("---")
    st.markdown("### 2. Chi Tiết Tối Ưu Hóa Tỷ Trọng Ledoit-Wolf Shrinkage (Min-Vol)")
    w_col1, w_col2 = st.columns([3, 2])

    with w_col1:
        st.markdown("""
        **Lý do lựa chọn Ledoit-Wolf Shrinkage kết hợp Min-Vol:**
        - **Khử nhiễu ma trận hiệp phương sai:** Ma trận mẫu truyền thống có độ nhiễu rất cao do số lượng quan sát hữu hạn. Ledoit-Wolf "co" ma trận mẫu về ma trận mục tiêu có cấu trúc chặt chẽ hơn.
        - **Miễn nhiễm với Overfitting lợi nhuận:** Không ước lượng tham số $\mu$ (Expected Returns) vì lợi nhuận kỳ vọng cực kỳ bất định và thường sai lệch nghiêm trọng ngoài mẫu. Chỉ tìm bộ trọng số làm cực tiểu phương sai $\sigma^2_p$.
        - **Ràng buộc tỷ trọng trần 40%:** Đảm bảo tính đa dạng hóa, ngăn thuật toán dồn 100% vào một cổ phiếu duy nhất.
        """)

        w_comp = pd.DataFrame({
            "Cổ phiếu": top5,
            "Tỷ trọng Equal Weight": ["20.00%"] * 5,
            "Tỷ trọng Shrinkage Min-Vol": [f"{cleaned_w[t]:.2%}" for t in top5]
        }).set_index("Cổ phiếu")
        st.dataframe(w_comp, use_container_width=True)

    with w_col2:
        fig_bar_w = px.bar(
            pd.DataFrame({"Ticker": top5, "Tỷ trọng": [cleaned_w[t] for t in top5]}),
            x="Ticker",
            y="Tỷ trọng",
            color="Tỷ trọng",
            color_continuous_scale="Teal",
            title="Bộ Tỷ Trọng Tối Ưu Shrinkage Min-Vol"
        )
        fig_bar_w.update_layout(height=320, showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_bar_w, use_container_width=True)

    st.markdown("---")
    st.markdown("### 3. Cơ Chế Định Thời Điểm: VNINDEX & Đường SMA200 (Lagged Signal)")

    # Plot VNINDEX vs SMA200 with shaded areas
    vni_df = pd.DataFrame({
        "VNINDEX": vnindex,
        "SMA200": vnindex_sma,
        "Signal": market_signal
    }).dropna()

    fig_vni = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)

    # Line VNINDEX & SMA200
    fig_vni.add_trace(go.Scatter(x=vni_df.index, y=vni_df["VNINDEX"], name="VNINDEX Close", line=dict(color="#0F172A", width=1.5)), row=1, col=1)
    fig_vni.add_trace(go.Scatter(x=vni_df.index, y=vni_df["SMA200"], name="SMA200", line=dict(color="#DC2626", width=2, dash="dash")), row=1, col=1)

    # Shaded signal
    fig_vni.add_trace(go.Scatter(
        x=vni_df.index, y=vni_df["Signal"],
        name="Tín hiệu Tham gia (1 = Invest, 0 = Cash)",
        line=dict(color="#059669", width=1.5),
        fill="tozeroy", fillcolor="rgba(5, 150, 105, 0.15)"
    ), row=2, col=1)

    # Train / Test vertical line divider
    fig_vni.add_vline(x="2022-01-01", line_width=2, line_dash="dot", line_color="#475569")
    fig_vni.add_annotation(x="2021-06-01", y=1400, text="TRAIN (UPTREND)", showarrow=False, font=dict(color="#059669", size=13))
    fig_vni.add_annotation(x="2022-06-01", y=1400, text="TEST (DOWNTREND)", showarrow=False, font=dict(color="#DC2626", size=13))

    fig_vni.update_layout(height=480, margin=dict(t=30, b=20, l=20, r=20), hovermode="x unified")
    st.plotly_chart(fig_vni, use_container_width=True)

    st.markdown("""
    > **⚠️ Không có Look-ahead Bias:** Giá đóng cửa hôm nay chỉ dùng để tạo tín hiệu cho ngày giao dịch hôm sau. Nếu hôm nay VNINDEX gãy đường SMA200, tài khoản sẽ bán ra chuyển về tiền mặt từ đầu phiên ngày mai.
    """)


# ==============================================================================
# VIEW 4: STAGE 4 — BACKTEST & STRATEGY DECISION
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

    # Rigorous Academic & Practical Conclusion
    st.markdown("---")
    st.markdown("### 4. Kết Luận Khoa Học & Quyết Định Chiến Lược Đầu Tư")
    st.markdown("""
    <div class="rec-box" style="background: #F8FAFC; border-color: #CBD5E1; border-left: 5px solid #2563EB; color: #0F172A;">
        <h4 style="margin: 0 0 10px 0; color: #1E40AF;">🎓 KẾT LUẬN & ĐÁNH GIÁ CHUYÊN SÂU DỰA TRÊN KẾT QUẢ BACKTEST</h4>
        <ol style="margin: 0 0 0 18px; padding: 0; line-height: 1.7;">
            <li><b>Chiến lược Buy & Hold thụ động hoàn toàn thất bại khi gặp suy thoái:</b> Trong năm 2022, chiến lược này ghi nhận sụt giảm tối đa <b>-78.87%</b> và thua lỗ <b>-68.04%</b>. Điều này chứng minh rằng việc đa dạng hóa danh mục theo cách thụ động (Equal Weight) không thể bảo vệ nhà đầu tư khi toàn thị trường rơi vào rủi ro hệ thống (Systemic Risk).</li>
            <li><b>Thuật toán Shrinkage Min-Vol một mình là chưa đủ:</b> Tối ưu hóa hiệp phương sai giúp giảm biến động (từ 52.53% xuống 50.84%) nhưng vẫn phải gánh chịu mức lỗ <b>-62.25%</b> và Max Drawdown <b>-75.70%</b> nếu không có cơ chế thoát khỏi thị trường.</li>
            <li><b>Market Timing là yếu tố quyết định thành bại:</b> Cơ chế đứng ngoài giữ 100% tiền mặt khi VNINDEX gãy đường SMA200 đã giúp danh mục né tránh <b>71.89% thời gian giảm giá khốc liệt nhất</b>, đưa mức sụt giảm tối đa từ -78.87% về chỉ còn <b>-26.20%</b>.</li>
            <li><b>Chiến lược Kết hợp (Combined Strategy) là lựa chọn tối ưu toàn diện:</b> Kết hợp tối ưu tỷ trọng phòng thủ (Ledoit-Wolf Min-Vol) trong các nhịp tăng và chuyển sang 100% Cash trong các pha giảm giúp đạt hiệu suất sinh lời cao nhất ngoài mẫu (<b>-22.77%</b>, vượt trội hơn +45.27% so với Buy & Hold), độ biến động thấp nhất (<b>23.28%</b>), và kiểm soát rủi ro hoàn hảo nhất.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


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
