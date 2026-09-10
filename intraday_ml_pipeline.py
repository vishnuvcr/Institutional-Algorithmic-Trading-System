#!/usr/bin/env python3
"""
INSTITUTIONAL UNIVERSAL QUANT ML ENGINE (NSE)
==============================================
Includes Full Feature Store:
- Volume: Relative Volume (RVOL), Normalized OBV, VWAP Distance, Money Flow Index (MFI)
- Momentum: Normalized RSI, Stochastic %K/%D, True Strength Index (TSI), ROC
- Trend: Supertrend Distance, ADX, Directional Differential (DMI), Multi-EMA Ribbons
- Volatility: Historical Volatility (HV), Normalized ATR (NATR), Bollinger Bands (%B)
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Strategy Parameters
TARGET_PROFIT_PCT  = 0.025      # 2.5% Target (Intraday Horizon)
ATR_MULTIPLIER     = 1.50       # ATR Stop Loss Multiplier
HORIZON_BARS       = 20         # Forward evaluation bars (~5 hours on 15m)
MIN_BARS_PER_STOCK = 150        # Minimum bars required
MAX_POOL_STOCKS    = 60         # Top liquid stocks pooled for training
RISK_FREE_RATE     = 0.065      # Baseline rate (~6.5%)


@dataclass
class UniversalOpportunity:
    ticker: str
    direction: str
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float
    universal_prob: float
    rvol: float
    vwap_dist: float
    atr: float


# =============================================================================
# 1. FULL INSTITUTIONAL FEATURE STORE (VOLUME + MOMENTUM + TREND + VOLATILITY)
# =============================================================================
def build_full_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # ------------------ A. VOLUME DYNAMICS ------------------
    # 1. RVOL (Relative Volume Surge vs 20-period average)
    vol_sma20 = volume.rolling(20).mean()
    data['RVOL'] = volume / (vol_sma20 + 1e-9)

    # 2. Intraday Anchored VWAP Distance (%)
    typical_p = (high + low + close) / 3.0
    cum_pv = (typical_p * volume).cumsum()
    cum_v = volume.cumsum()
    vwap = cum_pv / (cum_v + 1e-9)
    data['VWAP_Dist'] = (close - vwap) / (vwap + 1e-9)

    # 3. Normalized On-Balance Volume (OBV 20-bar Z-Score)
    direction_sign = np.sign(close.diff()).fillna(0)
    obv = (direction_sign * volume).cumsum()
    obv_mean = obv.rolling(20).mean()
    obv_std = obv.rolling(20).std(ddof=1)
    data['OBV_Z'] = (obv - obv_mean) / (obv_std + 1e-9)

    # 4. Money Flow Index (MFI 14 normalized to)
    rmf = typical_p * volume
    pos_flow = pd.Series(np.where(typical_p > typical_p.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    neg_flow = pd.Series(np.where(typical_p < typical_p.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    mfi_raw = 100.0 - (100.0 / (1.0 + (pos_flow / (neg_flow + 1e-9))))
    data['MFI_Norm'] = (mfi_raw - 50.0) / 50.0

    # ------------------ B. MOMENTUM METRICS ------------------
    # 5. Normalized RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    rsi_raw = 100.0 - (100.0 / (1.0 + rs))
    data['RSI_Norm'] = (rsi_raw - 50.0) / 50.0

    # 6. Stochastic Oscillator %K & %D (14, 3)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100.0 * ((close - low_14) / (high_14 - low_14 + 1e-9))
    data['Stoch_Norm'] = (stoch_k - 50.0) / 50.0

    # 7. True Strength Index (TSI 25, 13 normalized to)
    diff = close.diff()
    smooth1 = diff.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    abs_smooth1 = diff.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    data['TSI_Norm'] = smooth1 / (abs_smooth1 + 1e-9)

    # 8. Short-Term Price Return (ROC 4)
    data['ROC_4'] = close.pct_change(4)

    # ------------------ C. TREND METRICS ------------------
    # 9. EMA Ribbon Spread (8 vs 55 EMA)
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    data['Ribbon_Spread'] = (ema8 - ema55) / (ema55 + 1e-9)
    data['Fast_Spread'] = (ema8 - ema21) / (ema21 + 1e-9)

    # 10. Directional Movement Index (ADX 14 & DI Differential)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_s = tr.ewm(alpha=1/14, min_periods=14).mean()

    up_m = high - high.shift(1)
    down_m = low.shift(1) - low
    p_dm = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
    p_di = 100.0 * pd.Series(p_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    m_di = 100.0 * pd.Series(m_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di + 1e-9)
    data['ADX_Norm'] = dx.ewm(alpha=1/14, min_periods=14).mean() / 100.0
    data['DMI_Diff'] = (p_di - m_di) / 100.0

    # 11. Supertrend Distance (%)
    data['ATR'] = tr_s
    data['NATR'] = (data['ATR'] / close) * 100.0
    st_basic_ub = (high + low) / 2.0 + (3.0 * tr_s)
    st_basic_lb = (high + low) / 2.0 - (3.0 * tr_s)
    data['Supertrend_Dist'] = (close - st_basic_lb) / close

    # ------------------ D. VOLATILITY METRICS ------------------
    # 12. Bollinger Bands (%B normalized)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=1)
    bb_up = bb_mid + 2.0 * bb_std
    bb_low = bb_mid - 2.0 * bb_std
    data['BB_Norm'] = ((close - bb_low) / (bb_up - bb_low + 1e-9)) - 0.5

    # 13. Historical Volatility (Annualized log returns)
    log_ret = np.log(close / close.shift(1))
    data['Hist_Vol'] = log_ret.rolling(20).std() * np.sqrt(252 * 25)

    return data


# =============================================================================
# 2. TRIPLE-BARRIER LABELING
# =============================================================================
def generate_universal_labels(df: pd.DataFrame) -> pd.Series:
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - HORIZON_BARS):
        entry_p = close[i]
        target_p = entry_p * (1.0 + TARGET_PROFIT_PCT)
        stop_p = entry_p - (atr[i] * ATR_MULTIPLIER)

        for h in range(1, HORIZON_BARS + 1):
            if high[i + h] >= target_p:
                labels[i] = 1
                break
            elif low[i + h] <= stop_p:
                labels[i] = 0
                break
        else:
            labels[i] = 1 if close[i + HORIZON_BARS] > entry_p else 0

    return pd.Series(labels, index=df.index)


# =============================================================================
# 3. UNIVERSAL ENSEMBLE CLASSIFIER
# =============================================================================
def build_universal_ensemble() -> VotingClassifier:
    clf_gbm = HistGradientBoostingClassifier(max_iter=60, max_depth=4, learning_rate=0.05, random_state=42)
    clf_knn = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan')
    clf_lr  = LogisticRegression(C=0.1, max_iter=400, random_state=42)
    return VotingClassifier(estimators=[('gbm', clf_gbm), ('knn', clf_knn), ('lr', clf_lr)], voting='soft')


def generate_pine_script_v6(output_path: str = "strategy_v6.pine") -> str:
    pine_code = """//@version=6
strategy("Universal Quantitative ML Engine [v6]", 
         shorttitle="UNIV_ML_v6", 
         overlay=true, 
         initial_capital=1000000, 
         default_qty_type=strategy.percent_of_equity, 
         default_qty_value=10, 
         commission_type=strategy.commission.percent, 
         commission_value=0.03, 
         slippage=2,
         pyramiding=0)

// 1. CONFIGURATION
var string G_MODE       = "Universal Trading Mode"
i_tradeMode             = input.string("Intraday (15m)", "Mode", options=["Scalping (1m-5m)", "Intraday (15m)", "Swing (Daily)", "BTST (EOD)"], group=G_MODE)
i_enableShorts          = input.bool(true, "Enable Short Trades", group=G_MODE)

var string G_RISK       = "Universal Risk Controls"
i_targetProfitPct       = input.float(2.5, "Take Profit Target (%)", minval=0.5, step=0.25, group=G_RISK)
i_atrSlMultiplier       = input.float(1.50, "ATR Stop Loss Multiplier", minval=0.5, step=0.25, group=G_RISK)
i_atrLength             = input.int(14, "ATR Length", minval=1, group=G_RISK)
i_enableBreakeven       = input.bool(true, "Enable Breakeven Ratchet", group=G_RISK)
i_breakevenTriggerPct   = input.float(1.2, "Breakeven Gain (%)", minval=0.5, step=0.25, group=G_RISK)

var string G_ML         = "Universal Lorentzian Parameters"
i_kNeighbors            = input.int(8, "k-Nearest Neighbors (k)", minval=1, maxval=50, group=G_ML)
i_trainingWindow        = input.int(250, "Training Horizon (Bars)", minval=50, maxval=2000, group=G_ML)
i_confidenceThresh      = input.float(52.0, "Model Confidence (%)", minval=50.0, maxval=95.0, step=1.0, group=G_ML)

// 2. FEATURE EXTRACTION PIPELINE (VOLUME + MOMENTUM + TREND + VOLATILITY)
// Volume: RVOL & MFI
float volSma20 = ta.sma(volume, 20)
float rvol = volume / (volSma20 + 1e-9)
float f_rvol = math.min(rvol / 3.0, 1.0) // Normalized 0 to 1

float rawMfi = ta.mfi(hlc3, 14)
float f_mfi = (rawMfi - 50.0) / 50.0 // Normalized -1 to 1

// Momentum: RSI & TSI
float rawRsi = ta.rsi(close, 14)
float f_rsi = (rawRsi - 50.0) / 50.0

float rawTsi = ta.tsi(close, 25, 13)
float f_tsi = rawTsi / 100.0

// Trend: ADX & EMA Ribbon
[diPlus, diMinus, adxVal] = ta.dmi(14, 14)
float f_dmi = (diPlus - diMinus) / 100.0

float ema8 = ta.ema(close, 8)
float ema55 = ta.ema(close, 55)
float f_ribbon = math.max(math.min((ema8 - ema55) / (ema55 + 1e-9) * 10.0, 1.0), -1.0)

// Volatility: Bollinger %B
[bbMid, bbUp, bbLow] = ta.bb(close, 20, 2)
float f_bb = ((close - bbLow) / (bbUp - bbLow + 1e-9)) - 0.5

// 3. LORENTZIAN DISTANCE METRIC
f_lorentzian_dist(float x1, float x2, float x3, float x4, float x5, float x6, float x7,
                  float y1, float y2, float y3, float y4, float y5, float y6, float y7) =>
    math.log(1.0 + math.abs(x1 - y1)) +
    math.log(1.0 + math.abs(x2 - y2)) +
    math.log(1.0 + math.abs(x3 - y3)) +
    math.log(1.0 + math.abs(x4 - y4)) +
    math.log(1.0 + math.abs(x5 - y5)) +
    math.log(1.0 + math.abs(x6 - y6)) +
    math.log(1.0 + math.abs(x7 - y7))

var array<float> arr_f1     = array.new_float(0)
var array<float> arr_f2     = array.new_float(0)
var array<float> arr_f3     = array.new_float(0)
var array<float> arr_f4     = array.new_float(0)
var array<float> arr_f5     = array.new_float(0)
var array<float> arr_f6     = array.new_float(0)
var array<float> arr_f7     = array.new_float(0)
var array<int>   arr_labels = array.new_int(0)

var int lb = 4
int historicalLabel = close > close[lb] ? 1 : -1

if bar_index > 10
    array.push(arr_f1, f_rvol[lb])
    array.push(arr_f2, f_mfi[lb])
    array.push(arr_f3, f_rsi[lb])
    array.push(arr_f4, f_tsi[lb])
    array.push(arr_f5, f_dmi[lb])
    array.push(arr_f6, f_ribbon[lb])
    array.push(arr_f7, f_bb[lb])
    array.push(arr_labels, historicalLabel)
    if array.size(arr_labels) > i_trainingWindow
        array.shift(arr_f1)
        array.shift(arr_f2)
        array.shift(arr_f3)
        array.shift(arr_f4)
        array.shift(arr_f5)
        array.shift(arr_f6)
        array.shift(arr_f7)
        array.shift(arr_labels)

int countBull = 0
int countBear = 0
int totalSamples = array.size(arr_labels)

if totalSamples >= math.max(i_kNeighbors, 20)
    array<float> distances = array.new_float(totalSamples)
    array<int>   indices   = array.new_int(totalSamples)
    
    for i = 0 to totalSamples - 1
        float dist = f_lorentzian_dist(f_rvol, f_mfi, f_rsi, f_tsi, f_dmi, f_ribbon, f_bb,
                                      array.get(arr_f1, i), 
                                      array.get(arr_f2, i), 
                                      array.get(arr_f3, i), 
                                      array.get(arr_f4, i), 
                                      array.get(arr_f5, i),
                                      array.get(arr_f6, i),
                                      array.get(arr_f7, i))
        array.set(distances, i, dist)
        array.set(indices, i, i)

    int sortLimit = math.min(i_kNeighbors, totalSamples - 1)
    if sortLimit > 0
        for i = 0 to sortLimit - 1
            int minIdx = i
            for j = i + 1 to totalSamples - 1
                if array.get(distances, j) < array.get(distances, minIdx)
                    minIdx := j
            if minIdx != i
                float tempD = array.get(distances, i)
                array.set(distances, i, array.get(distances, minIdx))
                array.set(distances, minIdx, tempD)
                
                int tempI = array.get(indices, i)
                array.set(indices, i, array.get(indices, minIdx))
                array.set(indices, minIdx, tempI)

    for i = 0 to i_kNeighbors - 1
        int neighborIdx = array.get(indices, i)
        int lbl = array.get(arr_labels, neighborIdx)
        if lbl == 1
            countBull += 1
        else
            countBear += 1

float modelConfidence = 0.0
int mlDirection = 0

if (countBull + countBear) > 0
    if countBull > countBear
        modelConfidence := (float(countBull) / float(i_kNeighbors)) * 100.0
        if modelConfidence >= i_confidenceThresh
            mlDirection := 1
    else
        modelConfidence := (float(countBear) / float(i_kNeighbors)) * 100.0
        if modelConfidence >= i_confidenceThresh
            mlDirection := -1

bool modeConditionBuy  = (mlDirection == 1)
bool modeConditionSell = (mlDirection == -1) and i_enableShorts

float atrVal = ta.atr(i_atrLength)

var float entryPriceLocal = 0.0
var float targetPrice     = 0.0
var float stopLossPrice   = 0.0
var bool  isBreakeven     = false

var line lineTP    = na
var line lineSL    = na
var line lineEntry = na

bool inLongPosition  = strategy.position_size > 0
bool inShortPosition = strategy.position_size < 0

if modeConditionBuy and not inLongPosition and barstate.isconfirmed
    entryPriceLocal := close
    targetPrice     := entryPriceLocal * (1.0 + (i_targetProfitPct / 100.0))
    stopLossPrice   := entryPriceLocal - (atrVal * i_atrSlMultiplier)
    isBreakeven     := false
    strategy.entry("BUY", strategy.long)
    
    line.delete(lineTP)
    line.delete(lineSL)
    line.delete(lineEntry)
    lineEntry := line.new(bar_index, entryPriceLocal, bar_index + 10, entryPriceLocal, color=color.blue, width=2)
    lineTP    := line.new(bar_index, targetPrice,     bar_index + 10, targetPrice,     color=color.green, width=2, style=line.style_dashed)
    lineSL    := line.new(bar_index, stopLossPrice,   bar_index + 10, stopLossPrice,   color=color.red, width=2, style=line.style_dashed)

if modeConditionSell and not inShortPosition and barstate.isconfirmed
    entryPriceLocal := close
    targetPrice     := entryPriceLocal * (1.0 - (i_targetProfitPct / 100.0))
    stopLossPrice   := entryPriceLocal + (atrVal * i_atrSlMultiplier)
    isBreakeven     := false
    strategy.entry("SELL", strategy.short)
    
    line.delete(lineTP)
    line.delete(lineSL)
    line.delete(lineEntry)
    lineEntry := line.new(bar_index, entryPriceLocal, bar_index + 10, entryPriceLocal, color=color.blue, width=2)
    lineTP    := line.new(bar_index, targetPrice,     bar_index + 10, targetPrice,     color=color.green, width=2, style=line.style_dashed)
    lineSL    := line.new(bar_index, stopLossPrice,   bar_index + 10, stopLossPrice,   color=color.red, width=2, style=line.style_dashed)

if inLongPosition
    if i_enableBreakeven and not isBreakeven and (high >= entryPriceLocal * (1.0 + (i_breakevenTriggerPct / 100.0)))
        stopLossPrice := entryPriceLocal
        isBreakeven   := true
        line.set_y1(lineSL, stopLossPrice)
        line.set_y2(lineSL, stopLossPrice)
        line.set_color(lineSL, color.orange)
    strategy.exit("Exit_BUY", "BUY", limit=targetPrice, stop=stopLossPrice)

if inShortPosition
    if i_enableBreakeven and not isBreakeven and (low <= entryPriceLocal * (1.0 - (i_breakevenTriggerPct / 100.0)))
        stopLossPrice := entryPriceLocal
        isBreakeven   := true
        line.set_y1(lineSL, stopLossPrice)
        line.set_y2(lineSL, stopLossPrice)
        line.set_color(lineSL, color.orange)
    strategy.exit("Exit_SELL", "SELL", limit=targetPrice, stop=stopLossPrice)

if (inLongPosition or inShortPosition)
    line.set_x2(lineEntry, bar_index + 3)
    line.set_x2(lineTP, bar_index + 3)
    line.set_x2(lineSL, bar_index + 3)

var table hud = table.new(position.top_right, 2, 7, bgcolor=color.new(color.black, 15), border_width=1, border_color=color.gray)

if barstate.islast
    float rrRatio = 0.0
    if math.abs(entryPriceLocal - stopLossPrice) > 0.0001
        rrRatio := math.abs(targetPrice - entryPriceLocal) / math.abs(entryPriceLocal - stopLossPrice)
        
    table.cell(hud, 0, 0, "Metric", text_color=color.white, text_size=size.small, bgcolor=color.navy)
    table.cell(hud, 1, 0, "Value",  text_color=color.white, text_size=size.small, bgcolor=color.navy)
    
    table.cell(hud, 0, 1, "Universal ML", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 1, str.tostring(modelConfidence, "#.#") + "%", 
               text_color=modelConfidence >= i_confidenceThresh ? color.lime : color.gray, text_size=size.small)
    
    table.cell(hud, 0, 2, "Market Bias", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 2, mlDirection == 1 ? "BULLISH" : mlDirection == -1 ? "BEARISH" : "NEUTRAL", 
               text_color=mlDirection == 1 ? color.green : mlDirection == -1 ? color.red : color.gray, text_size=size.small)
    
    table.cell(hud, 0, 3, "Entry Price", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 3, inLongPosition or inShortPosition ? str.tostring(entryPriceLocal, "#.##") : "-", text_color=color.white, text_size=size.small)
    
    table.cell(hud, 0, 4, "Target", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 4, inLongPosition or inShortPosition ? str.tostring(targetPrice, "#.##") : "-", text_color=color.green, text_size=size.small)
    
    table.cell(hud, 0, 5, "Stop Loss", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 5, inLongPosition or inShortPosition ? str.tostring(stopLossPrice, "#.##") : "-", 
               text_color=isBreakeven ? color.orange : color.red, text_size=size.small)
    
    table.cell(hud, 0, 6, "Risk : Reward", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 6, inLongPosition or inShortPosition ? ("1 : " + str.tostring(rrRatio, "#.##")) : "-", text_color=color.yellow, text_size=size.small)

plotshape(modeConditionBuy and not inLongPosition, title="Buy Signal", style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small)
plotshape(modeConditionSell and not inShortPosition, title="Sell Signal", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small)
"""
    with open(output_path, "w") as f:
        f.write(pine_code)
    return pine_code


# =============================================================================
# 4. UNIVERSAL ENGINE: POOL 60 STOCKS -> TRAIN ENSEMBLE -> SCORE ALL
# =============================================================================
def download_single_ticker(ticker: str) -> Optional[pd.DataFrame]:
    clean_t = ticker.strip().upper()
    yf_symbol = f"{clean_t}.NS"
    try:
        df = yf.download(yf_symbol, period="60d", interval="15m", progress=False)
        if df is None or len(df) < MIN_BARS_PER_STOCK:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df_feat = build_full_feature_store(df).dropna()
        if len(df_feat) < MIN_BARS_PER_STOCK:
            return None
        df_feat['TICKER'] = clean_t
        return df_feat
    except Exception:
        return None


def main():
    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file):
        sample_tickers = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", 
            "SBIN", "LICI", "ITC", "HINDUNILVR", "LT", "BAJFINANCE", 
            "TATAMOTORS", "SUNPHARMA", "MARUTI", "JINDALSAW", "AXISBANK", 
            "KOTAKBANK", "TITAN", "ULTRACEMCO"
        ]
        with open(ticker_file, "w") as f:
            f.write("\n".join(sample_tickers))

    with open(ticker_file, "r") as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    pool_tickers = all_tickers[:MAX_POOL_STOCKS]

    print("=" * 115)
    print(f"UNIVERSAL ML ENGINE: INGESTING {len(pool_tickers)} EQUITIES INTO COMPREHENSIVE FEATURE STORE")
    print("=" * 115)

    stock_dataframes: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(download_single_ticker, t): t for t in pool_tickers}
        for future in as_completed(future_map):
            t = future_map[future]
            res = future.result()
            if res is not None:
                stock_dataframes[t] = res
                print(f"[+] Ingested {t:<12}: {len(res)} 15m bars with Volume & VWAP")

    # Unconditionally generate Pine Script v6 strategy
    generate_pine_script_v6("strategy_v6.pine")

    if len(stock_dataframes) < 3:
        print("[-] Insufficient data downloaded.")
        pd.DataFrame().to_csv("final_ranked_results.csv", index=False)
        return

    # Master Institutional Feature Vector
    feature_cols = [
        'RVOL', 'VWAP_Dist', 'OBV_Z', 'MFI_Norm', 
        'RSI_Norm', 'Stoch_Norm', 'TSI_Norm', 'ROC_4', 
        'Ribbon_Spread', 'Fast_Spread', 'ADX_Norm', 'DMI_Diff', 
        'Supertrend_Dist', 'BB_Norm', 'NATR', 'Hist_Vol'
    ]

    training_x_list = []
    training_y_list = []
    latest_rows = {}

    for t, df_stock in stock_dataframes.items():
        labels = generate_universal_labels(df_stock)
        valid_indices = df_stock.index[:-HORIZON_BARS]
        
        training_x_list.append(df_stock.loc[valid_indices, feature_cols])
        training_y_list.append(labels.loc[valid_indices])
        latest_rows[t] = df_stock.iloc[[-1]]

    X_universal = pd.concat(training_x_list, ignore_index=True)
    y_universal = pd.concat(training_y_list, ignore_index=True)

    print("-" * 115)
    print(f"[*] TOTAL POOLED TRAINING MATRIX: {len(X_universal):,} BARS × {len(feature_cols)} FEATURES")
    print(f"[*] Bullish Targets (+2.5%): {sum(y_universal == 1):,} | Other/Stops: {sum(y_universal == 0):,}")
    print("[-] Training Universal Multi-Model Ensemble...")

    # Train Single Universal Scaler & Ensemble Model
    scaler = StandardScaler()
    X_universal_scaled = scaler.fit_transform(X_universal)

    universal_model = build_universal_ensemble()
    universal_model.fit(X_universal_scaled, y_universal)
    print("[+] Universal Ensemble Trained Successfully!")
    print("-" * 115)

    # Score Every Stock with the Universal Model
    opportunities: List[UniversalOpportunity] = []
    row_first = 0
    col_win = 1

    for t, row_df in latest_rows.items():
        feat_vals = row_df[feature_cols]
        feat_scaled = scaler.transform(feat_vals)
        
        probs = universal_model.predict_proba(feat_scaled)
        live_prob = float(probs[row_first, col_win])

        last_close = float(row_df['Close'].iloc[0])
        last_atr   = float(row_df['ATR'].iloc[0])
        last_rvol  = float(row_df['RVOL'].iloc[0])
        last_vwap  = float(row_df['VWAP_Dist'].iloc[0]) * 100.0

        target_p   = last_close * (1.0 + TARGET_PROFIT_PCT)
        stop_p     = last_close - (last_atr * ATR_MULTIPLIER)
        risk_r     = abs(target_p - last_close) / (abs(last_close - stop_p) + 1e-9)

        opp = UniversalOpportunity(
            ticker=t,
            direction="BUY" if live_prob >= 0.50 else "WATCH",
            entry_price=round(last_close, 2),
            target_price=round(target_p, 2),
            stop_loss=round(stop_p, 2),
            risk_reward=round(risk_r, 2),
            universal_prob=round(live_prob * 100.0, 1),
            rvol=round(last_rvol, 2),
            vwap_dist=round(last_vwap, 2),
            atr=round(last_atr, 2)
        )
        opportunities.append(opp)

    opportunities.sort(key=lambda x: x.universal_prob, reverse=True)

    df_out = pd.DataFrame([asdict(o) for o in opportunities])
    df_out.to_csv("final_ranked_results.csv", index=False)
    print("[+] Saved Universal Ranking to 'final_ranked_results.csv'")

    # Print Full Dashboard
    print("\n" + "=" * 120)
    print(f"{'MASTER CROSS-SECTIONAL UNIVERSAL ML RANKING (NSE 15m)':^120}")
    print("=" * 120)
    header = f"{'Rank':<6}{'Ticker':<14}{'Signal':<8}{'Entry (₹)':<12}{'Target (+2.5%)':<16}{'Stop Loss':<12}{'R:R':<8}{'ML Prob':<10}{'RVOL':<8}{'VWAP Dist%':<12}{'ATR':<8}"
    print(header)
    print("-" * 120)
    for rank, o in enumerate(opportunities, start=1):
        print(f"{rank:<6}{o.ticker:<14}{o.direction:<8}{o.entry_price:<12.2f}{o.target_price:<16.2f}"
              f"{o.stop_loss:<12.2f}{o.risk_reward:<8.2f}{o.universal_prob:<9.1f}%{o.rvol:<8.2f}{o.vwap_dist:<12.2f}{o.atr:<8.2f}")

    print("=" * 120)
    tv_symbols = ", ".join([f"NSE:{o.ticker}" for o in opportunities[:20]])
    print("TOP 20 TRADINGVIEW WATCHLIST IMPORT STRING:")
    print(tv_symbols)
    print("=" * 120)


if __name__ == "__main__":
    main()
