#!/usr/bin/env python3
"""
INSTITUTIONAL INTRADAY ML ENGINE & PINE SCRIPT v6 GENERATOR (SHARDED)
====================================================================
- Calibrated for 15-Minute Intraday Indian Equities
- Always generates 'strategy_v6.pine' and 'final_ranked_results.csv'
- Multi-threaded per shard with automated aggregator merge
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Calibrated Strategy Parameters
TARGET_PROFIT_PCT = 0.025       # 2.5% Intraday Target (Realistic for 15m bars)
ATR_MULTIPLIER    = 1.50        # Volatility Stop Loss Multiplier
HORIZON_BARS      = 20          # Max forward bars (~5 hours of trading)
MIN_TRAIN_BARS    = 200         # Minimum historical bars required
TEST_FOLD_BARS    = 50          # Out-of-sample test window per fold
MIN_CONFIDENCE    = 0.52        # 52.0% Statistical Edge Threshold
RISK_FREE_RATE    = 0.065       # RBI 91-day T-Bill rate baseline (~6.5%)


@dataclass
class ScreenerOpportunity:
    ticker: str
    direction: str
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float
    ensemble_prob: float
    historical_win_rate: float
    max_drawdown: float
    fitness_score: float


def build_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # 2. Stochastic %K & %D (14, 3)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    data['Stoch_K'] = 100.0 * ((close - low_14) / (high_14 - low_14 + 1e-9))
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()

    # 3. ATR & NATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR'] = tr.ewm(alpha=1/14, min_periods=14).mean()
    data['NATR'] = (data['ATR'] / close) * 100.0

    # 4. Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=1)
    bb_up = bb_mid + 2.0 * bb_std
    bb_low = bb_mid - 2.0 * bb_std
    data['BB_PctB'] = (close - bb_low) / (bb_up - bb_low + 1e-9)
    data['BB_Width'] = (bb_up - bb_low) / (bb_mid + 1e-9)

    # 5. Money Flow Index (14)
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    pos_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    neg_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    data['MFI'] = 100.0 - (100.0 / (1.0 + (pos_flow / (neg_flow + 1e-9))))

    # 6. Trend: EMA Ribbon Dispersion
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    data['EMA_Ribbon_Spread'] = (ema8 - ema55) / (ema55 + 1e-9)
    data['EMA_Short_Spread'] = (ema8 - ema21) / (ema21 + 1e-9)

    # 7. ADX & Directional Differentials (14)
    up_m = high - high.shift(1)
    down_m = low.shift(1) - low
    p_dm = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
    tr_s = tr.ewm(alpha=1/14, min_periods=14).mean()
    p_di = 100.0 * pd.Series(p_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    m_di = 100.0 * pd.Series(m_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di + 1e-9)
    data['ADX'] = dx.ewm(alpha=1/14, min_periods=14).mean()
    data['DI_Diff'] = p_di - m_di

    # 8. Momentum Returns
    data['ROC_4'] = close.pct_change(4)
    data['ROC_12'] = close.pct_change(12)

    return data


def generate_triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - HORIZON_BARS):
        entry_price = close[i]
        target_price = entry_price * (1.0 + TARGET_PROFIT_PCT)
        stop_price = entry_price - (atr[i] * ATR_MULTIPLIER)

        for h in range(1, HORIZON_BARS + 1):
            if high[i + h] >= target_price:
                labels[i] = 1
                break
            elif low[i + h] <= stop_price:
                labels[i] = 0
                break
        else:
            labels[i] = 1 if close[i + HORIZON_BARS] > entry_price else 0

    return pd.Series(labels, index=df.index)


def build_ml_ensemble() -> VotingClassifier:
    clf_gbm = HistGradientBoostingClassifier(max_iter=50, max_depth=4, learning_rate=0.05, random_state=42)
    clf_knn = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan')
    clf_lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    return VotingClassifier(estimators=[('gbm', clf_gbm), ('knn', clf_knn), ('lr', clf_lr)], voting='soft')


def run_walk_forward_validation(X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame) -> Tuple[float, float, float]:
    n_samples = len(X)
    if n_samples < (MIN_TRAIN_BARS + TEST_FOLD_BARS):
        return 0.0, 100.0, 0.0

    oos_returns = []
    oos_trades = 0
    oos_wins = 0

    close_arr = df_raw['Close'].values
    high_arr = df_raw['High'].values
    low_arr = df_raw['Low'].values
    atr_arr = df_raw['ATR'].values

    for start_test in range(MIN_TRAIN_BARS, n_samples - TEST_FOLD_BARS, TEST_FOLD_BARS):
        end_test = start_test + TEST_FOLD_BARS
        purge_idx = start_test - HORIZON_BARS
        X_train, y_train = X.iloc[:purge_idx], y.iloc[:purge_idx]
        X_test = X.iloc[start_test:end_test]

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = build_ml_ensemble()
        model.fit(X_train_scaled, y_train)

        test_probs = model.predict_proba(X_test_scaled)

        for i_local, prob in enumerate(test_probs):
            if prob >= MIN_CONFIDENCE:
                bar_idx = start_test + i_local
                if bar_idx >= n_samples - HORIZON_BARS:
                    continue

                entry_p = close_arr[bar_idx]
                target_p = entry_p * (1.0 + TARGET_PROFIT_PCT)
                stop_p = entry_p - (atr_arr[bar_idx] * ATR_MULTIPLIER)

                oos_trades += 1
                trade_ret = 0.0

                for step in range(1, HORIZON_BARS + 1):
                    if high_arr[bar_idx + step] >= target_p:
                        trade_ret = TARGET_PROFIT_PCT
                        oos_wins += 1
                        break
                    elif low_arr[bar_idx + step] <= stop_p:
                        trade_ret = (stop_p - entry_p) / entry_p
                        break
                else:
                    trade_ret = (close_arr[bar_idx + HORIZON_BARS] - entry_p) / entry_p
                    if trade_ret > 0:
                        oos_wins += 1

                oos_returns.append(trade_ret)

    if oos_trades < 3:
        return 50.0, 5.0, 0.01

    win_rate = (oos_wins / oos_trades) * 100.0

    eq_curve = [1.0]
    for r in oos_returns:
        eq_curve.append(eq_curve[-1] * (1.0 + r))
    
    eq_series = pd.Series(eq_curve)
    drawdowns = (eq_series - eq_series.cummax()) / eq_series.cummax()
    max_dd = abs(drawdowns.min()) * 100.0
    if max_dd < 0.5:
        max_dd = 0.5

    total_ret = eq_series.iloc[-1] - 1.0
    ann_ret = total_ret * (1500.0 / max(len(oos_returns), 1))

    neg_rets = [r for r in oos_returns if r < 0]
    downside_std = np.std(neg_rets) if len(neg_rets) > 1 else 0.01
    sortino = (np.mean(oos_returns) * 1500.0 - RISK_FREE_RATE) / (downside_std * np.sqrt(1500) + 1e-9)
    sortino = max(sortino, 0.0)

    penalty = 0.2 if oos_trades < 6 else 0.0
    fitness = ((max(ann_ret, 0.0) * 100.0) / (max_dd ** 2)) * sortino * (1.0 - penalty)

    return win_rate, max_dd, fitness


def evaluate_ticker_ml(raw_ticker: str) -> Optional[ScreenerOpportunity]:
    clean_ticker = raw_ticker.strip().upper()
    if not clean_ticker:
        return None

    yf_symbol = f"{clean_ticker}.NS"
    try:
        df = yf.download(yf_symbol, period="60d", interval="15m", progress=False)
        if df is None or len(df) < (MIN_TRAIN_BARS + 30):
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df_feat = build_feature_store(df).dropna()
        if len(df_feat) < MIN_TRAIN_BARS:
            return None

        labels = generate_triple_barrier_labels(df_feat)

        feature_cols = [
            'RSI', 'Stoch_K', 'Stoch_D', 'NATR', 'BB_PctB', 'BB_Width', 
            'MFI', 'EMA_Ribbon_Spread', 'EMA_Short_Spread', 'ADX', 'DI_Diff', 
            'ROC_4', 'ROC_12'
        ]

        valid_indices = df_feat.index[:-HORIZON_BARS]
        X_train_full = df_feat.loc[valid_indices, feature_cols]
        y_train_full = labels.loc[valid_indices]

        win_rate, max_dd, fitness = run_walk_forward_validation(X_train_full, y_train_full, df_feat)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        
        ensemble = build_ml_ensemble()
        ensemble.fit(X_train_scaled, y_train_full)

        latest_features = df_feat.iloc[[-1]][feature_cols]
        latest_scaled = scaler.transform(latest_features)
        live_prob = float(ensemble.predict_proba(latest_scaled))

        # Accept trades with positive statistical edge
        if live_prob < MIN_CONFIDENCE:
            return None

        last_close = float(df_feat.iloc[-1]['Close'])
        last_atr = float(df_feat.iloc[-1]['ATR'])
        target_p = last_close * (1.0 + TARGET_PROFIT_PCT)
        stop_p = last_close - (last_atr * ATR_MULTIPLIER)
        rr_ratio = abs(target_p - last_close) / (abs(last_close - stop_p) + 1e-9)

        return ScreenerOpportunity(
            ticker=clean_ticker,
            direction="BUY",
            entry_price=round(last_close, 2),
            target_price=round(target_p, 2),
            stop_loss=round(stop_p, 2),
            risk_reward=round(rr_ratio, 2),
            ensemble_prob=round(live_prob * 100.0, 1),
            historical_win_rate=round(win_rate, 1),
            max_drawdown=round(max_dd, 1),
            fitness_score=round(fitness, 3)
        )
    except Exception:
        return None


def generate_pine_script_v6(output_path: str = "strategy_v6.pine") -> str:
    pine_code = """//@version=6
strategy("Institutional Quantitative Engine [v6]", 
         shorttitle="QUANT_V6", 
         overlay=true, 
         initial_capital=1000000, 
         default_qty_type=strategy.percent_of_equity, 
         default_qty_value=10, 
         commission_type=strategy.commission.percent, 
         commission_value=0.03, 
         slippage=2,
         pyramiding=0)

var string G_MODE     = "Operational Mode & Time Horizons"
i_tradeMode           = input.string("Intraday (15m)", "Trading Mode", options=["Scalping (1m-5m)", "Intraday (15m)", "Swing (Daily)", "BTST (EOD)"], group=G_MODE)
i_enableShorts        = input.bool(false, "Enable Short Trades (Intraday Only)", group=G_MODE)

var string G_RISK     = "Risk & Position Controls"
i_targetProfitPct     = input.float(2.5, "Take Profit Target (%)", minval=0.5, step=0.25, group=G_RISK)
i_atrSlMultiplier     = input.float(1.50, "ATR Stop Loss Multiplier", minval=0.5, step=0.25, group=G_RISK)
i_atrLength           = input.int(14, "ATR Length", minval=1, group=G_RISK)
i_enableBreakeven     = input.bool(true, "Enable Breakeven Ratchet", group=G_RISK)
i_breakevenTriggerPct = input.float(1.5, "Breakeven Activation Gain (%)", minval=0.5, step=0.25, group=G_RISK)

var string G_ML       = "Lorentzian Classification Parameters"
i_kNeighbors          = input.int(8, "k-Nearest Neighbors (k)", minval=1, maxval=50, group=G_ML)
i_trainingWindow      = input.int(250, "Training Historical Horizon (Bars)", minval=50, maxval=2000, group=G_ML)
i_confidenceThresh    = input.float(52.0, "Minimum Model Confidence (%)", minval=50.0, maxval=95.0, step=1.0, group=G_ML)

f_calc_rsi(int len) =>
    float rawRsi = ta.rsi(close, len)
    (rawRsi - 50.0) / 50.0

f_calc_cci(int len) =>
    float rawCci = ta.cci(close, len)
    math.max(math.min(rawCci / 200.0, 1.0), -1.0)

f_calc_tsi(int longLen, int shortLen) =>
    float rawTsi = ta.tsi(close, longLen, shortLen)
    rawTsi / 100.0

f_calc_mfi(int len) =>
    float rawMfi = ta.mfi(hlc3, len)
    (rawMfi - 50.0) / 50.0

f_calc_adx_diff(int len) =>
    [diPlus, diMinus, adxVal] = ta.dmi(len, len)
    float diff = (diPlus - diMinus) / 100.0
    math.max(math.min(diff, 1.0), -1.0)

f1 = f_calc_rsi(14)
f2 = f_calc_cci(20)
f3 = f_calc_tsi(25, 13)
f4 = f_calc_mfi(14)
f5 = f_calc_adx_diff(14)

f_lorentzian_dist(float x1, float x2, float x3, float x4, float x5, 
                  float y1, float y2, float y3, float y4, float y5) =>
    float d1 = math.log(1.0 + math.abs(x1 - y1))
    float d2 = math.log(1.0 + math.abs(x2 - y2))
    float d3 = math.log(1.0 + math.abs(x3 - y3))
    float d4 = math.log(1.0 + math.abs(x4 - y4))
    float d5 = math.log(1.0 + math.abs(x5 - y5))
    d1 + d2 + d3 + d4 + d5

var array<float> arr_f1     = array.new_float(0)
var array<float> arr_f2     = array.new_float(0)
var array<float> arr_f3     = array.new_float(0)
var array<float> arr_f4     = array.new_float(0)
var array<float> arr_f5     = array.new_float(0)
var array<int>   arr_labels = array.new_int(0)

int historicalLabel = close[0] > close ? 1 : -1

if bar_index > 4
    array.push(arr_f1, f1)
    array.push(arr_f2, f2)
    array.push(arr_f3, f3)
    array.push(arr_f4, f4)
    array.push(arr_f5, f5)
    array.push(arr_labels, historicalLabel)
    if array.size(arr_labels) > i_trainingWindow
        array.shift(arr_f1)
        array.shift(arr_f2)
        array.shift(arr_f3)
        array.shift(arr_f4)
        array.shift(arr_f5)
        array.shift(arr_labels)

int countBull = 0
int countBear = 0
int totalSamples = array.size(arr_labels)

if totalSamples >= math.max(i_kNeighbors, 20)
    array<float> distances = array.new_float(totalSamples)
    array<int>   indices   = array.new_int(totalSamples)
    
    for i = 0 to totalSamples - 1
        float dist = f_lorentzian_dist(f1, f2, f3, f4, f5, 
                                      array.get(arr_f1, i), 
                                      array.get(arr_f2, i), 
                                      array.get(arr_f3, i), 
                                      array.get(arr_f4, i), 
                                      array.get(arr_f5, i))
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

int istHour   = (hour + 5) + int((minute + 30) / 60)
int istMinute = (minute + 30) % 60
int timeMins  = istHour * 60 + istMinute

bool intradayEntryAllowed = timeMins >= (9 * 60 + 20) and timeMins <= (14 * 60 + 45)
bool intradayExitRequired = timeMins >= (15 * 60 + 15)
bool btstEntryAllowed     = timeMins >= (15 * 60 + 0)  and timeMins <= (15 * 60 + 25)
bool btstExitRequired    = timeMins >= (9 * 60 + 20)  and timeMins <= (9 * 60 + 35)

bool modeConditionBuy  = false
bool modeConditionSell = false

if i_tradeMode == "Scalping (1m-5m)" or i_tradeMode == "Intraday (15m)"
    modeConditionBuy  := intradayEntryAllowed and (mlDirection == 1)
    modeConditionSell := intradayEntryAllowed and (mlDirection == -1) and i_enableShorts
else if i_tradeMode == "BTST (EOD)"
    modeConditionBuy  := btstEntryAllowed and (mlDirection == 1)
else if i_tradeMode == "Swing (Daily)"
    modeConditionBuy  := (mlDirection == 1)
    modeConditionSell := (mlDirection == -1) and i_enableShorts

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
    
    alert("LONG: " + syminfo.ticker + " @ " + str.tostring(entryPriceLocal, "#.##") + 
          " | Target: " + str.tostring(targetPrice, "#.##") + 
          " | SL: " + str.tostring(stopLossPrice, "#.##"), 
          alert.freq_once_per_bar_close)

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
    
    alert("SHORT: " + syminfo.ticker + " @ " + str.tostring(entryPriceLocal, "#.##") + 
          " | Target: " + str.tostring(targetPrice, "#.##") + 
          " | SL: " + str.tostring(stopLossPrice, "#.##"), 
          alert.freq_once_per_bar_close)

if inLongPosition
    if i_enableBreakeven and not isBreakeven and (high >= entryPriceLocal * (1.0 + (i_breakevenTriggerPct / 100.0)))
        stopLossPrice := entryPriceLocal
        isBreakeven   := true
        line.set_y1(lineSL, stopLossPrice)
        line.set_y2(lineSL, stopLossPrice)
        line.set_color(lineSL, color.orange)
    strategy.exit("Long_Exit", "BUY", limit=targetPrice, stop=stopLossPrice)

if inShortPosition
    if i_enableBreakeven and not isBreakeven and (low <= entryPriceLocal * (1.0 - (i_breakevenTriggerPct / 100.0)))
        stopLossPrice := entryPriceLocal
        isBreakeven   := true
        line.set_y1(lineSL, stopLossPrice)
        line.set_y2(lineSL, stopLossPrice)
        line.set_color(lineSL, color.orange)
    strategy.exit("Short_Exit", "SELL", limit=targetPrice, stop=stopLossPrice)

if (i_tradeMode == "Scalping (1m-5m)" or i_tradeMode == "Intraday (15m)") and intradayExitRequired
    strategy.close_all(comment="Intraday Square-off")

if i_tradeMode == "BTST (EOD)" and btstExitRequired
    strategy.close_all(comment="BTST Morning Exit")

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
    
    table.cell(hud, 0, 1, "ML Confidence", text_color=color.silver, text_size=size.small)
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


def run_shard(shard_id: int, num_shards: int, ticker_file: str, output_csv: str) -> None:
    if not os.path.exists(ticker_file):
        print(f"[-] {ticker_file} not found.")
        sys.exit(1)

    with open(ticker_file, "r") as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()]

    shard_tickers = [t for i, t in enumerate(all_tickers) if (i % num_shards) == shard_id]

    print("=" * 90)
    print(f"SHARD {shard_id + 1}/{num_shards}: PROCESSING {len(shard_tickers)} TICKERS")
    print("=" * 90)

    opportunities: List[ScreenerOpportunity] = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(evaluate_ticker_ml, t): t for t in shard_tickers}
        for future in as_completed(future_map):
            res = future.result()
            if res is not None:
                opportunities.append(res)
                print(f"[+] Flagged: {res.ticker} | Prob: {res.ensemble_prob}% | WinRate: {res.historical_win_rate}%")

    empty_cols = ["ticker", "direction", "entry_price", "target_price", "stop_loss", 
                  "risk_reward", "ensemble_prob", "historical_win_rate", "max_drawdown", "fitness_score"]

    if opportunities:
        df_out = pd.DataFrame([asdict(o) for o in opportunities])
        df_out.to_csv(output_csv, index=False)
        print(f"[+] Shard {shard_id} saved {len(opportunities)} opportunities to {output_csv}")
    else:
        pd.DataFrame(columns=empty_cols).to_csv(output_csv, index=False)
        print(f"[-] Shard {shard_id}: No candidates met the threshold.")


def merge_and_display() -> None:
    print("\n" + "=" * 115)
    print(f"{'MERGING ALL SHARD ARTIFACTS & COMPUTING MASTER RANKING':^115}")
    print("=" * 115)

    # 1. ALWAYS GENERATE PINE SCRIPT v6 SO ARTIFACT UPLOAD NEVER FAILS
    generate_pine_script_v6("strategy_v6.pine")
    print("[+] Successfully generated 'strategy_v6.pine'.")

    empty_cols = ["ticker", "direction", "entry_price", "target_price", "stop_loss", 
                  "risk_reward", "ensemble_prob", "historical_win_rate", "max_drawdown", "fitness_score"]

    csv_files = glob.glob("results_shard_*.csv")
    if not csv_files:
        print("[-] No shard CSV files found.")
        pd.DataFrame(columns=empty_cols).to_csv("final_ranked_results.csv", index=False)
        return

    dfs = []
    for f in csv_files:
        try:
            if os.path.exists(f) and os.path.getsize(f) > 0:
                df_temp = pd.read_csv(f)
                if not df_temp.empty:
                    dfs.append(df_temp)
        except Exception:
            pass

    if not dfs:
        print("[-] All shard files were empty. Saving empty template.")
        pd.DataFrame(columns=empty_cols).to_csv("final_ranked_results.csv", index=False)
        return

    merged_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ticker"])
    merged_df.sort_values(by="fitness_score", ascending=False, inplace=True)
    merged_df.to_csv("final_ranked_results.csv", index=False)

    print("\n" + "=" * 115)
    print(f"{'TOP INTRADAY MACHINE LEARNING CANDIDATES (NSE)':^115}")
    print("=" * 115)
    header = f"{'Ticker':<12}{'Signal':<6}{'Entry (₹)':<12}{'Target':<14}{'Stop Loss':<12}{'R:R':<8}{'ML Prob (%)':<14}{'OOS Win %':<12}{'Max DD %':<10}{'Fitness':<10}"
    print(header)
    print("-" * 115)
    for _, row in merged_df.iterrows():
        print(f"{row['ticker']:<12}{row['direction']:<6}{row['entry_price']:<12.2f}{row['target_price']:<14.2f}"
              f"{row['stop_loss']:<12.2f}{row['risk_reward']:<8.2f}{row['ensemble_prob']:<14.1f}"
              f"{row['historical_win_rate']:<12.1f}{row['max_drawdown']:<10.1f}{row['fitness_score']:<10.3f}")

    print("=" * 115)
    tv_symbols = ", ".join([f"NSE:{t}" for t in merged_df['ticker'].tolist()])
    print("TRADINGVIEW WATCHLIST EXPORT:")
    print(tv_symbols)
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(description="Distributed Intraday ML Engine")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ticker-file", type=str, default="tickers.txt")
    parser.add_argument("--output-csv", type=str, default="shard_results.csv")
    parser.add_argument("--merge", action="store_true")

    args = parser.parse_args()

    if args.merge:
        merge_and_display()
    else:
        run_shard(args.shard_id, args.num_shards, args.ticker_file, args.output_csv)


if __name__ == "__main__":
    main()
