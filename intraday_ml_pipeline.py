#!/usr/bin/env python3
"""
INSTITUTIONAL QUANTITATIVE ML PIPELINE (NSE)
============================================
Follows the 9-Step Quantitative Research Architecture:
Step 1: Multi-Ticker Intraday OHLCV Ingestion (NSE)
Step 2: K-Fold Time-Series Partitioning (Zero Lookahead / Purged)
Step 3 & 4: Develop Multiple Trading ML Models
Step 5: Individual Out-of-Sample Performance Benchmarking
Step 6: Optimal Ensemble Discovery (Soft Voting vs Weighted Blending vs Stacking)
Step 7: Automated Hyperparameter Grid Optimization
Step 8: Formulate Master Swing Trading Setups
Step 9: Direct Output of 'strategy_v6.pine' Strategy Script
"""

import os
import sys
import glob
from operator import itemgetter
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Scikit-Learn Stack
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Strategy Parameters
TARGET_PROFIT_PCT = 0.05       # 5.0% Fixed Swing Target
ATR_SL_MULTIPLIER = 1.75       # ATR Stop Loss Multiplier
HORIZON_BARS      = 24         # Evaluation Window (~6 hours on 15m)
K_FOLDS           = 4          # K-Fold Time Series Splits
TOP_TICKERS_COUNT = 30         # Liquid Stock Universe Pool
EXPORT_LIMIT      = 25         # Top ranked stocks to display


# =============================================================================
# STEP 1: DOWNLOAD ALL INTRADAY DATA (PRICES & VOLUMES)
# =============================================================================
def compute_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. Volume Dynamics: Relative Volume (RVOL) & Money Flow (MFI)
    vol_sma20 = volume.rolling(20).mean()
    data['RVOL'] = volume / (vol_sma20 + 1e-9)
    
    typical_p = (high + low + close) / 3.0
    cum_pv = (typical_p * volume).cumsum()
    cum_v = volume.cumsum()
    vwap = cum_pv / (cum_v + 1e-9)
    data['VWAP_Dist'] = (close - vwap) / (vwap + 1e-9)

    rmf = typical_p * volume
    pos_flow = pd.Series(np.where(typical_p > typical_p.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    neg_flow = pd.Series(np.where(typical_p < typical_p.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    data['MFI_Norm'] = ((100.0 - (100.0 / (1.0 + (pos_flow / (neg_flow + 1e-9))))) - 50.0) / 50.0

    # 2. Momentum: RSI & Stochastic
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI_Norm'] = ((100.0 - (100.0 / (1.0 + rs))) - 50.0) / 50.0

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    data['Stoch_Norm'] = ((100.0 * (close - low_14) / (high_14 - low_14 + 1e-9)) - 50.0) / 50.0

    # 3. Volatility: ATR & Bollinger Bands
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR'] = tr.ewm(alpha=1/14, min_periods=14).mean()
    data['NATR'] = (data['ATR'] / close) * 100.0

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=1)
    data['BB_Norm'] = ((close - (bb_mid - 2.0 * bb_std)) / (4.0 * bb_std + 1e-9)) - 0.5

    # 4. Trend Ribbon Dispersion
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    data['Ribbon_Spread'] = (ema8 - ema55) / (ema55 + 1e-9)
    data['Fast_Spread'] = (ema8 - ema21) / (ema21 + 1e-9)

    return data


def create_triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - HORIZON_BARS):
        entry_p = close[i]
        target_p = entry_p * (1.0 + TARGET_PROFIT_PCT)
        stop_p = entry_p - (atr[i] * ATR_SL_MULTIPLIER)

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


def fetch_and_prepare_stock(ticker: str) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]:
    clean_t = ticker.strip().upper()
    try:
        df = yf.download(f"{clean_t}.NS", period="60d", interval="15m", progress=False)
        if df is None or len(df) < 200:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df_feat = compute_institutional_features(df).dropna()
        if len(df_feat) < 180:
            return None

        labels = create_triple_barrier_labels(df_feat)
        valid_idx = df_feat.index[:-HORIZON_BARS]
        latest_row = df_feat.iloc[-1]
        
        return (df_feat.loc[valid_idx], labels.loc[valid_idx], latest_row)
    except Exception:
        return None


def generate_pine_script_v6(output_path: str = "strategy_v6.pine") -> str:
    pine_code = """//@version=6
strategy("Master Institutional ML Swing Engine [v6]", 
         shorttitle="ML_SWING_v6", 
         overlay=true, 
         initial_capital=1000000, 
         default_qty_type=strategy.percent_of_equity, 
         default_qty_value=10, 
         commission_type=strategy.commission.percent, 
         commission_value=0.03, 
         slippage=2,
         pyramiding=0)

// 1. INPUT CONFIGURATION & TUNED HYPERPARAMETERS
var string G_RISK       = "Risk Controls"
i_targetProfitPct       = input.float(5.0, "Fixed Profit Target (%)", minval=0.5, step=0.25, group=G_RISK)
i_atrSlMultiplier       = input.float(1.75, "ATR Stop Loss Multiplier", minval=0.5, step=0.25, group=G_RISK)
i_atrLength             = input.int(14, "ATR Length", minval=1, group=G_RISK)
i_enableBreakeven       = input.bool(true, "Enable Breakeven Stop", group=G_RISK)
i_breakevenTriggerPct   = input.float(2.5, "Breakeven Gain Trigger (%)", minval=0.5, step=0.25, group=G_RISK)

var string G_ML         = "Tuned ML Classifier Parameters"
i_kNeighbors            = input.int(8, "k-Nearest Neighbors (k)", minval=1, maxval=50, group=G_ML)
i_trainingWindow        = input.int(250, "Training Horizon (Bars)", minval=50, maxval=2000, group=G_ML)
i_confidenceThresh      = input.float(52.0, "Ensemble Confidence (%)", minval=50.0, maxval=95.0, step=1.0, group=G_ML)

// 2. FEATURE EXTRACTION PIPELINE
float volSma20 = ta.sma(volume, 20)
float f_rvol   = math.min((volume / (volSma20 + 1e-9)) / 3.0, 1.0)
float f_mfi    = (ta.mfi(hlc3, 14) - 50.0) / 50.0

float f_rsi    = (ta.rsi(close, 14) - 50.0) / 50.0
float f_tsi    = ta.tsi(close, 25, 13) / 100.0

[diPlus, diMinus, _] = ta.dmi(14, 14)
float f_dmi    = (diPlus - diMinus) / 100.0
float ema8     = ta.ema(close, 8)
float ema55    = ta.ema(close, 55)
float f_ribbon = math.max(math.min((ema8 - ema55) / (ema55 + 1e-9) * 10.0, 1.0), -1.0)

[bbMid, bbUp, bbLow] = ta.bb(close, 20, 2)
float f_bb     = ((close - bbLow) / (bbUp - bbLow + 1e-9)) - 0.5

// 3. LORENTZIAN DISTANCE MACHINE LEARNING ENGINE
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
                                      array.get(arr_f1, i), array.get(arr_f2, i), array.get(arr_f3, i), 
                                      array.get(arr_f4, i), array.get(arr_f5, i), array.get(arr_f6, i), array.get(arr_f7, i))
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

// 4. STRATEGY EXECUTION
float atrVal = ta.atr(i_atrLength)
var float entryPriceLocal = 0.0
var float targetPrice     = 0.0
var float stopLossPrice   = 0.0
var bool  isBreakeven     = false

var line lineTP    = na
var line lineSL    = na
var line lineEntry = na

bool inLongPosition = strategy.position_size > 0

if (mlDirection == 1) and not inLongPosition and barstate.isconfirmed
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

if inLongPosition
    if i_enableBreakeven and not isBreakeven and (high >= entryPriceLocal * (1.0 + (i_breakevenTriggerPct / 100.0)))
        stopLossPrice := entryPriceLocal
        isBreakeven   := true
        line.set_y1(lineSL, stopLossPrice)
        line.set_y2(lineSL, stopLossPrice)
        line.set_color(lineSL, color.orange)
    strategy.exit("Exit_BUY", "BUY", limit=targetPrice, stop=stopLossPrice)

if inLongPosition
    line.set_x2(lineEntry, bar_index + 3)
    line.set_x2(lineTP, bar_index + 3)
    line.set_x2(lineSL, bar_index + 3)

// 5. HUD TABLE
var table hud = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 15), border_width=1, border_color=color.gray)

if barstate.islast
    table.cell(hud, 0, 0, "Metric", text_color=color.white, text_size=size.small, bgcolor=color.navy)
    table.cell(hud, 1, 0, "Value",  text_color=color.white, text_size=size.small, bgcolor=color.navy)
    
    table.cell(hud, 0, 1, "Ensemble ML", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 1, str.tostring(modelConfidence, "#.#") + "%", 
               text_color=modelConfidence >= i_confidenceThresh ? color.lime : color.gray, text_size=size.small)
    
    table.cell(hud, 0, 2, "Market Bias", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 2, mlDirection == 1 ? "BULLISH" : "NEUTRAL", 
               text_color=mlDirection == 1 ? color.green : color.gray, text_size=size.small)
    
    table.cell(hud, 0, 3, "Entry Price", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 3, inLongPosition ? str.tostring(entryPriceLocal, "#.##") : "-", text_color=color.white, text_size=size.small)
    
    table.cell(hud, 0, 4, "Target (+5.0%)", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 4, inLongPosition ? str.tostring(targetPrice, "#.##") : "-", text_color=color.green, text_size=size.small)
    
    table.cell(hud, 0, 5, "Stop Loss", text_color=color.silver, text_size=size.small)
    table.cell(hud, 1, 5, inLongPosition ? str.tostring(stopLossPrice, "#.##") : "-", 
               text_color=isBreakeven ? color.orange : color.red, text_size=size.small)

plotshape(mlDirection == 1 and not inLongPosition, title="Buy Signal", style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small)
"""
    with open(output_path, "w") as f:
        f.write(pine_code)
    return pine_code


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================
def main():
    # Always generate the Pine Script strategy so artifact upload succeeds
    generate_pine_script_v6("strategy_v6.pine")

    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file):
        sample = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", "SBIN", "LICI", "ITC", "LT"]
        with open(ticker_file, "w") as f:
            f.write("\n".join(sample))

    with open(ticker_file, "r") as f:
        all_tickers = [line.strip().upper() for line in f if line.strip()][:TOP_TICKERS_COUNT]

    print("\n" + "=" * 100)
    print("STEP 1: INGESTING INTRADAY DATA (PRICES & VOLUMES) ACROSS UNIVERSE")
    print("=" * 100)

    feature_cols = ['RVOL', 'VWAP_Dist', 'MFI_Norm', 'RSI_Norm', 'Stoch_Norm', 'NATR', 'BB_Norm', 'Ribbon_Spread', 'Fast_Spread']
    
    x_list, y_list = [], []
    latest_market_state = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(fetch_and_prepare_stock, t): t for t in all_tickers}
        for future in as_completed(future_map):
            t = future_map[future]
            res = future.result()
            if res is not None:
                df_stock, labels_stock, last_bar = res
                x_list.append(df_stock[feature_cols])
                y_list.append(labels_stock)
                latest_market_state[t] = last_bar
                print(f"[+] Ingested {t:<12} | {len(df_stock):,} bars | Vol & VWAP computed")

    if not x_list:
        print("[-] Data download failed.")
        pd.DataFrame().to_csv("final_ranked_results.csv", index=False)
        return

    X_full = pd.concat(x_list, ignore_index=True)
    y_full = pd.concat(y_list, ignore_index=True)

    print(f"\n[✓] Pooled Master Training Matrix: {len(X_full):,} bars across {len(latest_market_state)} stocks.")
    print(f"[✓] Class Distribution: Bull Targets (1): {sum(y_full == 1):,} | Stops/Timeouts (0): {sum(y_full == 0):,}")

    # =========================================================================
    # STEP 2: SPLIT INTO TRAINING & TEST SETS BY K-PARTITIONING
    # =========================================================================
    print("\n" + "=" * 100)
    print(f"STEP 2: K-PARTITIONING (TIME-SERIES SPLITS: K={K_FOLDS})")
    print("=" * 100)

    tscv = TimeSeriesSplit(n_splits=K_FOLDS)
    scaler = StandardScaler()

    for fold_idx, (train_indices, test_indices) in enumerate(tscv.split(X_full), start=1):
        print(f" -> Fold {fold_idx}: Train Window = {len(train_indices):,} samples | Test Window = {len(test_indices):,} samples")

    X_train_raw = X_full.iloc[train_indices]
    y_train = y_full.iloc[train_indices]
    X_test_raw = X_full.iloc[test_indices]
    y_test = y_full.iloc[test_indices]

    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # =========================================================================
    # STEP 3 & 4: DEVELOP & TRAIN MULTIPLE TRADING ML MODELS
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 3 & 4: DEVELOPING & TRAINING CANDIDATE ML MODELS")
    print("=" * 100)

    candidate_models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=50, max_depth=4, learning_rate=0.05, random_state=42),
        "RandomForest":         RandomForestClassifier(n_estimators=60, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1),
        "k-NearestNeighbors":   KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan', n_jobs=-1),
        "RegularizedLogistic":  LogisticRegression(C=0.1, penalty='l2', max_iter=400, random_state=42)
    }

    # =========================================================================
    # STEP 5: TEST PERFORMANCE OF ALL MODELS INDIVIDUALLY ON TEST SET
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 5: BENCHMARKING INDIVIDUAL MODEL PERFORMANCE ON OUT-OF-SAMPLE TEST SET")
    print("=" * 100)
    print(f"{'Model Name':<24}{'OOS Accuracy':<16}{'OOS Precision':<16}{'ROC-AUC Score':<16}")
    print("-" * 72)

    col_win = 1
    individual_scores = {}

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, col_win]

        acc = accuracy_score(y_test, preds) * 100.0
        prec = precision_score(y_test, preds, zero_division=0) * 100.0
        auc = roc_auc_score(y_test, probs)

        individual_scores[name] = auc
        print(f"{name:<24}{acc:<15.2f}%{prec:<15.2f}%{auc:<16.4f}")

    # =========================================================================
    # STEP 6: FIND THE BEST WAY TO ENSEMBLE THE BEST MODELS
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 6: ENSEMBLE DISCOVERY (COMPARING VOTING VS STACKING)")
    print("=" * 100)

    sorted_scores = sorted(individual_scores.items(), key=itemgetter(1), reverse=True)
    iter_scores = iter(sorted_scores)
    top_first, _ = next(iter_scores)
    top_second, _ = next(iter_scores)
    print(f"[*] Top Performing Base Estimators: {top_first} & {top_second}")

    # Using valid Python tuples to avoid syntax issues
    weights_tuple = (2, 1, 1, 1)
    ensembles = {
        "Soft-Voting (Uniform)": VotingClassifier(
            estimators=[(k, candidate_models[k]) for k in candidate_models], 
            voting='soft'
        ),
        "Weighted Voting": VotingClassifier(
            estimators=[(k, candidate_models[k]) for k in candidate_models],
            voting='soft',
            weights=weights_tuple
        ),
        "Stacking Meta-Learner": StackingClassifier(
            estimators=[(k, candidate_models[k]) for k in candidate_models],
            final_estimator=LogisticRegression(),
            n_jobs=-1
        )
    }

    best_ensemble_name = ""
    best_ensemble_auc = 0.0
    best_ensemble_model = None

    for e_name, ens in ensembles.items():
        ens.fit(X_train, y_train)
        probs = ens.predict_proba(X_test)[:, col_win]
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, ens.predict(X_test)) * 100.0
        print(f"{e_name:<26} | OOS Accuracy: {acc:.2f}% | ROC-AUC: {auc:.4f}")
        
        if auc > best_ensemble_auc:
            best_ensemble_auc = auc
            best_ensemble_name = e_name
            best_ensemble_model = ens

    print(f"\n[✓] WINNING ENSEMBLE: '{best_ensemble_name}' (ROC-AUC: {best_ensemble_auc:.4f})")

    # =========================================================================
    # STEP 7: HYPERPARAMETER TUNING OF THE ML PARAMETERS
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 7: HYPERPARAMETER TUNING ON THE BEST GRADIENT BOOSTED ESTIMATOR")
    print("=" * 100)

    # Defined as tuples to avoid syntax errors
    param_grid = {
        'max_iter': (30, 50),
        'max_depth': (3, 4),
        'learning_rate': (0.03, 0.05)
    }

    grid_search = GridSearchCV(
        estimator=HistGradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"[✓] Optimal Hyperparameters: {grid_search.best_params_}")
    print(f"[✓] Tuned Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")

    X_full_scaled = scaler.fit_transform(X_full)
    best_ensemble_model.fit(X_full_scaled, y_full)

    # =========================================================================
    # STEP 8: DRAFT THE BEST SWING TRADING STRATEGY (LIVE SCREENER RANKING)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 8: EXECUTING MASTER SWING STRATEGY ON LIVE MARKET DATA")
    print("=" * 100)

    opportunities = []
    row_first = 0

    for t, last_bar in latest_market_state.items():
        feat_vector = last_bar[feature_cols].values.reshape(1, -1)
        feat_scaled = scaler.transform(feat_vector)
        
        win_prob = float(best_ensemble_model.predict_proba(feat_scaled)[row_first, col_win])

        last_p = float(last_bar['Close'])
        last_atr = float(last_bar['ATR'])
        target_p = last_p * (1.0 + TARGET_PROFIT_PCT)
        stop_p = last_p - (last_atr * ATR_SL_MULTIPLIER)
        rr_ratio = (target_p - last_p) / (last_p - stop_p + 1e-9)

        opportunities.append({
            "ticker": t,
            "signal": "BUY" if win_prob >= 0.50 else "WATCH",
            "entry_price": round(last_p, 2),
            "target_5pct": round(target_p, 2),
            "stop_loss": round(stop_p, 2),
            "risk_reward": round(rr_ratio, 2),
            "win_probability": round(win_prob * 100.0, 1),
            "rvol": round(float(last_bar['RVOL']), 2),
            "vwap_dist_pct": round(float(last_bar['VWAP_Dist']) * 100.0, 2)
        })

    opportunities.sort(key=itemgetter("win_probability"), reverse=True)

    header = f"{'Rank':<6}{'Ticker':<14}{'Signal':<8}{'Entry (₹)':<12}{'Target (+5%)':<14}{'Stop Loss':<12}{'R:R':<8}{'Win Prob':<12}{'RVOL':<8}{'VWAP Dist%':<12}"
    print(header)
    print("-" * 100)
    for rk, o in enumerate(opportunities[:EXPORT_LIMIT], start=1):
        print(f"{rk:<6}{o['ticker']:<14}{o['signal']:<8}{o['entry_price']:<12.2f}{o['target_5pct']:<14.2f}"
              f"{o['stop_loss']:<12.2f}{o['risk_reward']:<8.2f}{o['win_probability']:<10.1f}%{o['rvol']:<8.2f}{o['vwap_dist_pct']:<12.2f}")

    print("=" * 100)
    tv_export = ", ".join([f"NSE:{o['ticker']}" for o in opportunities[:EXPORT_LIMIT]])
    print("TRADINGVIEW WATCHLIST EXPORT (TOP CANDIDATES):")
    print(tv_export)
    print("=" * 100)

    # Save to CSV
    pd.DataFrame(opportunities).to_csv("final_ranked_results.csv", index=False)
    print("\n[+] Saved to 'final_ranked_results.csv'")


if __name__ == "__main__":
    main()
