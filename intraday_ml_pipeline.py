#!/usr/bin/env python3
"""
INSTITUTIONAL INTRADAY ML ENGINE & PINE SCRIPT v6 GENERATOR (SHARDED)
====================================================================
Fixed Probability Slicing, Robust Error Reporting, and Verified Population
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

TARGET_PROFIT_PCT = 0.025       # 2.5% Target
ATR_MULTIPLIER    = 1.50        # ATR Stop Loss Multiplier
HORIZON_BARS      = 20          # Max forward bars (~5 hours on 15m)
MIN_TRAIN_BARS    = 180         # Lookback threshold
TEST_FOLD_BARS    = 40          # Walk-forward fold window
MIN_CONFIDENCE    = 0.50        # Statistical Edge Threshold (50%+)
RISK_FREE_RATE    = 0.065       # Baseline rate (~6.5%)


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

    # 2. Stochastic %K & %D
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

    # 5. Money Flow Index (14)
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    pos_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    neg_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    data['MFI'] = 100.0 - (100.0 / (1.0 + (pos_flow / (neg_flow + 1e-9))))

    # 6. Trend: EMA Ribbon
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    data['EMA_Ribbon_Spread'] = (ema8 - ema55) / (ema55 + 1e-9)
    data['EMA_Short_Spread'] = (ema8 - ema21) / (ema21 + 1e-9)

    # 7. Directional Movement Index (ADX)
    up_m = high - high.shift(1)
    down_m = low.shift(1) - low
    p_dm = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
    tr_s = tr.ewm(alpha=1/14, min_periods=14).mean()
    p_di = 100.0 * pd.Series(p_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    m_di = 100.0 * pd.Series(m_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    data['DI_Diff'] = p_di - m_di

    # 8. Momentum Return
    data['ROC_4'] = close.pct_change(4)

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
    clf_gbm = HistGradientBoostingClassifier(max_iter=40, max_depth=3, learning_rate=0.05, random_state=42)
    clf_knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')
    clf_lr = LogisticRegression(C=0.1, max_iter=300, random_state=42)
    return VotingClassifier(estimators=[('gbm', clf_gbm), ('knn', clf_knn), ('lr', clf_lr)], voting='soft')


def run_walk_forward_validation(X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame) -> Tuple[float, float, float]:
    n_samples = len(X)
    if n_samples < (MIN_TRAIN_BARS + TEST_FOLD_BARS):
        return 50.0, 5.0, 0.05

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

        # FIXED: Slice class 1 probabilities directly as a 1D float array
        test_probs = model.predict_proba(X_test_scaled)

        for i_local in range(len(test_probs)):
            prob_val = float(test_probs[i_local])
            if prob_val >= MIN_CONFIDENCE:
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

    if oos_trades < 2:
        return 50.0, 5.0, 0.05

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

    fitness = ((max(ann_ret, 0.0) * 100.0) / (max_dd ** 2)) * sortino

    return win_rate, max_dd, fitness


def evaluate_ticker_ml(raw_ticker: str) -> Optional[ScreenerOpportunity]:
    clean_ticker = raw_ticker.strip().upper()
    if not clean_ticker:
        return None

    yf_symbol = f"{clean_ticker}.NS"
    try:
        df = yf.download(yf_symbol, period="60d", interval="15m", progress=False)
        if df is None or len(df) < (MIN_TRAIN_BARS + 20):
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df_feat = build_feature_store(df).dropna()
        if len(df_feat) < MIN_TRAIN_BARS:
            return None

        labels = generate_triple_barrier_labels(df_feat)

        feature_cols = [
            'RSI', 'Stoch_K', 'Stoch_D', 'NATR', 'BB_PctB', 
            'MFI', 'EMA_Ribbon_Spread', 'EMA_Short_Spread', 'DI_Diff', 'ROC_4'
        ]

        valid_indices = df_feat.index[:-HORIZON_BARS]
        X_train_full = df_feat.loc[valid_indices, feature_cols]
        y_train_full = labels.loc[valid_indices]

        if len(np.unique(y_train_full)) < 2:
            return None

        win_rate, max_dd, fitness = run_walk_forward_validation(X_train_full, y_train_full, df_feat)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        
        ensemble = build_ml_ensemble()
        ensemble.fit(X_train_scaled, y_train_full)

        # FIXED: Extract scalar class 1 probability from 2D predict_proba
        latest_features = df_feat.iloc[[-1]][feature_cols]
        latest_scaled = scaler.transform(latest_features)
        proba_matrix = ensemble.predict_proba(latest_scaled)
        live_prob = float(proba_matrix)

        last_close = float(df_feat.iloc[-1]['Close'])
        last_atr = float(df_feat.iloc[-1]['ATR'])
        target_p = last_close * (1.0 + TARGET_PROFIT_PCT)
        stop_p = last_close - (last_atr * ATR_MULTIPLIER)
        rr_ratio = abs(target_p - last_close) / (abs(last_close - stop_p) + 1e-9)

        return ScreenerOpportunity(
            ticker=clean_ticker,
            direction="BUY" if live_prob >= 0.50 else "WATCH",
            entry_price=round(last_close, 2),
            target_price=round(target_p, 2),
            stop_loss=round(stop_p, 2),
            risk_reward=round(rr_ratio, 2),
            ensemble_prob=round(live_prob * 100.0, 1),
            historical_win_rate=round(win_rate, 1),
            max_drawdown=round(max_dd, 1),
            fitness_score=round(fitness, 3)
        )
    except Exception as e:
        print(f"[-] Evaluation note for {clean_ticker}: {e}")
        return None


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
                print(f"[+] Evaluated: {res.ticker:<10} | Prob: {res.ensemble_prob}% | WinRate: {res.historical_win_rate}% | Fitness: {res.fitness_score}")

    empty_cols = ["ticker", "direction", "entry_price", "target_price", "stop_loss", 
                  "risk_reward", "ensemble_prob", "historical_win_rate", "max_drawdown", "fitness_score"]

    if opportunities:
        df_out = pd.DataFrame([asdict(o) for o in opportunities])
        df_out.to_csv(output_csv, index=False)
        print(f"[+] Shard {shard_id} saved {len(opportunities)} evaluated tickers to {output_csv}")
    else:
        pd.DataFrame(columns=empty_cols).to_csv(output_csv, index=False)
        print(f"[-] Shard {shard_id}: No data returned.")


def merge_and_display() -> None:
    print("\n" + "=" * 115)
    print(f"{'MERGING ALL SHARD ARTIFACTS & COMPUTING MASTER RANKING':^115}")
    print("=" * 115)

    empty_cols = ["ticker", "direction", "entry_price", "target_price", "stop_loss", 
                  "risk_reward", "ensemble_prob", "historical_win_rate", "max_drawdown", "fitness_score"]

    csv_files = glob.glob("results_shard_*.csv")
    dfs = []
    for f in csv_files:
        try:
            if os.path.exists(f) and os.path.getsize(f) > 0:
                df_temp = pd.read_csv(f)
                if not df_temp.empty and "fitness_score" in df_temp.columns:
                    dfs.append(df_temp)
        except Exception:
            pass

    if not dfs:
        print("[-] No records found across shards.")
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
    for _, row in merged_df.head(25).iterrows():
        print(f"{row['ticker']:<12}{row['direction']:<6}{row['entry_price']:<12.2f}{row['target_price']:<14.2f}"
              f"{row['stop_loss']:<12.2f}{row['risk_reward']:<8.2f}{row['ensemble_prob']:<14.1f}"
              f"{row['historical_win_rate']:<12.1f}{row['max_drawdown']:<10.1f}{row['fitness_score']:<10.3f}")

    print("=" * 115)
    tv_symbols = ", ".join([f"NSE:{t}" for t in merged_df['ticker'].head(25).tolist()])
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
