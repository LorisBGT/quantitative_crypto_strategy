import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from datetime import datetime

# Chemin vers le fichier enrichi
DATA_FILE = "/content/drive/My Drive/ProjetCrypto/bitcoin_indicators.csv"

class DataLoader:
    def __init__(self, csv_file, start_date="2012-01-01"):
        self.csv_file = csv_file
        self.start_date = start_date  # format "YYYY-MM-DD"
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_file, index_col="Timestamp", parse_dates=True)
            self.df.sort_index(inplace=True)
        except Exception as e:
            raise IOError(f"Erreur lors du chargement du fichier CSV : {e}")
        return self.df

    def clean_data(self, puell_threshold, mvrv_threshold, rsi_threshold, fg_threshold):
        self.df = self.df.ffill().bfill()
        self.df = self.df[self.df.index >= self.start_date]
        self.df['buy_signal'] = (
            (self.df['Puell_Multiple'] < puell_threshold) &
            (self.df['MVRV'] < mvrv_threshold) &
            (self.df['RSI'] < rsi_threshold) &
            (self.df['Fear_Greed_Index'] < fg_threshold)
        ).astype(int)
        return self.df

def run_signal_strategy(df, initial_capital=1000.0):
    capital = initial_capital
    position = 0.0
    in_position = False
    trades = []
    portfolio_history = []

    for date, row in df.iterrows():
        price = row['close']
        if (not in_position) and (row.get('buy_signal', 0) == 1):
            position = capital / price
            capital = 0.0
            in_position = True
            trades.append({'date': date, 'type': 'buy', 'price': price})
        if in_position:
            pc111 = row.get('Pi_Cycle_111DMA', np.nan)
            pc350 = row.get('Pi_Cycle_350DMA', np.nan)
            if (not pd.isna(pc111)) and (not pd.isna(pc350)) and (pc111 > pc350):
                capital = position * price
                position = 0.0
                in_position = False
                trades.append({'date': date, 'type': 'sell', 'price': price})
        current_value = capital + position * price
        portfolio_history.append({'date': date, 'portfolio': current_value})
    if in_position:
        final_price = df['close'].iloc[-1]
        capital = position * final_price
        trades.append({'date': df.index[-1], 'type': 'sell', 'price': final_price})
        portfolio_history[-1]['portfolio'] = capital
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df.set_index('date', inplace=True)
    return trades, portfolio_df

def performance_report(price_df, portfolio_df, trades, initial_capital=1000.0):
    report = {}
    final_value = portfolio_df['portfolio'].iloc[-1]
    strategy_return = final_value / initial_capital - 1
    report["Valeur finale"] = f"{final_value:,.2f} €"
    report["Rendement total (Stratégie)"] = f"{strategy_return*100:.2f} %"
    start_date = portfolio_df.index[0]
    end_date = portfolio_df.index[-1]
    days = (end_date - start_date).days
    years = days / 365.25 if days > 0 else 1
    CAGR = (final_value / initial_capital) ** (1 / years) - 1
    report["CAGR"] = f"{CAGR*100:.2f} %"
    df_perf = portfolio_df.copy()
    df_perf['daily_return'] = df_perf['portfolio'].pct_change()
    mean_daily = df_perf['daily_return'].mean()
    std_daily = df_perf['daily_return'].std()
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily != 0 else np.nan
    report["Sharpe Ratio"] = f"{sharpe:.2f}"
    cumulative_max = portfolio_df['portfolio'].cummax()
    drawdown = (portfolio_df['portfolio'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    report["Max Drawdown"] = f"{max_drawdown*100:.2f} %"
    avg_drawdown = drawdown[drawdown < 0].mean() if any(drawdown < 0) else 0
    report["Drawdown moyen"] = f"{avg_drawdown*100:.2f} %"
    first_price = price_df['close'].iloc[0]
    last_price = price_df['close'].iloc[-1]
    benchmark_final = (initial_capital / first_price) * last_price
    benchmark_return = benchmark_final / initial_capital - 1
    report["Rendement total (Benchmark)"] = f"{benchmark_return*100:.2f} %"
    perf_diff = strategy_return - benchmark_return
    report["Différence de rendement (Stratégie - Benchmark)"] = f"{perf_diff*100:.2f} %"
    round_trades = []
    trade_durations = []
    buy_trade = None
    for trade in trades:
        if trade['type'] == 'buy':
            buy_trade = trade
        elif trade['type'] == 'sell' and buy_trade is not None:
            ret = trade['price'] / buy_trade['price'] - 1
            round_trades.append(ret)
            duration = (trade['date'] - buy_trade['date']).total_seconds() / (3600 * 24)
            trade_durations.append(duration)
            buy_trade = None
    if round_trades:
        num_trades = len(round_trades)
        win_trades = [r for r in round_trades if r > 0]
        loss_trades = [r for r in round_trades if r <= 0]
        win_rate = len(win_trades) / num_trades * 100
        avg_trade_perf = np.mean(round_trades) if round_trades else 0
        avg_gain = np.mean(win_trades) if win_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        profit_factor = (sum(win_trades) / abs(sum(loss_trades))) if loss_trades and sum(loss_trades) != 0 else np.nan
        avg_duration = np.mean(trade_durations) if trade_durations else np.nan
        report["Nombre de trades"] = num_trades
        report["Win Rate"] = f"{win_rate:.2f} %"
        report["Performance moyenne par trade"] = f"{avg_trade_perf*100:.2f} %"
        report["Gain moyen par trade"] = f"{avg_gain*100:.2f} %"
        report["Perte moyenne par trade"] = f"{avg_loss*100:.2f} %"
        report["Profit Factor"] = f"{profit_factor:.2f}"
        report["Durée moyenne par trade (jours)"] = f"{avg_duration:.2f}"
    else:
        report["Nombre de trades"] = 0
    return report

def plot_backtest_results(df, trades, portfolio_df, initial_capital=1000.0, title_suffix=""):
    df_plot = df.copy().join(portfolio_df, how='left')
    df_plot['strategy_return'] = df_plot['portfolio'] / initial_capital
    buy_signals = [t for t in trades if t['type'] == 'buy']
    sell_signals = [t for t in trades if t['type'] == 'sell']
    fig, ax1 = plt.subplots(figsize=(14,7))
    ax1.plot(df_plot.index, df_plot['close'], label="Cours BTC", color='blue', linewidth=1.5)
    if buy_signals:
        ax1.scatter([t['date'] for t in buy_signals],
                    [t['price'] for t in buy_signals],
                    marker='^', color='green', s=100, label="Signal Achat")
    if sell_signals:
        ax1.scatter([t['date'] for t in sell_signals],
                    [t['price'] for t in sell_signals],
                    marker='v', color='red', s=100, label="Signal Vente")
    ax1.set_ylabel("Cours BTC (USD)")
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ax2.plot(df_plot.index, df_plot['strategy_return'], label="Rendement Stratégie", color='purple', linewidth=1.5)
    ax2.set_ylabel("Multiplicateur du Capital", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_yscale('log')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    ax1.set_title(f"Cours BTC et Rendement Cumulé {title_suffix}")
    ax1.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

def optimize_objective(trial, start_date="2012-01-01", initial_capital=1000.0):
    puell_threshold = trial.suggest_float("puell_threshold", 0.5, 2.5)
    mvrv_threshold  = trial.suggest_float("mvrv_threshold", 0.5, 2.5)
    rsi_threshold   = trial.suggest_int("rsi_threshold", 10, 50)
    fg_threshold    = trial.suggest_int("fg_threshold", 20, 80)
    loader = DataLoader(DATA_FILE, start_date=start_date)
    df = loader.load_data()
    df = loader.clean_data(puell_threshold, mvrv_threshold, rsi_threshold, fg_threshold)
    split_index = int(len(df) * 0.7)
    df_train = df.iloc[:split_index]
    _, portfolio_train = run_signal_strategy(df_train, initial_capital=initial_capital)
    final_value = portfolio_train['portfolio'].iloc[-1]
    return -final_value

if __name__ == "__main__":
    INITIAL_CAPITAL = 1000.0
    START_DATE = "2012-01-01"
    loader = DataLoader(DATA_FILE, start_date=START_DATE)
    df_full = loader.load_data()
    
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: optimize_objective(trial, start_date=START_DATE, initial_capital=INITIAL_CAPITAL), n_trials=50)
    best_params = study.best_params
    print("=== Optimisation terminée ===")
    print("Meilleurs paramètres trouvés :")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    df_clean = loader.clean_data(best_params["puell_threshold"],
                                 best_params["mvrv_threshold"],
                                 best_params["rsi_threshold"],
                                 best_params["fg_threshold"])
    
    split_index = int(len(df_clean) * 0.7)
    df_train = df_clean.iloc[:split_index]
    df_test  = df_clean.iloc[split_index:]
    
    trades_train, portfolio_train = run_signal_strategy(df_train, initial_capital=INITIAL_CAPITAL)
    print("\n=== Backtest Période d'optimisation (70% des données) ===")
    plot_backtest_results(df_train, trades_train, portfolio_train, initial_capital=INITIAL_CAPITAL, title_suffix="(Optimisation)")
    report_train = performance_report(df_train, portfolio_train, trades_train, initial_capital=INITIAL_CAPITAL)
    print("\n--- Rapport de Performance - Période d'optimisation ---")
    for key, value in report_train.items():
        print(f"{key}: {value}")
    
    trades_test, portfolio_test = run_signal_strategy(df_test, initial_capital=INITIAL_CAPITAL)
    print("\n=== Backtest Période de test (30% des données) ===")
    plot_backtest_results(df_test, trades_test, portfolio_test, initial_capital=INITIAL_CAPITAL, title_suffix="(Test)")
    report_test = performance_report(df_test, portfolio_test, trades_test, initial_capital=INITIAL_CAPITAL)
    print("\n--- Rapport de Performance - Période de test ---")
    for key, value in report_test.items():
        print(f"{key}: {value}")
