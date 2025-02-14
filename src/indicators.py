import os
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

class BitcoinIndicators:
    def __init__(self, btc_file, fng_file, output_file, mvrv_file=None):
        self.btc_file = btc_file
        self.fng_file = fng_file
        self.output_file = output_file
        self.mvrv_file = mvrv_file
        self.data = self.load_bitcoin_data()

    def load_bitcoin_data(self):
        try:
            df = pd.read_csv(self.btc_file, index_col="Timestamp", parse_dates=True)
            df.index = pd.to_datetime(df.index).normalize()
            return df
        except Exception as e:
            print(f"Erreur lors du chargement du fichier Bitcoin : {e}")
            return pd.DataFrame()

    def calculate_puell_multiple(self):
        miner_revenue = self.data["close"] * 900
        ma_365 = miner_revenue.rolling(window=365).mean()
        self.data["Puell_Multiple"] = miner_revenue / ma_365

    def calculate_pi_cycle_top(self):
        self.data["Pi_Cycle_111DMA"] = self.data["close"].rolling(window=111).mean()
        self.data["Pi_Cycle_350DMA"] = self.data["close"].rolling(window=350).mean() * 2

    def calculate_2y_ma_multiplier(self):
        self.data["MA_2Y"] = self.data["close"].rolling(window=730).mean()
        self.data["MA_2Y_Multiplier"] = self.data["MA_2Y"] * 5

    def fetch_fear_and_greed(self):
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                fng_data = [{
                    "Timestamp": datetime.utcfromtimestamp(int(entry["timestamp"])),
                    "Fear_Greed_Index": entry["value"],
                    "FNG_Classification": entry["value_classification"]
                } for entry in data["data"]]
                df_fng = pd.DataFrame(fng_data)
                df_fng["Timestamp"] = pd.to_datetime(df_fng["Timestamp"]).dt.normalize()
                df_fng.set_index("Timestamp", inplace=True)
                self.data = self.data.join(df_fng, how="left")
            else:
                print("Aucune donnée historique disponible pour le Fear & Greed Index.")
        else:
            print(f"Erreur API Fear & Greed : {response.status_code}")

    def calculate_mvrv_z_score(self):
        if not self.mvrv_file:
            print("Aucun fichier MVRV fourni.")
            return
        try:
            with open(self.mvrv_file, "r") as file:
                data = json.load(file)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier MVRV : {e}")
            return
        mvrv_data = data.get("mvrv", [])
        if not mvrv_data:
            print("Aucune donnée MVRV trouvée.")
            return
        timestamps = []
        mvrv_values = []
        for point in mvrv_data:
            ts = point["x"]
            if ts > 1e10:
                ts = ts / 1000
            date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
            timestamps.append(date_str)
            mvrv_values.append(point["y"])
        mean_mvrv = sum(mvrv_values) / len(mvrv_values)
        std_mvrv = (sum((x - mean_mvrv) ** 2 for x in mvrv_values) / len(mvrv_values)) ** 0.5
        z_scores = [(value - mean_mvrv) / std_mvrv for value in mvrv_values]
        df_mvrv = pd.DataFrame({
            "Timestamp": timestamps,
            "MVRV": mvrv_values,
            "MVRV_Z_Score": z_scores
        })
        df_mvrv["Timestamp"] = pd.to_datetime(df_mvrv["Timestamp"], format="%Y-%m-%d")
        df_mvrv.set_index("Timestamp", inplace=True)
        self.data = self.data.join(df_mvrv, how="left")

    def calculate_rsi(self, period=14):
        delta = self.data["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        self.data["RSI"] = 100 - (100 / (1 + rs))

    def calculate_macd(self, span_short=12, span_long=26, signal_span=9):
        ema_short = self.data["close"].ewm(span=span_short, adjust=False).mean()
        ema_long = self.data["close"].ewm(span=span_long, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_span, adjust=False).mean()
        histogram = macd - signal
        self.data["MACD"] = macd
        self.data["MACD_Signal"] = signal
        self.data["MACD_Hist"] = histogram

    def calculate_bollinger_bands(self, window=20, num_std=2):
        self.data["BB_Middle"] = self.data["close"].rolling(window=window).mean()
        self.data["BB_Std"] = self.data["close"].rolling(window=window).std()
        self.data["BB_Upper"] = self.data["BB_Middle"] + (num_std * self.data["BB_Std"])
        self.data["BB_Lower"] = self.data["BB_Middle"] - (num_std * self.data["BB_Std"])

    def export_to_csv(self):
        try:
            self.data.to_csv(self.output_file)
            print(f"Données enrichies sauvegardées dans {self.output_file}")
        except Exception as e:
            print(f"Erreur lors de l'exportation vers CSV : {e}")

    def run(self):
        self.calculate_puell_multiple()
        self.calculate_pi_cycle_top()
        self.calculate_2y_ma_multiplier()
        self.fetch_fear_and_greed()
        self.calculate_mvrv_z_score()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.export_to_csv()

if __name__ == "__main__":
    btc_file = "/content/drive/My Drive/ProjetCrypto/btc_usdt_1h.csv"         # adaptez les chemins
    fng_file = "/content/drive/My Drive/ProjetCrypto/fear_and_greed_history.csv"
    indicators_output_file = "/content/drive/My Drive/ProjetCrypto/bitcoin_indicators.csv"
    mvrv_file = "/content/drive/My Drive/ProjetCrypto/mvrv.json"
    btc_indicators = BitcoinIndicators(btc_file, fng_file, indicators_output_file, mvrv_file)
    btc_indicators.run()
