import os
import requests
import pandas as pd
from datetime import datetime

def btc_data(output_file, start_date="2010-07-17"):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "limit": 2000,
        "toTs": int(datetime.now().timestamp())
    }
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    all_data = []
    while True:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json().get("Data", {}).get("Data", [])
            if not data:
                break
            all_data.extend(data)
            earliest_timestamp = data[0]["time"]
            if earliest_timestamp <= start_timestamp:
                break
            params["toTs"] = earliest_timestamp
        else:
            print(f"Erreur API : {response.status_code}")
            break
    df = pd.DataFrame(all_data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"time": "Timestamp"}, inplace=True)
    df.set_index("Timestamp", inplace=True)
    df = df[df.index >= start_date]
    df.sort_index(inplace=True)

    for col in ["high", "low", "open", "close"]:
        df[col] = df[col].apply(lambda x: round(x) if x > 1000 else x)
    df.to_csv(output_file)
    print(f"Données sauvegardées dans {output_file}")

if __name__ == "__main__":
    btc_csv_file = "/content/drive/My Drive/ProjetCrypto/btc_usdt_1h.csv"  # adaptez le chemin si nécessaire
    btc_data(btc_csv_file, start_date="2010-07-17")
