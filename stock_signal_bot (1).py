"""
Basic AI Stock Signal Bot (v1 — starter version)
--------------------------------------------------
A first, simple AI-based stock signal bot. Pulls historical data,
builds a few basic features, trains a Logistic Regression model to
predict next-day price direction, and prints a basic entry / exit /
stoploss suggestion.

NOTE: This is a LEARNING project, not financial advice. It uses one
stock, one train/test split, and very few features — it is a
foundation to build on, not a system to trade real money with.

Install requirements first:
    pip install yfinance pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression

# ---------------- CONFIG ----------------
TICKER = "RELIANCE.NS"      # change to any symbol, e.g. "AAPL", "TCS.NS"
PERIOD = "2y"                # how much history to pull
STOPLOSS_PCT = 0.02          # 2% stoploss
TAKE_PROFIT_PCT = 0.04       # 4% take-profit
CONFIDENCE_THRESHOLD = 0.55  # only act if model is at least this confident
# -----------------------------------------


def fetch_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    # yfinance sometimes returns multi-level columns (e.g. ("Close", "RELIANCE.NS"))
    # even for a single ticker — flatten it so "Close" etc. are plain columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def add_features(df):
    df["return_1d"] = df["Close"].pct_change()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]
    df["volatility_10"] = df["return_1d"].rolling(10).std()

    # basic RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df = df.dropna()
    return df


def add_label(df):
    # 1 = price goes up next day, 0 = it doesn't
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df


def train_model(df):
    features = ["return_1d", "sma_ratio", "volatility_10", "rsi_14"]
    X = df[features]
    y = df["target"]

    # IMPORTANT: time-based split — never randomly shuffle time series data,
    # or you leak future information into training (a very common beginner bug)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.2%}")
    print(f"Test accuracy:  {test_acc:.2%}  <- trust this number, not the train one")

    return model, features


def latest_signal(df, model, features):
    latest_row = df[features].iloc[[-1]]
    prediction = model.predict(latest_row)[0]
    probability = model.predict_proba(latest_row)[0][1]

    last_close = df["Close"].iloc[-1]
    stoploss_price = last_close * (1 - STOPLOSS_PCT)
    target_price = last_close * (1 + TAKE_PROFIT_PCT)

    print("\n--- Latest Signal ---")
    print(f"Ticker: {TICKER}")
    print(f"Last close: {last_close:.2f}")
    print(f"Model prediction: {'UP' if prediction == 1 else 'DOWN'} (confidence: {probability:.2%})")

    if prediction == 1 and probability > CONFIDENCE_THRESHOLD:
        print("Suggested action: BUY / ENTER")
        print(f"  Stoploss:    {stoploss_price:.2f}  (-{STOPLOSS_PCT:.0%})")
        print(f"  Take-profit: {target_price:.2f}  (+{TAKE_PROFIT_PCT:.0%})")
    else:
        print("Suggested action: HOLD / NO TRADE (low confidence or predicting down)")


def main():
    df = fetch_data(TICKER, PERIOD)
    if df.empty:
        print(f"\nNo data came back for '{TICKER}'.")
        print("This is usually a temporary Yahoo Finance hiccup, not a real problem.")
        print("Try again in a minute, try a different ticker (e.g. 'AAPL'), or run:")
        print("    pip install --upgrade yfinance")
        return
    df = add_features(df)
    df = add_label(df)
    model, features = train_model(df)
    latest_signal(df, model, features)


if __name__ == "__main__":
    main()
