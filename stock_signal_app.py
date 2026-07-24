"""
AI Stock Signal Bot — Web App (Streamlit version)
----------------------------------------------------
Turns the basic AI stock signal bot into a simple web app.
Enter any stock ticker in the sidebar, click "Run Model", and see
the model's prediction + suggested entry/exit/stoploss.

NOTE: Learning project, not financial advice.

Install requirements first:
    pip install streamlit yfinance pandas numpy scikit-learn

Run with:
    streamlit run stock_signal_app.py

This opens a local web page in your browser automatically.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import hashlib
from sklearn.linear_model import LogisticRegression

# ---------------- Core logic (same model as before) ----------------

@st.cache_data(ttl=3600)  # cache for 1 hour so we don't re-download for every click
def fetch_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


def add_features(df):
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]
    df["volatility_10"] = df["return_1d"].rolling(10).std()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df = df.dropna()
    return df


def add_label(df):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df


def train_model(df):
    features = ["return_1d", "sma_ratio", "volatility_10", "rsi_14"]
    X = df[features]
    y = df["target"]

    # time-based split — never shuffle time series randomly
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LogisticRegression()    
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    return model, features, train_acc, test_acc


def get_signal(df, model, features, stoploss_pct, take_profit_pct, confidence_threshold):
    latest_row = df[features].iloc[[-1]]
    prediction = model.predict(latest_row)[0]
    probability = model.predict_proba(latest_row)[0][1]

    last_close = float(df["Close"].iloc[-1])
    stoploss_price = last_close * (1 - stoploss_pct)
    target_price = last_close * (1 + take_profit_pct)

    action = "HOLD / NO TRADE"
    if prediction == 1 and probability > confidence_threshold:
        action = "BUY / ENTER"

    return {
        "last_close": last_close,
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability": probability,
        "action": action,
        "stoploss_price": stoploss_price,
        "target_price": target_price,
    }


# ---------------- Page config (must be the very first Streamlit command) ----------------

st.set_page_config(page_title="AI Stock Signal Bot", layout="centered")

# ---------------- Real backend: SQLite-based user accounts ----------------
# This is a genuine backend for a learning/resume project: real accounts,
# stored in a local database file (users.db), with hashed passwords.
# Honest limitation: SHA-256 hashing here is simple, not the gold standard
# (bcrypt/argon2 are stronger) — fine for a demo project, not for a system
# handling sensitive real-world data.

DB_FILE = "users.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username TEXT PRIMARY KEY,
               password_hash TEXT NOT NULL
           )"""
    )
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def add_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # username already taken
    finally:
        conn.close()


def check_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row is not None and row[0] == hash_password(password)


init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.logged_in:
    st.title("🔒 AI Stock Signal Bot")
    st.caption("Create a real account or log in.")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if check_user(login_user, login_pass):
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        signup_user = st.text_input("Choose a username", key="signup_user")
        signup_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Create account"):
            if not signup_user or not signup_pass:
                st.warning("Please fill in both fields.")
            elif add_user(signup_user, signup_pass):
                st.success("Account created! Go to the Login tab to sign in.")
            else:
                st.error("That username is already taken.")

    st.stop()  # nothing below this runs until logged in

# ---------------- Streamlit UI (this is the "web app" part) ----------------

st.title("📈 AI Stock Signal Bot")
st.caption("Learning project — not financial advice. For education only.")

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.username}**")
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    st.header("Settings")
    ticker = st.text_input("Stock ticker", value="RELIANCE.NS")
    period = st.selectbox("History to use", ["1y", "2y", "5y"], index=1)
    stoploss_pct = st.slider("Stoploss %", 1, 10, 2) / 100
    take_profit_pct = st.slider("Take-profit %", 1, 20, 4) / 100
    confidence_threshold = st.slider("Confidence threshold %", 50, 90, 55) / 100
    run_button = st.button("Run Model")

if run_button:
    with st.spinner("Fetching data and training model..."):
        try:
            df = fetch_data(ticker, period)
            if df.empty:
                st.error("No data found for that ticker. Check the symbol (e.g. AAPL, TCS.NS, RELIANCE.NS).")
            else:
                df = add_features(df)
                df = add_label(df)
                model, features, train_acc, test_acc = train_model(df)
                signal = get_signal(df, model, features, stoploss_pct, take_profit_pct, confidence_threshold)

                col1, col2 = st.columns(2)
                col1.metric("Train accuracy", f"{train_acc:.1%}")
                col2.metric("Test accuracy (trust this one)", f"{test_acc:.1%}")

                st.subheader(f"Latest signal for {ticker}")
                st.write(f"**Last close:** {signal['last_close']:.2f}")
                st.write(f"**Model prediction:** {signal['prediction']} (confidence: {signal['probability']:.1%})")

                if signal["action"].startswith("BUY"):
                    st.success(f"Suggested action: {signal['action']}")
                    st.write(f"Stoploss: {signal['stoploss_price']:.2f}")
                    st.write(f"Take-profit: {signal['target_price']:.2f}")
                else:
                    st.info(f"Suggested action: {signal['action']}")

                st.subheader("Price chart (last 6 months)")
                st.line_chart(df[["Close", "sma_10", "sma_50"]].tail(126))

        except Exception as e:
            st.error(f"Something went wrong: {e}")
else:
    st.write("👈 Set your options in the sidebar and click **Run Model** to get a signal.")
