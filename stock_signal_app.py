"""
Copyright (c) 2026 Raj Jadhav. All rights reserved.
This source code is made publicly viewable for hosting purposes only.
No permission is granted to copy, modify, or redistribute without consent.

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

    # RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD histogram (12, 26, 9) — trend + momentum
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line

    # Bollinger Band %B — where price sits within its volatility bands (0-1 range typically)
    sma_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    df["bb_percent_b"] = (df["Close"] - lower_band) / (upper_band - lower_band)

    # ATR (14-day) — average true range, a volatility measure independent of direction
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # Volume change — is trading activity picking up or fading?
    df["volume_change"] = df["Volume"].pct_change()

    # Safety net: some tickers have zero-volume days or flat prices, which can
    # produce infinity from division (e.g. volume_change, bb_percent_b).
    # Treat infinities as missing data rather than letting them crash the model.
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna()
    return df


def add_label(df):
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()
    return df


def train_model(df):
    features = [
        "return_1d",
        "sma_ratio",
        "volatility_10",
        "rsi_14",
        "macd_hist",
        "bb_percent_b",
        "atr_14",
        "volume_change",
    ]
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


def get_signal(df, model, features, stoploss_pct, take_profit_pct, confidence_threshold, stoploss_mode="fixed", atr_multiplier=1.5):
    latest_row = df[features].iloc[[-1]]
    prediction = model.predict(latest_row)[0]
    probability = model.predict_proba(latest_row)[0][1]

    last_close = float(df["Close"].iloc[-1])
    last_atr = float(df["atr_14"].iloc[-1])

    if stoploss_mode == "atr":
        stoploss_price = last_close - (atr_multiplier * last_atr)
    else:
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

# IMPORTANT: change this to YOUR OWN username (the one you sign up with)
# so only you can see the admin panel below.
ADMIN_USERNAME = "changeme_to_your_username"


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


def add_premium_column():
    # Safe to call every time — only adds the column if it's missing.
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE users ADD COLUMN is_premium INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.close()


def is_user_premium(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT is_premium FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return bool(row and row[0])


def get_all_users():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, is_premium FROM users")
    rows = c.fetchall()
    conn.close()
    return rows


def set_premium(username, value):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET is_premium = ? WHERE username = ?", (int(value), username))
    conn.commit()
    conn.close()


init_db()
add_premium_column()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.logged_in:
    st.title("📈 AI Stock Signal Bot")
    st.caption("AI-powered trading signals — entry, exit, and stoploss guidance in one place.")

    col1, col2, col3 = st.columns(3)
    col1.markdown("**🤖 ML-Powered**\n\nModel trains fresh on real market data every run")
    col2.markdown("**📊 8 Indicators**\n\nRSI, MACD, Bollinger Bands, ATR, and more")
    col3.markdown("**🎯 Smart Stoploss**\n\nFixed % or volatility-adjusted (ATR) options")

    st.divider()

    with st.expander("💳 Free vs Premium"):
        st.markdown(
            "| | Free | Premium ⭐ |\n"
            "|---|---|---|\n"
            "| History length | 1 year | 2–5 years |\n"
            "| All 8 indicators | ✅ | ✅ |\n"
            "| Stoploss options | ✅ | ✅ |\n"
            "| Price | ₹0 | Contact for pricing |\n"
        )

    st.subheader("Get started")
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if check_user(login_user.strip(), login_pass.strip()):
                st.session_state.logged_in = True
                st.session_state.username = login_user.strip()
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        signup_user = st.text_input("Choose a username", key="signup_user")
        signup_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Create account"):
            clean_user = signup_user.strip()
            clean_pass = signup_pass.strip()
            if not clean_user or not clean_pass:
                st.warning("Please fill in both fields.")
            elif " " in clean_user:
                st.warning("Username cannot contain spaces.")
            elif len(clean_user) < 3:
                st.warning("Username must be at least 3 characters.")
            elif len(clean_pass) < 4:
                st.warning("Password must be at least 4 characters.")
            elif add_user(clean_user, clean_pass):
                st.success("Account created! Go to the Login tab to sign in.")
            else:
                st.error("That username is already taken.")

    st.stop()  # nothing below this runs until logged in

# ---------------- Streamlit UI (this is the "web app" part) ----------------

st.title("📈 AI Stock Signal Bot")
st.caption("Learning project — not financial advice. For education only.")

if st.session_state.username == ADMIN_USERNAME:
    with st.expander("🛠️ Admin Panel — manage users"):
        users = get_all_users()
        if not users:
            st.write("No users yet.")
        for uname, prem in users:
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.write(uname)
            col2.write("⭐ Premium" if prem else "Free")
            if prem:
                if col3.button("Revoke", key=f"revoke_{uname}"):
                    set_premium(uname, False)
                    st.rerun()
            else:
                if col3.button("Grant", key=f"grant_{uname}"):
                    set_premium(uname, True)
                    st.rerun()

        st.divider()
        try:
            with open(DB_FILE, "rb") as f:
                st.download_button("⬇️ Download database file (users.db)", f, file_name="users.db")
        except FileNotFoundError:
            st.write("Database file not found yet.")

with st.sidebar:
    st.write(f"Logged in as **{st.session_state.username}**")
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    with st.expander("ℹ️ How to use this app"):
        st.markdown(
            "1. Enter a stock ticker (e.g. `AAPL`, `RELIANCE.NS`, `TCS.NS`)\n"
            "2. Adjust stoploss / take-profit / confidence if you like\n"
            "3. Click **Run Model**\n"
            "4. Check the **Signal** tab for the prediction, **Chart** tab for the trend\n\n"
            "This is a learning project — not financial advice."
        )

    is_premium = is_user_premium(st.session_state.username)

    if is_premium:
        st.success("⭐ Premium plan")
    else:
        st.info("Free plan")
        with st.expander("Upgrade to Premium ⭐"):
            st.write("Premium unlocks 2-year and 5-year history for the model.")
            # Replace this URL with your own Stripe/Razorpay Payment Link
            st.link_button("Upgrade now", "https://buy.stripe.com/your-payment-link-here")
            st.caption(
                "After paying, your account is upgraded manually (see grant_premium.py) "
                "until an automated version is set up."
            )

    st.header("Settings")
    ticker = st.text_input(
        "Stock ticker",
        value="RELIANCE.NS",
        help="Examples: AAPL, MSFT, RELIANCE.NS, TCS.NS, INFY.NS",
    )

    if is_premium:
        period = st.selectbox("History to use", ["1y", "2y", "5y"], index=1)
    else:
        period = "1y"
        st.caption("Free plan: limited to 1 year of history.")

    stoploss_mode_label = st.radio(
        "Stoploss method",
        ["Fixed %", "ATR-based (smarter, adapts to volatility)"],
        help="ATR-based stoploss widens automatically for volatile stocks and tightens for calm ones.",
    )
    stoploss_mode = "atr" if stoploss_mode_label.startswith("ATR") else "fixed"
    stoploss_pct = st.slider("Stoploss % (used if Fixed selected)", 1, 10, 2) / 100
    take_profit_pct = st.slider("Take-profit %", 1, 20, 4) / 100
    confidence_threshold = st.slider("Confidence threshold %", 50, 90, 55) / 100
    run_button = st.button("Run Model")

if run_button:
    ticker_clean = ticker.strip().upper()

    if not ticker_clean:
        st.warning("Please enter a stock ticker first.")
    else:
        try:
            with st.status(f"Running model for {ticker_clean}...", expanded=True) as status:
                st.write("📥 Fetching price data...")
                df = fetch_data(ticker_clean, period)

                if df.empty or len(df) < 60:
                    status.update(label="No usable data found", state="error")
                    st.error(
                        f"Couldn't get enough data for **{ticker_clean}**. "
                        "Double-check the symbol format (e.g. `AAPL`, `MSFT`, `RELIANCE.NS`, `TCS.NS`). "
                        "If the symbol looks correct, Yahoo Finance may be having a temporary hiccup — try again in a minute."
                    )
                else:
                    st.write("🧮 Building features...")
                    df = add_features(df)
                    df = add_label(df)

                    if len(df) < 30:
                        status.update(label="Not enough history to train reliably", state="error")
                        st.error(
                            "Not enough historical data after processing. "
                            "Try a longer history period (if you're on Premium) or a different ticker."
                        )
                    else:
                        st.write("🤖 Training model...")
                        model, features, train_acc, test_acc = train_model(df)
                        signal = get_signal(
                            df, model, features, stoploss_pct, take_profit_pct,
                            confidence_threshold, stoploss_mode=stoploss_mode,
                        )
                        status.update(label="Done!", state="complete")

                        tab_signal, tab_chart = st.tabs(["📊 Signal", "📈 Chart"])

                        with tab_signal:
                            col1, col2 = st.columns(2)
                            col1.metric("Train accuracy", f"{train_acc:.1%}")
                            col2.metric("Test accuracy (trust this one)", f"{test_acc:.1%}")

                            st.subheader(f"Latest signal for {ticker_clean}")
                            st.metric("Last close", f"{signal['last_close']:.2f}")
                            st.write(
                                f"**Model prediction:** {signal['prediction']} "
                                f"(confidence: {signal['probability']:.1%})"
                            )

                            if signal["action"].startswith("BUY"):
                                st.success(f"✅ Suggested action: {signal['action']}")
                                stoploss_label = "Stoploss (ATR-based)" if stoploss_mode == "atr" else "Stoploss (fixed %)"
                                colA, colB = st.columns(2)
                                colA.metric(stoploss_label, f"{signal['stoploss_price']:.2f}")
                                colB.metric("Take-profit", f"{signal['target_price']:.2f}")
                            else:
                                st.info(f"⏸️ Suggested action: {signal['action']}")

                        with tab_chart:
                            st.subheader("Price chart (last 6 months)")
                            st.line_chart(df[["Close", "sma_10", "sma_50"]].tail(126))

        except Exception as e:
            st.error(
                "Something went wrong while running the model. "
                "This is usually a temporary data issue — try again in a minute, or try a different ticker."
            )
            with st.expander("Technical details (for debugging)"):
                st.code(str(e))
else:
    st.write("👈 Set your options in the sidebar and click **Run Model** to get a signal.")


st.markdown("---")
st.caption("© 2026 Raj Jadhav | All Rights Reserved")
