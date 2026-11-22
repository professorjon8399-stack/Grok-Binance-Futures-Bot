
import streamlit as st
import requests
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime, timezone

st.set_page_config(page_title="Grok Binance AI", page_icon="📈", layout="centered")

st.title("🤖 Grok Binance Futures AI")
st.caption("15-minute market analysis • Signals + Paper Trading")

st.warning(
    "Educational tool only. Futures trading is high risk. "
    "This app does NOT place real trades."
)

# ---------- Market data ----------
@st.cache_data(ttl=60)
def get_market_data(symbol):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "15m", "limit": 150}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    rows = response.json()
    columns = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(rows, columns=columns)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    # Remove the still-forming candle
    df = df.iloc[:-1].copy()

    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    change = df["close"].diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    previous_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    return df.dropna().reset_index(drop=True)


# ---------- Sidebar ----------
st.sidebar.header("Trading settings")
symbol = st.sidebar.selectbox(
    "Futures pair",
    ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
)
balance = st.sidebar.number_input(
    "Demo balance (USDT)", min_value=10.0, value=20.0, step=5.0
)
risk_pct = st.sidebar.slider("Illustrative risk per trade (%)", 0.5, 2.0, 1.0, 0.5)

# ---------- Paper log ----------
if "paper_log" not in st.session_state:
    st.session_state.paper_log = []

# ---------- Analysis ----------
if st.button("🔍 Analyze Market with Grok", type="primary", use_container_width=True):
    try:
        with st.spinner("Getting Binance market data..."):
            df = get_market_data(symbol)
            last = df.iloc[-1]

        st.subheader(f"{symbol} • 15-minute chart")
        st.metric("Last closed price", f"{last['close']:,.4f}")
        st.write({
            "EMA20": round(float(last["EMA20"]), 4),
            "EMA50": round(float(last["EMA50"]), 4),
            "RSI14": round(float(last["RSI14"]), 2),
            "ATR14": round(float(last["ATR14"]), 4)
        })

        recent = df.tail(30)[
            ["open", "high", "low", "close", "volume", "EMA20", "EMA50", "RSI14", "ATR14"]
        ].round(5).to_dict(orient="records")

        market_summary = {
            "symbol": symbol,
            "timeframe": "15m",
            "last_closed_candle": {
                "close": float(last["close"]),
                "EMA20": float(last["EMA20"]),
                "EMA50": float(last["EMA50"]),
                "RSI14": float(last["RSI14"]),
                "ATR14": float(last["ATR14"])
            },
            "recent_30_candles": recent
        }

        api_key = st.secrets.get("XAI_API_KEY", "")
        if not api_key:
            st.error(
                "Grok API key missing. Add XAI_API_KEY in "
                "Streamlit → App settings → Secrets."
            )
            st.stop()

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=120.0
        )

        prompt = f"""
You are a cautious crypto futures market analyst.
Analyze the supplied Binance USD-M Futures data for {symbol} on 15m.

Return:
1. Signal: BUY, SELL, or WAIT
2. Confidence: 0-100%
3. Trend and momentum
4. Entry area (or say no entry)
5. Stop-loss and take-profit idea
6. Main risks and invalidation condition

Use only the supplied data. Do not invent live prices or news.
If evidence is mixed or weak, choose WAIT.
Do not promise profit. This is analysis, not an instruction to execute a trade.

Market data:
{market_summary}
"""

        with st.spinner("Grok is analyzing the market..."):
            result = client.responses.create(
                model="grok-4.7",
                input=prompt,
                reasoning={"effort": "medium"},
                store=False,
                max_output_tokens=1200
            )

        analysis = result.output_text
        st.session_state.last_analysis = analysis
        st.session_state.last_symbol = symbol
        st.session_state.last_price = float(last["close"])
        st.session_state.last_atr = float(last["ATR14"])

        st.success("Analysis complete")
        st.markdown(analysis)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check internet connection, API key, and API account access.")

# ---------- Illustrative risk calculator ----------
st.divider()
st.subheader("🧮 Illustrative risk calculator")

entry = st.number_input("Possible entry price", min_value=0.0, value=0.0)
atr = st.number_input("ATR value", min_value=0.0, value=0.0)
direction = st.selectbox("Trade direction", ["BUY", "SELL"])

if st.button("Calculate example SL / TP"):
    if entry <= 0 or atr <= 0:
        st.error("Enter a valid entry price and ATR.")
    else:
        risk_amount = balance * (risk_pct / 100)
        stop_distance = atr * 1.5
        target_distance = stop_distance * 2

        if direction == "BUY":
            sl = entry - stop_distance
            tp = entry + target_distance
        else:
            sl = entry + stop_distance
            tp = entry - target_distance

        estimated_quantity = risk_amount / stop_distance

        st.write(f"**Stop loss:** {sl:.5f}")
        st.write(f"**Take profit (2R example):** {tp:.5f}")
        st.write(f"**Illustrative risk:** {risk_amount:.2f} USDT")
        st.write(f"**Estimated quantity:** {estimated_quantity:.6f} units")
        st.caption(
            "Estimate only. Fees, slippage, leverage, contract rules and "
            "liquidation risk are not included. Verify Binance contract sizing."
        )

# ---------- Paper trading ----------
st.divider()
st.subheader("📝 Paper trade journal")

if st.button("Save latest analysis to paper journal"):
    if "last_analysis" not in st.session_state:
        st.warning("Pehle market analysis karein.")
    else:
        st.session_state.paper_log.append({
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "symbol": st.session_state.last_symbol,
            "price": st.session_state.last_price,
            "analysis": st.session_state.last_analysis
        })
        st.success("Analysis saved in this session.")

if st.session_state.paper_log:
    for item in reversed(st.session_state.paper_log):
        with st.expander(f"{item['symbol']} • {item['time']}"):
            st.write(f"Reference price: {item['price']}")
            st.write(item["analysis"])
else:
    st.caption("Abhi koi paper journal entry nahi hai.")

st.caption("Signals are not financial advice. API usage may incur charges.")
