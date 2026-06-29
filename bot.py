"""
XAUUSD Gold Alert Bot
=====================
Professional ICT/SMC-based alert system for gold trading
Sends alerts to Telegram when A+ setups are detected

Architecture:
- Fetches OHLCV data via yfinance (free, no API key needed)
- Analyzes: Order Blocks, FVG, BOS, Kill Zones, Premium/Discount
- Sends rich Telegram alerts with full setup details
- Runs 24/7 on Railway/Render (free tier)
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("GoldBot")

# ─────────────────────────────────────────────
# CONFIG  (از environment variables خوانده می‌شود)
# ─────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL", "300"))   # هر ۵ دقیقه
SYMBOL = "GC=F"   # طلا در yfinance
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "60"))        # حداقل فاصله بین دو آلرت

# ─────────────────────────────────────────────
# KILL ZONES  (UTC)
# ─────────────────────────────────────────────
KILL_ZONES = {
    "London Open":   {"start": (7, 0),  "end": (9, 0)},
    "New York Open": {"start": (13, 0), "end": (15, 0)},
    "London Close":  {"start": (16, 30),"end": (17, 30)},
}

# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class OrderBlock:
    high: float
    low: float
    mid: float
    direction: str        # "bullish" | "bearish"
    timestamp: pd.Timestamp
    has_fvg: bool = False
    strength: int = 0     # 1-5


@dataclass
class SetupSignal:
    direction: str          # "BUY" | "SELL"
    entry_zone_high: float
    entry_zone_low: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rr1: float
    rr2: float
    kill_zone: str
    bias: str
    ob: OrderBlock
    liquidity_swept: bool
    bos_confirmed: bool
    score: int              # 0-10


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    """Send message to Telegram. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set — skipping alert.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("✅ Telegram alert sent.")
            return True
        else:
            log.error(f"Telegram error: {resp.status_code} — {resp.text}")
            return False
    except Exception as e:
        log.error(f"Telegram exception: {e}")
        return False


def send_startup_message():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = (
        "🤖 <b>Gold Alert Bot Started</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ Time: {now_utc}\n"
        f"📊 Symbol: XAUUSD (GC=F)\n"
        f"🔄 Check Interval: Every {CHECK_INTERVAL_SECONDS//60} min\n"
        f"⏳ Alert Cooldown: {COOLDOWN_MINUTES} min\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ System is live and monitoring markets."
    )
    send_telegram(msg)


# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────
def fetch_ohlcv(interval: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data with retry logic."""
    for attempt in range(3):
        try:
            ticker = yf.Ticker(SYMBOL)
            df = ticker.history(interval=interval, period=period)
            if df is None or df.empty:
                raise ValueError("Empty dataframe returned")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            log.debug(f"Fetched {len(df)} candles [{interval}]")
            return df
        except Exception as e:
            log.warning(f"Fetch attempt {attempt+1}/3 failed: {e}")
            time.sleep(5)
    log.error(f"Failed to fetch data for interval={interval}")
    return None


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────

def get_active_kill_zone() -> Optional[str]:
    """Return name of active kill zone or None."""
    now = datetime.now(timezone.utc)
    current_minutes = now.hour * 60 + now.minute
    for name, zone in KILL_ZONES.items():
        start = zone["start"][0] * 60 + zone["start"][1]
        end = zone["end"][0] * 60 + zone["end"][1]
        if start <= current_minutes <= end:
            return name
    return None


def is_weekend() -> bool:
    """Markets closed on weekends."""
    return datetime.now(timezone.utc).weekday() >= 5  # Sat=5, Sun=6


def detect_market_bias(df_daily: pd.DataFrame) -> str:
    """
    Determine overall market bias using:
    - Last 5 daily candles structure
    - 50-period EMA direction
    """
    if len(df_daily) < 20:
        return "NEUTRAL"

    close = df_daily["Close"]
    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()

    last_close = close.iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]

    # Higher highs / higher lows check (last 5 candles)
    highs = df_daily["High"].tail(5).values
    lows = df_daily["Low"].tail(5).values
    hh = all(highs[i] >= highs[i-1] for i in range(1, len(highs)))
    hl = all(lows[i] >= lows[i-1] for i in range(1, len(lows)))
    lh = all(highs[i] <= highs[i-1] for i in range(1, len(highs)))
    ll = all(lows[i] <= lows[i-1] for i in range(1, len(lows)))

    bullish_signals = 0
    bearish_signals = 0

    if last_close > last_ema20: bullish_signals += 1
    if last_close < last_ema20: bearish_signals += 1
    if last_ema20 > last_ema50: bullish_signals += 1
    if last_ema20 < last_ema50: bearish_signals += 1
    if hh and hl: bullish_signals += 2
    if lh and ll: bearish_signals += 2

    if bullish_signals >= 3:
        return "BULLISH"
    elif bearish_signals >= 3:
        return "BEARISH"
    return "NEUTRAL"


def get_premium_discount(df_daily: pd.DataFrame) -> str:
    """
    Check if price is in Premium (sell) or Discount (buy) zone.
    Based on 20-day equilibrium (50% of range).
    """
    high_20 = df_daily["High"].tail(20).max()
    low_20 = df_daily["Low"].tail(20).min()
    eq = (high_20 + low_20) / 2
    last_close = df_daily["Close"].iloc[-1]
    return "PREMIUM" if last_close > eq else "DISCOUNT"


def detect_order_blocks(df: pd.DataFrame, direction: str, lookback: int = 30) -> list[OrderBlock]:
    """
    Detect Order Blocks:
    - Bullish OB: Last bearish candle before a strong bullish move
    - Bearish OB: Last bullish candle before a strong bearish move
    """
    obs = []
    df = df.tail(lookback).copy().reset_index(drop=True)
    avg_body = abs(df["Close"] - df["Open"]).mean()

    for i in range(2, len(df) - 2):
        c = df.iloc[i]
        body = abs(c["Close"] - c["Open"])
        is_bearish = c["Close"] < c["Open"]
        is_bullish = c["Close"] > c["Open"]

        if direction == "bullish" and is_bearish:
            # Check if followed by strong bullish move
            next1 = df.iloc[i+1]
            next2 = df.iloc[i+2]
            move = next1["Close"] - c["Low"]
            if move > avg_body * 1.5 and next1["Close"] > next1["Open"]:
                strength = min(5, int(move / avg_body))
                obs.append(OrderBlock(
                    high=c["High"],
                    low=c["Low"],
                    mid=(c["High"] + c["Low"]) / 2,
                    direction="bullish",
                    timestamp=df.index[i] if hasattr(df.index[i], 'date') else pd.Timestamp.now(),
                    strength=strength
                ))

        elif direction == "bearish" and is_bullish:
            # Check if followed by strong bearish move
            next1 = df.iloc[i+1]
            move = c["High"] - next1["Close"]
            if move > avg_body * 1.5 and next1["Close"] < next1["Open"]:
                strength = min(5, int(move / avg_body))
                obs.append(OrderBlock(
                    high=c["High"],
                    low=c["Low"],
                    mid=(c["High"] + c["Low"]) / 2,
                    direction="bearish",
                    timestamp=df.index[i] if hasattr(df.index[i], 'date') else pd.Timestamp.now(),
                    strength=strength
                ))

    return obs


def detect_fvg(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """
    Detect Fair Value Gaps (3-candle pattern):
    Bullish FVG: candle[i-1].high < candle[i+1].low
    Bearish FVG: candle[i-1].low > candle[i+1].high
    """
    fvgs = []
    df = df.tail(lookback).copy().reset_index(drop=True)
    for i in range(1, len(df) - 1):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        nxt = df.iloc[i+1]

        # Bullish FVG
        if prev["High"] < nxt["Low"]:
            fvgs.append({
                "type": "bullish",
                "high": nxt["Low"],
                "low": prev["High"],
                "mid": (nxt["Low"] + prev["High"]) / 2,
                "size": nxt["Low"] - prev["High"]
            })

        # Bearish FVG
        elif prev["Low"] > nxt["High"]:
            fvgs.append({
                "type": "bearish",
                "high": prev["Low"],
                "low": nxt["High"],
                "mid": (prev["Low"] + nxt["High"]) / 2,
                "size": prev["Low"] - nxt["High"]
            })

    return fvgs


def detect_bos(df_15m: pd.DataFrame, direction: str) -> bool:
    """
    Break of Structure on 15M:
    Bullish BOS: price breaks above a recent swing high
    Bearish BOS: price breaks below a recent swing low
    """
    df = df_15m.tail(20).copy().reset_index(drop=True)
    if len(df) < 6:
        return False

    if direction == "bullish":
        # Find swing high in first half, check if broken in second half
        first_half = df.head(10)
        second_half = df.tail(10)
        swing_high = first_half["High"].max()
        return second_half["Close"].max() > swing_high

    elif direction == "bearish":
        first_half = df.head(10)
        second_half = df.tail(10)
        swing_low = first_half["Low"].min()
        return second_half["Close"].min() < swing_low

    return False


def detect_liquidity_sweep(df: pd.DataFrame, direction: str) -> bool:
    """
    Check if price recently swept liquidity (equal highs/lows).
    Bullish: swept below equal lows, then recovered
    Bearish: swept above equal highs, then dropped
    """
    df = df.tail(15).copy().reset_index(drop=True)
    if len(df) < 5:
        return False

    last_close = df["Close"].iloc[-1]

    if direction == "bullish":
        recent_low = df["Low"].tail(10).min()
        prev_lows = df["Low"].tail(15).values
        # Check for wick below recent support with close back above
        for i in range(len(df) - 5, len(df) - 1):
            if df["Low"].iloc[i] <= recent_low * 1.001:  # touched or breached
                if df["Close"].iloc[-1] > df["Low"].iloc[i] * 1.002:
                    return True

    elif direction == "bearish":
        recent_high = df["High"].tail(10).max()
        for i in range(len(df) - 5, len(df) - 1):
            if df["High"].iloc[i] >= recent_high * 0.999:
                if df["Close"].iloc[-1] < df["High"].iloc[i] * 0.998:
                    return True

    return False


def price_in_ob(current_price: float, ob: OrderBlock, buffer_pct: float = 0.001) -> bool:
    """Check if current price is inside or touching an Order Block."""
    low = ob.low * (1 - buffer_pct)
    high = ob.high * (1 + buffer_pct)
    return low <= current_price <= high


def ob_has_fvg(ob: OrderBlock, fvgs: list[dict]) -> bool:
    """Check if any FVG overlaps with the Order Block."""
    for fvg in fvgs:
        if fvg["type"] == ob.direction:
            # Check overlap
            overlap = min(ob.high, fvg["high"]) - max(ob.low, fvg["low"])
            if overlap > 0:
                return True
    return False


def calculate_targets(direction: str, entry: float, ob: OrderBlock,
                       df_4h: pd.DataFrame) -> dict:
    """Calculate SL, TP1, TP2, TP3 levels."""
    atr = calculate_atr(df_4h, period=14)
    buffer = atr * 0.3

    if direction == "BUY":
        sl = ob.low - buffer
        risk = entry - sl
        tp1 = entry + risk * 2.0
        tp2 = entry + risk * 3.5
        tp3 = entry + risk * 5.0
    else:
        sl = ob.high + buffer
        risk = sl - entry
        tp1 = entry - risk * 2.0
        tp2 = entry - risk * 3.5
        tp3 = entry - risk * 5.0

    rr1 = round(abs(tp1 - entry) / abs(entry - sl), 1)
    rr2 = round(abs(tp2 - entry) / abs(entry - sl), 1)

    return {
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "tp3": round(tp3, 2),
        "rr1": rr1,
        "rr2": rr2,
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def score_setup(bias: str, zone: str, kill_zone: str,
                ob: OrderBlock, liquidity_swept: bool,
                bos: bool, fvg_present: bool) -> int:
    """
    Score the setup from 0-10.
    Only signal if score >= 7.
    """
    score = 0

    # Bias alignment (2 pts)
    if (bias == "BULLISH" and zone == "DISCOUNT") or \
       (bias == "BEARISH" and zone == "PREMIUM"):
        score += 2

    # Kill Zone active (2 pts)
    if kill_zone:
        score += 2

    # Order Block strength (1 pt)
    score += min(1, ob.strength // 2)

    # FVG present (2 pts)
    if fvg_present:
        score += 2

    # Liquidity swept (1 pt)
    if liquidity_swept:
        score += 1

    # BOS confirmed (2 pts)
    if bos:
        score += 2

    return score


# ─────────────────────────────────────────────
# ALERT FORMATTER
# ─────────────────────────────────────────────
def format_alert(signal: SetupSignal, current_price: float) -> str:
    emoji = "🟢" if signal.direction == "BUY" else "🔴"
    dir_label = "BUY (Long) 📈" if signal.direction == "BUY" else "SELL (Short) 📉"
    bias_emoji = "🐂" if signal.bias == "BULLISH" else "🐻"
    score_bar = "⭐" * (signal.score // 2)

    tehran_offset = timedelta(hours=3, minutes=30)
    now_tehran = (datetime.now(timezone.utc) + tehran_offset).strftime("%H:%M")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    msg = (
        f"{emoji} <b>GOLD ALERT — {signal.direction}</b> {emoji}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Direction:</b> {dir_label}\n"
        f"💰 <b>Current Price:</b> ${current_price:,.2f}\n"
        f"\n"
        f"📍 <b>Entry Zone</b>\n"
        f"   High: ${signal.entry_zone_high:,.2f}\n"
        f"   Low:  ${signal.entry_zone_low:,.2f}\n"
        f"\n"
        f"🛑 <b>Stop Loss:</b> ${signal.stop_loss:,.2f}\n"
        f"\n"
        f"🎯 <b>Targets</b>\n"
        f"   TP1 (1:{signal.rr1}): ${signal.tp1:,.2f}\n"
        f"   TP2 (1:{signal.rr2}): ${signal.tp2:,.2f}\n"
        f"   TP3 (Swing):  ${signal.tp3:,.2f}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>Setup Analysis</b>\n"
        f"   {bias_emoji} Bias: {signal.bias}\n"
        f"   ⏰ Kill Zone: {signal.kill_zone}\n"
        f"   🧱 OB Quality: {'⭐' * signal.ob.strength}\n"
        f"   💧 Liquidity Swept: {'✅' if signal.liquidity_swept else '❌'}\n"
        f"   📉 BOS Confirmed: {'✅' if signal.bos_confirmed else '❌'}\n"
        f"   📐 FVG Present: {'✅' if signal.ob.has_fvg else '❌'}\n"
        f"\n"
        f"🏆 <b>Setup Score: {signal.score}/10</b>  {score_bar}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ <b>Risk Management</b>\n"
        f"   Max Risk: 1% of account per trade\n"
        f"   Move SL to BE after TP1 hit\n"
        f"\n"
        f"🕐 Tehran: {now_tehran} | UTC: {now_utc}\n"
        f"<i>Always confirm on 15M before entry</i>"
    )
    return msg


def format_no_setup_log(reason: str):
    log.info(f"No setup: {reason}")


# ─────────────────────────────────────────────
# MAIN ANALYSIS LOOP
# ─────────────────────────────────────────────
class GoldAlertBot:
    def __init__(self):
        self.last_alert_time: Optional[datetime] = None
        self.last_alert_direction: Optional[str] = None
        self.alerts_sent_today: int = 0

    def can_send_alert(self, direction: str) -> bool:
        """Rate limiting: prevent alert spam."""
        if self.last_alert_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_alert_time).total_seconds() / 60
        if elapsed < COOLDOWN_MINUTES:
            log.info(f"Cooldown active ({elapsed:.0f}/{COOLDOWN_MINUTES} min)")
            return False
        return True

    def analyze(self) -> Optional[SetupSignal]:
        """Full market analysis. Returns SetupSignal or None."""

        if is_weekend():
            log.info("Weekend — markets closed, skipping analysis.")
            return None

        # ── 1. Fetch data
        df_daily = fetch_ohlcv("1d", "3mo")
        df_4h    = fetch_ohlcv("1h", "30d")   # yfinance: use 1h as proxy for 4h
        df_1h    = fetch_ohlcv("1h", "14d")
        df_15m   = fetch_ohlcv("15m", "5d")

        if any(df is None for df in [df_daily, df_4h, df_1h, df_15m]):
            log.error("Data fetch failed — skipping cycle.")
            return None

        # ── 2. Kill Zone check
        kill_zone = get_active_kill_zone()
        if not kill_zone:
            log.info("Not in a Kill Zone — skipping detailed analysis.")
            return None

        log.info(f"Kill Zone active: {kill_zone}")

        # ── 3. Bias & Zone
        bias = detect_market_bias(df_daily)
        zone = get_premium_discount(df_daily)
        current_price = df_15m["Close"].iloc[-1]

        log.info(f"Price: {current_price:.2f} | Bias: {bias} | Zone: {zone}")

        if bias == "NEUTRAL":
            format_no_setup_log("Bias is NEUTRAL")
            return None

        # ── 4. Direction filter
        direction = "BUY" if bias == "BULLISH" else "SELL"
        ob_direction = "bullish" if direction == "BUY" else "bearish"

        # Confirm zone alignment
        if direction == "BUY" and zone != "DISCOUNT":
            format_no_setup_log("BUY bias but price in PREMIUM zone")
            return None
        if direction == "SELL" and zone != "PREMIUM":
            format_no_setup_log("SELL bias but price in DISCOUNT zone")
            return None

        # ── 5. Order Block detection (on 1H)
        obs_1h = detect_order_blocks(df_1h, ob_direction, lookback=50)
        if not obs_1h:
            format_no_setup_log("No Order Blocks found on 1H")
            return None

        # Find nearest OB to current price
        nearest_ob = None
        min_dist = float("inf")
        for ob in obs_1h:
            dist = abs(ob.mid - current_price)
            if dist < min_dist and price_in_ob(current_price, ob):
                min_dist = dist
                nearest_ob = ob

        if nearest_ob is None:
            format_no_setup_log("Price not inside any Order Block")
            return None

        log.info(f"OB found: {nearest_ob.low:.2f} - {nearest_ob.high:.2f} (strength={nearest_ob.strength})")

        # ── 6. FVG detection
        fvgs_1h = detect_fvg(df_1h, lookback=30)
        fvg_present = ob_has_fvg(nearest_ob, fvgs_1h)
        nearest_ob.has_fvg = fvg_present
        log.info(f"FVG present: {fvg_present}")

        # ── 7. Liquidity sweep
        liquidity_swept = detect_liquidity_sweep(df_15m, ob_direction)
        log.info(f"Liquidity swept: {liquidity_swept}")

        # ── 8. BOS on 15M
        bos = detect_bos(df_15m, ob_direction)
        log.info(f"BOS confirmed: {bos}")

        # ── 9. Score setup
        score = score_setup(
            bias=bias,
            zone=zone,
            kill_zone=kill_zone,
            ob=nearest_ob,
            liquidity_swept=liquidity_swept,
            bos=bos,
            fvg_present=fvg_present,
        )
        log.info(f"Setup score: {score}/10")

        if score < 7:
            format_no_setup_log(f"Score too low ({score}/10 < 7)")
            return None

        # ── 10. Calculate targets
        targets = calculate_targets(direction, current_price, nearest_ob, df_4h)

        signal = SetupSignal(
            direction=direction,
            entry_zone_high=nearest_ob.high,
            entry_zone_low=nearest_ob.low,
            stop_loss=targets["sl"],
            tp1=targets["tp1"],
            tp2=targets["tp2"],
            tp3=targets["tp3"],
            rr1=targets["rr1"],
            rr2=targets["rr2"],
            kill_zone=kill_zone,
            bias=bias,
            ob=nearest_ob,
            liquidity_swept=liquidity_swept,
            bos_confirmed=bos,
            score=score,
        )

        return signal

    def run_cycle(self):
        """Single analysis cycle."""
        log.info("─" * 40)
        log.info("Running analysis cycle...")
        try:
            signal = self.analyze()
            if signal:
                if self.can_send_alert(signal.direction):
                    current_price = signal.entry_zone_low  # approximate
                    message = format_alert(signal, current_price)
                    if send_telegram(message):
                        self.last_alert_time = datetime.now(timezone.utc)
                        self.last_alert_direction = signal.direction
                        self.alerts_sent_today += 1
                        log.info(f"Alert sent! Total today: {self.alerts_sent_today}")
            else:
                log.info("No signal this cycle.")
        except Exception as e:
            log.error(f"Unexpected error in cycle: {e}", exc_info=True)

    def run(self):
        """Main loop — runs forever."""
        log.info("=" * 50)
        log.info("   XAUUSD Gold Alert Bot — Starting")
        log.info("=" * 50)
        send_startup_message()

        while True:
            self.run_cycle()
            log.info(f"Next check in {CHECK_INTERVAL_SECONDS // 60} minutes...")
            time.sleep(CHECK_INTERVAL_SECONDS)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    bot = GoldAlertBot()
    bot.run()
