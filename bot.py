"""
XAUUSD Gold Alert Bot v2
========================
Professional ICT/SMC-based alert system for gold trading
Data source: Stooq (رایگان، بدون API key، روی Railway کار می‌کند)
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
import requests
import pandas as pd
import numpy as np
import io

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
# CONFIG
# ─────────────────────────────────────────────
TELEGRAM_TOKEN        = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID      = os.environ.get("TELEGRAM_CHAT_ID", "")
CHECK_INTERVAL        = int(os.environ.get("CHECK_INTERVAL", "300"))
COOLDOWN_MINUTES      = int(os.environ.get("COOLDOWN_MINUTES", "60"))
MIN_SCORE             = int(os.environ.get("MIN_SCORE", "7"))

# ─────────────────────────────────────────────
# KILL ZONES (UTC)
# ─────────────────────────────────────────────
KILL_ZONES = {
    "London Open":   (7*60,   9*60),
    "New York Open": (13*60, 15*60),
    "London Close":  (16*60+30, 17*60+30),
}

# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class OrderBlock:
    high: float
    low: float
    mid: float
    direction: str
    strength: int = 1
    has_fvg: bool = False

@dataclass
class Signal:
    direction: str
    entry_high: float
    entry_low: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rr1: float
    rr2: float
    kill_zone: str
    bias: str
    score: int
    ob: OrderBlock
    liquidity_swept: bool
    bos: bool

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials missing.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        if r.status_code == 200:
            log.info("✅ Telegram sent.")
            return True
        log.error(f"Telegram {r.status_code}: {r.text}")
    except Exception as e:
        log.error(f"Telegram error: {e}")
    return False

def send_startup():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    send_telegram(
        f"🤖 <b>Gold Alert Bot v2 — Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {now}\n"
        f"📊 Symbol: XAUUSD\n"
        f"🔄 Check: Every {CHECK_INTERVAL//60} min\n"
        f"⏳ Cooldown: {COOLDOWN_MINUTES} min\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Bot is live and monitoring!"
    )

# ─────────────────────────────────────────────
# DATA FETCH — TwelveData (رایگان، 800 req/day، بدون مسدودی)
# ─────────────────────────────────────────────
TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY", "demo")

TD_INTERVALS = {
    "1d":  "1day",
    "1h":  "1h",
    "15m": "15min",
    "5m":  "5min",
}

def fetch_ohlcv(interval: str, bars: int = 200) -> Optional[pd.DataFrame]:
    """
    Fetch gold data from TwelveData — رایگان (نیاز به API key رایگان)
    Symbol: XAU/USD
    """
    td_interval = TD_INTERVALS.get(interval, "1day")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": td_interval,
        "outputsize": min(bars, 500),
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
    }

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15,
                              headers={"User-Agent": "Mozilla/5.0"})
            data = r.json()

            if "code" in data and data.get("code") != 200:
                raise ValueError(f"API error: {data.get('message', data)}")

            if "values" not in data:
                raise ValueError(f"No 'values' in response: {data}")

            df = pd.DataFrame(data["values"])
            df.rename(columns={
                "open": "Open", "high": "High",
                "low": "Low", "close": "Close",
            }, inplace=True)

            needed = ["Open", "High", "Low", "Close"]
            df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
            df = df.iloc[::-1].reset_index(drop=True)  # TwelveData چینش نزولی می‌دهد، برعکسش می‌کنیم

            if len(df) < 10:
                raise ValueError("Too few rows")

            log.info(f"✅ Fetched {len(df)} bars [{interval}] from TwelveData")
            return df

        except Exception as e:
            log.warning(f"TwelveData attempt {attempt+1}/3 [{interval}]: {e}")
            time.sleep(3)

    log.error(f"❌ All fetch attempts failed for [{interval}]")
    return None

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
def get_kill_zone() -> Optional[str]:
    now = datetime.now(timezone.utc)
    mins = now.hour * 60 + now.minute
    for name, (start, end) in KILL_ZONES.items():
        if start <= mins <= end:
            return name
    return None

def is_weekend() -> bool:
    return datetime.now(timezone.utc).weekday() >= 5

def market_bias(df: pd.DataFrame) -> str:
    if len(df) < 20:
        return "NEUTRAL"
    c = df["Close"]
    ema20 = c.ewm(span=20).mean().iloc[-1]
    ema50 = c.ewm(span=min(50, len(c)-1)).mean().iloc[-1]
    highs = df["High"].tail(5).values
    lows  = df["Low"].tail(5).values
    # اضافه: EMA200 برای تأیید روند بلندمدت
    ema200 = c.ewm(span=min(200, len(c)-1)).mean().iloc[-1]
    bull = sum([
        c.iloc[-1] > ema20,
        ema20 > ema50,
        c.iloc[-1] > ema200,
        all(highs[i] >= highs[i-1] for i in range(1,5)),
    ])
    bear = sum([
        c.iloc[-1] < ema20,
        ema20 < ema50,
        c.iloc[-1] < ema200,
        all(highs[i] <= highs[i-1] for i in range(1,5)),
    ])
    if bull >= 2: return "BULLISH"
    if bear >= 2: return "BEARISH"
    return "NEUTRAL"

def price_zone(df: pd.DataFrame) -> str:
    h = df["High"].tail(20).max()
    l = df["Low"].tail(20).min()
    eq = (h + l) / 2
    return "PREMIUM" if df["Close"].iloc[-1] > eq else "DISCOUNT"

def find_order_blocks(df: pd.DataFrame, direction: str) -> list:
    obs = []
    avg_body = (df["Close"] - df["Open"]).abs().mean()
    data = df.tail(40).reset_index(drop=True)
    for i in range(1, len(data) - 2):
        c = data.iloc[i]
        n = data.iloc[i+1]
        bearish = c["Close"] < c["Open"]
        bullish = c["Close"] > c["Open"]
        if direction == "bullish" and bearish:
            move = n["Close"] - c["Low"]
            if move > avg_body * 1.5 and n["Close"] > n["Open"]:
                obs.append(OrderBlock(
                    high=round(c["High"], 2),
                    low=round(c["Low"], 2),
                    mid=round((c["High"]+c["Low"])/2, 2),
                    direction="bullish",
                    strength=min(5, int(move/avg_body))
                ))
        elif direction == "bearish" and bullish:
            move = c["High"] - n["Close"]
            if move > avg_body * 1.5 and n["Close"] < n["Open"]:
                obs.append(OrderBlock(
                    high=round(c["High"], 2),
                    low=round(c["Low"], 2),
                    mid=round((c["High"]+c["Low"])/2, 2),
                    direction="bearish",
                    strength=min(5, int(move/avg_body))
                ))
    return obs

def price_in_ob(price: float, ob: OrderBlock) -> bool:
    buf = (ob.high - ob.low) * 0.1
    return (ob.low - buf) <= price <= (ob.high + buf)

def find_fvg(df: pd.DataFrame, direction: str) -> bool:
    data = df.tail(20).reset_index(drop=True)
    for i in range(1, len(data)-1):
        p, n = data.iloc[i-1], data.iloc[i+1]
        if direction == "bullish" and p["High"] < n["Low"]:
            return True
        if direction == "bearish" and p["Low"] > n["High"]:
            return True
    return False

def check_bos(df: pd.DataFrame, direction: str) -> bool:
    data = df.tail(20).reset_index(drop=True)
    if len(data) < 10:
        return False
    first, second = data.head(10), data.tail(10)
    if direction == "bullish":
        return second["Close"].max() > first["High"].max()
    return second["Close"].min() < first["Low"].min()

def check_liquidity(df: pd.DataFrame, direction: str) -> bool:
    data = df.tail(15).reset_index(drop=True)
    if len(data) < 5:
        return False
    if direction == "bullish":
        recent_low = data["Low"].min()
        for i in range(len(data)-4, len(data)-1):
            if data["Low"].iloc[i] <= recent_low * 1.001:
                if data["Close"].iloc[-1] > data["Low"].iloc[i] * 1.002:
                    return True
    else:
        recent_high = data["High"].max()
        for i in range(len(data)-4, len(data)-1):
            if data["High"].iloc[i] >= recent_high * 0.999:
                if data["Close"].iloc[-1] < data["High"].iloc[i] * 0.998:
                    return True
    return False

def calc_atr(df: pd.DataFrame, p: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h-l, (h-c).abs(), (l-c).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean().iloc[-1]

def calc_targets(direction: str, entry: float, ob: OrderBlock, atr: float) -> dict:
    buf = atr * 0.3
    if direction == "BUY":
        sl   = round(ob.low - buf, 2)
        risk = entry - sl
        return dict(sl=sl,
                    tp1=round(entry + risk*2.0, 2),
                    tp2=round(entry + risk*3.5, 2),
                    tp3=round(entry + risk*5.0, 2),
                    rr1=2.0, rr2=3.5)
    else:
        sl   = round(ob.high + buf, 2)
        risk = sl - entry
        return dict(sl=sl,
                    tp1=round(entry - risk*2.0, 2),
                    tp2=round(entry - risk*3.5, 2),
                    tp3=round(entry - risk*5.0, 2),
                    rr1=2.0, rr2=3.5)

def score_setup(bias, zone, kz, ob, liq, bos, fvg) -> int:
    s = 0
    if (bias=="BULLISH" and zone=="DISCOUNT") or (bias=="BEARISH" and zone=="PREMIUM"): s+=2
    if kz:   s+=2
    if fvg:  s+=2
    if bos:  s+=2
    if liq:  s+=1
    s += min(1, ob.strength//2)
    return s

# ─────────────────────────────────────────────
# ALERT FORMAT
# ─────────────────────────────────────────────
def format_alert(sig: Signal, price: float) -> str:
    e  = "🟢" if sig.direction=="BUY" else "🔴"
    be = "🐂" if sig.bias=="BULLISH" else "🐻"
    stars = "⭐" * (sig.score // 2)
    thr = (datetime.now(timezone.utc) + timedelta(hours=3,minutes=30)).strftime("%H:%M")
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    return (
        f"{e} <b>GOLD ALERT — {sig.direction}</b> {e}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>Price:</b> ${price:,.2f}\n\n"
        f"📍 <b>Entry Zone</b>\n"
        f"   High: ${sig.entry_high:,.2f}\n"
        f"   Low:  ${sig.entry_low:,.2f}\n\n"
        f"🛑 <b>Stop Loss:</b> ${sig.stop_loss:,.2f}\n\n"
        f"🎯 <b>Targets</b>\n"
        f"   TP1 (1:{sig.rr1}): ${sig.tp1:,.2f}\n"
        f"   TP2 (1:{sig.rr2}): ${sig.tp2:,.2f}\n"
        f"   TP3 (Swing):  ${sig.tp3:,.2f}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 <b>Setup Details</b>\n"
        f"   {be} Bias: {sig.bias}\n"
        f"   ⏰ Kill Zone: {sig.kill_zone}\n"
        f"   🧱 OB: {sig.ob.low} — {sig.ob.high}\n"
        f"   📐 FVG: {'✅' if sig.ob.has_fvg else '❌'}\n"
        f"   💧 Liquidity Swept: {'✅' if sig.liquidity_swept else '❌'}\n"
        f"   📉 BOS: {'✅' if sig.bos else '❌'}\n\n"
        f"🏆 <b>Score: {sig.score}/10</b> {stars}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ Max risk 1% | Move SL to BE after TP1\n"
        f"🕐 Tehran: {thr} | UTC: {utc}\n"
        f"<i>Confirm entry on 15M chart</i>"
    )

# ─────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────
class GoldBot:
    def __init__(self):
        self.last_alert: Optional[datetime] = None

    def can_alert(self) -> bool:
        if not self.last_alert:
            return True
        return (datetime.now(timezone.utc) - self.last_alert).seconds/60 >= COOLDOWN_MINUTES

    def analyze(self) -> Optional[Signal]:
        if is_weekend():
            log.info("Weekend — skipping.")
            return None

        kz = get_kill_zone()
        if not kz:
            log.info("Not in Kill Zone.")
            return None

        log.info(f"Kill Zone: {kz}")

        df_d  = fetch_ohlcv("1d",  200)
        df_1h = fetch_ohlcv("1h",  200)
        df_15 = fetch_ohlcv("15m", 100)

        if any(d is None for d in [df_d, df_1h, df_15]):
            log.error("Data fetch failed.")
            return None

        bias  = market_bias(df_d)
        zone  = price_zone(df_d)
        price = df_15["Close"].iloc[-1]

        log.info(f"Price: {price:.2f} | Bias: {bias} | Zone: {zone}")

        if bias == "NEUTRAL":
            log.info("Bias neutral — skip.")
            return None

        direction  = "BUY" if bias == "BULLISH" else "SELL"
        ob_dir     = "bullish" if direction == "BUY" else "bearish"

        # Zone filter نرم‌تر: فقط extreme مخالف را رد کن
        h20 = df_d["High"].tail(20).max()
        l20 = df_d["Low"].tail(20).min()
        rng = h20 - l20
        eq  = l20 + rng * 0.5
        extreme_premium  = price > l20 + rng * 0.75
        extreme_discount = price < l20 + rng * 0.25
        if direction == "BUY"  and extreme_premium:
            log.info("BUY but EXTREME PREMIUM — skip."); return None
        if direction == "SELL" and extreme_discount:
            log.info("SELL but EXTREME DISCOUNT — skip."); return None

        log.info(f"Zone OK — proceeding ({zone})")

        # OB روی 4H (هر 4 کندل 1H را ادغام می‌کنیم)
        df_4h = df_1h.groupby(df_1h.index // 4).agg(
            {"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
        obs_4h = find_order_blocks(df_4h, ob_dir)
        obs_1h = find_order_blocks(df_1h, ob_dir)
        obs = obs_4h + obs_1h  # 4H اول، 1H دوم
        if not obs:
            log.info("No OB found.")
            return None

        nearest = None
        for ob in obs:
            if price_in_ob(price, ob):
                if nearest is None or abs(ob.mid - price) < abs(nearest.mid - price):
                    nearest = ob

        if not nearest:
            log.info("Price not in any OB.")
            return None

        fvg  = find_fvg(df_1h, ob_dir)
        bos  = check_bos(df_15, ob_dir)
        liq  = check_liquidity(df_15, ob_dir)
        nearest.has_fvg = fvg

        sc = score_setup(bias, zone, kz, nearest, liq, bos, fvg)
        log.info(f"Score: {sc}/10 | FVG:{fvg} BOS:{bos} LIQ:{liq}")

        if sc < MIN_SCORE:
            log.info(f"Score {sc} < {MIN_SCORE} — skip.")
            return None

        atr  = calc_atr(df_1h)
        tgts = calc_targets(direction, price, nearest, atr)

        return Signal(
            direction=direction,
            entry_high=nearest.high,
            entry_low=nearest.low,
            stop_loss=tgts["sl"],
            tp1=tgts["tp1"], tp2=tgts["tp2"], tp3=tgts["tp3"],
            rr1=tgts["rr1"], rr2=tgts["rr2"],
            kill_zone=kz, bias=bias, score=sc,
            ob=nearest, liquidity_swept=liq, bos=bos,
        )

    def run(self):
        log.info("=" * 50)
        log.info("  Gold Alert Bot v2 — Starting")
        log.info("=" * 50)
        send_startup()

        while True:
            log.info("─" * 40)
            try:
                sig = self.analyze()
                if sig and self.can_alert():
                    msg = format_alert(sig, sig.entry_low)
                    if send_telegram(msg):
                        self.last_alert = datetime.now(timezone.utc)
                else:
                    log.info("No signal.")
            except Exception as e:
                log.error(f"Cycle error: {e}", exc_info=True)

            log.info(f"Next check in {CHECK_INTERVAL//60} min...")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    GoldBot().run()
