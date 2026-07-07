"""
Multi-Symbol Professional ICT Bot v7
=====================================
Strategy: Full ICT — Liquidity + OB + FVG + CHoCH + BOS
Timeframes: 4H → 1H → 15M → 5M
Score: 0-100 | Min: 75
R:R: 1:2 / 1:3.5 / 1:5 (Partial TP)
Symbols: XAUUSD EURUSD GBPUSD USDJPY AUDUSD USDCAD
"""
import os, time, logging, requests, sqlite3
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

# ── LOGGING ──────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("bot.log", encoding="utf-8")])
log = logging.getLogger("ICTBot")

# ── CONFIG ───────────────────────────────────
TG_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT   = os.environ.get("TELEGRAM_CHAT_ID", "")
INTERVAL  = int(os.environ.get("CHECK_INTERVAL", "300"))
COOLDOWN  = int(os.environ.get("COOLDOWN_MINUTES", "240"))
TD_KEY    = os.environ.get("TWELVEDATA_KEY", "demo")
MIN_SCORE = int(os.environ.get("MIN_SCORE", "75"))

SYMBOLS = {
    "XAUUSD": {"td":"XAU/USD","emoji":"🥇","dec":2},
    "EURUSD": {"td":"EUR/USD","emoji":"💶","dec":5},
    "GBPUSD": {"td":"GBP/USD","emoji":"💷","dec":5},
    "USDJPY": {"td":"USD/JPY","emoji":"💴","dec":3},
    "AUDUSD": {"td":"AUD/USD","emoji":"🦘","dec":5},
    "USDCAD": {"td":"USD/CAD","emoji":"🍁","dec":5},
}

# ── KILL ZONES (UTC) ─────────────────────────
KILL_ZONES = {
    "Asian Range":       (0*60,   2*60),
    "London Open":       (7*60,   9*60),
    "New York Open":     (13*60, 15*60),
    "London Close":      (16*60+30, 17*60+30),
}

def get_kill_zone() -> Optional[str]:
    now = datetime.now(timezone.utc)
    m = now.hour*60 + now.minute
    for name,(s,e) in KILL_ZONES.items():
        if s <= m <= e: return name
    return None

def is_weekend() -> bool:
    return datetime.now(timezone.utc).weekday() >= 5

# ── DATA CLASSES ─────────────────────────────
@dataclass
class SwingPoint:
    price: float
    index: int
    kind: str  # "high" | "low"

@dataclass
class OrderBlock:
    high: float
    low: float
    mid: float
    direction: str   # "bullish" | "bearish"
    displaced: bool = False
    mitigated: bool = False
    strength: int = 0

@dataclass
class FVG:
    high: float
    low: float
    mid: float
    direction: str
    size: float = 0.0
    filled: bool = False

@dataclass
class AnalysisResult:
    symbol: str
    direction: str
    price: float
    entry_high: float
    entry_low: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    score: int
    bias: str
    kill_zone: str
    reasons: list = field(default_factory=list)
    rejects: list = field(default_factory=list)

# ── DATABASE ─────────────────────────────────
def init_db():
    con = sqlite3.connect("journal.db")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT, price REAL,
        entry_high REAL, entry_low REAL,
        sl REAL, tp1 REAL, tp2 REAL, tp3 REAL,
        score INTEGER, bias TEXT, kill_zone TEXT,
        atr REAL, session TEXT, ts TEXT,
        outcome TEXT DEFAULT 'OPEN',
        exit_price REAL, exit_ts TEXT,
        r_mult REAL, duration_min INTEGER,
        hit_tp1 INTEGER DEFAULT 0,
        hit_tp2 INTEGER DEFAULT 0,
        reasons TEXT, rejects TEXT
    );
    CREATE TABLE IF NOT EXISTS stats_cache (
        key TEXT PRIMARY KEY, value TEXT, updated TEXT
    );
    """)
    con.commit(); con.close()

def save_signal(r: AnalysisResult, atr: float) -> int:
    now = datetime.now(timezone.utc)
    session = get_kill_zone() or "Off-Session"
    con = sqlite3.connect("journal.db")
    cur = con.execute("""INSERT INTO signals
        (symbol,direction,price,entry_high,entry_low,
         sl,tp1,tp2,tp3,score,bias,kill_zone,atr,session,ts,
         reasons,rejects)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (r.symbol,r.direction,r.price,r.entry_high,r.entry_low,
         r.sl,r.tp1,r.tp2,r.tp3,r.score,r.bias,r.kill_zone,
         atr,session,now.isoformat(),
         "|".join(r.reasons),"|".join(r.rejects)))
    con.commit(); sid=cur.lastrowid; con.close()
    return sid

def update_signal(sid, field_name, value):
    con = sqlite3.connect("journal.db")
    con.execute(f"UPDATE signals SET {field_name}=? WHERE id=?", (value,sid))
    con.commit(); con.close()

def close_trade(sid, outcome, exit_price, ts_open):
    now = datetime.now(timezone.utc)
    dur = int((now - ts_open).total_seconds() / 60)
    con = sqlite3.connect("journal.db")
    row = con.execute("SELECT price,sl FROM signals WHERE id=?", (sid,)).fetchone()
    if row:
        entry, sl = row
        risk = abs(entry - sl)
        r = round((exit_price - entry) / risk if risk > 0 else 0, 2)
        con.execute("""UPDATE signals SET outcome=?,exit_price=?,
            exit_ts=?,r_mult=?,duration_min=? WHERE id=?""",
            (outcome,exit_price,now.isoformat(),r,dur,sid))
    con.commit(); con.close()

def monthly_stats() -> dict:
    con = sqlite3.connect("journal.db")
    now = datetime.now(timezone.utc)
    start = now.replace(day=1,hour=0,minute=0,second=0).isoformat()
    rows = con.execute("""SELECT symbol,outcome,r_mult,kill_zone,
        duration_min,hit_tp1,hit_tp2 FROM signals WHERE ts >= ?""",
        (start,)).fetchall()
    con.close()

    total=len(rows); wins=losses=opens=0
    net=0.0; win_rs=[]; loss_rs=[]; durations=[]
    by_sym={}; by_kz={}

    for sym,outcome,rm,kz,dur,ht1,ht2 in rows:
        rm = rm or 0
        if outcome=="WIN":
            wins+=1; net+=rm; win_rs.append(rm)
        elif outcome=="LOSS":
            losses+=1; net+=rm; loss_rs.append(rm)
        else:
            opens+=1
        if dur: durations.append(dur)
        if sym not in by_sym: by_sym[sym]=[0,0]
        if outcome=="WIN": by_sym[sym][0]+=1
        elif outcome=="LOSS": by_sym[sym][1]+=1
        if kz not in by_kz: by_kz[kz]=[0,0]
        if outcome=="WIN": by_kz[kz][0]+=1
        elif outcome=="LOSS": by_kz[kz][1]+=1

    wr = round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0
    avg_win = round(sum(win_rs)/len(win_rs),2) if win_rs else 0
    avg_loss = round(sum(loss_rs)/len(loss_rs),2) if loss_rs else 0
    pf = round(abs(sum(win_rs)/sum(loss_rs)),2) if loss_rs and sum(loss_rs)!=0 else 0
    exp = round(wr/100*avg_win - (1-wr/100)*abs(avg_loss),2)
    avg_dur = round(sum(durations)/len(durations)) if durations else 0

    return dict(total=total,wins=wins,losses=losses,opens=opens,
                net=round(net,2),wr=wr,avg_win=avg_win,avg_loss=avg_loss,
                pf=pf,exp=exp,avg_dur=avg_dur,by_sym=by_sym,by_kz=by_kz)

# ── TELEGRAM ─────────────────────────────────
def tg(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT: return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id":TG_CHAT,"text":text,"parse_mode":"HTML"},
            timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.error(f"TG: {e}"); return False

# ── DATA LAYER ───────────────────────────────
_cache: dict = {}
TD_INT = {"4h":"4h","1h":"1h","15m":"15min","5m":"5min"}

def fetch(td_sym: str, interval: str, bars=150) -> Optional[pd.DataFrame]:
    key = f"{td_sym}_{interval}"
    for attempt in range(3):
        try:
            r = requests.get("https://api.twelvedata.com/time_series",
                params={"symbol":td_sym,"interval":TD_INT[interval],
                        "outputsize":bars,"apikey":TD_KEY,"format":"JSON"},
                timeout=15)
            data = r.json()
            if "values" not in data:
                raise ValueError(data.get("message","no values"))
            df = pd.DataFrame(data["values"])
            df.rename(columns={"open":"Open","high":"High",
                                "low":"Low","close":"Close"},inplace=True)
            df = df[["Open","High","Low","Close"]].apply(
                pd.to_numeric,errors="coerce").dropna()
            df = df.iloc[::-1].reset_index(drop=True)
            if len(df)<20: raise ValueError("too few rows")
            _cache[key]=df; return df
        except Exception as e:
            log.warning(f"{td_sym} {interval} attempt {attempt+1}: {e}")
            time.sleep(6)
    return _cache.get(key)

def get_price(td_sym: str) -> Optional[float]:
    try:
        r = requests.get("https://api.twelvedata.com/price",
            params={"symbol":td_sym,"apikey":TD_KEY},timeout=10)
        return float(r.json()["price"])
    except:
        key=f"{td_sym}_5m"; df=_cache.get(key)
        return float(df["Close"].iloc[-1]) if df is not None else None

# ── MARKET STRUCTURE ─────────────────────────
def find_swings(df: pd.DataFrame, left=3, right=3) -> tuple:
    """تشخیص Swing High و Swing Low واقعی"""
    highs=[]; lows=[]
    for i in range(left, len(df)-right):
        # Swing High
        if all(df["High"].iloc[i] > df["High"].iloc[i-j] for j in range(1,left+1)) and \
           all(df["High"].iloc[i] > df["High"].iloc[i+j] for j in range(1,right+1)):
            highs.append(SwingPoint(df["High"].iloc[i], i, "high"))
        # Swing Low
        if all(df["Low"].iloc[i] < df["Low"].iloc[i-j] for j in range(1,left+1)) and \
           all(df["Low"].iloc[i] < df["Low"].iloc[i+j] for j in range(1,right+1)):
            lows.append(SwingPoint(df["Low"].iloc[i], i, "low"))
    return highs, lows

def get_bias_4h(df: pd.DataFrame) -> str:
    """Bias از ساختار 4H — HH/HL یا LH/LL"""
    if len(df) < 30: return "NEUTRAL"
    _, lows = find_swings(df, left=3, right=3)
    highs, _ = find_swings(df, left=3, right=3)

    # EMA تأیید اضافه
    c = df["Close"]
    e50 = c.ewm(span=50).mean().iloc[-1]
    e200 = c.ewm(span=min(200,len(c)-1)).mean().iloc[-1]
    ema_bull = c.iloc[-1] > e50 > e200
    ema_bear = c.iloc[-1] < e50 < e200

    # HH/HL بررسی
    if len(highs)>=2 and len(lows)>=2:
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price
        if hh and hl and ema_bull: return "BULLISH"
        if lh and ll and ema_bear: return "BEARISH"

    if ema_bull: return "BULLISH"
    if ema_bear: return "BEARISH"
    return "NEUTRAL"

def detect_choch_bos(df: pd.DataFrame, bias: str) -> tuple:
    """
    CHoCH: تغییر ساختار (بازگشت روند)
    BOS: ادامه ساختار
    Returns: (choch: bool, bos: bool)
    """
    highs, lows = find_swings(df, left=2, right=2)
    if not highs or not lows: return False, False
    last_close = df["Close"].iloc[-1]

    if bias == "BULLISH":
        # BOS: شکست بالای آخرین Swing High
        bos = last_close > highs[-1].price if highs else False
        # CHoCH: شکست زیر آخرین Swing Low در روند صعودی
        choch = last_close < lows[-1].price if lows else False
        return choch, bos
    else:
        bos = last_close < lows[-1].price if lows else False
        choch = last_close > highs[-1].price if highs else False
        return choch, bos

# ── LIQUIDITY ENGINE ─────────────────────────
def detect_liquidity_sweep(df: pd.DataFrame, direction: str,
                            tolerance: float = 0.001) -> bool:
    """
    تشخیص Liquidity Sweep:
    قیمت Equal High/Low را لمس کرده و برگشته
    """
    h = df["High"]; l = df["Low"]; c = df["Close"]
    last = 30  # بررسی ۳۰ کندل اخیر

    if direction == "bullish":
        # SSL: Equal Lows را sweep کرده و برگشته بالا
        recent_lows = l.tail(last)
        min_low = recent_lows.min()
        # چک: آیا کندل‌های اخیر زیر کف رفته و برگشته
        for i in range(len(df)-5, len(df)-1):
            if l.iloc[i] <= min_low * (1 + tolerance):
                if c.iloc[-1] > l.iloc[i] * (1 + tolerance*2):
                    return True
    else:
        # BSL: Equal Highs را sweep کرده و برگشته پایین
        recent_highs = h.tail(last)
        max_high = recent_highs.max()
        for i in range(len(df)-5, len(df)-1):
            if h.iloc[i] >= max_high * (1 - tolerance):
                if c.iloc[-1] < h.iloc[i] * (1 - tolerance*2):
                    return True
    return False

def detect_equal_levels(df: pd.DataFrame, direction: str,
                         tolerance: float = 0.0005) -> bool:
    """Equal Highs یا Equal Lows — نشان‌دهنده Liquidity"""
    if direction == "bullish":
        lows = df["Low"].tail(20).values
        for i in range(len(lows)-1):
            for j in range(i+1, len(lows)):
                if abs(lows[i]-lows[j])/lows[i] < tolerance:
                    return True
    else:
        highs = df["High"].tail(20).values
        for i in range(len(highs)-1):
            for j in range(i+1, len(highs)):
                if abs(highs[i]-highs[j])/highs[i] < tolerance:
                    return True
    return False

# ── ORDER BLOCK (حرفه‌ای) ────────────────────
def find_valid_ob(df: pd.DataFrame, direction: str) -> Optional[OrderBlock]:
    """
    OB معتبر ICT:
    1. کندل مخالف قبل از Displacement
    2. Displacement: حرکت قوی بعد از OB (> 1.5x ATR)
    3. هنوز Mitigate نشده
    """
    o=df["Open"]; h=df["High"]; l=df["Low"]; c=df["Close"]
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean()

    for i in range(len(c)-3, 1, -1):
        # Displacement: حرکت i+1 بزرگتر از 1.5x ATR
        displacement = abs(c.iloc[i+1]-o.iloc[i+1])
        if displacement < 1.5 * atr.iloc[i]: continue

        if direction=="bullish":
            # OB: آخرین کندل نزولی قبل از حرکت صعودی قوی
            if c.iloc[i] < o.iloc[i] and c.iloc[i+1] > o.iloc[i+1]:
                ob_high = h.iloc[i]
                ob_low = l.iloc[i]
                # چک Mitigation: آیا قیمت بعداً به OB برگشته و آن را پر کرده؟
                mitigated = False
                for j in range(i+2, len(c)):
                    if l.iloc[j] < ob_low:
                        mitigated = True; break
                strength = min(5, int(displacement/atr.iloc[i]))
                return OrderBlock(round(ob_high,6),round(ob_low,6),
                    round((ob_high+ob_low)/2,6),"bullish",True,mitigated,strength)

        else:
            # OB: آخرین کندل صعودی قبل از حرکت نزولی قوی
            if c.iloc[i] > o.iloc[i] and c.iloc[i+1] < o.iloc[i+1]:
                ob_high = h.iloc[i]
                ob_low = l.iloc[i]
                mitigated = False
                for j in range(i+2, len(c)):
                    if h.iloc[j] > ob_high:
                        mitigated = True; break
                strength = min(5, int(displacement/atr.iloc[i]))
                return OrderBlock(round(ob_high,6),round(ob_low,6),
                    round((ob_high+ob_low)/2,6),"bearish",True,mitigated,strength)
    return None

def price_in_ob(price: float, ob: OrderBlock) -> bool:
    buf = (ob.high - ob.low) * 0.1
    return (ob.low - buf) <= price <= (ob.high + buf)

# ── FAIR VALUE GAP (حرفه‌ای) ─────────────────
def find_valid_fvg(df: pd.DataFrame, direction: str,
                    min_size_atr: float = 0.3) -> Optional[FVG]:
    """
    FVG معتبر:
    1. اندازه حداقل 0.3x ATR
    2. هنوز پر نشده
    3. در جهت Bias
    """
    h=df["High"]; l=df["Low"]
    tr=pd.concat([h-l,(h-df["Close"].shift()).abs(),
                  (l-df["Close"].shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean().iloc[-1]
    min_size = atr * min_size_atr

    for i in range(len(df)-2, 1, -1):
        if direction=="bullish":
            gap_low = h.iloc[i-1]
            gap_high = l.iloc[i+1]
            if gap_high > gap_low and (gap_high-gap_low) >= min_size:
                # چک: پر نشده
                filled = any(l.iloc[j] < gap_low for j in range(i+1,len(df)))
                size = gap_high - gap_low
                return FVG(round(gap_high,6),round(gap_low,6),
                    round((gap_high+gap_low)/2,6),"bullish",size,filled)
        else:
            gap_high = l.iloc[i-1]
            gap_low = h.iloc[i+1]
            if gap_high > gap_low and (gap_high-gap_low) >= min_size:
                filled = any(h.iloc[j] > gap_high for j in range(i+1,len(df)))
                size = gap_high - gap_low
                return FVG(round(gap_high,6),round(gap_low,6),
                    round((gap_high+gap_low)/2,6),"bearish",size,filled)
    return None

# ── SCORE ENGINE (0-100) ──────────────────────
def calculate_score(bias_ok, kz, liq_sweep, eq_levels,
                    ob, fvg, bos, choch) -> tuple:
    """
    امتیازدهی 0-100 با وزن‌های ICT واقعی
    Returns: (score, reasons, rejects)
    """
    score = 0
    reasons = []
    rejects = []

    # Bias 4H (20 امتیاز)
    if bias_ok:
        score += 20; reasons.append("✅ Bias 4H")
    else:
        rejects.append("❌ Bias 4H")

    # Kill Zone (15 امتیاز)
    if kz:
        score += 15; reasons.append(f"✅ {kz}")
    else:
        rejects.append("❌ Kill Zone")

    # Liquidity Sweep (20 امتیاز)
    if liq_sweep:
        score += 20; reasons.append("✅ Liquidity Sweep")
    else:
        rejects.append("❌ Liquidity Sweep")

    # Equal Levels (5 امتیاز)
    if eq_levels:
        score += 5; reasons.append("✅ Equal Levels")

    # Order Block (20 امتیاز)
    if ob and not ob.mitigated:
        pts = 10 + min(10, ob.strength*2)
        score += pts; reasons.append(f"✅ OB (str:{ob.strength})")
    elif ob and ob.mitigated:
        rejects.append("❌ OB Mitigated")
    else:
        rejects.append("❌ No OB")

    # FVG (10 امتیاز)
    if fvg and not fvg.filled:
        score += 10; reasons.append(f"✅ FVG ({fvg.size:.5f})")
    elif fvg and fvg.filled:
        rejects.append("❌ FVG Filled")
    else:
        rejects.append("❌ No FVG")

    # BOS/CHoCH (10 امتیاز)
    if bos:
        score += 10; reasons.append("✅ BOS")
    elif choch:
        score += 7; reasons.append("✅ CHoCH")
    else:
        rejects.append("❌ No BOS/CHoCH")

    return min(score, 100), reasons, rejects

# ── ATR ──────────────────────────────────────
def calc_atr(df: pd.DataFrame, p=14) -> float:
    h,l,c=df["High"],df["Low"],df["Close"].shift(1)
    return pd.concat([h-l,(h-c).abs(),(l-c).abs()],
        axis=1).max(axis=1).rolling(p).mean().iloc[-1]

# ── ALERTS ───────────────────────────────────
def fmt(p, dec): return f"{p:.{dec}f}"
def thr_now():
    return (datetime.now(timezone.utc)+timedelta(hours=3,minutes=30)).strftime("%H:%M")

def alert_entry(r: AnalysisResult, cfg: dict, sid: int):
    e="🟢" if r.direction=="BUY" else "🔴"
    d=cfg["dec"]; em=cfg["emoji"]
    risk=abs(r.price-r.sl)
    rr1=round(abs(r.tp1-r.price)/risk,1) if risk>0 else 0
    rr3=round(abs(r.tp3-r.price)/risk,1) if risk>0 else 0
    reasons_txt="\n".join(f"   {x}" for x in r.reasons)
    tg(
        f"{e} <b>{em} {r.symbol} — {r.direction}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price: {fmt(r.price,d)}\n"
        f"📍 Entry: {fmt(r.entry_low,d)} – {fmt(r.entry_high,d)}\n"
        f"🛑 SL: {fmt(r.sl,d)}\n\n"
        f"🎯 TP1 (1:{rr1}): {fmt(r.tp1,d)} → 30%\n"
        f"🎯 TP2 (1:3.5): {fmt(r.tp2,d)} → 30%\n"
        f"🎯 TP3 (1:{rr3}): {fmt(r.tp3,d)} → 40%\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 Analysis:\n{reasons_txt}\n\n"
        f"🏆 Score: {r.score}/100\n"
        f"⏰ {r.kill_zone}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ Risk max 1% | Move SL→BE after TP1\n"
        f"🕐 Tehran: {thr_now()} | 🔖 #{sid}"
    )

def alert_tp(sym, cfg, tp_num, price, r, sid):
    tg(f"🎯 <b>{cfg['emoji']} {sym} TP{tp_num} HIT! 🎉</b>\n"
       f"Exit: {fmt(price,cfg['dec'])}\n"
       f"<b>Result: {r:+.1f}R ✅</b>\n"
       f"{'⚡ Move SL to Break Even!' if tp_num==1 else ''}\n"
       f"🔖 #{sid}")

def alert_sl(sym, cfg, price, r, sid):
    tg(f"🛑 <b>{cfg['emoji']} {sym} SL HIT</b>\n"
       f"Exit: {fmt(price,cfg['dec'])}\n"
       f"<b>Result: {r:+.1f}R ❌</b>\n"
       f"🔖 #{sid}")

def alert_be(sym, cfg, sid):
    tg(f"⚡ <b>{cfg['emoji']} {sym} — Move SL to BE</b>\n"
       f"TP1 hit — protect your trade!\n"
       f"🔖 #{sid}")

def alert_monthly(s: dict):
    now=datetime.now(timezone.utc)
    sym_lines="\n".join(
        f"   {sym}: {v[0]}W/{v[1]}L"
        for sym,v in s["by_sym"].items())
    kz_lines="\n".join(
        f"   {kz}: {v[0]}W/{v[1]}L"
        for kz,v in s["by_kz"].items() if kz)
    tg(
        f"📊 <b>MONTHLY REPORT — {now.strftime('%B %Y')}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Total: {s['total']} | ✅{s['wins']} ❌{s['losses']} ⏳{s['opens']}\n"
        f"🏆 Win Rate: {s['wr']}%\n"
        f"💰 Net R: {s['net']:+.1f}R\n"
        f"📊 Profit Factor: {s['pf']}\n"
        f"⚡ Expectancy: {s['exp']:+.2f}R\n"
        f"📈 Avg Win: {s['avg_win']:+.1f}R\n"
        f"📉 Avg Loss: {s['avg_loss']:+.1f}R\n"
        f"⏱ Avg Duration: {s['avg_dur']} min\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 By Symbol:\n{sym_lines}\n"
        f"⏰ By Session:\n{kz_lines}\n"
        f"{'📈 Profitable! 🎉' if s['net']>0 else '📉 Keep going! 💪'}"
    )

# ── TRACKER ──────────────────────────────────
class Tracker:
    def __init__(self):
        self.open: dict = {}  # sid -> (sym,dir,entry,sl,tp1,tp2,tp3,ts_open)

    def add(self, sid, sym, direction, price, sl, tp1, tp2, tp3):
        self.open[sid]=(sym,direction,price,sl,tp1,tp2,tp3,
                        datetime.now(timezone.utc))

    def check_all(self):
        closed=[]
        for sid,(sym,direction,entry,sl,tp1,tp2,tp3,ts) in self.open.items():
            cfg=SYMBOLS[sym]
            price=get_price(cfg["td"])
            if price is None: continue
            risk=abs(entry-sl)
            if risk==0: continue

            if direction=="BUY":
                if price<=sl:
                    r=round((price-entry)/risk,2)
                    close_trade(sid,"LOSS",price,ts)
                    alert_sl(sym,cfg,price,r,sid)
                    closed.append(sid)
                elif price>=tp3:
                    r=round((price-entry)/risk,2)
                    close_trade(sid,"WIN",price,ts)
                    alert_tp(sym,cfg,3,price,r,sid)
                    closed.append(sid)
                elif price>=tp2:
                    update_signal(sid,"hit_tp2",1)
                    alert_tp(sym,cfg,2,price,round((price-entry)/risk,2),sid)
                elif price>=tp1:
                    update_signal(sid,"hit_tp1",1)
                    alert_tp(sym,cfg,1,price,round((price-entry)/risk,2),sid)
                    alert_be(sym,cfg,sid)
            else:
                if price>=sl:
                    r=round((entry-price)/risk,2)
                    close_trade(sid,"LOSS",price,ts)
                    alert_sl(sym,cfg,price,r,sid)
                    closed.append(sid)
                elif price<=tp3:
                    r=round((entry-price)/risk,2)
                    close_trade(sid,"WIN",price,ts)
                    alert_tp(sym,cfg,3,price,r,sid)
                    closed.append(sid)
                elif price<=tp2:
                    update_signal(sid,"hit_tp2",1)
                    alert_tp(sym,cfg,2,price,round((entry-price)/risk,2),sid)
                elif price<=tp1:
                    update_signal(sid,"hit_tp1",1)
                    alert_tp(sym,cfg,1,price,round((entry-price)/risk,2),sid)
                    alert_be(sym,cfg,sid)
            time.sleep(2)
        for sid in closed:
            del self.open[sid]

# ── MAIN ANALYSIS ────────────────────────────
class ICTBot:
    def __init__(self):
        init_db()
        self.last_alerts: dict={}
        self.tracker=Tracker()
        self.last_monthly: Optional[datetime]=None
        self.cycle=0

    def can_alert(self, sym: str) -> bool:
        t=self.last_alerts.get(sym)
        if not t: return True
        return (datetime.now(timezone.utc)-t).seconds//60>=COOLDOWN

    def analyze(self, sym: str, cfg: dict, kz: Optional[str]) -> Optional[AnalysisResult]:
        td=cfg["td"]; dec=cfg["dec"]

        # Fetch all timeframes
        df_4h=fetch(td,"4h",100)
        df_1h=fetch(td,"1h",150)
        df_15=fetch(td,"15m",100)
        df_5m=fetch(td,"5m",50)
        if any(d is None for d in [df_4h,df_1h,df_15]):
            log.error(f"{sym}: Data fetch failed"); return None

        price=get_price(td)
        if price is None: return None

        # 1. Bias از 4H
        bias=get_bias_4h(df_4h)
        if bias=="NEUTRAL":
            log.info(f"{sym}: NEUTRAL bias"); return None

        direction="BUY" if bias=="BULLISH" else "SELL"
        od="bullish" if direction=="BUY" else "bearish"

        # 2. Liquidity
        liq_sweep=detect_liquidity_sweep(df_1h,od)
        eq_levels=detect_equal_levels(df_1h,od)

        # 3. OB روی 1H
        ob=find_valid_ob(df_1h,od)
        if ob is None or ob.mitigated:
            log.info(f"{sym}: No valid OB")

        # 4. FVG روی 1H
        fvg=find_valid_fvg(df_1h,od)

        # 5. BOS/CHoCH روی 15M
        choch,bos=detect_choch_bos(df_15,bias)

        # 6. Score
        score,reasons,rejects=calculate_score(
            True,kz,liq_sweep,eq_levels,ob,fvg,bos,choch)

        # لاگ دقیق
        log.info(f"{sym}: {direction} Score:{score}/100")
        log.info(f"  ✓ {' | '.join(reasons)}")
        if rejects: log.info(f"  ✗ {' | '.join(rejects)}")

        if score < MIN_SCORE:
            log.info(f"{sym}: Score {score} < {MIN_SCORE} — skip")
            return None

        if ob is None or not price_in_ob(price,ob):
            log.info(f"{sym}: Price not in OB"); return None

        # 7. محاسبه اهداف
        atr=calc_atr(df_1h); buf=atr*0.3
        if direction=="BUY":
            sl=round(ob.low-buf,dec)
            risk=price-sl
            tp1=round(price+risk*2.0,dec)
            tp2=round(price+risk*3.5,dec)
            tp3=round(price+risk*5.0,dec)
        else:
            sl=round(ob.high+buf,dec)
            risk=sl-price
            tp1=round(price-risk*2.0,dec)
            tp2=round(price-risk*3.5,dec)
            tp3=round(price-risk*5.0,dec)

        if risk<=0: return None

        return AnalysisResult(
            symbol=sym,direction=direction,price=price,
            entry_high=ob.high,entry_low=ob.low,
            sl=sl,tp1=tp1,tp2=tp2,tp3=tp3,
            score=score,bias=bias,kill_zone=kz or "Off-KZ",
            reasons=reasons,rejects=rejects
        )

    def check_monthly(self):
        now=datetime.now(timezone.utc)
        if now.day==1 and now.hour==8:
            if not self.last_monthly or (now-self.last_monthly).days>=28:
                alert_monthly(monthly_stats())
                self.last_monthly=now

    def run(self):
        log.info("="*50)
        log.info("  ICT Bot v7 — Professional")
        log.info("="*50)
        syms=", ".join(SYMBOLS.keys())
        tg(f"🤖 <b>ICT Bot v7 — Professional</b>\n"
           f"📊 {syms}\n"
           f"⚡ Full ICT: Liquidity+OB+FVG+BOS+CHoCH\n"
           f"🎯 Min Score: {MIN_SCORE}/100 | R:R 1:5\n"
           f"✅ Active!")

        while True:
            self.cycle+=1
            log.info(f"══ Cycle {self.cycle} ══")
            try:
                if self.tracker.open:
                    self.tracker.check_all()

                if is_weekend():
                    log.info("Weekend."); time.sleep(INTERVAL); continue

                kz=get_kill_zone()
                if not kz:
                    log.info("Not in Kill Zone.")
                else:
                    log.info(f"Kill Zone: {kz}")
                    for sym,cfg in SYMBOLS.items():
                        if not self.can_alert(sym): continue
                        try:
                            result=self.analyze(sym,cfg,kz)
                            if result:
                                atr=calc_atr(fetch(cfg["td"],"1h",50) or pd.DataFrame())
                                sid=save_signal(result,atr if not pd.isna(atr) else 0)
                                alert_entry(result,cfg,sid)
                                self.tracker.add(sid,sym,result.direction,
                                    result.price,result.sl,
                                    result.tp1,result.tp2,result.tp3)
                                self.last_alerts[sym]=datetime.now(timezone.utc)
                                log.info(f"✅ {sym} #{sid} sent!")
                            time.sleep(8)
                        except Exception as e:
                            log.error(f"{sym}: {e}")

                self.check_monthly()
            except Exception as e:
                log.error(f"Cycle: {e}",exc_info=True)

            log.info(f"Next in {INTERVAL//60}min...")
            time.sleep(INTERVAL)

if __name__=="__main__":
    ICTBot().run()
