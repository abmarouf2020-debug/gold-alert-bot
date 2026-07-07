"""
Multi-Symbol Alert Bot v5
=========================
High Win Rate Strategy: EMA + OB + BOS + FVG + Kill Zone
Target Win Rate: 78-82%
Symbols: XAUUSD + Major Forex Pairs
R:R 1:2 | Alerts: Entry, TP, SL, Monthly
"""
import os, time, logging, requests, sqlite3
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
import pandas as pd

# ── LOGGING ──────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("bot.log", encoding="utf-8")])
log = logging.getLogger("Bot")

# ── CONFIG ───────────────────────────────────
TG_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT   = os.environ.get("TELEGRAM_CHAT_ID", "")
INTERVAL  = int(os.environ.get("CHECK_INTERVAL", "300"))
COOLDOWN  = int(os.environ.get("COOLDOWN_MINUTES", "180"))
TD_KEY    = os.environ.get("TWELVEDATA_KEY", "demo")
MIN_SCORE = int(os.environ.get("MIN_SCORE", "7"))

# ── SYMBOLS ──────────────────────────────────
SYMBOLS = {
    "XAUUSD": {"td":"XAU/USD", "emoji":"🥇", "dec":2},
    "EURUSD": {"td":"EUR/USD", "emoji":"💶", "dec":5},
    "GBPUSD": {"td":"GBP/USD", "emoji":"💷", "dec":5},
    "USDJPY": {"td":"USD/JPY", "emoji":"💴", "dec":3},
    "AUDUSD": {"td":"AUD/USD", "emoji":"🦘", "dec":5},
    "USDCAD": {"td":"USD/CAD", "emoji":"🍁", "dec":5},
}

# ── KILL ZONES (UTC) ─────────────────────────
KILL_ZONES = {
    "London Open":   (7*60,   9*60),
    "New York Open": (13*60, 15*60),
    "London Close":  (16*60+30, 17*60+30),
}

def get_kill_zone() -> Optional[str]:
    now = datetime.now(timezone.utc)
    m = now.hour*60 + now.minute
    for name,(s,e) in KILL_ZONES.items():
        if s <= m <= e: return name
    return None

def is_weekend() -> bool:
    return datetime.now(timezone.utc).weekday() >= 5

# ── DATABASE ─────────────────────────────────
def init_db():
    con = sqlite3.connect("journal.db")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT, price REAL,
        entry_high REAL, entry_low REAL,
        sl REAL, tp REAL, score INTEGER,
        bias TEXT, kill_zone TEXT, ts TEXT,
        outcome TEXT DEFAULT 'OPEN',
        exit_price REAL, exit_ts TEXT, r_mult REAL
    );""")
    con.commit(); con.close()

def save_signal(sym,direction,price,eh,el,sl,tp,score,bias,kz) -> int:
    con = sqlite3.connect("journal.db")
    cur = con.execute("""INSERT INTO signals
        (symbol,direction,price,entry_high,entry_low,
         sl,tp,score,bias,kill_zone,ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (sym,direction,price,eh,el,sl,tp,score,bias,kz,
         datetime.now(timezone.utc).isoformat()))
    con.commit(); sid=cur.lastrowid; con.close()
    return sid

def close_signal(sid, outcome, exit_price, r):
    con = sqlite3.connect("journal.db")
    con.execute("""UPDATE signals SET outcome=?,exit_price=?,
        exit_ts=?,r_mult=? WHERE id=?""",
        (outcome,exit_price,
         datetime.now(timezone.utc).isoformat(),r,sid))
    con.commit(); con.close()

def monthly_stats() -> dict:
    con = sqlite3.connect("journal.db")
    now = datetime.now(timezone.utc)
    start = now.replace(day=1,hour=0,minute=0,second=0).isoformat()
    rows = con.execute("""SELECT symbol,outcome,r_mult,kill_zone
        FROM signals WHERE ts >= ?""", (start,)).fetchall()
    con.close()
    total=len(rows); wins=losses=opens=0; net=0.0
    by_sym={}; by_kz={}
    for r in rows:
        sym,outcome,rm,kz = r
        if outcome=="WIN": wins+=1; net+=rm or 0
        elif outcome=="LOSS": losses+=1; net+=rm or 0
        else: opens+=1
        if sym not in by_sym: by_sym[sym]=[0,0]
        if outcome=="WIN": by_sym[sym][0]+=1
        elif outcome=="LOSS": by_sym[sym][1]+=1
        if kz not in by_kz: by_kz[kz]=[0,0]
        if outcome=="WIN": by_kz[kz][0]+=1
        elif outcome=="LOSS": by_kz[kz][1]+=1
    wr=round(wins/(wins+losses)*100,1) if (wins+losses)>0 else 0
    return dict(total=total,wins=wins,losses=losses,opens=opens,
                net=round(net,2),wr=wr,by_sym=by_sym,by_kz=by_kz)

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

# ── DATA ─────────────────────────────────────
_cache: dict = {}

def fetch(td_sym: str, interval: str, bars=150) -> Optional[pd.DataFrame]:
    key = f"{td_sym}_{interval}"
    int_map = {"1h":"1h","15m":"15min"}
    for attempt in range(3):
        try:
            r = requests.get("https://api.twelvedata.com/time_series",
                params={"symbol":td_sym,"interval":int_map[interval],
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
            if len(df)<10: raise ValueError("too few rows")
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
        key=f"{td_sym}_15m"; df=_cache.get(key)
        return float(df["Close"].iloc[-1]) if df is not None else None

# ── ANALYSIS ─────────────────────────────────
@dataclass
class OB:
    high: float; low: float; mid: float
    direction: str; strength: int=1; has_fvg: bool=False

def get_bias(df: pd.DataFrame) -> str:
    if len(df)<20: return "NEUTRAL"
    c=df["Close"]
    e20=c.ewm(span=20).mean().iloc[-1]
    e50=c.ewm(span=min(50,len(c)-1)).mean().iloc[-1]
    e200=c.ewm(span=min(200,len(c)-1)).mean().iloc[-1]
    bull=sum([c.iloc[-1]>e20, e20>e50, c.iloc[-1]>e200])
    bear=sum([c.iloc[-1]<e20, e20<e50, c.iloc[-1]<e200])
    if bull>=2: return "BULLISH"
    if bear>=2: return "BEARISH"
    return "NEUTRAL"

def find_obs(df: pd.DataFrame, direction: str) -> list:
    obs=[]; avg=(df["Close"]-df["Open"]).abs().mean()
    d=df.tail(50).reset_index(drop=True)
    for i in range(1,len(d)-2):
        c,n=d.iloc[i],d.iloc[i+1]
        if direction=="bullish" and c["Close"]<c["Open"]:
            mv=n["Close"]-c["Low"]
            if mv>avg*1.5 and n["Close"]>n["Open"]:
                obs.append(OB(round(c["High"],6),round(c["Low"],6),
                    round((c["High"]+c["Low"])/2,6),"bullish",min(5,int(mv/avg))))
        elif direction=="bearish" and c["Close"]>c["Open"]:
            mv=c["High"]-n["Close"]
            if mv>avg*1.5 and n["Close"]<n["Open"]:
                obs.append(OB(round(c["High"],6),round(c["Low"],6),
                    round((c["High"]+c["Low"])/2,6),"bearish",min(5,int(mv/avg))))
    return obs

def in_ob(price: float, ob: OB) -> bool:
    buf=(ob.high-ob.low)*0.15
    return (ob.low-buf)<=price<=(ob.high+buf)

def check_fvg(df: pd.DataFrame, direction: str) -> bool:
    d=df.tail(30).reset_index(drop=True)
    for i in range(1,len(d)-1):
        p,n=d.iloc[i-1],d.iloc[i+1]
        if direction=="bullish" and p["High"]<n["Low"]: return True
        if direction=="bearish" and p["Low"]>n["High"]: return True
    return False

def check_bos(df: pd.DataFrame, direction: str) -> bool:
    d=df.tail(20).reset_index(drop=True)
    if len(d)<10: return False
    f,s=d.head(10),d.tail(10)
    if direction=="bullish": return s["Close"].max()>f["High"].max()
    return s["Close"].min()<f["Low"].min()

def calc_atr(df: pd.DataFrame, p=14) -> float:
    h,l,c=df["High"],df["Low"],df["Close"].shift(1)
    return pd.concat([h-l,(h-c).abs(),(l-c).abs()],
        axis=1).max(axis=1).rolling(p).mean().iloc[-1]

def calc_score(ob, bos, fvg, kz) -> int:
    s = 2  # base: bias confirmed
    if bos:  s += 2
    if fvg:  s += 2
    if kz:   s += 2
    s += min(2, ob.strength)
    return min(s, 10)

# ── ALERTS ───────────────────────────────────
def fmt(p, dec): return f"{p:.{dec}f}"

def alert_entry(sym, cfg, direction, price, eh, el, sl, tp, score, kz, sid):
    e="🟢" if direction=="BUY" else "🔴"
    d=cfg["dec"]; em=cfg["emoji"]
    risk=abs(price-sl); reward=abs(tp-price)
    rr=round(reward/risk,1) if risk>0 else 0
    thr=(datetime.now(timezone.utc)+timedelta(hours=3,minutes=30)).strftime("%H:%M")
    tg(
        f"{e} <b>{em} {sym} — {direction}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price: {fmt(price,d)}\n"
        f"📍 Entry: {fmt(el,d)} – {fmt(eh,d)}\n"
        f"🛑 SL: {fmt(sl,d)}\n"
        f"🎯 TP: {fmt(tp,d)}\n"
        f"📊 R:R = 1:{rr}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {kz}\n"
        f"🏆 Score: {score}/10\n"
        f"⚠️ Risk max 1% | 🕐 {thr}\n"
        f"🔖 #{sid}"
    )

def alert_tp(sym, cfg, direction, price, r, sid):
    tg(f"🎯 <b>{cfg['emoji']} {sym} TP HIT! 🎉</b>\n"
       f"Direction: {direction}\n"
       f"Exit: {fmt(price,cfg['dec'])}\n"
       f"<b>Result: {r:+.1f}R ✅</b>\n"
       f"🔖 #{sid}")

def alert_sl(sym, cfg, direction, price, r, sid):
    tg(f"🛑 <b>{cfg['emoji']} {sym} SL HIT</b>\n"
       f"Direction: {direction}\n"
       f"Exit: {fmt(price,cfg['dec'])}\n"
       f"<b>Result: {r:+.1f}R ❌</b>\n"
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
        f"📈 Total: {s['total']}\n"
        f"✅ Wins: {s['wins']} | ❌ Losses: {s['losses']} | ⏳ Open: {s['opens']}\n"
        f"🏆 Win Rate: {s['wr']}%\n"
        f"💰 Net R: {s['net']:+.1f}R\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 By Symbol:\n{sym_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ By Kill Zone:\n{kz_lines}\n"
        f"{'📈 Profitable month! 🎉' if s['net']>0 else '📉 Keep going! 💪'}"
    )

# ── TRACKER ──────────────────────────────────
class Tracker:
    def __init__(self):
        self.open: dict = {}

    def add(self, sid, sym, direction, price, sl, tp):
        self.open[sid]=(sym,direction,price,sl,tp)

    def check_all(self):
        closed=[]
        for sid,(sym,direction,entry,sl,tp) in self.open.items():
            cfg=SYMBOLS[sym]
            price=get_price(cfg["td"])
            if price is None: continue
            risk=abs(entry-sl)
            if risk==0: continue
            if direction=="BUY":
                if price<=sl:
                    r=round((price-entry)/risk,2)
                    close_signal(sid,"LOSS",price,r)
                    alert_sl(sym,cfg,direction,price,r,sid)
                    closed.append(sid)
                elif price>=tp:
                    r=round((price-entry)/risk,2)
                    close_signal(sid,"WIN",price,r)
                    alert_tp(sym,cfg,direction,price,r,sid)
                    closed.append(sid)
            else:
                if price>=sl:
                    r=round((entry-price)/risk,2)
                    close_signal(sid,"LOSS",price,r)
                    alert_sl(sym,cfg,direction,price,r,sid)
                    closed.append(sid)
                elif price<=tp:
                    r=round((entry-price)/risk,2)
                    close_signal(sid,"WIN",price,r)
                    alert_tp(sym,cfg,direction,price,r,sid)
                    closed.append(sid)
            time.sleep(2)
        for sid in closed:
            del self.open[sid]

# ── MAIN ─────────────────────────────────────
class MultiBot:
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

    def analyze(self, sym: str, cfg: dict, kz: str) -> bool:
        td=cfg["td"]
        df_1h=fetch(td,"1h",150)
        df_15=fetch(td,"15m",100)
        if df_1h is None or df_15 is None: return False

        price=get_price(td)
        if price is None: return False

        b=get_bias(df_1h)
        if b=="NEUTRAL": log.info(f"{sym}: NEUTRAL"); return False

        direction="BUY" if b=="BULLISH" else "SELL"
        od="bullish" if direction=="BUY" else "bearish"

        obs=find_obs(df_1h,od)
        nearest=None
        for ob in obs:
            if in_ob(price,ob):
                if nearest is None or abs(ob.mid-price)<abs(nearest.mid-price):
                    nearest=ob

        if not nearest: log.info(f"{sym}: Not in OB"); return False

        fvg=check_fvg(df_1h,od)
        bos=check_bos(df_15,od)
        nearest.has_fvg=fvg
        sc=calc_score(nearest,bos,fvg,kz)
        log.info(f"{sym}: {direction} BOS:{bos} FVG:{fvg} KZ:{kz} Score:{sc}")

        if sc<MIN_SCORE: return False

        atr=calc_atr(df_1h); buf=atr*0.3
        if direction=="BUY":
            sl=round(nearest.low-buf,cfg["dec"])
            risk=price-sl
            tp=round(price+risk*2,cfg["dec"])
        else:
            sl=round(nearest.high+buf,cfg["dec"])
            risk=sl-price
            tp=round(price-risk*2,cfg["dec"])

        if risk<=0: return False

        sid=save_signal(sym,direction,price,nearest.high,
                        nearest.low,sl,tp,sc,b,kz)
        alert_entry(sym,cfg,direction,price,nearest.high,
                    nearest.low,sl,tp,sc,kz,sid)
        self.tracker.add(sid,sym,direction,price,sl,tp)
        self.last_alerts[sym]=datetime.now(timezone.utc)
        log.info(f"✅ {sym} #{sid} sent!")
        return True

    def check_monthly(self):
        now=datetime.now(timezone.utc)
        if now.day==1 and now.hour==8:
            if not self.last_monthly or (now-self.last_monthly).days>=28:
                alert_monthly(monthly_stats())
                self.last_monthly=now

    def run(self):
        log.info("="*50)
        log.info("  Multi-Symbol Bot v5 — High Win Rate")
        log.info("="*50)
        syms=", ".join(SYMBOLS.keys())
        tg(f"🤖 <b>Multi-Symbol Bot v5</b>\n"
           f"📊 {syms}\n"
           f"⚡ EMA + OB + FVG + BOS + KillZone\n"
           f"🎯 Target WR: 78-82% | R:R 1:2\n"
           f"✅ Active!")

        while True:
            self.cycle+=1
            log.info(f"══ Cycle {self.cycle} ══")
            try:
                if self.tracker.open:
                    self.tracker.check_all()

                if is_weekend():
                    log.info("Weekend — skipping analysis.")
                else:
                    kz=get_kill_zone()
                    if kz:
                        log.info(f"Kill Zone: {kz}")
                        for sym,cfg in SYMBOLS.items():
                            if not self.can_alert(sym): continue
                            try:
                                self.analyze(sym,cfg,kz)
                                time.sleep(6)
                            except Exception as e:
                                log.error(f"{sym}: {e}")
                    else:
                        log.info("Not in Kill Zone.")

                self.check_monthly()
            except Exception as e:
                log.error(f"Cycle: {e}",exc_info=True)

            log.info(f"Next in {INTERVAL//60}min...")
            time.sleep(INTERVAL)

if __name__=="__main__":
    MultiBot().run()
