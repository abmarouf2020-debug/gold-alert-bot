# ═══════════════════════════════════════════════════
# تابع تحلیل ارتقاء‌یافته با سه لنز وایکوفی
# ═══════════════════════════════════════════════════
def analyze_symbol(sym: str, df_1h: pd.DataFrame, df_15m: pd.DataFrame) -> Optional[Dict]:
    """
    تحلیل یک نماد با SMC (OB, BOS, FVG) + بهبودهای وایکوف:
    1. تست محدوده (Test)
    2. چشمه معتبر (Valid Spring)
    3. تلاش و نتیجه (Effort vs Result)
    """

    # ── ۱. تشخیص روند اصلی (EMA 200 روی ۱h) ──
    df_1h['EMA200'] = df_1h['close'].ewm(span=200, adjust=False).mean()
    trend_bias = "Bullish" if df_1h['close'].iloc[-1] > df_1h['EMA200'].iloc[-1] else "Bearish"

    # ── ۲. آماده‌سازی دیتافریم ۱۵m ──
    o = df_15m['open'].astype(float)
    h = df_15m['high'].astype(float)
    l = df_15m['low'].astype(float)
    c = df_15m['close'].astype(float)

    # محاسبه ATR 14 برای اندازه‌گیری نوسان
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    avg_body = (c - o).abs().rolling(20).mean().iloc[-1]  # میانگین بدنه‌ی کندل‌ها

    # ── ۳. یافتن آخرین BOS (شکست ساختار) ──
    # برای سادگی، از سوئینگ‌های ۱۵ کندل اخیر استفاده می‌کنیم.
    lookback = 10
    recent_high = h.iloc[-lookback:].max()
    recent_low  = l.iloc[-lookback:].min()
    last_close  = c.iloc[-1]

    bos_bull = last_close > recent_high   # شکست سقف قبلی
    bos_bear = last_close < recent_low    # شکست کف قبلی

    # ── ۴. تشخیص Order Block (OB) ──
    # یک روش ساده: برای خرید، آخرین کندل نزولی قبل از یک حرکت صعودی قوی
    # (کندلی که Low آن پایین‌تر از کندل قبلی است و سپس قیمت برگشته بالا)
    ob_high = ob_low = None
    ob_valid = False

    # --- جستجوی OB صعودی (برای BUY) ---
    if trend_bias == "Bullish":
        # آخرین کندل کاملاً نزولی (close < open) که قبل از یک برگشت بوده
        for i in range(len(c)-2, 1, -1):
            if c[i] < o[i] and c[i+1] > o[i+1] and h[i] < h[i+1]:  # یک کندل نزولی و سپس صعودی
                ob_high = h[i]
                ob_low  = l[i]
                # کیفیت: بدنه بزرگ‌تر از میانگین و سایه‌های کوچک
                body = abs(c[i] - o[i])
                upper_wick = h[i] - max(c[i], o[i])
                lower_wick = min(c[i], o[i]) - l[i]
                if body > 1.5 * avg_body and upper_wick < 0.3 * body and lower_wick < 0.3 * body:
                    ob_valid = True
                    break
    # --- جستجوی OB نزولی (برای SELL) ---
    else:
        for i in range(len(c)-2, 1, -1):
            if c[i] > o[i] and c[i+1] < o[i+1] and l[i] > l[i+1]:  # کندل صعودی و سپس نزولی
                ob_high = h[i]
                ob_low  = l[i]
                body = abs(c[i] - o[i])
                upper_wick = h[i] - max(c[i], o[i])
                lower_wick = min(c[i], o[i]) - l[i]
                if body > 1.5 * avg_body and upper_wick < 0.3 * body and lower_wick < 0.3 * body:
                    ob_valid = True
                    break

    if not ob_valid:
        # اگر OB معتبر یافت نشد، ستاپی وجود ندارد
        return None

    # ── ۵. تشخیص FVG (شکاف ارزش منصفانه) ──
    fvg_bull = fvg_bear = False
    # FVG صعودی: Low کندل فعلی > High دو کندل قبل
    if len(c) >= 3 and l.iloc[-1] > h.iloc[-3]:
        fvg_bull = True
    # FVG نزولی: High کندل فعلی < Low دو کندل قبل
    if len(c) >= 3 and h.iloc[-1] < l.iloc[-3]:
        fvg_bear = True

    # ── ۶. بهبودهای وایکوف ──

    # ۶.۱ تست محدوده (Test): قیمت باید به داخل OB برگشته باشد
    # یعنی کندل جاری (یا قبلی) با OB تماس داشته باشد
    latest_low  = l.iloc[-1]
    latest_high = h.iloc[-1]
    price_in_ob = (ob_low <= latest_low <= ob_high) or (ob_low <= latest_high <= ob_high)

    # ۶.۲ چشمه معتبر (Spring):
    # برای خرید: قیمت کمی از کف OB پایین‌تر رفته و برگشته (سایهٔ بلند پایینی)
    spring_bull = False
    if trend_bias == "Bullish":
        candle_low   = l.iloc[-1]
        candle_close = c.iloc[-1]
        candle_open  = o.iloc[-1]
        body = abs(candle_close - candle_open)
        lower_wick = min(candle_close, candle_open) - candle_low
        # اگر کندل از کف OB پایین‌تر رفته، اما بسته‌شدن بالای کف OB باشد
        # و بدنه حداقل ۷۰٪ سایه را پوشش داده و بیش از ۵۰٪ بدنه درون OB بسته شده
        if candle_low < ob_low and candle_close > ob_low:
            if body > 0.7 * lower_wick and (candle_close - ob_low) > 0.5 * body:
                spring_bull = True

    spring_bear = False
    if trend_bias == "Bearish":
        candle_high  = h.iloc[-1]
        candle_close = c.iloc[-1]
        candle_open  = o.iloc[-1]
        body = abs(candle_close - candle_open)
        upper_wick = candle_high - max(candle_close, candle_open)
        if candle_high > ob_high and candle_close < ob_high:
            if body > 0.7 * upper_wick and (ob_high - candle_close) > 0.5 * body:
                spring_bear = True

    # ۶.۳ تلاش و نتیجه (Effort vs Result):
    # کندل قبلی بزرگ بوده و نتوانسته OB را بشکند، سپس کندل جاری آن را بلعیده
    effort_bull = effort_bear = False
    if len(c) >= 2:
        prev_body = abs(c.iloc[-2] - o.iloc[-2])
        curr_body = abs(c.iloc[-1] - o.iloc[-1])
        # برای خرید: کندل قبلی نزولی بزرگ بوده، اما به بالای کف OB نخورده،
        # و کندل جاری صعودی و بدنه‌اش از کندل قبلی بزرگتر (انگالفینگ صعودی)
        if (c.iloc[-2] < o.iloc[-2] and                 # کندل قبلی نزولی
            c.iloc[-1] > o.iloc[-1] and                 # کندل جاری صعودی
            curr_body > prev_body and                   # انگالفینگ
            l.iloc[-2] > ob_low and                     # کندل قبلی نتوانسته OB را بشکند
            c.iloc[-1] > o.iloc[-2]):                  # بسته‌شدن بالای بازشدن قبلی
            effort_bull = True

        # برای فروش: کندل قبلی صعودی بزرگ، نتواسته سقف OB را بشکند، انگالفینگ نزولی
        if (c.iloc[-2] > o.iloc[-2] and
            c.iloc[-1] < o.iloc[-1] and
            curr_body > prev_body and
            h.iloc[-2] < ob_high and
            c.iloc[-1] < o.iloc[-2]):
            effort_bear = True

    # ── ۷. محاسبهٔ امتیاز نهایی ──
    score = 0
    direction = None

    # امتیازدهی برای سیگنال خرید (Bullish Setup)
    if trend_bias == "Bullish":
        if bos_bull:                 score += 2
        if fvg_bull:                score += 2
        if price_in_ob:             score += 2
        if spring_bull:             score += 3
        if effort_bull:             score += 3
        if ob_valid:                score += 1
        direction = "BUY"
        sl = ob_low - 0.5 * atr
        # هدف: نزدیک‌ترین سقف بالاتر از قیمت فعلی (از کندل‌های بعد از OB)
        # برای سادگی، ۲ برابر ATR به‌عنوان TP (در عمل می‌توانی سوئینگ بعدی را بگیری)
        tp = c.iloc[-1] + 2 * atr
        entry_price = c.iloc[-1]   # قیمت لحظه‌ای (ورود در کندل بعد)
        entry_high = h.iloc[-1]
        entry_low  = l.iloc[-1]

    # امتیازدهی برای فروش (Bearish Setup)
    elif trend_bias == "Bearish":
        if bos_bear:                score += 2
        if fvg_bear:               score += 2
        if price_in_ob:            score += 2
        if spring_bear:            score += 3
        if effort_bear:            score += 3
        if ob_valid:               score += 1
        direction = "SELL"
        sl = ob_high + 0.5 * atr
        tp = c.iloc[-1] - 2 * atr
        entry_price = c.iloc[-1]
        entry_high = h.iloc[-1]
        entry_low  = l.iloc[-1]
    else:
        return None

    # اگر امتیاز حداقل ۴ (بدون احتساب KZ) نباشد، رد می‌کنیم.
    # (KZ را در حلقهٔ اصلی +۱ می‌کنیم تا به MIN_SCORE برسیم)
    if score < 4:
        return None

    return {
        "direction": direction,
        "price": round(entry_price, SYMBOLS[sym]["dec"]),
        "entry_high": round(entry_high, SYMBOLS[sym]["dec"]),
        "entry_low": round(entry_low, SYMBOLS[sym]["dec"]),
        "sl": round(sl, SYMBOLS[sym]["dec"]),
        "tp": round(tp, SYMBOLS[sym]["dec"]),
        "score": score,
        "bias": trend_bias
    }
