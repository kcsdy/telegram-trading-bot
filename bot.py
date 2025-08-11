
"""
Telegram Trading Prediction Bot — Telegram-controlled portfolio, predictions, alerts
(Complete build: 4-hour DROP & PROFIT alerts + news-driven predictions + price/portfolio controls)

New:
- Automatic 4-hour alerts every 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 local time
  • Drop alert when % change <= -threshold (default 3%)
  • Profit/surge alert when % change >= +threshold (default 3%)
- Controls:
  • /alert4h on|off           — enable/disable 4h drop alerts
  • /alert4hup on|off         — enable/disable 4h profit alerts
  • /set4hthresh PCT          — set drop threshold (e.g., 3)
  • /set4hupthresh PCT        — set profit threshold (e.g., 3)

Core features preserved:
- Manage portfolio: /add, /remove, /portfolio
- Prices & alerts: /price, /prices, /setalert, /unsetalert, /alerts
- Push frequency: /setfreq 30m|2h|1d
- Predictions: /predict [TICKER]  (news sentiment + simple model)
- News: /news [TICKER]
- Intraperiod change: /change4h [TICKER]
- Timezone: /settz Area/City

Disclaimer: Educational use only. Not investment advice.
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Any

import requests
import numpy as np
import pandas as pd

# Data
import yfinance as yf

# Scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Utils
from dateutil import tz
from dotenv import load_dotenv

try:
    import feedparser  # for Google News RSS
except Exception:
    feedparser = None

# Load .env if present
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

# Ensure VADER is ready (first run will download the lexicon)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --------------------------- Config ---------------------------
DATA_PATH = os.getenv("BOT_STATE_PATH", "./bot_state.json")
DEFAULT_TZ = "Europe/London"
DEFAULT_FREQ_MIN = int(os.getenv("DEFAULT_FREQ_MIN", "1440"))  # daily
NEWS_DAYS = int(os.getenv("NEWS_DAYS", "5"))
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # optional
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LONG_POLL_TIMEOUT = int(os.getenv("LONG_POLL_TIMEOUT", "50"))
DEFAULT_4H_THRESHOLD = float(os.getenv("FOURH_THRESHOLD_PCT", "3.0"))       # drop threshold (%)
DEFAULT_4H_UP_THRESHOLD = float(os.getenv("FOURH_UP_THRESHOLD_PCT", "3.0")) # profit threshold (%)

if TELEGRAM_BOT_TOKEN is None:
    raise SystemExit("ERROR: TELEGRAM_BOT_TOKEN must be set (in environment or .env).")

# ----------------------- Persistence Layer -------------------
def load_state() -> Dict[str, Any]:
    if not os.path.exists(DATA_PATH):
        return {"chats": {}}
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"chats": {}}

def save_state(state: Dict[str, Any]) -> None:
    tmp = DATA_PATH + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, DATA_PATH)

def get_chat(state: Dict[str, Any], chat_id: str) -> Dict[str, Any]:
    chats = state.setdefault("chats", {})
    chat = chats.setdefault(chat_id, {
        "tz": DEFAULT_TZ,
        "freq_min": DEFAULT_FREQ_MIN,
        "portfolio": {},  # ticker -> quantity (float)
        "alerts": {},      # ticker -> {"above": price or None, "below": price or None}
        "symbols": ["AAPL", "TSLA"],  # default for predictions
        # 4h alert settings
        "fourh_enabled": True,                       # drop alerts
        "fourh_threshold_pct": DEFAULT_4H_THRESHOLD,
        "fourh_up_enabled": True,                    # profit alerts
        "fourh_up_threshold_pct": DEFAULT_4H_UP_THRESHOLD,
    })
    return chat

# ----------------------- Telegram Helpers --------------------
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def tg_send(chat_id: str, text: str, disable_web_page_preview: bool = False) -> None:
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': disable_web_page_preview
    }
    try:
        r = requests.post(f"{API_BASE}/sendMessage", json=payload, timeout=20)
        if r.status_code != 200:
            print("Telegram send error:", r.text)
    except Exception as e:
        print("Telegram send failed:", e)

def tg_get_updates(offset: int | None, timeout: int = LONG_POLL_TIMEOUT) -> Dict[str, Any]:
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(f"{API_BASE}/getUpdates", params=params, timeout=timeout+5)
    r.raise_for_status()
    return r.json()

# ----------------------- Market Data & NLP -------------------
def get_quote(symbol: str) -> tuple[float | None, float | None]:
    """
    Returns (live_last_price, previous_close) as floats.
    previous_close is the official raw close for the prior regular session.
    live_last_price may be None outside market hours or if source is unavailable.
    """
    tkr = yf.Ticker(symbol)

    live = None
    prev_close = None

    # 1) Fast path via fast_info
    try:
        fi = tkr.fast_info or {}
        if "previous_close" in fi and fi["previous_close"] is not None:
            prev_close = float(fi["previous_close"])
        if "last_price" in fi and fi["last_price"] is not None:
            live = float(fi["last_price"])
    except Exception:
        pass

    # 2) Fallback for previous close via .info
    if prev_close is None:
        try:
            info = tkr.info or {}
            if "previousClose" in info and info["previousClose"] is not None:
                prev_close = float(info["previousClose"])
        except Exception:
            pass

    # 3) Final fallback for previous close via daily history (raw, no adjustments)
    if prev_close is None:
        try:
            df_daily = tkr.history(period="5d", interval="1d", auto_adjust=False)
            if not df_daily.empty:
                prev_close = float(df_daily["Close"].iloc[-1])
        except Exception:
            pass

    # 4) Fallback for live using 1-minute bar (can be None outside hours)
    if live is None:
        try:
            intraday = yf.download(
                symbol, period="1d", interval="1m",
                prepost=True, auto_adjust=False, progress=False
            )
            if not intraday.empty:
                live = float(intraday["Close"].iloc[-1])
        except Exception:
            pass

    return live, prev_close

def fetch_news_newsapi(query: str, days: int = NEWS_DAYS) -> List[Dict]:
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    from_param = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
    params = {
        'q': query,
        'from': from_param,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': 50,
        'apiKey': NEWSAPI_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    articles = data.get('articles', [])
    return [{
        'title': a.get('title'),
        'description': a.get('description'),
        'url': a.get('url'),
        'publishedAt': a.get('publishedAt')
    } for a in articles if a.get('title')]

def fetch_news_google_rss(query: str, days: int = NEWS_DAYS) -> List[Dict]:
    if feedparser is None:
        return []
    import urllib.parse as up
    url = f"https://news.google.com/rss/search?q={up.quote(query)}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
    d = feedparser.parse(url)
    out = []
    for e in d.entries[:50]:
        out.append({
            'title': getattr(e, 'title', None),
            'description': getattr(e, 'summary', None),
            'url': getattr(e, 'link', None),
            'publishedAt': getattr(e, 'published', None)
        })
    return out

def score_sentiment(texts: List[str]) -> float:
    if not texts:
        return 0.0
    sia = SentimentIntensityAnalyzer()
    scores = []
    for t in texts:
        if not t:
            continue
        s = sia.polarity_scores(t)
        scores.append(s['compound'])
    return float(np.mean(scores)) if scores else 0.0

def get_sentiment_for_symbol(symbol: str) -> Tuple[float, List[Dict]]:
    """Return (sentiment_score, top_headlines[])"""
    info = {}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        pass
    name = info.get('shortName') or info.get('longName') or symbol
    articles: List[Dict] = []
    if NEWSAPI_KEY:
        articles.extend(fetch_news_newsapi(f"{name} OR {symbol}", days=NEWS_DAYS))
    articles.extend(fetch_news_google_rss(f"{name} OR {symbol}", days=NEWS_DAYS))
    seen, dedup = set(), []
    for a in articles:
        t = (a.get('title') or '').strip()
        if t and t not in seen:
            seen.add(t)
            dedup.append(a)
    texts = [a.get('title') for a in dedup if a.get('title')]
    sent = score_sentiment(texts)
    return sent, dedup[:5]

# --------------------- Predictions ----------------------------
def train_and_predict(symbol: str):
    """
    Fetch daily prices & news sentiment for symbol, train a simple model,
    and predict the next day's % change.

    Returns tuple:
      (prediction_pct, r2_score, mae, last_close, sentiment_score)
    """
    # 1) Past 120 days of daily prices
    df = yf.download(symbol, period="120d", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No price data for " + symbol)

    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    # 2) Sentiment
    sentiment_score, _ = get_sentiment_for_symbol(symbol)

    # 3) Simple features: lagged return + sentiment
    X, y = [], []
    for i in range(1, len(df)):
        X.append([df["Return"].iloc[i-1], sentiment_score])
        y.append(df["Return"].iloc[i])
    X = np.array(X)
    y = np.array(y)

    if len(X) < 10:
        raise RuntimeError("Not enough history to model " + symbol)

    # 4) Train/test split (time order)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    else:
        r2, mae = float("nan"), float("nan")

    # 5) Predict next day's return
    last_return = df["Return"].iloc[-1]
    next_pred = float(model.predict([[last_return, sentiment_score]])[0])

    return next_pred * 100.0, r2, mae, float(df["Close"].iloc[-1]), float(sentiment_score)

# --------------------- 4h Change Logic ------------------------
def change_last_hours(symbol: str, hours: int = 4):
    """
    Returns tuple: (pct_change, start_price, end_price, start_ts, end_ts)
    Uses 1m intraday bars with pre/post; falls back to 5m if needed.
    """
    def fetch(interval: str):
        df = yf.download(
            symbol,
            period="2d",
            interval=interval,
            prepost=True,
            auto_adjust=False,
            progress=False,
        )
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        return df

    df = fetch("1m")
    if df is None or df.empty:
        df = fetch("5m")
    if df is None or df.empty:
        raise RuntimeError("no intraday data")

    end_ts = df.index[-1]
    start_cutoff = end_ts - timedelta(hours=hours)
    window = df.loc[df.index >= start_cutoff]

    if window.empty:
        approx_rows = int(hours * 60)
        window = df.tail(max(approx_rows, 2))

    if len(window) < 2:
        raise RuntimeError("not enough data in last hours")

    start_price = float(window["Close"].iloc[0])
    end_price = float(window["Close"].iloc[-1])
    if start_price <= 0:
        raise RuntimeError("invalid starting price")

    pct = (end_price - start_price) / start_price * 100.0
    start_ts = window.index[0]
    end_ts = window.index[-1]
    return pct, start_price, end_price, start_ts, end_ts

# ----------------------- Core Bot Functions ------------------
def parse_freq(text: str) -> int | None:
    text = text.strip().lower()
    if text.endswith('min') or text.endswith('m'):
        n = text[:-1] if text.endswith('m') else text[:-3]
        return int(n)
    if text.endswith('h') or text.endswith('hr') or text.endswith('hrs'):
        n = text.split('h')[0]
        return int(float(n) * 60)
    if text.endswith('d'):
        n = text[:-1]
        return int(float(n) * 1440)
    try:
        return int(text)  # minutes
    except Exception:
        return None

def portfolio_value(portfolio: Dict[str, float]) -> Tuple[float, List[str]]:
    total = 0.0
    lines = []
    for sym, qty in portfolio.items():
        try:
            live, prev = get_quote(sym)
            use_px = live if live is not None else prev
            if use_px is None:
                raise RuntimeError("no price available")
            val = use_px * float(qty)
            total += val
            px_label = f"${use_px:.2f} (live)" if live is not None else f"${prev:.2f} (prev close)"
            lines.append(f"• {sym}: {qty} @ {px_label} = ${val:.2f}")
        except Exception as e:
            lines.append(f"• {sym}: price error ({e})")
    return total, lines

def periodic_push(chat_id: str, chat: Dict[str, Any]) -> None:
    tz_local = tz.gettz(chat.get("tz", DEFAULT_TZ))
    asof = datetime.now(tz_local).strftime('%Y-%m-%d %H:%M')
    lines = [f"*Portfolio Update*  ", f"As of {asof} {chat.get('tz', DEFAULT_TZ)}"]
    if chat.get("portfolio"):
        total, details = portfolio_value(chat["portfolio"])
        lines.append(f"Total value: ${total:.2f}")
        lines.extend(details)
    else:
        lines.append("Portfolio is empty. Add with `/add TICKER [QTY]`.")
    tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

def fourh_scan(chat_id: str, chat: Dict[str, Any]) -> None:
    """Check last 4h changes for portfolio; alert on drops/profits vs thresholds."""
    if not chat.get("portfolio"):
        return

    tz_local = tz.gettz(chat.get("tz", DEFAULT_TZ))
    down_enabled = bool(chat.get("fourh_enabled", True))
    up_enabled = bool(chat.get("fourh_up_enabled", True))
    down_thresh = float(chat.get("fourh_threshold_pct", DEFAULT_4H_THRESHOLD))
    up_thresh = float(chat.get("fourh_up_threshold_pct", DEFAULT_4H_UP_THRESHOLD))

    drops = []
    surges = []
    notes = []

    for sym in chat["portfolio"].keys():
        try:
            pct, p0, p1, t0, t1 = change_last_hours(sym, hours=4)
            if down_enabled and pct <= -down_thresh:
                drops.append((sym, pct, p0, p1, t0, t1))
            if up_enabled and pct >= up_thresh:
                surges.append((sym, pct, p0, p1, t0, t1))
        except Exception as e:
            notes.append(f"• {sym}: check failed ({e})")

    if drops:
        lines = [f"🚨 *4h Drop Alert* (≤ -{down_thresh:.1f}%):"]
        for sym, pct, p0, p1, t0, t1 in drops:
            t0l = t0.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            t1l = t1.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            lines.append(f"• {sym}: {pct:.2f}%  ({t0l}: ${p0:.2f} → {t1l}: ${p1:.2f})")
        if notes:
            lines.append("—"); lines.extend(notes)
        tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

    if surges:
        lines = [f"📈 *4h Profit Alert* (≥ +{up_thresh:.1f}%):"]
        for sym, pct, p0, p1, t0, t1 in surges:
            t0l = t0.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            t1l = t1.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            lines.append(f"• {sym}: {pct:+.2f}%  ({t0l}: ${p0:.2f} → {t1l}: ${p1:.2f})")
        if notes and not drops:
            lines.append("—"); lines.extend(notes)
        tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

# ---------------- Scheduler & Jobs ----------------
scheduler = BackgroundScheduler(timezone=DEFAULT_TZ)

def reschedule_chat_job(chat_id: str, chat: Dict[str, Any]) -> None:
    job_id = f"job_{chat_id}"
    try:
        job = scheduler.get_job(job_id)
        if job:
            job.remove()
    except Exception:
        pass
    freq = int(chat.get("freq_min", DEFAULT_FREQ_MIN))
    scheduler.add_job(lambda: periodic_push(chat_id, chat), IntervalTrigger(minutes=freq), id=job_id, replace_existing=True)

def reschedule_4h_job(chat_id: str, chat: Dict[str, Any]) -> None:
    job_id = f"job4h_{chat_id}"
    try:
        job = scheduler.get_job(job_id)
        if job:
            job.remove()
    except Exception:
        pass
    tz_name = chat.get("tz", DEFAULT_TZ)
    trigger = CronTrigger(hour="0,4,8,12,16,20", minute=0, timezone=tz_name)
    scheduler.add_job(lambda: fourh_scan(chat_id, chat), trigger=trigger, id=job_id, replace_existing=True)

# -------------------------- Commands -------------------------
HELP = (
    "*Commands*\n"
    "• /add TICKER [QTY] — add to portfolio (default QTY=1)\n"
    "• /remove TICKER — remove from portfolio\n"
    "• /portfolio — show portfolio and value\n"
    "• /prices — current prices for portfolio\n"
    "• /price TICKER — live + official previous close\n"
    "• /setfreq 30m|2h|1d — set price update frequency\n"
    "• /setalert TICKER ABOVE|BELOW PRICE — price alert\n"
    "• /unsetalert TICKER — clear alerts for ticker\n"
    "• /alerts — list alerts\n"
    "• /predict [TICKER] — model next-day prediction\n"
    "• /news [TICKER] — latest headlines + sentiment\n"
    "• /change4h [TICKER] — show last 4h % change\n"
    "• /alert4h on|off — enable/disable automatic 4h drop alerts\n"
    "• /alert4hup on|off — enable/disable automatic 4h profit alerts\n"
    "• /set4hthresh PCT — set 4h drop alert threshold (default 3)\n"
    "• /set4hupthresh PCT — set 4h profit alert threshold (default 3)\n"
    "• /settz Area/City — set timezone (e.g., Europe/London)\n"
    "• /help — this help"
)

def handle_command(state: Dict[str, Any], update: Dict[str, Any]) -> None:
    msg = update.get('message') or update.get('edited_message')
    if not msg:
        return
    chat_id = str(msg['chat']['id'])
    text = (msg.get('text') or '').strip()
    chat = get_chat(state, chat_id)

    if text.startswith('/'):
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ('/start', '/help'):
            tg_send(chat_id, HELP, disable_web_page_preview=True)

        elif cmd == '/add' and len(parts) >= 2:
            sym = parts[1].upper()
            try:
                qty = float(parts[2]) if len(parts) >= 3 else 1.0
            except Exception:
                qty = 1.0
            chat['portfolio'][sym] = chat['portfolio'].get(sym, 0.0) + qty
            save_state(state)
            tg_send(chat_id, f"Added {qty} {sym}. Use /portfolio to view.")

        elif cmd == '/remove' and len(parts) >= 2:
            sym = parts[1].upper()
            if sym in chat['portfolio']:
                del chat['portfolio'][sym]
                save_state(state)
                tg_send(chat_id, f"Removed {sym}.")
            else:
                tg_send(chat_id, f"{sym} not in portfolio.")

        elif cmd == '/portfolio':
            total, details = portfolio_value(chat['portfolio'])
            lines = ["*Your Portfolio*", *details, f"Total: ${total:.2f}"] if details else ["Portfolio is empty."]
            tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

        elif cmd == '/prices':
            if not chat['portfolio']:
                tg_send(chat_id, "Portfolio is empty. Add with /add TICKER [QTY].")
            else:
                lines = ["*Current Prices*"]
                for sym in chat['portfolio'].keys():
                    try:
                        live, prev = get_quote(sym)
                        if live is not None:
                            lines.append(f"• {sym}: ${live:.2f} (live)")
                        if prev is not None:
                            lines.append(f"  {sym} prev close: ${prev:.2f}")
                    except Exception as e:
                        lines.append(f"• {sym}: error {e}")
                tg_send(chat_id, "\n".join(lines))

        elif cmd == '/price' and len(parts) >= 2:
            sym = parts[1].upper()
            try:
                live, prev_close = get_quote(sym)
                lines = [f"*{sym}*"]
                if live is not None:
                    lines.append(f"Live: ${live:.2f}")
                if prev_close is not None:
                    lines.append(f"Prev close: ${prev_close:.2f}")
                tg_send(chat_id, "\n".join(lines))
            except Exception as e:
                tg_send(chat_id, f"Error fetching {sym}: {e}")

        elif cmd == '/setfreq' and len(parts) >= 2:
            freq_min = parse_freq(parts[1])
            if freq_min and freq_min >= 5:
                chat['freq_min'] = int(freq_min)
                save_state(state)
                reschedule_chat_job(chat_id, chat)
                tg_send(chat_id, f"Frequency set to every {freq_min} minutes.")
            else:
                tg_send(chat_id, "Invalid frequency. Try 30m, 2h, or 1d. Minimum 5 minutes.")

        elif cmd == '/setalert' and len(parts) >= 4:
            sym = parts[1].upper()
            direction = parts[2].upper()
            try:
                level = float(parts[3])
            except Exception:
                tg_send(chat_id, "Usage: /setalert TICKER ABOVE|BELOW PRICE")
                return
            cfg = chat['alerts'].setdefault(sym, {"above": None, "below": None})
            if direction == 'ABOVE':
                cfg['above'] = level
            elif direction == 'BELOW':
                cfg['below'] = level
            else:
                tg_send(chat_id, "Direction must be ABOVE or BELOW")
                return
            save_state(state)
            tg_send(chat_id, f"Alert set for {sym}: {direction} {level}")

        elif cmd == '/unsetalert' and len(parts) >= 2:
            sym = parts[1].upper()
            if sym in chat['alerts']:
                del chat['alerts'][sym]
                save_state(state)
                tg_send(chat_id, f"Alerts cleared for {sym}.")
            else:
                tg_send(chat_id, f"No alerts for {sym}.")

        elif cmd == '/alerts':
            if not chat['alerts']:
                tg_send(chat_id, "No alerts set.")
            else:
                lines = ["*Alerts*"]
                for sym, cfg in chat['alerts'].items():
                    ab = f"ABOVE {cfg['above']}" if cfg.get('above') is not None else ""
                    be = f"BELOW {cfg['below']}" if cfg.get('below') is not None else ""
                    join = ", ".join([x for x in (ab, be) if x])
                    lines.append(f"• {sym}: {join}")
                tg_send(chat_id, "\n".join(lines))

        elif cmd == '/predict':
            sym = parts[1].upper() if len(parts) >= 2 else None
            symbols = [sym] if sym else list(chat.get('portfolio').keys()) or chat.get('symbols', ["AAPL"])
            tz_local = tz.gettz(chat.get("tz", DEFAULT_TZ))
            asof = datetime.now(tz_local).strftime('%Y-%m-%d %H:%M')
            lines = [f"*Predictions*  As of {asof} {chat.get('tz', DEFAULT_TZ)}"]
            for s in symbols:
                try:
                    pred_pct, r2, mae, last_close, sent = train_and_predict(s)
                    lines.append(
                        f"\n*{s}*\nClose ${last_close:.2f}\n"
                        f"Pred {pred_pct:+.2f}% (R² {r2:.2f}, MAE {mae*100:.2f}%)\n"
                        f"NewsSent {sent:+.2f}"
                    )
                except Exception as e:
                    lines.append(f"{s}: error {e}")
            tg_send(chat_id, "\n".join(lines))

        elif cmd == '/news':
            sym = parts[1].upper() if len(parts) >= 2 else None
            if not sym:
                tg_send(chat_id, "Usage: /news TICKER")
                return
            sent, headlines = get_sentiment_for_symbol(sym)
            lines = [f"*{sym} News*\nSentiment: {sent:+.2f}"]
            for h in headlines:
                title, url = h.get('title'), h.get('url')
                if title and url:
                    lines.append(f"• [{title}]({url})")
            tg_send(chat_id, "\n".join(lines))

        elif cmd == '/change4h':
            sym = parts[1].upper() if len(parts) >= 2 else None
            targets = [sym] if sym else (list(chat.get('portfolio').keys()) or chat.get('symbols', ["AAPL"]))
            tz_local = tz.gettz(chat.get("tz", DEFAULT_TZ))
            lines = ["*Last 4h change*"]
            for s in targets:
                try:
                    pct, p0, p1, t0, t1 = change_last_hours(s, hours=4)
                    t0l = t0.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
                    t1l = t1.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
                    lines.append(f"• {s}: {pct:+.2f}%  ({t0l}: ${p0:.2f} → {t1l}: ${p1:.2f})")
                except Exception as e:
                    lines.append(f"• {s}: error {e}")
            tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

        elif cmd == '/alert4h' and len(parts) >= 2:
            arg = parts[1].lower()
            if arg == 'on':
                chat['fourh_enabled'] = True
                save_state(state)
                reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, "4h drop alerts enabled.")
            elif arg == 'off':
                chat['fourh_enabled'] = False
                save_state(state)
                reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, "4h drop alerts disabled.")
            else:
                tg_send(chat_id, "Usage: /alert4h on|off")

        elif cmd == '/alert4hup' and len(parts) >= 2:
            arg = parts[1].lower()
            if arg == 'on':
                chat['fourh_up_enabled'] = True
                save_state(state)
                reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, "4h profit alerts enabled.")
            elif arg == 'off':
                chat['fourh_up_enabled'] = False
                save_state(state)
                reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, "4h profit alerts disabled.")
            else:
                tg_send(chat_id, "Usage: /alert4hup on|off")

        elif cmd == '/set4hthresh' and len(parts) >= 2:
            try:
                pct = float(parts[1])
                if pct <= 0 or pct > 50:
                    raise ValueError()
                chat['fourh_threshold_pct'] = pct
                save_state(state)
                reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, f"4h drop alert threshold set to {pct:.1f}% down.")
            except Exception:
                tg_send(chat_id, "Usage: /set4hthresh PCT   (e.g., 3)")

        elif cmd == '/set4hupthresh' and len(parts) >= 2:
            try:
                pct = float(parts[1])
                if pct <= 0 or pct > 50:
                    raise ValueError()
                chat['fourh_up_threshold_pct'] = pct
                save_state(state)
                reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, f"4h profit alert threshold set to {pct:.1f}% up.")
            except Exception:
                tg_send(chat_id, "Usage: /set4hupthresh PCT   (e.g., 3)")

        elif cmd == '/settz' and len(parts) >= 2:
            chat['tz'] = parts[1]
            save_state(state)
            reschedule_chat_job(chat_id, chat)
            reschedule_4h_job(chat_id, chat)
            tg_send(chat_id, f"Timezone set to {chat['tz']}")

        else:
            tg_send(chat_id, "Unknown or malformed command. Try /help")

# --------------------------- Main Loop ------------------------
def main() -> None:
    state = load_state()
    # Start scheduler
    if not scheduler.running:
        scheduler.start()
    # Ensure jobs exist for all known chats
    for chat_id, chat in state.get('chats', {}).items():
        reschedule_chat_job(chat_id, chat)
        reschedule_4h_job(chat_id, chat)

    last_update_id = None
    print("Bot is running. Waiting for Telegram messages…")
    while True:
        try:
            resp = tg_get_updates(last_update_id + 1 if last_update_id is not None else None)
            ok = resp.get('ok')
            if not ok:
                time.sleep(1)
                continue
            for upd in resp.get('result', []):
                last_update_id = upd['update_id']
                handle_command(state, upd)
        except requests.exceptions.ReadTimeout:
            continue
        except Exception as e:
            print("Loop error:", e)
            time.sleep(2)

if __name__ == "__main__":
    main()
