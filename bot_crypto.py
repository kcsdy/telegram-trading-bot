
"""
Telegram Crypto Portfolio Bot — with 1-minute uptick/downtick, min 1-minute frequency,
and in-memory 1-minute price snapshots.

Commands:
  /add SYMBOL [QTY]            — add to portfolio (e.g., /add BTC/USDT 0.1)
  /remove SYMBOL               — remove from portfolio
  /portfolio                   — show holdings & value
  /prices                      — current prices (includes 1m tick if available)
  /price SYMBOL                — last & prev daily close + 1m change (📈/📉)
  /setfreq 1|5|30m|2h|1d       — push frequency (min 1 minute)
  /setalert SYMBOL ABOVE|BELOW PRICE — price alert
  /unsetalert SYMBOL           — clear alerts
  /alerts                      — list alerts
  /news SYMBOL                 — latest headlines + sentiment
  /change4h [SYMBOL]           — last 4h % change
  /alert4h on|off              — 4h drop alerts
  /alert4hup on|off            — 4h profit alerts
  /set4hthresh P               — set drop threshold
  /set4hupthresh P             — set profit threshold
  /settz Area/City             — time zone
  /help                        — this help

Env vars (namespaced to co-exist with stocks bot; falls back to generic if unset):
  CRYPTO_TELEGRAM_BOT_TOKEN       (fallback TELEGRAM_BOT_TOKEN)
  CRYPTO_EXCHANGE                 (fallback EXCHANGE, default binance)
  CRYPTO_BOT_STATE_PATH           (fallback BOT_STATE_PATH, default ./bot_state_crypto.json)
  CRYPTO_DEFAULT_FREQ_MIN         (fallback DEFAULT_FREQ_MIN, default 60)
  CRYPTO_LONG_POLL_TIMEOUT        (fallback LONG_POLL_TIMEOUT, default 50)
  CRYPTO_FOURH_THRESHOLD_PCT      (fallback FOURH_THRESHOLD_PCT, default 3.0)
  CRYPTO_FOURH_UP_THRESHOLD_PCT   (fallback FOURH_UP_THRESHOLD_PCT, default 3.0)
  CRYPTO_CRYPTOPANIC_API_KEY      (fallback CRYPTOPANIC_API_KEY)

Disclaimer: Educational use only. Not investment advice.
"""

import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Set

import requests
import numpy as np

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

import ccxt
import feedparser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from dateutil import tz
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# ---------------- Env ----------------
TELEGRAM_BOT_TOKEN = os.getenv("CRYPTO_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
EXCHANGE_ID = os.getenv("CRYPTO_EXCHANGE") or os.getenv("EXCHANGE") or "binance"
DATA_PATH = os.getenv("CRYPTO_BOT_STATE_PATH") or os.getenv("BOT_STATE_PATH") or "./bot_state_crypto.json"
DEFAULT_TZ = "Europe/London"
DEFAULT_FREQ_MIN = int(os.getenv("CRYPTO_DEFAULT_FREQ_MIN") or os.getenv("DEFAULT_FREQ_MIN") or "60")
FOURH_DOWN = float(os.getenv("CRYPTO_FOURH_THRESHOLD_PCT") or os.getenv("FOURH_THRESHOLD_PCT") or "3.0")
FOURH_UP = float(os.getenv("CRYPTO_FOURH_UP_THRESHOLD_PCT") or os.getenv("FOURH_UP_THRESHOLD_PCT") or "3.0")
LONG_POLL_TIMEOUT = int(os.getenv("CRYPTO_LONG_POLL_TIMEOUT") or os.getenv("LONG_POLL_TIMEOUT") or "50")
CRYPTOPANIC_API_KEY = os.getenv("CRYPTO_CRYPTOPANIC_API_KEY") or os.getenv("CRYPTOPANIC_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN is required (CRYPTO_TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN)")

API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ---------------- Exchange ----------------
def get_exchange():
    ex_class = getattr(ccxt, EXCHANGE_ID)
    ex = ex_class({'enableRateLimit': True})
    return ex

EX = get_exchange()

# ---------------- Persistence ----------------
def load_state() -> Dict[str, Any]:
    if not os.path.exists(DATA_PATH):
        return {"chats": {}}
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"chats": {}}

def save_state(state: Dict[str, Any]) -> None:
    tmp = DATA_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, DATA_PATH)

def get_chat(state: Dict[str, Any], chat_id: str) -> Dict[str, Any]:
    chats = state.setdefault("chats", {})
    chat = chats.setdefault(chat_id, {
        "tz": DEFAULT_TZ,
        "freq_min": DEFAULT_FREQ_MIN,
        "portfolio": {},  # "BTC/USDT": qty
        "alerts": {},     # symbol -> {"above": x, "below": y}
        "fourh_enabled": True,
        "fourh_up_enabled": True,
        "fourh_threshold_pct": FOURH_DOWN,
        "fourh_up_threshold_pct": FOURH_UP,
        "symbols": ["BTC/USDT","ETH/USDT"]
    })
    return chat

# ---------------- Telegram helpers ----------------
def tg_send(chat_id: str, text: str, disable_web_page_preview: bool = False) -> None:
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown', 'disable_web_page_preview': disable_web_page_preview}
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

# ---------------- Data helpers ----------------
def get_quote(symbol: str) -> Tuple[float | None, float | None]:
    """
    Returns (last_price, prev_day_close) for a crypto symbol on the configured exchange.
    prev_day_close computed from last completed 1d OHLCV candle.
    """
    last = None
    prev_close = None
    try:
        t = EX.fetch_ticker(symbol)
        last = float(t['last']) if t['last'] is not None else None
    except Exception:
        pass
    try:
        ohlcv = EX.fetch_ohlcv(symbol, timeframe='1d', limit=2)
        if ohlcv and len(ohlcv) >= 1:
            prev_close = float(ohlcv[-1][4])
    except Exception:
        pass
    return last, prev_close

def change_last_hours(symbol: str, hours: int = 4):
    """
    Percent change over the last `hours` using 1m OHLCV (falls back to 5m).
    Returns (pct, start_price, end_price, start_ts, end_ts) in UTC.
    """
    def fetch(tf: str, limit: int):
        return EX.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    limit = min(1000, hours*60 + 5)
    ohlcv = None
    try:
        ohlcv = fetch('1m', limit)
    except Exception:
        try:
            limit = min(1000, int(hours*60/5)+5)
            ohlcv = fetch('5m', limit)
        except Exception:
            pass
    if not ohlcv or len(ohlcv) < 2:
        raise RuntimeError("no intraday data")
    end_ts_ms = ohlcv[-1][0]
    start_cutoff_ms = end_ts_ms - hours*60*60*1000
    window = [c for c in ohlcv if c[0] >= start_cutoff_ms]
    if len(window) < 2:
        window = ohlcv[-max(2, int(hours*60/5)):]  # fallback slice
    start_price = float(window[0][4])
    end_price = float(window[-1][4])
    pct = (end_price - start_price) / start_price * 100.0
    start_ts = datetime.fromtimestamp(window[0][0]/1000, tz=timezone.utc)
    end_ts = datetime.fromtimestamp(window[-1][0]/1000, tz=timezone.utc)
    return pct, start_price, end_price, start_ts, end_ts

# ---------------- News & Sentiment ----------------
def cryptopanic_news(query: str) -> List[Dict]:
    if not CRYPTOPANIC_API_KEY:
        return []
    try:
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": CRYPTOPANIC_API_KEY, "currencies": query.replace('/',''), "public": "true"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        out = []
        for item in data.get("results", [])[:20]:
            out.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "publishedAt": item.get("published_at")
            })
        return out
    except Exception:
        return []

def google_news(query: str) -> List[Dict]:
    import urllib.parse as up
    url = f"https://news.google.com/rss/search?q={up.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    d = feedparser.parse(url)
    out = []
    for e in d.entries[:20]:
        out.append({
            "title": getattr(e, "title", None),
            "url": getattr(e, "link", None),
            "publishedAt": getattr(e, "published", None),
        })
    return out

def get_sentiment_for_symbol(symbol: str) -> Tuple[float, List[Dict]]:
    base = symbol.split('/')[0]  # BTC from BTC/USDT
    articles = []
    articles += cryptopanic_news(base)
    if len(articles) < 5:
        articles += google_news(f"{base} crypto")
    seen = set(); dedup = []
    for a in articles:
        t = (a.get("title") or "").strip()
        if t and t not in seen:
            seen.add(t); dedup.append(a)
    sia = SentimentIntensityAnalyzer()
    scores = []
    for a in dedup:
        t = a.get("title"); 
        if t:
            scores.append(sia.polarity_scores(t)["compound"])
    sent = float(np.mean(scores)) if scores else 0.0
    return sent, dedup[:5]

# ---------------- In-memory 1-minute snapshots ----------------
LAST_MINUTE: Dict[str, Dict[str, float]] = {}  # symbol -> {"ts": epoch_sec, "price": float}
TRACKED_SYMBOLS: Set[str] = set()

def snapshot_prices_for_all_chats(state: Dict[str, Any]) -> None:
    """Background job every 1 minute: snapshot last price for all symbols in all chats' portfolios."""
    global LAST_MINUTE, TRACKED_SYMBOLS
    symbols: Set[str] = set()
    for chat in state.get("chats", {}).values():
        for s in chat.get("portfolio", {}).keys():
            symbols.add(s)
    symbols |= TRACKED_SYMBOLS  # include any ad-hoc tracked symbols (queried via /price)
    if not symbols:
        return
    tickers = {}
    for s in symbols:
        try:
            t = EX.fetch_ticker(s)
            if t and t.get("last") is not None:
                tickers[s] = float(t["last"])
        except Exception:
            continue
    now = time.time()
    for s, px in tickers.items():
        LAST_MINUTE[s] = {"ts": now, "price": px}

def get_1min_change(symbol: str, current_price: float) -> Tuple[float | None, str]:
    """
    Compare current price to snapshot from ~1 minute ago (if available).
    Returns (pct_change, label) where label is "📈", "📉", or "".
    If no snapshot in the last ~120s, fallback to previous 1m candle close.
    """
    snap = LAST_MINUTE.get(symbol)
    now = time.time()
    if snap and (now - float(snap["ts"])) <= 120:
        prev = float(snap["price"])
    else:
        try:
            ohlcv = EX.fetch_ohlcv(symbol, timeframe="1m", limit=2)
            if ohlcv and len(ohlcv) >= 1:
                prev = float(ohlcv[-1][4])
            else:
                prev = None
        except Exception:
            prev = None
    if prev is None or current_price is None or prev <= 0:
        return None, ""
    pct = (current_price - prev) / prev * 100.0
    label = "📈" if pct > 0 else ("📉" if pct < 0 else "")
    return pct, label

# ---------------- Portfolio & alerts ----------------
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
        return int(text)
    except Exception:
        return None

def portfolio_value(portfolio: Dict[str, float]) -> Tuple[float, List[str]]:
    total = 0.0
    lines = []
    for sym, qty in portfolio.items():
        try:
            last, prev = get_quote(sym)
            px = last if last is not None else prev
            if px is None:
                raise RuntimeError("no price")
            val = float(qty) * px
            total += val
            label = f"${px:.4f}"
            lines.append(f"• {sym}: {qty} @ {label} = ${val:.2f}")
        except Exception as e:
            lines.append(f"• {sym}: price error ({e})")
    return total, lines

def periodic_push(chat_id: str, chat: Dict[str, Any]) -> None:
    tz_local = tz.gettz(chat.get("tz", DEFAULT_TZ))
    asof = datetime.now(tz_local).strftime('%Y-%m-%d %H:%M')
    lines = [f"*Crypto Portfolio Update*  ", f"As of {asof} {chat.get('tz', DEFAULT_TZ)}"]
    if chat.get("portfolio"):
        total, details = portfolio_value(chat["portfolio"])
        lines.append(f"Total value: ${total:.2f}")
        lines.extend(details)
    else:
        lines.append("Portfolio is empty. Add with `/add BTC/USDT [QTY]`.")
    tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

def fourh_scan(chat_id: str, chat: Dict[str, Any]) -> None:
    if not chat.get("portfolio"):
        return
    tz_local = tz.gettz(chat.get("tz", DEFAULT_TZ))
    down_en = bool(chat.get("fourh_enabled", True))
    up_en = bool(chat.get("fourh_up_enabled", True))
    down_th = float(chat.get("fourh_threshold_pct", FOURH_DOWN))
    up_th = float(chat.get("fourh_up_threshold_pct", FOURH_UP))

    drops, surges, notes = [], [], []
    for sym in chat["portfolio"].keys():
        try:
            pct, p0, p1, t0, t1 = change_last_hours(sym, 4)
            if down_en and pct <= -down_th: drops.append((sym, pct, p0, p1, t0, t1))
            if up_en and pct >= up_th: surges.append((sym, pct, p0, p1, t0, t1))
        except Exception as e:
            notes.append(f"• {sym}: {e}")
    if drops:
        lines = [f"🚨 *4h Drop Alert* (≤ -{down_th:.1f}%):"]
        for sym, pct, p0, p1, t0, t1 in drops:
            t0l = t0.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            t1l = t1.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            lines.append(f"• {sym}: {pct:.2f}%  ({t0l}: {p0:.4f} → {t1l}: {p1:.4f})")
        if notes: lines += ["—"] + notes
        tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)
    if surges:
        lines = [f"📈 *4h Profit Alert* (≥ +{up_th:.1f}%):"]
        for sym, pct, p0, p1, t0, t1 in surges:
            t0l = t0.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            t1l = t1.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
            lines.append(f"• {sym}: {pct:+.2f}%  ({t0l}: {p0:.4f} → {t1l}: {p1:.4f})")
        if notes and not drops: lines += ["—"] + notes
        tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

# ---------------- Scheduler ----------------
scheduler = BackgroundScheduler(timezone=DEFAULT_TZ)

def reschedule_chat_job(chat_id: str, chat: Dict[str, Any]) -> None:
    job_id = f"job_{chat_id}"
    try:
        j = scheduler.get_job(job_id)
        if j: j.remove()
    except Exception: pass
    freq = int(chat.get("freq_min", DEFAULT_FREQ_MIN))
    scheduler.add_job(lambda: periodic_push(chat_id, chat), IntervalTrigger(minutes=freq), id=job_id, replace_existing=True)

def reschedule_4h_job(chat_id: str, chat: Dict[str, Any]) -> None:
    job_id = f"job4h_{chat_id}"
    try:
        j = scheduler.get_job(job_id)
        if j: j.remove()
    except Exception: pass
    tz_name = chat.get("tz", DEFAULT_TZ)
    trigger = CronTrigger(hour="0,4,8,12,16,20", minute=0, timezone=tz_name)
    scheduler.add_job(lambda: fourh_scan(chat_id, chat), trigger=trigger, id=job_id, replace_existing=True)

# ---------------- Commands ----------------
HELP = (
    "*Crypto Bot Commands*\n"
    "• /add SYMBOL [QTY] — add to portfolio (e.g., /add BTC/USDT 0.1)\n"
    "• /remove SYMBOL — remove from portfolio\n"
    "• /portfolio — show holdings & value\n"
    "• /prices — current prices\n"
    "• /price SYMBOL — last & prev daily close + 1m change\n"
    "• /setfreq 1|5|30m|2h|1d — push frequency (min 1 minute)\n"
    "• /setalert SYMBOL ABOVE|BELOW PRICE — price alert\n"
    "• /unsetalert SYMBOL — clear alerts\n"
    "• /alerts — list alerts\n"
    "• /news SYMBOL — latest headlines + sentiment\n"
    "• /change4h [SYMBOL] — last 4h % change\n"
    "• /alert4h on|off — 4h drop alerts\n"
    "• /alert4hup on|off — 4h profit alerts\n"
    "• /set4hthresh P — set drop threshold\n"
    "• /set4hupthresh P — set profit threshold\n"
    "• /settz Area/City — time zone\n"
    "• /help — this help"
)

def handle_command(state: Dict[str, Any], update: Dict[str, Any]) -> None:
    global TRACKED_SYMBOLS
    msg = update.get('message') or update.get('edited_message')
    if not msg: return
    chat_id = str(msg['chat']['id'])
    text = (msg.get('text') or '').strip()
    chat = get_chat(state, chat_id)

    if text.startswith('/'):
        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ('/start','/help'):
            tg_send(chat_id, HELP, disable_web_page_preview=True)

        elif cmd == '/add' and len(parts) >= 2:
            sym = parts[1].upper()
            try: qty = float(parts[2]) if len(parts)>=3 else 1.0
            except: qty = 1.0
            chat['portfolio'][sym] = chat['portfolio'].get(sym, 0.0) + qty
            save_state(state); tg_send(chat_id, f"Added {qty} {sym}.")
            TRACKED_SYMBOLS.add(sym)

        elif cmd == '/remove' and len(parts) >= 2:
            sym = parts[1].upper()
            if sym in chat['portfolio']:
                del chat['portfolio'][sym]; save_state(state); tg_send(chat_id, f"Removed {sym}.")
            else:
                tg_send(chat_id, f"{sym} not in portfolio.")

        elif cmd == '/portfolio':
            total, details = portfolio_value(chat['portfolio'])
            lines = ["*Your Crypto Portfolio*"] + (details if details else ["(empty)"]) + [f"Total: ${total:.2f}"]
            tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

        elif cmd == '/prices':
            if not chat['portfolio']:
                tg_send(chat_id, "Portfolio empty. Add with /add BTC/USDT [QTY].")
            else:
                lines = ["*Current Prices*"]; 
                for s in chat['portfolio'].keys():
                    try:
                        last, prev = get_quote(s)
                        pct1m, label = get_1min_change(s, last if last is not None else prev)
                        if last is not None:
                            if pct1m is not None:
                                lines.append(f"• {s}: {last:.4f} {label} {pct1m:+.2f}% (1m)")
                            else:
                                lines.append(f"• {s}: {last:.4f}")
                        if prev is not None:
                            lines.append(f"  {s} prev daily close: {prev:.4f}")
                    except Exception as e:
                        lines.append(f"• {s}: error {e}")
                tg_send(chat_id, "\n".join(lines))

        elif cmd == '/price' and len(parts) >= 2:
            s = parts[1].upper()
            TRACKED_SYMBOLS.add(s)
            try:
                last, prev = get_quote(s)
                lines = [f"*{s}*"]
                if last is not None:
                    pct1m, label = get_1min_change(s, last)
                    if pct1m is not None:
                        lines.append(f"Last: {last:.4f} {label} {pct1m:+.2f}% (last 1 min)")
                    else:
                        lines.append(f"Last: {last:.4f}")
                if prev is not None:
                    lines.append(f"Prev daily close: {prev:.4f}")
                tg_send(chat_id, "\n".join(lines))
            except Exception as e:
                tg_send(chat_id, f"Error fetching {s}: {e}")

        elif cmd == '/setfreq' and len(parts) >= 2:
            freq = parse_freq(parts[1])
            if freq and freq >= 1:  # min 1 minute
                chat['freq_min'] = int(freq); save_state(state); reschedule_chat_job(chat_id, chat)
                tg_send(chat_id, f"Frequency set to {freq} min.")
            else:
                tg_send(chat_id, "Invalid frequency; minimum is 1 minute.")

        elif cmd == '/setalert' and len(parts) >= 4:
            s = parts[1].upper(); direction = parts[2].upper()
            try: level = float(parts[3])
            except: tg_send(chat_id, "Usage: /setalert SYMBOL ABOVE|BELOW PRICE"); return
            cfg = chat['alerts'].setdefault(s, {"above": None,"below": None})
            if direction == 'ABOVE': cfg['above'] = level
            elif direction == 'BELOW': cfg['below'] = level
            else: tg_send(chat_id,"Direction must be ABOVE or BELOW"); return
            save_state(state); tg_send(chat_id, f"Alert set for {s}: {direction} {level}")
            TRACKED_SYMBOLS.add(s)

        elif cmd == '/unsetalert' and len(parts) >= 2:
            s = parts[1].upper()
            if s in chat['alerts']: del chat['alerts'][s]; save_state(state); tg_send(chat_id,"Cleared.")
            else: tg_send(chat_id,"No alert for that symbol.")

        elif cmd == '/alerts':
            if not chat['alerts']: tg_send(chat_id, "No alerts set.")
            else:
                lines = ["*Alerts*"]
                for s, cfg in chat['alerts'].items():
                    ab = f"ABOVE {cfg['above']}" if cfg.get('above') is not None else ""
                    be = f"BELOW {cfg['below']}" if cfg.get('below') is not None else ""
                    join = ", ".join([x for x in (ab, be) if x])
                    lines.append(f"• {s}: {join}")
                tg_send(chat_id, "\n".join(lines))

        elif cmd == '/news' and len(parts) >= 2:
            s = parts[1].upper()
            sent, headlines = get_sentiment_for_symbol(s)
            lines = [f"*{s} News*  Sentiment: {sent:+.2f}"]
            for h in headlines:
                if h.get("title") and h.get("url"):
                    lines.append(f"• [{h['title']}]({h['url']})")
            tg_send(chat_id, "\n".join(lines))

        elif cmd == '/change4h':
            s = parts[1].upper() if len(parts) >= 2 else None
            targets = [s] if s else (list(chat.get("portfolio").keys()) or chat.get("symbols", ["BTC/USDT"]))
            tz_local = tz.gettz(chat.get("tz", DEFAULT_TZ))
            lines = ["*Last 4h change*"]
            for sym in targets:
                try:
                    pct, p0, p1, t0, t1 = change_last_hours(sym, 4)
                    t0l = t0.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
                    t1l = t1.astimezone(tz_local).strftime("%Y-%m-%d %H:%M")
                    lines.append(f"• {sym}: {pct:+.2f}%  ({t0l}: {p0:.4f} → {t1l}: {p1:.4f})")
                except Exception as e:
                    lines.append(f"• {sym}: error {e}")
            tg_send(chat_id, "\n".join(lines), disable_web_page_preview=True)

        elif cmd == '/alert4h' and len(parts) >= 2:
            arg = parts[1].lower()
            chat['fourh_enabled'] = (arg == 'on')
            save_state(state); reschedule_4h_job(chat_id, chat)
            tg_send(chat_id, f"4h drop alerts {'enabled' if chat['fourh_enabled'] else 'disabled'}.")

        elif cmd == '/alert4hup' and len(parts) >= 2:
            arg = parts[1].lower()
            chat['fourh_up_enabled'] = (arg == 'on')
            save_state(state); reschedule_4h_job(chat_id, chat)
            tg_send(chat_id, f"4h profit alerts {'enabled' if chat['fourh_up_enabled'] else 'disabled'}.")

        elif cmd == '/set4hthresh' and len(parts) >= 2:
            try:
                pct = float(parts[1]); 
                if pct <= 0 or pct > 100: raise ValueError()
                chat['fourh_threshold_pct'] = pct; save_state(state); reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, f"4h drop threshold set to {pct:.1f}%.")
            except Exception:
                tg_send(chat_id, "Usage: /set4hthresh P")

        elif cmd == '/set4hupthresh' and len(parts) >= 2:
            try:
                pct = float(parts[1]); 
                if pct <= 0 or pct > 100: raise ValueError()
                chat['fourh_up_threshold_pct'] = pct; save_state(state); reschedule_4h_job(chat_id, chat)
                tg_send(chat_id, f"4h profit threshold set to {pct:.1f}%.")
            except Exception:
                tg_send(chat_id, "Usage: /set4hupthresh P")

        elif cmd == '/settz' and len(parts) >= 2:
            chat['tz'] = parts[1]; save_state(state); reschedule_chat_job(chat_id, chat); reschedule_4h_job(chat_id, chat)
            tg_send(chat_id, f"Timezone set to {chat['tz']}")

        else:
            tg_send(chat_id, "Unknown or malformed command. Try /help")

# ---------------- Main loop ----------------
def main() -> None:
    state = load_state()
    if not scheduler.running: scheduler.start()

    # Ensure per-chat jobs are scheduled
    for chat_id, chat in state.get("chats", {}).items():
        reschedule_chat_job(chat_id, chat)
        reschedule_4h_job(chat_id, chat)

    # Global 1-minute snapshot job
    scheduler.add_job(lambda: snapshot_prices_for_all_chats(state),
                      IntervalTrigger(minutes=1),
                      id="snapshot_1m",
                      replace_existing=True)

    last_update_id = None
    print("Crypto bot running…")
    while True:
        try:
            resp = tg_get_updates(last_update_id + 1 if last_update_id is not None else None)
            if not resp.get('ok'): time.sleep(1); continue
            for upd in resp.get('result', []):
                last_update_id = upd['update_id']
                handle_command(state, upd)
        except requests.exceptions.ReadTimeout:
            continue
        except Exception as e:
            print("Loop error:", e); time.sleep(2)

if __name__ == "__main__":
    main()
