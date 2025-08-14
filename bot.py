import os, re, time, html
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

import requests, feedparser, yaml
from bs4 import BeautifulSoup
from dateutil import parser as dtparser, tz
from readability import Document
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ========= Config =========
HEADERS = {"User-Agent": "Mozilla/5.0 (news-bot; +https://github.com/)"}
LONDON = tz.gettz("Europe/London")
MAX_ITEMS_PER_SECTION = 3
MAX_BULLETS_PER_ITEM = 4
UPDATE_KEYWORDS = {"update", "/update"}  # Telegram command for on-demand send
FRESHNESS_WINDOW_SEC = 900  # how long an 'update' stays valid (15 minutes)
HISTORY_LOOKBACK_DAYS = 7   # avoid resending links from the last N days

# Importance weights by domain (higher = more trusted/important)
SOURCE_WEIGHTS = {
    # Carbon / policy
    "climate.ec.europa.eu": 1.0, "unfccc.int": 0.9, "icapcarbonaction.com": 0.9,
    "gov.uk": 1.0, "iea.org": 0.9, "carbon-pulse.com": 0.85,
    "climatechangenews.com": 0.7, "weforum.org": 0.6,

    # AI
    "deepmind.google": 1.0, "blog.google": 0.9, "openai.com": 1.0,
    "huggingface.co": 0.85, "mistral.ai": 0.85, "anthropic.com": 0.9,
    "stability.ai": 0.7, "aisi.gov.uk": 0.9, "europa.eu": 0.9,

    # Crypto
    "fca.org.uk": 1.0, "coindesk.com": 0.8, "cointelegraph.com": 0.6,
    "cryptoslate.com": 0.6, "theblock.co": 0.7, "decrypt.co": 0.7,
    "binance.com": 0.6, "coinbase.com": 0.6, "blog.ethereum.org": 0.8,
}

# Keyword boosts nudging “impactful” stories higher
KEYWORD_BOOSTS = {
    # Generic impact
    "ban": 0.5, "fine": 0.5, "lawsuit": 0.6, "warning": 0.5, "recall": 0.5,
    "regulator": 0.4, "regulation": 0.4, "policy": 0.4, "investigation": 0.5,
    "breach": 0.5, "security": 0.3, "outage": 0.4, "vulnerability": 0.4,
    "acquisition": 0.3, "merger": 0.3, "funding": 0.3, "partnership": 0.2,
    "launch": 0.2, "release": 0.2, "guidance": 0.3, "consultation": 0.3,
    "auction": 0.2, "price": 0.2, "ets": 0.4, "emissions": 0.3,

    # Crypto simple signals
    "meme": 0.6, "dogecoin": 0.6, "shib": 0.5, "pepe": 0.5,
    "scam": 0.6, "rug": 0.6, "hack": 0.6, "exploit": 0.6,
}

# ========= Time / Telegram =========
def now_london():
    return datetime.now(LONDON)

def within_send_window():
    # Override if FORCE_SEND=1 or a recent 'update' message exists
    if os.getenv("FORCE_SEND") == "1" or user_sent_update_recently(FRESHNESS_WINDOW_SEC):
        return True
    t = now_london()
    return t.hour == 7 and 30 <= t.minute < 35

def telegram_api_url(method: str) -> str:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    return f"https://api.telegram.org/bot{token}/{method}"

def send_to_telegram(text: str):
    payload = {
        "chat_id": os.environ["TELEGRAM_CHAT_ID"],
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    r = requests.post(telegram_api_url("sendMessage"), json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

def telegram_get_updates():
    r = requests.get(telegram_api_url("getUpdates"), timeout=20)
    r.raise_for_status()
    return r.json().get("result", [])

def user_sent_update_recently(max_age_seconds=300) -> bool:
    chat_id = str(os.environ.get("TELEGRAM_CHAT_ID", ""))
    if not chat_id:
        return False
    updates = telegram_get_updates()
    latest = None
    for u in updates:
        msg = u.get("message") or u.get("edited_message")
        if not msg:
            continue
        if str(msg.get("chat", {}).get("id")) != chat_id:
            continue
        latest = msg if (latest is None or msg.get("date", 0) > latest.get("date", 0)) else latest
    if not latest:
        return False
    text = (latest.get("text") or "").strip().lower()
    if text not in UPDATE_KEYWORDS:
        return False
    age = int(time.time()) - int(latest.get("date", 0))
    return age <= max_age_seconds

# ========= Fetching / Parsing =========
def clean_url(u: str) -> str:
    try:
        parts = list(urlsplit(u))
        if parts[3]:
            qs = [(k, v) for k, v in parse_qsl(parts[3]) if not k.lower().startswith("utm")]
            parts[3] = urlencode(qs)
        return urlunsplit(parts)
    except Exception:
        return u

def extract_readable(url: str) -> str:
    res = requests.get(url, headers=HEADERS, timeout=20)
    res.raise_for_status()
    doc = Document(res.text)
    soup = BeautifulSoup(doc.summary(html_partial=True), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def fetch_rss(url: str):
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:10]:
        u = clean_url(getattr(e, "link", ""))
        t = getattr(e, "title", "").strip()
        date = None
        for key in ("published", "updated"):
            val = getattr(e, key, None)
            if val:
                try:
                    date = dtparser.parse(val)
                    break
                except:
                    pass
        if u and t:
            items.append({"title": t, "url": u, "date": date})
    return items

def fetch_list_html(url: str):
    res = requests.get(url, headers=HEADERS, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")
    out, seen = [], set()
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if href.startswith("#"):
            continue
        if href.startswith("/"):
            base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            href = base + href
        title = a.get_text(strip=True)
        if len(title.split()) >= 4:
            u = clean_url(href)
            if u not in seen:
                seen.add(u)
                out.append({"title": title, "url": u, "date": None})
        if len(out) >= 10:
            break
    return out

def load_sources(path="sources.yml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ========= Ranking / Dedupe =========
def domain_weight(url: str) -> float:
    dom = urlparse(url).netloc.lower().replace("www.", "")
    if dom == "ethereum.foundation":
        dom = "blog.ethereum.org"
    return SOURCE_WEIGHTS.get(dom, 0.5)

def headline_score(title: str, section: str) -> float:
    t = title.lower()
    score = sum(w for k, w in KEYWORD_BOOSTS.items() if k in t)
    if section == "crypto":
        if any(k in t for k in ["fca", "warning", "ban", "scam", "hack", "meme", "doge", "shib", "pepe", "price"]):
            score += 0.4
        if any(k in t for k in ["sharding", "rollup", "zk", "consensus", "opcodes"]):
            score -= 0.2
    if section == "ai":
        if any(k in t for k in ["openai", "gemini", "deepmind", "anthropic", "mistral", "aisi", "eu ai act"]):
            score += 0.4
    if section == "carbon":
        if any(k in t for k in ["ets", "auction", "cap", "allowance", "uk ets", "eu ets"]):
            score += 0.3
    return score

def recency_score(date_dt):
    if not date_dt:
        return 0.1
    age_hours = max(0.0, (datetime.utcnow() - date_dt.replace(tzinfo=None)).total_seconds() / 3600.0)
    if age_hours <= 24: return 0.8
    if age_hours <= 48: return 0.4
    return 0.1

def jaccard_title_sim(a: str, b: str) -> float:
    A = set(re.findall(r"[a-z0-9]+", a.lower()))
    B = set(re.findall(r"[a-z0-9]+", b.lower()))
    return (len(A & B) / len(A | B)) if A and B else 0.0

def dedupe(items):
    seen, out = set(), []
    for it in items:
        key = (it["title"].strip().lower(), it["url"])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def fetch_recent_sent_urls_from_telegram(days_back=HISTORY_LOOKBACK_DAYS) -> set:
    sent = set()
    try:
        updates = telegram_get_updates()
        cutoff = int(time.time()) - days_back * 86400
        for u in updates:
            msg = u.get("message") or u.get("edited_message")
            if not msg or msg.get("date", 0) < cutoff:
                continue
            if str(msg.get("chat", {}).get("id")) != str(os.environ.get("TELEGRAM_CHAT_ID")):
                continue
            ents = msg.get("entities") or []
            for e in ents:
                if e.get("type") == "text_link" and e.get("url"):
                    sent.add(clean_url(e["url"]))
    except Exception:
        pass
    return sent

def fetch_section(name, conf):
    items = []
    for r in conf.get("rss", []):
        try: items.extend(fetch_rss(r))
        except Exception: pass
    for h in conf.get("html", []):
        try: items.extend(fetch_list_html(h))
        except Exception: pass

    items = dedupe(items)

    scored = []
    for it in items:
        url = it["url"]
        title = it["title"].strip()
        s = domain_weight(url) + headline_score(title, name) + recency_score(it.get("date"))
        scored.append((s, it))

    scored.sort(key=lambda x: x[0], reverse=True)

    final, seen_urls = [], set()
    for _, it in scored:
        url = clean_url(it["url"])
        if url in seen_urls:
            continue
        dup = any(jaccard_title_sim(existing["title"], it["title"]) >= 0.8 for existing in final)
        if not dup:
            seen_urls.add(url)
            it["url"] = url
            final.append(it)
        if len(final) >= 12:
            break
    return final

# ========= Summarisation / Formatting =========
def summarize_to_sentences(text: str, max_sents=4):
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text.split()) < 40:
        return [text] if text else []
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = TextRankSummarizer()
    sents = [str(s) for s in summ(parser.document, max_sents)]
    if not sents:
        sents = re.split(r"(?<=[.!?])\s+", text)[:max_sents]
    sents = [s if s.endswith(('.', '!', '?')) else s + '.' for s in sents]
    return [s for s in sents if len(s.split()) > 3][:max_sents]

def bold_key_bits(s: str) -> str:
    patterns = [
        r"(£|\$|€)\s?\d[\d,\.]*",
        r"\b\d{1,3}(,\d{3})+(\.\d+)?\b",
        r"\b\d+(\.\d+)?\s?%",
        r"\b(?:Q[1-4]|H[1-2]|FY\d{2}|FY\d{4})\b",
        r"\b(?:\d{4}|\d{1,2}\s?[A-Z][a-z]{2,9})\b"
    ]
    for p in patterns:
        s = re.sub(p, lambda m: f"<b>{m.group(0)}</b>", s)
    return s

def html_link(text, url):
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(text)}</a>'

def make_bullets(text: str, max_sents=4):
    return [f"- {bold_key_bits(s)}" for s in summarize_to_sentences(text, max_sents=max_sents)]

def country_flag_for_ai(source_domain: str, title: str):
    domain = source_domain.lower()
    t = title.lower()
    mapping = {
        "openai.com": "🇺🇸", "blog.google": "🇺🇸", "deepmind.google": "🇬🇧",
        "anthropic.com": "🇺🇸", "mistral.ai": "🇪🇺", "stability.ai": "🇬🇧",
        "deepseek": "🇨🇳", "baidu": "🇨🇳", "alibaba": "🇨🇳", "qwen": "🇨🇳",
        "europa.eu": "🇪🇺", "aisi.gov.uk": "🇬🇧"
    }
    for key, flag in mapping.items():
        if key in domain or key in t:
            return flag
    return ""

def article_block(item, section="carbon", add_flag=False):
    title = item["title"].strip()
    url = item["url"]
    source = urlparse(url).netloc.replace("www.", "")
    if section == "ai" and add_flag:
        flag = country_flag_for_ai(source, title)
        if flag:
            title = f"{flag} {title}"
    try:
        text = extract_readable(url)
    except Exception:
        text = title
    bullets = make_bullets(text, MAX_BULLETS_PER_ITEM)
    link = html_link("Source", url)
    lines = [f"📰 {html.escape(title)}"] + bullets + [link]
    return "\n".join(lines)

# ========= Crypto prices / Meme watch =========
def coingecko_prices(ids):
    qs = ",".join(ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={qs}&vs_currencies=usd&include_24hr_change=true"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def crypto_prices_block():
    ids_map = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "ripple": "XRP",
        "solana": "SOL",
        "dogecoin": "DOGE",
        "shiba-inu": "SHIB",
        "pepe": "PEPE"
    }
    data = coingecko_prices(list(ids_map.keys()))
    lines = ["💰 Crypto Prices (07:30 UK)"]
    for k in ["bitcoin", "ethereum", "ripple", "solana"]:
        if k in data:
            sym = ids_map[k]
            price = data[k]["usd"]
            chg = data[k].get("usd_24h_change", 0.0)
            lines.append(f"- {sym}: ${price:,.2f} ({chg:+.1f}%)")
    notable = []
    for k in ["dogecoin", "shiba-inu", "pepe"]:
        if k in data:
            chg = data[k].get("usd_24h_change", 0.0)
            if abs(chg) >= 10:
                notable.append((k, chg, data[k]["usd"]))
    if notable:
        lines.append("\nMeme coin watch")
        for k, chg, price in sorted(notable, key=lambda x: -abs(x[1])):
            sym = ids_map[k]
            lines.append(f"- {sym} moved {chg:+.1f}% in 24h to ${price:,.6f}".rstrip("0").rstrip("."))
    return "\n".join(lines)

# ========= LinkedIn (public-only) =========
def linkedin_blocks(urls):
    blocks = []
    for u in urls or []:
        try:
            res = requests.get(u, headers=HEADERS, timeout=20)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "lxml")
            title = soup.find("meta", property="og:title")
            desc = soup.find("meta", property="og:description")
            t = title["content"].strip() if title and title.get("content") else "LinkedIn post"
            d = desc["content"].strip() if desc and desc.get("content") else "Public LinkedIn update."
            parts = re.split(r"(?<=[.!?])\s+", d)
            d = " ".join(parts[:2])
            d = bold_key_bits(d)
            blocks.append(f"{html.escape(t)}\n- {d}\n{html_link('View post', u)}")
        except Exception:
            pass
    return blocks[:3]

# ========= Build & Send =========
def build_message(sections):
    d = now_london()
    header = f"🌅 Daily Carbon–AI–Crypto — {d.strftime('%a, %d %b %Y')} (07:30 UK)"
    parts = [header]

    parts.append("\n🌍 Carbon Markets\n" + "━"*16)
    parts.extend(sections.get("carbon_blocks") or ["- No major updates worth your time today."])

    parts.append("\n🤖 AI Trends\n" + "━"*16)
    parts.extend(sections.get("ai_blocks") or ["- No major updates worth your time today."])

    parts.append("\n₿ Crypto\n" + "━"*16)
    parts.extend(sections.get("crypto_blocks") or ["- No major updates worth your time today."])

    if sections.get("prices_block"):
        parts.append(sections["prices_block"])

    if sections.get("linkedin_blocks"):
        parts.append("\n🧵 From LinkedIn\n" + "━"*16)
        parts.extend(sections["linkedin_blocks"])

    # Telegram 4096 char limit
    full = "\n\n".join(parts).strip()
    chunks = []
    while len(full) > 4096:
        cut = full.rfind("\n\n", 0, 3900)
        if cut == -1:
            cut = 3900
        chunks.append(full[:cut])
        full = full[cut:]
    chunks.append(full)
    return chunks

# (Optional) helpful debug prints so you can see the decision path
def within_send_window():
    print(f"[dbg] London now: {now_london().strftime('%Y-%m-%d %H:%M:%S')}")
    if os.getenv("FORCE_SEND") == "1":
        print("[dbg] FORCE_SEND=1 -> sending.")
        return True
    if user_sent_update_recently(max_age_seconds=FRESHNESS_WINDOW_SEC):
        print("[dbg] Recent 'update' detected -> sending.")
        return True
    t = now_london()
    inwin = (t.hour == 7 and 30 <= t.minute < 35)
    print(f"[dbg] In 07:30 window? {inwin}")
    return inwin

def main():
    if not within_send_window():
        print("Not in send window; exiting without sending.")
        return

    conf = load_sources("sources.yml")
    sections = {}

    # Avoid repeats across days
    already_sent = fetch_recent_sent_urls_from_telegram(days_back=HISTORY_LOOKBACK_DAYS)

    for cat in ("carbon", "ai", "crypto"):
        raw_items = fetch_section(cat, conf.get(cat, {}))
        fresh_items = [it for it in raw_items if clean_url(it["url"]) not in already_sent]
        items = fresh_items or raw_items
        blocks, picked = [], 0
        for it in items:
            try:
                blocks.append(article_block(it, section=cat, add_flag=True))
                picked += 1
                if picked >= MAX_ITEMS_PER_SECTION:
                    break
            except Exception as e:
                print(f"Skip {cat} item: {e}")
        sections[f"{cat}_blocks"] = blocks

    try:
        sections["prices_block"] = crypto_prices_block()
    except Exception as e:
        print(f"Prices error: {e}")
        sections["prices_block"] = ""

    sections["linkedin_blocks"] = linkedin_blocks(conf.get("linkedin_public_posts", []))

    # Acknowledge manual update
    if user_sent_update_recently(max_age_seconds=FRESHNESS_WINDOW_SEC):
        try:
            send_to_telegram("✅ Update received — sending the latest digest now.")
        except Exception:
            pass

    for i, m in enumerate(build_message(sections), 1):
        resp = send_to_telegram(m)
        print(f"[dbg] sent chunk {i}, length={len(m)} chars, ok={bool(resp)}")
        time.sleep(1)

if __name__ == "__main__":
    main()

