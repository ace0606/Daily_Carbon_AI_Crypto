import os, re, textwrap, time, html
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode

import requests, feedparser, yaml
from bs4 import BeautifulSoup
from dateutil import parser as dtparser, tz
from readability import Document
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ---------- Config
HEADERS = {"User-Agent": "Mozilla/5.0 (news-bot; +https://github.com/)"}
LONDON = tz.gettz("Europe/London")
MAX_ITEMS_PER_SECTION = 3
MAX_BULLETS_PER_ITEM = 4
UPDATE_KEYWORDS = {"update", "/update"}  # messages that trigger on-demand send

# ---------- Time helpers
def now_london():
    return datetime.now(LONDON)

def within_send_window():
    # Allow env override OR user 'update' in last 5 minutes
    if os.getenv("FORCE_SEND") == "1" or user_sent_update_recently(max_age_seconds=300):
        return True
    t = now_london()
    return t.hour == 7 and 30 <= t.minute < 35

# ---------- Telegram helpers
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
    """True if the latest message from YOUR chat says 'update' or '/update' and is recent."""
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

# ---------- Fetch/parse helpers
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

def dedupe(items):
    seen, out = set(), []
    for it in items:
        key = (it["title"].strip().lower(), it["url"])
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

# ---------- Summarisation & formatting
def summarize_to_sentences(text: str, max_sents=4):
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text.split()) < 40:
        return [text] if text else []
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = TextRankSummarizer()
    sents = [str(s) for s in summ(parser.document, max_sents)]
    if not sents:
        # fallback: first N sentences
        sents = re.split(r"(?<=[.!?])\s+", text)[:max_sents]
    # Full sentences with terminal punctuation
    sents = [s if s.endswith(('.', '!', '?')) else s + '.' for s in sents]
    return [s for s in sents if len(s.split()) > 3][:max_sents]

def bold_key_bits(s: str) -> str:
    # Bold numbers, currencies, percents, dates, short all-caps (≤5)
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

def fetch_section(name, conf):
    items = []
    for r in conf.get("rss", []):
        try:
            items.extend(fetch_rss(r))
        except Exception:
            pass
    for h in conf.get("html", []):
        try:
            items.extend(fetch_list_html(h))
        except Exception:
            pass
    items = dedupe(items)[:8]
    return items

# ---------- Crypto prices & meme watcher
def coingecko_prices(ids):
    qs = ",".join(ids)
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={qs}&vs_currencies=usd&include_24hr_change=true"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def crypto_prices_block():
    ids_map = {
        "bitcoin": "BTC", "ethereum": "ETH", "ripple": "XRP", "solana": "SOL",
        "dogecoin": "DOGE", "shiba-inu": "SHIB", "pepe": "PEPE"
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

# ---------- LinkedIn (public URLs only)
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
            d = re.split(r"(?<=[.!?])\s+", d)
            d = " ".join(d[:2])  # up to 2 sentences
            d = bold_key_bits(d)
            blocks.append(f"{html.escape(t)}\n- {d}\n{html_link('View post', u)}")
        except Exception:
            pass
    return blocks[:3]

# ---------- Build & send
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

    # Telegram 4096 chars limit
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

def main():
    if not within_send_window():
        print("Not in send window; exiting without sending.")
        return

    conf = load_sources("sources.yml")

    sections = {}
    # Collect for each category
    for cat in ("carbon", "ai", "crypto"):
        items = fetch_section(cat, conf.get(cat, {}))
        blocks = []
        for it in items[:MAX_ITEMS_PER_SECTION]:
            try:
                blocks.append(article_block(it, section=cat, add_flag=True))
            except Exception as e:
                print(f"Skip {cat} item: {e}")
        sections[f"{cat}_blocks"] = blocks

    # Prices
    try:
        sections["prices_block"] = crypto_prices_block()
    except Exception as e:
        print(f"Prices error: {e}")
        sections["prices_block"] = ""

    # LinkedIn (public only)
    sections["linkedin_blocks"] = linkedin_blocks(conf.get("linkedin_public_posts", []))

    # Acknowledge manual update
    if user_sent_update_recently():
        try:
            send_to_telegram("✅ Update received — sending the latest digest now.")
        except Exception:
            pass

    messages = build_message(sections)
    for m in messages:
        send_to_telegram(m)
        time.sleep(1)

if __name__ == "__main__":
    main()

