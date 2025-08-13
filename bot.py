{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os, re, textwrap, time, json, html\
from datetime import datetime, timedelta\
from urllib.parse import urlparse\
import requests, feedparser, yaml\
from bs4 import BeautifulSoup\
from dateutil import parser as dtparser, tz\
from readability import Document\
from sumy.parsers.plaintext import PlaintextParser\
from sumy.nlp.tokenizers import Tokenizer\
from sumy.summarizers.text_rank import TextRankSummarizer\
\
LONDON = tz.gettz("Europe/London")\
HEADERS = \{"User-Agent": "Mozilla/5.0 (news-bot; +https://github.com/)"\}\
\
# ---------- Utilities\
\
def now_london():\
    return datetime.now(LONDON)\
\
def within_send_window():\
    # Only send between 07:30 and 07:35 London time (job runs 06:30 & 07:30 UTC)\
    t = now_london()\
    return t.hour == 7 and 30 <= t.minute < 35\
\
def clean_url(u: str) -> str:\
    # Strip tracking params\
    try:\
        from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode\
        parts = list(urlsplit(u))\
        if parts[3]:\
            qs = [(k,v) for k,v in parse_qsl(parts[3]) if not k.lower().startswith("utm")]\
            parts[3] = urlencode(qs)\
        return urlunsplit(parts)\
    except Exception:\
        return u\
\
def bold_key_bits(s: str) -> str:\
    # Bold numbers, currencies, percents, dates, and all-caps abbreviations \uc0\u8804 5 chars\
    patterns = [\
        r"(\'a3|\\$|\'80)\\s?\\d[\\d\\,\\.]*", r"\\b\\d\{1,3\}(,\\d\{3\})+(\\.\\d+)?\\b",\
        r"\\b\\d+(\\.\\d+)?\\s?%","\\\\b(?:Q[1-4]|H[1-2]|FY\\d\{2\}|FY\\d\{4\})\\\\b",\
        r"\\b(?:\\d\{4\}|\\d\{1,2\}\\s?[A-Z][a-z]\{2,9\})\\b"\
    ]\
    def repl(m): return f"<b>\{m.group(0)\}</b>"\
    for p in patterns:\
        s = re.sub(p, repl, s)\
    return s\
\
def html_link(text, url):\
    return f'<a href="\{html.escape(url, quote=True)\}">\{html.escape(text)\}</a>'\
\
def clamp_sentences(text, max_sentences=4):\
    # Ensure 3\'964 full sentences.\
    sents = re.split(r'(?<=[.!?])\\s+', text.strip())\
    sents = [s for s in sents if len(s.split()) > 3]\
    return " ".join(sents[:max_sentences])\
\
def summarize_to_sentences(text, max_sents=4):\
    text = re.sub(r'\\s+', ' ', text).strip()\
    if not text or len(text.split()) < 40:\
        return [text] if text else []\
    parser = PlaintextParser.from_string(text, Tokenizer("english"))\
    summ = TextRankSummarizer()\
    sents = [str(s) for s in summ(parser.document, max_sents)]\
    # Fallback if TextRank returns nothing\
    if not sents:\
        sents = clamp_sentences(text, max_sents).split(". ")\
    # Make each a full sentence\
    sents = [s if s.endswith(('.', '!', '?')) else s + '.' for s in sents]\
    return sents[:max_sents]\
\
def extract_readable(url):\
    res = requests.get(url, headers=HEADERS, timeout=20)\
    res.raise_for_status()\
    doc = Document(res.text)\
    content_html = doc.summary(html_partial=True)\
    soup = BeautifulSoup(content_html, "lxml")\
    # Remove scripts/styles\
    for tag in soup(["script", "style", "noscript"]): tag.decompose()\
    text = soup.get_text(separator=" ", strip=True)\
    return text\
\
def fetch_rss(url):\
    feed = feedparser.parse(url)\
    items = []\
    for e in feed.entries[:10]:\
        url = clean_url(e.link)\
        title = e.title\
        date = None\
        for key in ("published", "updated"):\
            if getattr(e, key, None):\
                try:\
                    date = dtparser.parse(getattr(e,key))\
                    break\
                except: pass\
        items.append(\{"title": title, "url": url, "date": date\})\
    return items\
\
def fetch_list_html(url):\
    # Fetch list page and grab obvious article links\
    res = requests.get(url, headers=HEADERS, timeout=20)\
    res.raise_for_status()\
    soup = BeautifulSoup(res.text, "lxml")\
    links = []\
    for a in soup.select("a[href]"):\
        href = a["href"]\
        if href.startswith("#"): continue\
        if href.startswith("/"): href = f"\{urlparse(url).scheme\}://\{urlparse(url).netloc\}\{href\}"\
        text = a.get_text(strip=True)\
        # Heuristic: keep longer texts that look like headlines\
        if len(text.split()) >= 4:\
            links.append((text, clean_url(href)))\
    # Deduplicate by URL\
    seen, out = set(), []\
    for t,u in links:\
        if u not in seen:\
            seen.add(u)\
            out.append(\{"title": t, "url": u, "date": None\})\
    return out[:10]\
\
def load_sources(path="sources.yml"):\
    with open(path, "r", encoding="utf-8") as f:\
        return yaml.safe_load(f)\
\
def country_flag_for_ai(source_domain, title):\
    # Map by domain or keywords\
    domain = source_domain.lower()\
    t = title.lower()\
    mapping = \{\
        "openai.com": "\uc0\u55356 \u56826 \u55356 \u56824 ", "blog.google": "\u55356 \u56826 \u55356 \u56824 ", "deepmind.google": "\u55356 \u56812 \u55356 \u56807 ",\
        "anthropic.com": "\uc0\u55356 \u56826 \u55356 \u56824 ", "mistral.ai": "\u55356 \u56810 \u55356 \u56826 ", "stability.ai": "\u55356 \u56812 \u55356 \u56807 ",\
        "deepseek": "\uc0\u55356 \u56808 \u55356 \u56819 ", "baidu": "\u55356 \u56808 \u55356 \u56819 ", "alibaba": "\u55356 \u56808 \u55356 \u56819 ", "qwen": "\u55356 \u56808 \u55356 \u56819 ",\
        "europa.eu": "\uc0\u55356 \u56810 \u55356 \u56826 ", "aisi.gov.uk": "\u55356 \u56812 \u55356 \u56807 "\
    \}\
    for key, flag in mapping.items():\
        if key in domain or key in t:\
            return flag\
    return ""  # none\
\
def dedupe(items):\
    seen = set(); out=[]\
    for it in items:\
        key = (it["title"].strip().lower(), it["url"])\
        if key not in seen:\
            seen.add(key); out.append(it)\
    return out\
\
def recent_only(items, days=2):\
    cutoff = datetime.utcnow() - timedelta(days=days)\
    out=[]\
    for it in items:\
        if it["date"] is None:\
            out.append(it)  # keep when unknown but will be filtered by content later\
        else:\
            if it["date"].replace(tzinfo=None) >= cutoff:\
                out.append(it)\
    return out\
\
def make_bullets(text, max_sents=4):\
    sents = summarize_to_sentences(text, max_sents=max_sents)\
    bullets = []\
    for s in sents:\
        s = bold_key_bits(s)\
        bullets.append(f"- \{s\}")\
    return bullets\
\
def article_to_block(item, section="carbon", add_flag=False):\
    title = item["title"].strip()\
    url = item["url"]\
    source = urlparse(url).netloc.replace("www.","")\
    flag = ""\
    if section=="ai" and add_flag:\
        flag = country_flag_for_ai(source, title)\
        title = f"\{flag\} \{title\}" if flag else title\
    # Get content\
    try:\
        text = extract_readable(url)\
    except Exception:\
        text = title\
    bullets = make_bullets(text, max_sents=4)[:4]\
    link = html_link("Source", url)\
    block = []\
    block.append(f"\uc0\u55357 \u56560  \{html.escape(title)\}")\
    block.extend(bullets)\
    block.append(link)\
    return "\\n".join(block)\
\
def fetch_section(name, conf):\
    items=[]\
    for r in conf.get("rss", []):\
        try: items.extend(fetch_rss(r))\
        except: pass\
    for h in conf.get("html", []):\
        try: items.extend(fetch_list_html(h))\
        except: pass\
    items = dedupe(items)\
    items = items[:8]\
    return items\
\
def coingecko_prices(ids):\
    # Free, no key\
    qs = ",".join(ids)\
    url = f"https://api.coingecko.com/api/v3/simple/price?ids=\{qs\}&vs_currencies=usd&include_24hr_change=true"\
    r = requests.get(url, timeout=20, headers=HEADERS)\
    r.raise_for_status()\
    return r.json()\
\
def crypto_prices_block():\
    ids_map = \{\
        "bitcoin":"BTC","ethereum":"ETH","ripple":"XRP","solana":"SOL",\
        "dogecoin":"DOGE","shiba-inu":"SHIB","pepe":"PEPE"\
    \}\
    data = coingecko_prices(list(ids_map.keys()))\
    # Headline prices\
    wanted = ["bitcoin","ethereum","ripple","solana"]\
    lines=["\uc0\u55357 \u56496  Crypto Prices (07:30 UK)"]\
    for k in wanted:\
        sym = ids_map[k]\
        if k in data:\
            price = data[k]["usd"]\
            chg = data[k].get("usd_24h_change", 0.0)\
            lines.append(f"- \{sym\}: $\{price:,.2f\} (\{chg:+.1f\}%)")\
    # Meme coin watch if notable\
    notable=[]\
    for mem in ["dogecoin","shiba-inu","pepe"]:\
        if mem in data:\
            chg = data[mem].get("usd_24h_change", 0.0)\
            if abs(chg) >= 10:\
                notable.append((mem, chg, data[mem]["usd"]))\
    if notable:\
        lines.append("\\nMeme coin watch")\
        for mem, chg, price in sorted(notable, key=lambda x: -abs(x[1])):\
            sym = ids_map[mem]\
            lines.append(f"- \{sym\} moved \{chg:+.1f\}% in 24h to $\{price:,.6f\}".rstrip("0").rstrip("."))\
    return "\\n".join(lines)\
\
def build_message(sections):\
    d = now_london()\
    header = f"\uc0\u55356 \u57093  Daily Carbon\'96AI\'96Crypto \'97 \{d.strftime('%a, %d %b %Y')\} (07:30 UK)"\
    parts = [header]\
    # Carbon\
    parts.append("\\n\uc0\u55356 \u57101  Carbon Markets\\n" + "\u9473 "*16)\
    parts.extend(sections.get("carbon_blocks", []) or ["- No major updates worth your time today."])\
    # AI\
    parts.append("\\n\uc0\u55358 \u56598  AI Trends\\n" + "\u9473 "*16)\
    parts.extend(sections.get("ai_blocks", []) or ["- No major updates worth your time today."])\
    # Crypto\
    parts.append("\\n\uc0\u8383  Crypto\\n" + "\u9473 "*16)\
    parts.extend(sections.get("crypto_blocks", []) or ["- No major updates worth your time today."])\
    parts.append(sections.get("prices_block",""))\
    # LinkedIn (optional)\
    if sections.get("linkedin_blocks"):\
        parts.append("\\n\uc0\u55358 \u56821  From LinkedIn\\n" + "\u9473 "*16)\
        parts.extend(sections["linkedin_blocks"])\
    # Telegram message limit is 4096 chars; split if needed\
    full = "\\n\\n".join(parts).strip()\
    chunks=[]\
    while len(full) > 4096:\
        cut = full.rfind("\\n\\n", 0, 3900)\
        if cut == -1: cut = 3900\
        chunks.append(full[:cut])\
        full = full[cut:]\
    chunks.append(full)\
    return chunks\
\
def linkedin_blocks(urls):\
    blocks=[]\
    for u in urls or []:\
        try:\
            res = requests.get(u, headers=HEADERS, timeout=20)\
            res.raise_for_status()\
            soup = BeautifulSoup(res.text, "lxml")\
            title = soup.find("meta", property="og:title")\
            desc = soup.find("meta", property="og:description")\
            t = title["content"].strip() if title and title.get("content") else "LinkedIn post"\
            d = desc["content"].strip() if desc and desc.get("content") else "Public LinkedIn update."\
            d = clamp_sentences(d, 2)\
            d = bold_key_bits(d)\
            blocks.append(f"\{html.escape(t)\}\\n- \{d\}\\n\{html_link('View post', u)\}")\
        except: \
            pass\
    return blocks[:3]\
\
def send_to_telegram(text):\
    token = os.environ["TELEGRAM_BOT_TOKEN"]\
    chat_id = os.environ["TELEGRAM_CHAT_ID"]\
    url = f"https://api.telegram.org/bot\{token\}/sendMessage"\
    payload = \{\
        "chat_id": chat_id,\
        "text": text,\
        "parse_mode": "HTML",\
        "disable_web_page_preview": True\
    \}\
    r = requests.post(url, json=payload, timeout=20)\
    r.raise_for_status()\
    return r.json()\
\
def main():\
    if not within_send_window():\
        print("Not in 07:30 London window; exiting without sending.")\
        return\
\
    conf = load_sources("sources.yml")\
\
    # Collect sections\
    sections = \{\}\
    for cat in ("carbon","ai","crypto"):\
        items = fetch_section(cat, conf.get(cat, \{\}))\
        blocks=[]\
        for it in items[:3]:  # up to 3 stories per section\
            try:\
                blocks.append(article_to_block(it, section=cat, add_flag=True))\
            except Exception as e:\
                print(f"Skip \{cat\} item error: \{e\}")\
        sections[f"\{cat\}_blocks"] = blocks\
\
    # Crypto prices & meme coins\
    try:\
        sections["prices_block"] = crypto_prices_block()\
    except Exception as e:\
        sections["prices_block"] = ""\
\
    # LinkedIn (optional)\
    sections["linkedin_blocks"] = linkedin_blocks(conf.get("linkedin_public_posts", []))\
\
    # Build & send\
    messages = build_message(sections)\
    for m in messages:\
        send_to_telegram(m)\
        time.sleep(1)\
\
if __name__ == "__main__":\
    main()\
}