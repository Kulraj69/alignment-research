#!/usr/bin/env python3
"""
Fetch real-world alignment signals and save them as dated artifacts.

Sources:
- OpenAI News RSS (official)
- Anthropic sitemap (official)
- arXiv API (papers)

Outputs:
- data/real_world/alignment_signals_<YYYY-MM-DD>.json
- data/real_world/alignment_signals_<YYYY-MM-DD>.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

USER_AGENT = "Mozilla/5.0 (compatible; CVAT-ResearchBot/0.1; +https://github.com/Kulraj69/alignment-research)"

OPENAI_RSS_URL = "https://openai.com/news/rss.xml"
ANTHROPIC_SITEMAP_URL = "https://www.anthropic.com/sitemap.xml"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

SAFETY_KEYWORDS = [
    "alignment faking",
    "ai alignment",
    "alignment safety",
    "model alignment",
    "superalignment",
    "evaluation awareness",
    "safety",
    "preparedness",
    "system card",
    "red team",
    "red teaming",
    "scheming",
    "deception",
    "oversight",
    "reward hacking",
    "jailbreak",
    "faithfulness",
    "mechanistic interpretability",
    "sleeper agent",
]

AI_CONTEXT_KEYWORDS = [
    "ai",
    "artificial intelligence",
    "language model",
    "llm",
    "transformer",
    "agentic",
]

OPENAI_FOCUS_KEYWORDS = [
    "system card",
    "preparedness",
    "model spec",
    "red team",
    "bug bounty",
    "cyber resilience",
    "safety",
    "security",
    "policy",
    "responsibility",
]

ARXIV_QUERY = (
    'cat:cs.AI AND (all:"alignment" OR all:"ai safety" OR '
    'all:"mechanistic interpretability" OR all:"deception" OR all:"oversight" '
    'OR all:"red teaming" OR all:"reward hacking" OR all:"jailbreak")'
)

TARGET_ARXIV_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL"}


def fetch_text(url: str, retries: int = 3, backoff_seconds: float = 2.0, optional: bool = False) -> str:
    """
    Fetch URL text with retry/backoff for transient failures.

    If optional=True, returns empty string after retries instead of raising.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    transient_codes = {429, 500, 502, 503, 504}

    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            should_retry = exc.code in transient_codes and attempt < retries
            if should_retry:
                sleep_s = backoff_seconds * (2 ** attempt)
                print(f"[Fetch] HTTP {exc.code} for {url} (retry in {sleep_s:.1f}s)")
                time.sleep(sleep_s)
                continue
            if optional:
                print(f"[Fetch] WARNING: optional source failed ({exc.code}) for {url}. Continuing without it.")
                return ""
            raise
        except Exception as exc:  # noqa: BLE001
            if attempt < retries:
                sleep_s = backoff_seconds * (2 ** attempt)
                print(f"[Fetch] ERROR for {url}: {exc} (retry in {sleep_s:.1f}s)")
                time.sleep(sleep_s)
                continue
            if optional:
                print(f"[Fetch] WARNING: optional source failed for {url}: {exc}. Continuing without it.")
                return ""
            raise

    if optional:
        return ""
    raise RuntimeError(f"Failed to fetch {url} after retries")


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def score_keywords(text: str, keywords: List[str]) -> Dict[str, Any]:
    hay = text.lower()
    matched = [kw for kw in keywords if kw in hay]
    return {"score": len(matched), "matched_keywords": matched}


def compute_relevance(text: str) -> Dict[str, Any]:
    safety = score_keywords(text, SAFETY_KEYWORDS)
    ai_ctx = score_keywords(text, AI_CONTEXT_KEYWORDS)
    total = safety["score"] + ai_ctx["score"]
    return {
        "score": total,
        "safety_score": safety["score"],
        "ai_context_score": ai_ctx["score"],
        "safety_keywords": safety["matched_keywords"],
        "ai_context_keywords": ai_ctx["matched_keywords"],
        "matched_keywords": sorted(set(safety["matched_keywords"] + ai_ctx["matched_keywords"])),
    }


def iso_date_from_any(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = value.strip()

    # RSS style: Tue, 15 Apr 2025 12:00:00 +0000
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        return parsed.date().isoformat()
    except Exception:
        pass

    # ISO timestamp
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            parsed = dt.datetime.strptime(value, fmt)
            return parsed.date().isoformat()
        except Exception:
            continue

    return None


def within_lookback(date_iso: Optional[str], lookback_days: int, today: dt.date) -> bool:
    if not date_iso:
        return True
    try:
        event_date = dt.date.fromisoformat(date_iso)
    except ValueError:
        return True
    return (today - event_date).days <= lookback_days


def parse_openai_rss(xml_text: str, lookback_days: int, today: dt.date) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    items: List[Dict[str, Any]] = []
    for item in channel.findall("item"):
        title = norm_text(item.findtext("title") or "")
        link = norm_text(item.findtext("link") or "")
        pub_raw = norm_text(item.findtext("pubDate") or "")
        desc = norm_text(item.findtext("description") or "")
        date_iso = iso_date_from_any(pub_raw)

        if not within_lookback(date_iso, lookback_days, today):
            continue

        keyword_info = compute_relevance(f"{title} {desc}")
        focus_info = score_keywords(f"{title} {link} {desc}", OPENAI_FOCUS_KEYWORDS)
        if keyword_info["safety_score"] == 0 or focus_info["score"] == 0:
            continue

        items.append(
            {
                "source": "openai_news_rss",
                "title": title,
                "url": link,
                "date": date_iso,
                "snippet": desc,
                "score": keyword_info["score"],
                "safety_score": keyword_info["safety_score"],
                "ai_context_score": keyword_info["ai_context_score"],
                "matched_keywords": keyword_info["matched_keywords"],
                "openai_focus_keywords": focus_info["matched_keywords"],
            }
        )

    return items


def slug_to_title(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    slug = slug.replace("-", " ")
    return slug.title()


def parse_anthropic_sitemap(xml_text: str, lookback_days: int, today: dt.date) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    items: List[Dict[str, Any]] = []
    for url_node in root.findall("sm:url", ns):
        loc = norm_text(url_node.findtext("sm:loc", default="", namespaces=ns))
        if "/research/" not in loc:
            continue

        lastmod_raw = norm_text(url_node.findtext("sm:lastmod", default="", namespaces=ns))
        date_iso = iso_date_from_any(lastmod_raw)
        if not within_lookback(date_iso, lookback_days, today):
            continue

        title_guess = slug_to_title(loc)
        keyword_info = compute_relevance(f"{title_guess} {loc}")
        if keyword_info["safety_score"] == 0:
            continue

        items.append(
            {
                "source": "anthropic_sitemap",
                "title": title_guess,
                "url": loc,
                "date": date_iso,
                "snippet": "Matched via Anthropic research sitemap.",
                "score": keyword_info["score"],
                "safety_score": keyword_info["safety_score"],
                "ai_context_score": keyword_info["ai_context_score"],
                "matched_keywords": keyword_info["matched_keywords"],
            }
        )

    return items


def parse_arxiv_atom(xml_text: str, lookback_days: int, today: dt.date) -> List[Dict[str, Any]]:
    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    items: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ns):
        title = norm_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = norm_text(entry.findtext("atom:summary", default="", namespaces=ns))
        id_url = norm_text(entry.findtext("atom:id", default="", namespaces=ns))
        published_raw = norm_text(entry.findtext("atom:published", default="", namespaces=ns))
        updated_raw = norm_text(entry.findtext("atom:updated", default="", namespaces=ns))

        date_iso = iso_date_from_any(updated_raw) or iso_date_from_any(published_raw)
        if not within_lookback(date_iso, lookback_days, today):
            continue

        authors = [
            norm_text(a.findtext("atom:name", default="", namespaces=ns))
            for a in entry.findall("atom:author", ns)
        ]
        categories = [
            cat.attrib.get("term", "")
            for cat in entry.findall("atom:category", ns)
            if cat.attrib.get("term")
        ]
        if categories and not any(c in TARGET_ARXIV_CATEGORIES for c in categories):
            continue

        keyword_info = compute_relevance(f"{title} {summary}")
        # arXiv is broad: require both safety signal and AI context to reduce noise
        if keyword_info["safety_score"] == 0 or keyword_info["ai_context_score"] == 0:
            continue

        items.append(
            {
                "source": "arxiv_api",
                "title": title,
                "url": id_url,
                "date": date_iso,
                "snippet": summary[:420],
                "authors": authors,
                "categories": categories,
                "score": keyword_info["score"],
                "safety_score": keyword_info["safety_score"],
                "ai_context_score": keyword_info["ai_context_score"],
                "matched_keywords": keyword_info["matched_keywords"],
            }
        )

    return items


def sort_key(item: Dict[str, Any]) -> Any:
    date_str = item.get("date") or "1900-01-01"
    return (item.get("score", 0), date_str)


def build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Real-World Alignment Signals")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at_utc']}")
    lines.append(f"Lookback window: {payload['lookback_days']} days")
    lines.append("")

    totals = payload["counts"]
    lines.append("## Source Counts")
    lines.append("")
    lines.append(f"- OpenAI RSS matches: {totals['openai']}" )
    lines.append(f"- Anthropic sitemap matches: {totals['anthropic']}" )
    lines.append(f"- arXiv API matches: {totals['arxiv']}" )
    lines.append(f"- Total: {totals['total']}" )
    lines.append("")

    lines.append("## Top Signals")
    lines.append("")
    top_items = payload["top_signals"]
    if not top_items:
        lines.append("- No matching signals found for current filter.")
    else:
        for item in top_items:
            date = item.get("date") or "unknown-date"
            kws = ", ".join(item.get("matched_keywords", []))
            lines.append(f"- [{item['title']}]({item['url']}) | {item['source']} | {date} | kws: {kws}")
    lines.append("")

    lines.append("## Why This Helps CVAT")
    lines.append("")
    lines.append("- Grounds experiment design in latest external safety discourse.")
    lines.append("- Provides fresh candidates for watched/unwatched prompt templates.")
    lines.append("- Supports weekly updates with reproducible data collection.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real-world alignment signals.")
    parser.add_argument("--lookback-days", type=int, default=180, help="Keep items within this many days.")
    parser.add_argument("--arxiv-max-results", type=int, default=40, help="arXiv max results to request.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/real_world"),
        help="Directory to write artifacts.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    date_tag = today.isoformat()
    json_path = args.out_dir / f"alignment_signals_{date_tag}.json"
    md_path = args.out_dir / f"alignment_signals_{date_tag}.md"

    print("[Fetch] OpenAI RSS")
    openai_xml = fetch_text(OPENAI_RSS_URL)
    print("[Fetch] Anthropic sitemap")
    anthropic_xml = fetch_text(ANTHROPIC_SITEMAP_URL)

    arxiv_params = {
        "search_query": ARXIV_QUERY,
        "start": 0,
        "max_results": args.arxiv_max_results,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
    }
    arxiv_url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(arxiv_params)}"
    print("[Fetch] arXiv API")
    arxiv_xml = fetch_text(arxiv_url, optional=True)

    openai_items = parse_openai_rss(openai_xml, args.lookback_days, today)
    anthropic_items = parse_anthropic_sitemap(anthropic_xml, args.lookback_days, today)
    arxiv_items = parse_arxiv_atom(arxiv_xml, args.lookback_days, today)

    if not arxiv_xml.strip() and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            backup_arxiv = [item for item in existing_payload.get("all_signals", []) if item.get("source") == "arxiv_api"]
            if backup_arxiv:
                print(f"[Fetch] Reusing {len(backup_arxiv)} cached arXiv items from {json_path}")
                arxiv_items = backup_arxiv
        except Exception as exc:  # noqa: BLE001
            print(f"[Fetch] WARNING: failed to reuse cached arXiv data: {exc}")

    all_items = sorted(openai_items + anthropic_items + arxiv_items, key=sort_key, reverse=True)

    payload = {
        "generated_at_utc": generated_at,
        "lookback_days": args.lookback_days,
        "query": {
            "openai_rss": OPENAI_RSS_URL,
            "anthropic_sitemap": ANTHROPIC_SITEMAP_URL,
            "arxiv_api_url": arxiv_url,
            "safety_keywords": SAFETY_KEYWORDS,
            "ai_context_keywords": AI_CONTEXT_KEYWORDS,
            "openai_focus_keywords": OPENAI_FOCUS_KEYWORDS,
        },
        "counts": {
            "openai": len(openai_items),
            "anthropic": len(anthropic_items),
            "arxiv": len(arxiv_items),
            "total": len(all_items),
        },
        "top_signals": all_items[:25],
        "all_signals": all_items,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    markdown = build_markdown(payload)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print("Saved:")
    print(f"- {json_path}")
    print(f"- {md_path}")
    print(f"Counts: {payload['counts']}")


if __name__ == "__main__":
    main()
