"""
web_scraper.py — Smart web scraper for the College RAG Chatbot.

Owner: Opus (Antigravity)
Handles: Fetching college website pages, extracting clean content text,
         stripping navigation/header/footer boilerplate, and producing chunks.

Designed specifically for college websites like nitte.edu.in/nmit/ which have
heavy navigation menus that would pollute the knowledge base if not filtered.

Integration contract:
  - scrape_url(url) → str                  (returns clean text)
  - scrape_and_chunk(url, ...) → List[Dict] (returns chunked dicts)
  - scrape_multiple_urls(urls) → List[Dict] (batch scrape + chunk)
"""

import re
import time
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup, Comment

from document_processor import chunk_text


# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════

# Default request headers (polite scraping)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# HTML elements to completely remove (they never contain useful content)
NOISE_ELEMENTS = [
    "script", "style", "noscript", "iframe",
    "nav", "header", "footer",
    "form", "button", "input", "select",
    "meta", "link",
]

# Common CSS class/id patterns for navigation and boilerplate
# These get stripped to avoid indexing menu items as content
NOISE_PATTERNS = [
    r"nav", r"menu", r"sidebar", r"footer", r"header",
    r"breadcrumb", r"social", r"share", r"cookie",
    r"popup", r"modal", r"banner", r"advertisement",
    r"quick-link", r"quicklink", r"top-bar", r"topbar",
]


# ═══════════════════════════════════════════════════════
#  CORE SCRAPING
# ═══════════════════════════════════════════════════════

def scrape_url(url: str, timeout: int = 15) -> str:
    """
    Fetch a URL and extract clean, meaningful text content.

    Strips navigation menus, headers, footers, scripts, styles,
    and other boilerplate that college websites are full of.

    Args:
        url:     The URL to scrape.
        timeout: Request timeout in seconds.

    Returns:
        Clean text content from the page. Empty string on failure.
    """
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Step 1: Remove noise elements entirely
        _remove_noise_elements(soup)

        # Step 2: Remove elements with noisy class/id names
        _remove_noise_by_attributes(soup)

        # Step 3: Remove HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Step 4: Try to find the main content area
        main_content = _find_main_content(soup)

        # Step 5: Extract and clean text
        if main_content:
            raw_text = main_content.get_text(separator="\n", strip=True)
        else:
            raw_text = soup.get_text(separator="\n", strip=True)

        # Step 6: Clean the extracted text
        clean = _clean_scraped_text(raw_text)

        return clean

    except requests.exceptions.RequestException as e:
        print(f"[web_scraper] Request error for {url}: {e}")
        return ""
    except Exception as e:
        print(f"[web_scraper] Unexpected error for {url}: {e}")
        return ""


def scrape_and_chunk(
    url: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """
    Scrape a URL and return chunked text with metadata.

    Args:
        url:        The URL to scrape.
        chunk_size: Words per chunk.
        overlap:    Word overlap between chunks.

    Returns:
        List of chunk dicts with keys: text, source, chunk_id.
    """
    text = scrape_url(url)
    if not text:
        return []

    # Use a clean source name (domain + path, no query params)
    source_name = _url_to_source_name(url)
    return chunk_text(text, source_name=source_name, chunk_size=chunk_size, overlap=overlap)


def scrape_multiple_urls(
    urls: List[str],
    chunk_size: int = 500,
    overlap: int = 50,
    delay: float = 1.0,
) -> List[Dict[str, str]]:
    """
    Scrape multiple URLs and return all chunks combined.

    Includes a polite delay between requests to avoid overwhelming the server.

    Args:
        urls:       List of URLs to scrape.
        chunk_size: Words per chunk.
        overlap:    Word overlap between chunks.
        delay:      Seconds to wait between requests.

    Returns:
        Combined list of chunk dicts from all URLs.
    """
    all_chunks = []

    for i, url in enumerate(urls):
        print(f"[web_scraper] Scraping ({i + 1}/{len(urls)}): {url}")
        chunks = scrape_and_chunk(url, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
        print(f"  → Got {len(chunks)} chunks")

        # Polite delay (skip after last URL)
        if i < len(urls) - 1 and delay > 0:
            time.sleep(delay)

    return all_chunks


# ═══════════════════════════════════════════════════════
#  NMIT-SPECIFIC: PREDEFINED URLs
# ═══════════════════════════════════════════════════════

# Key pages from nitte.edu.in/nmit/ that contain useful college info
NMIT_DEFAULT_URLS = [
    "https://nitte.edu.in/nmit/about-nmit-bangalore.php",
    "https://nitte.edu.in/nmit/placement.php",
    "https://nitte.edu.in/nmit/admissions.php",
    "https://nitte.edu.in/nmit/btech-computer-science-engineering.php",
    "https://nitte.edu.in/nmit/btech-in-artificialintelligence-and-datascience.php",
    "https://nitte.edu.in/nmit/btech-in-artificialintelligence-and-machinelearning.php",
    "https://nitte.edu.in/nmit/btech-information-science-and-engineering.php",
    "https://nitte.edu.in/nmit/btech-electronics-and-communication-engineering.php",
    "https://nitte.edu.in/nmit/btech-mechanical-engineering.php",
    "https://nitte.edu.in/nmit/btech-aeronautical-engineering.php",
    "https://nitte.edu.in/nmit/btech-electrical-and-electronics-engineering.php",
    "https://nitte.edu.in/nmit/btech-civil-engineering.php",
    "https://nitte.edu.in/nmit/hostels.php",
    "https://nitte.edu.in/nmit/library-and-information-center.php",
    "https://nitte.edu.in/nmit/sports.php",
    "https://nitte.edu.in/nmit/transportation.php",
    "https://nitte.edu.in/nmit/vision-mission.php",
    "https://nitte.edu.in/nmit/contact-nmit.php",
]


def get_nmit_default_urls() -> List[str]:
    """Return the predefined list of NMIT page URLs to scrape."""
    return NMIT_DEFAULT_URLS.copy()


# ═══════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════

def _remove_noise_elements(soup: BeautifulSoup) -> None:
    """Remove HTML elements that never contain useful content."""
    for tag_name in NOISE_ELEMENTS:
        for el in soup.find_all(tag_name):
            el.decompose()


def _remove_noise_by_attributes(soup: BeautifulSoup) -> None:
    """Remove elements whose class or id matches noise patterns."""
    noise_regex = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

    # Collect elements to remove first (avoid modifying tree during iteration)
    to_remove = []
    for el in soup.find_all(True):  # All elements
        try:
            classes = " ".join(el.get("class", []) or [])
            el_id = el.get("id", "") or ""
            combined = f"{classes} {el_id}"

            if noise_regex.search(combined):
                to_remove.append(el)
        except Exception:
            continue

    for el in to_remove:
        try:
            el.decompose()
        except Exception:
            pass


def _find_main_content(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Try to find the main content area of the page.

    Looks for common content container patterns used by college websites.
    Returns None if no obvious main content area is found.
    """
    # Try standard HTML5 <main> element
    main = soup.find("main")
    if main:
        return main

    # Try common content class/id patterns
    content_patterns = [
        {"id": re.compile(r"content|main|body", re.I)},
        {"class_": re.compile(r"content|main-content|page-content|article", re.I)},
        {"role": "main"},
    ]

    for pattern in content_patterns:
        result = soup.find("div", pattern)
        if result:
            return result

    # Fallback: return <body> if it exists
    body = soup.find("body")
    return body


def _clean_scraped_text(text: str) -> str:
    """
    Clean scraped text by removing duplicated navigation items,
    excessive whitespace, and other noise.
    """
    lines = text.split("\n")
    seen_lines = set()
    clean_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (will add spacing later)
        if not stripped:
            continue

        # Skip very short lines that are likely nav items (< 4 words, no sentence structure)
        if len(stripped.split()) < 3 and not stripped.endswith((".", "!", "?")):
            # Allow it only if it looks like a heading/title (common patterns)
            if not any(c.isdigit() for c in stripped) and len(stripped) < 20:
                continue

        # Skip exact duplicate lines (common with nav menus that repeat)
        if stripped in seen_lines:
            continue
        seen_lines.add(stripped)

        # Skip lines that are just URLs
        if stripped.startswith(("http://", "https://")) and " " not in stripped:
            continue

        clean_lines.append(stripped)

    # Join with newlines, then collapse multiple blanks
    result = "\n".join(clean_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r" {2,}", " ", result)

    return result.strip()


def _url_to_source_name(url: str) -> str:
    """
    Convert a URL to a clean, human-readable source name.

    Examples:
        https://nitte.edu.in/nmit/placement.php → nmit/placement.php
        https://nitte.edu.in/nmit/about-nmit-bangalore.php → nmit/about-nmit-bangalore.php
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    # For nitte.edu.in, strip the domain part
    if "nitte.edu.in" in parsed.netloc:
        return path if path else parsed.netloc

    # For other domains, include domain + path
    return f"{parsed.netloc}/{path}" if path else parsed.netloc