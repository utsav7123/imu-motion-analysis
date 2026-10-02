from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAGES = [SITE / "index.html", SITE / "methodology" / "index.html", SITE / "404.html"]
PUBLIC_PAGES = PAGES[:-1]


def test_pages_have_basic_metadata_and_one_h1():
    titles = []
    descriptions = []
    canonicals = []

    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        title = re.search(r"<title>(.+?)</title>", text, re.I | re.S)
        description = re.search(r'<meta name="description" content="(.+?)">', text, re.I | re.S)
        assert title, page
        assert description, page
        assert len(re.findall(r"<h1(?:\s|>)", text, re.I)) == 1, page
        assert "—" not in text, page
        assert "placeholder" not in text.lower(), page
        assert "TODO" not in text, page
        titles.append(title.group(1))
        descriptions.append(description.group(1))

        canonical = re.search(r'<link rel="canonical" href="(.+?)">', text, re.I | re.S)
        if page.name != "404.html":
            assert canonical, page
            canonicals.append(canonical.group(1))

    assert len(titles) == len(set(titles))
    assert len(descriptions) == len(set(descriptions))
    assert len(canonicals) == len(set(canonicals))


def test_public_pages_have_social_and_structured_metadata():
    for page in PUBLIC_PAGES:
        text = page.read_text(encoding="utf-8")
        assert 'type="application/ld+json"' in text, page
        assert 'property="og:title"' in text, page
        assert 'property="og:description"' in text, page
        assert 'property="og:image"' in text, page


def test_site_discovery_files_exist():
    for name in ("robots.txt", "sitemap.xml", "llms.txt"):
        assert (SITE / name).exists()
    assert (SITE / "assets" / "favicon.svg").exists()
    assert (SITE / "assets" / "social-card.svg").exists()


def test_relative_internal_links_resolve():
    href_pattern = re.compile(r'href="([^"]+)"')
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        for href in href_pattern.findall(text):
            parsed = urlparse(href)
            if parsed.scheme or href.startswith("#"):
                continue
            target = (page.parent / parsed.path).resolve()
            if parsed.path.endswith("/") or parsed.path in {".", "..", "./", "../"}:
                target = target / "index.html"
            assert target.exists(), f"Broken link {href} in {page}"


def test_production_javascript_has_no_source_map_or_console_error():
    for path in (SITE / "assets").glob("*.js"):
        text = path.read_text(encoding="utf-8")
        assert "sourceMappingURL" not in text
        assert "console.error" not in text
        assert "—" not in text
