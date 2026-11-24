"""
Web Scraping Module
-------------------
A clean, modular web scraper that extracts main webpage content 
while removing headers, footers, navigation, scripts, and styling elements.

Design follows SOLID principles:
- Single Responsibility: Separate classes for fetching, parsing/cleaning, and orchestrating.
- Open/Closed: Tag removal lists and fetcher/extractor logic can be extended without modifying core orchestration.
- Liskov Substitution: Alternative fetchers (e.g., Playwright/Selenium) or extractors can be swapped in seamlessly.
- Interface Segregation: Small, focused abstract interfaces.
- Dependency Inversion: Orchestrator depends on abstract interfaces, not concrete implementations.
"""

import re
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Set
import requests
from bs4 import BeautifulSoup, Tag


# ============================================================================
# 1. Abstractions (Interfaces) - Dependency Inversion & Interface Segregation
# ============================================================================

class PageFetcher(ABC):
    """Abstract interface responsible solely for retrieving raw webpage content."""

    @abstractmethod
    def fetch(self, url: str) -> str:
        """
        Fetch raw HTML markup from a given URL.

        Args:
            url: Target URL to retrieve.

        Returns:
            Raw HTML string.
        """
        pass


class ContentExtractor(ABC):
    """Abstract interface responsible solely for parsing and cleaning HTML markup."""

    @abstractmethod
    def extract(self, raw_html: str) -> str:
        """
        Parse raw HTML and return clean extracted content.

        Args:
            raw_html: The raw HTML content.

        Returns:
            Sanitized content string.
        """
        pass


# ============================================================================
# 2. Concrete Implementations - Single Responsibility Principle
# ============================================================================

class RequestsFetcher(PageFetcher):
    """
    HTTP client fetcher using the `requests` library.
    Handles network requests, timeouts, and standard headers.
    """

    def __init__(
        self,
        timeout_seconds: int = 15,
        user_agent: Optional[str] = None
    ) -> None:
        """
        Initialize the HTTP client with optional custom settings.

        Args:
            timeout_seconds: Timeout for network requests in seconds.
            user_agent: Custom User-Agent string to avoid basic bot detection.
        """
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()

        # Standard desktop browser user-agent to prevent 403 blocks
        default_ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self._session.headers.update({"User-Agent": user_agent or default_ua})

    def fetch(self, url: str) -> str:
        """
        Retrieves the HTML content of the target URL.

        Args:
            url: Webpage URL to fetch.

        Returns:
            HTML string content.

        Raises:
            requests.RequestException: If network error or non-200 HTTP status occurs.
        """
        response = self._session.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.text


class Bs4BodyExtractor(ContentExtractor):
    """
    HTML parser and cleaner using BeautifulSoup.
    Locates the <body> element, removes unwanted tags (headers, footers, etc.),
    and returns sanitized content.
    """

    # Tags typically associated with non-content elements or structural noise
    DEFAULT_EXCLUDED_TAGS: Set[str] = {
        "header",
        "footer",
        "nav",
        "aside",
        "script",
        "style",
        "noscript",
        "iframe",
        "form"
    }

    # Regex pattern to identify non-content structural elements (footers, sitemaps, etc.) by class/id
    STRUCTURAL_NOISE_PATTERN = re.compile(
        r"(footer|sitemap|copyright|breadcrumb|site-header|navbar)",
        re.IGNORECASE
    )

    def __init__(
        self,
        excluded_tags: Optional[Iterable[str]] = None,
        as_text: bool = True
    ) -> None:
        """
        Args:
            excluded_tags: Tag names to strip from the DOM.
            as_text: If True, returns clean plain text. If False, returns cleaned inner HTML.
        """
        self._excluded_tags = set(excluded_tags) if excluded_tags else self.DEFAULT_EXCLUDED_TAGS
        self._as_text = as_text

    def extract(self, raw_html: str) -> str:
        """
        Removes blacklisted tags, hidden elements, and structural noise (headers/footers)
        and extracts the cleaned body content.

        Args:
            raw_html: The HTML document string.

        Returns:
            Clean body content (text or HTML) excluding header, footer, etc.
        """
        soup = BeautifulSoup(raw_html, "html.parser")

        # Focus on the <body> tag; fallback to full document if <body> is missing
        target_container = soup.body if soup.body else soup

        # 1. Decompose (remove from DOM tree) unwanted tags
        for tag_name in self._excluded_tags:
            for element in target_container.find_all(tag_name):
                element.decompose()

        # 2. Decompose hidden elements (e.g. style="display: none", hidden attribute)
        for element in list(target_container.find_all(True)):
            if not isinstance(element, Tag) or element.attrs is None:
                continue
            if getattr(element, "decomposed", False) or element.parent is None:
                continue

            if self._is_hidden_element(element):
                element.decompose()
                continue

            # 3. Decompose non-heading elements with footer/header/sitemap in class or id
            if element.name not in ["h1", "h2", "h3", "h4", "h5", "h6", "body", "html"]:
                class_attr = element.get("class", [])
                classes = " ".join(class_attr) if isinstance(class_attr, list) else str(class_attr or "")
                el_id = str(element.get("id") or "")
                if self.STRUCTURAL_NOISE_PATTERN.search(classes) or self.STRUCTURAL_NOISE_PATTERN.search(el_id):
                    element.decompose()

        if self._as_text:
            # Extract plain text with normalized whitespace
            raw_text = target_container.get_text(separator="\n")
            return self._clean_text(raw_text)

        # Return cleaned inner HTML
        return target_container.decode_contents().strip()

    @staticmethod
    def _is_hidden_element(element: Tag) -> bool:
        """Check if an element is hidden via attributes or inline CSS styles."""
        if not isinstance(element, Tag) or element.attrs is None:
            return False
        if element.has_attr("hidden") or element.get("aria-hidden") == "true":
            return True
        style_attr = element.get("style", "")
        if isinstance(style_attr, list):
            style_attr = " ".join(style_attr)
        elif not isinstance(style_attr, str):
            style_attr = str(style_attr or "")
        style = style_attr.replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
        return False

    @classmethod
    def _clean_text(cls, raw_text: str) -> str:
        """Normalize line breaks, strip trailing spaces, and remove debug session dumps."""
        # Remove stray PHP/debug session dumps like "Array ( [session_id] => ... )"
        cleaned = re.sub(
            r"Array\s*\(\s*\[session_id\].*?\)\s*(\*{3,})?",
            "",
            raw_text,
            flags=re.DOTALL
        )

        lines = (line.strip() for line in cleaned.splitlines())
        clean_lines = [line for line in lines if line]
        return "\n".join(clean_lines)


# ============================================================================
# 3. Orchestration Service - Open/Closed & Dependency Inversion
# ============================================================================

class WebScraperService:
    """
    High-level facade coordinating fetching and extraction.
    Depends on abstractions (PageFetcher, ContentExtractor).
    """

    def __init__(
        self,
        fetcher: Optional[PageFetcher] = None,
        extractor: Optional[ContentExtractor] = None
    ) -> None:
        """
        Initialize the scraper service with dependency injection.

        Args:
            fetcher: Instance conforming to PageFetcher interface.
            extractor: Instance conforming to ContentExtractor interface.
        """
        self._fetcher = fetcher or RequestsFetcher()
        self._extractor = extractor or Bs4BodyExtractor()

    def scrape(self, url: str) -> str:
        """
        Scrapes a webpage URL and returns the clean body content.

        Args:
            url: The webpage URL to scrape.

        Returns:
            Clean string content of the page body.
        """
        raw_html = self._fetcher.fetch(url)
        return self._extractor.extract(raw_html)


# ============================================================================
# 4. Standalone Convenience Function
# ============================================================================

def scrape_webpage_body(url: str, as_text: bool = True) -> str:
    """
    Convenience function to quickly scrape the body of a webpage.

    Args:
        url: URL of the webpage to scrape.
        as_text: If True, returns clean readable text. If False, returns clean HTML.

    Returns:
        Clean body content without header or footer.
    """
    extractor = Bs4BodyExtractor(as_text=as_text)
    service = WebScraperService(extractor=extractor)
    return service.scrape(url)
