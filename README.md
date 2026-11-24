# Web Scraper (Clean Body Content)

A modular, clean Python web scraper that retrieves webpage body content while excluding headers, footers, navigation bars, aside sidebars, scripts, and styling elements.

## Features & SOLID Architecture

- **Single Responsibility Principle (SRP):**
  - `RequestsFetcher`: Responsible only for HTTP networking, timeouts, and headers.
  - `Bs4BodyExtractor`: Responsible only for HTML parsing, tag decomposition, and whitespace cleaning.
  - `WebScraperService`: Coordinates fetching and extraction.
- **Open / Closed Principle (OCP):**
  - Easily extendable to headless browsers (Selenium, Playwright) by implementing `PageFetcher` without modifying existing classes.
  - Custom excluded tags can be passed to `Bs4BodyExtractor`.
- **Liskov Substitution Principle (LSP):**
  - Any custom implementation of `PageFetcher` or `ContentExtractor` can replace default implementations seamlessly.
- **Interface Segregation Principle (ISP):**
  - Small, focused interfaces with single abstract methods (`fetch`, `extract`).
- **Dependency Inversion Principle (DIP):**
  - High-level `WebScraperService` depends on abstract protocols, injected via constructors.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Simple Function Call

```python
from scraper import scrape_webpage_body

# Extract clean plain text
text = scrape_webpage_body("https://en.wikipedia.org/wiki/Web_scraping", as_text=True)
print(text)

# Or extract clean sanitized HTML
html = scrape_webpage_body("https://en.wikipedia.org/wiki/Web_scraping", as_text=False)
print(html)
```

### 2. Run Example Script

```bash
python main.py
```
