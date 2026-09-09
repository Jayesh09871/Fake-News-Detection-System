"""
Web Article Extractor Module
Extracts news article text, title, and metadata from web URLs using trafilatura.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import sys
from pathlib import Path
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False


logger = logging.getLogger("ArticleExtractor")


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate that a given string is a syntactically correct HTTP/HTTPS URL.

    Args:
        url: URL string to validate.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    url = url.strip()
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False, "Invalid URL protocol. URL must start with http:// or https://"
        if not parsed.netloc:
            return False, "Invalid URL. Missing domain or host name."
        # Basic check to avoid localhost/internal addresses if desired
        hostname = (parsed.hostname or "").lower()
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            return False, "Localhost URLs are not allowed for security reasons."
        return True, ""
    except Exception as e:
        return False, f"URL parsing error: {e}"


def extract_article_from_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Fetch and extract article content and metadata from a given URL.

    Args:
        url: The web URL to extract.
        timeout: Network request timeout in seconds.

    Returns:
        Dictionary containing:
            - success: bool
            - title: str
            - text: str
            - source_url: str
            - error: str (if success is False)
    """
    is_valid, err_msg = validate_url(url)
    if not is_valid:
        return {
            "success": False,
            "title": "",
            "text": "",
            "source_url": url,
            "error": err_msg,
        }

    if not TRAFILATURA_AVAILABLE:
        return {
            "success": False,
            "title": "",
            "text": "",
            "source_url": url,
            "error": "The 'trafilatura' library is not installed. Run 'pip install trafilatura' to enable URL article extraction or paste the article text manually.",
        }

    try:
        logger.info("Fetching webpage from: %s", url)
        # Fetch raw HTML using trafilatura config for timeout
        config = trafilatura.settings.use_config()
        config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout))
        downloaded = trafilatura.fetch_url(url, config=config)
        if downloaded is None:
            return {
                "success": False,
                "title": "",
                "text": "",
                "source_url": url,
                "error": "Failed to fetch webpage content. The server may be unreachable, blocking automated requests, or timed out.",
            }

        # Extract text content and metadata
        extracted_text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            output_format="txt",
        )

        # Extract metadata (such as title)
        metadata = trafilatura.extract_metadata(downloaded)
        extracted_title = metadata.title if metadata and metadata.title else ""

        if not extracted_text or not extracted_text.strip():
            return {
                "success": False,
                "title": extracted_title or "",
                "text": "",
                "source_url": url,
                "error": (
                    "Could not extract meaningful article text from this page. "
                    "The webpage may rely entirely on client-side JavaScript rendering, paywalls, or bot protection. "
                    "Please copy and paste the article text directly into the text field."
                ),
            }

        return {
            "success": True,
            "title": extracted_title.strip() if extracted_title else "",
            "text": extracted_text.strip(),
            "source_url": url,
            "error": "",
        }

    except Exception as e:
        logger.error("Exception during article extraction: %s", e)
        return {
            "success": False,
            "title": "",
            "text": "",
            "source_url": url,
            "error": f"An unexpected error occurred while extracting the article: {str(e)}. Please paste the article text manually.",
        }
