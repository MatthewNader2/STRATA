"""
src/external_rag.py
Active, domain-agnostic asynchronous retrieval module for API documentation,
package metadata, and code context synthesis.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models & Cache
# ============================================================================

@dataclass(frozen=True)
class RAGResponse:
    """Represents normalized documentation or schema retrieved externally."""
    source: str
    identifier: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None


class TTLCache:
    """Thread-safe and async-safe in-memory cache with time-to-live expiration."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1024):
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_entries = max_entries
        self._cache: Dict[str, tuple[datetime, RAGResponse]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[RAGResponse]:
        async with self._lock:
            if key not in self._cache:
                return None
            inserted_at, response = self._cache[key]
            if datetime.utcnow() - inserted_at > self._ttl:
                del self._cache[key]
                return None
            return response

    async def set(self, key: str, value: RAGResponse) -> None:
        async with self._lock:
            if len(self._cache) >= self._max_entries:
                # Evict oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (datetime.utcnow(), value)


# ============================================================================
# Abstract Base Fetcher
# ============================================================================

class BaseFetcher(abc.ABC):
    """Abstract interface for external documentation and package fetchers."""

    def __init__(self, timeout: float = 5.0, user_agent: str = "GEVR-Synthesis-RAG/2.0"):
        self.timeout = timeout
        self.user_agent = user_agent

    @abc.abstractmethod
    async def fetch(self, query: str, context: Optional[Dict[str, Any]] = None) -> RAGResponse:
        """Asynchronously retrieve documentation or symbol metadata."""
        pass

    async def _http_get(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        """Asynchronous HTTP GET wrapper offloaded from the event loop."""
        req_headers = {"User-Agent": self.user_agent}
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, headers=req_headers)

        def _blocking_call() -> str:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")

        return await asyncio.to_thread(_blocking_call)


# ============================================================================
# Domain-Agnostic Fetcher Implementations
# ============================================================================

class PackageRegistryFetcher(BaseFetcher):
    """
    Domain-agnostic fetcher for ecosystem registries (PyPI, Crates.io, NPM, etc.).
    Configurable via endpoint templates and response extraction callbacks.
    """

    def __init__(
        self,
        name: str,
        url_template: str,
        extractor: Callable[[str], str],
        timeout: float = 5.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(timeout=timeout)
        self.name = name
        self.url_template = url_template
        self.extractor = extractor
        self.custom_headers = headers or {}

    async def fetch(self, query: str, context: Optional[Dict[str, Any]] = None) -> RAGResponse:
        clean_name = urllib.parse.quote(query.strip())
        target_url = self.url_template.format(name=clean_name)

        try:
            raw_data = await self._http_get(target_url, headers=self.custom_headers)
            extracted_doc = self.extractor(raw_data)
            return RAGResponse(
                source=self.name,
                identifier=query,
                content=extracted_doc,
                metadata={"url": target_url},
                success=True,
            )
        except Exception as exc:
            logger.warning(f"[{self.name}] Failed to fetch package metadata for '{query}': {exc}")
            return RAGResponse(
                source=self.name,
                identifier=query,
                content="",
                success=False,
                error_message=str(exc),
            )


class WebSearchFetcher(BaseFetcher):
    """
    Domain-agnostic web documentation fetcher.
    Avoids hardcoding specific ecosystems, allowing query construction via context templates.
    """

    HTML_TAG_CLEANER: Pattern = re.compile(r"<[^>]+>")

    def __init__(
        self,
        endpoint_url: str = "https://html.duckduckgo.com/html/",
        default_query_template: str = "{query}",
        timeout: float = 6.0,
    ):
        super().__init__(timeout=timeout)
        self.endpoint_url = endpoint_url
        self.default_query_template = default_query_template

    async def fetch(self, query: str, context: Optional[Dict[str, Any]] = None) -> RAGResponse:
        context = context or {}
        template = context.get("query_template", self.default_query_template)
        qualifier = context.get("qualifier", "")
        formatted_query = template.format(query=query, qualifier=qualifier).strip()

        encoded_query = urllib.parse.urlencode({"q": formatted_query})
        target_url = f"{self.endpoint_url}?{encoded_query}"

        try:
            html = await self._http_get(target_url)
            snippets = self._extract_snippets(html)
            return RAGResponse(
                source="WebSearchFetcher",
                identifier=query,
                content="\n".join(snippets),
                metadata={"formatted_query": formatted_query, "url": target_url},
                success=True,
            )
        except Exception as exc:
            logger.error(f"[WebSearchFetcher] Query '{formatted_query}' failed: {exc}")
            return RAGResponse(
                source="WebSearchFetcher",
                identifier=query,
                content="",
                success=False,
                error_message=str(exc),
            )

    @classmethod
    def _extract_snippets(cls, html: str, max_snippets: int = 5) -> List[str]:
        """Extract plain-text result snippets from search HTML without external dependencies."""
        snippets: List[str] = []
        pattern = re.compile(r'class="result__snippet[^"]*">(.*?)</a>', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(html)
        for match in matches[:max_snippets]:
            clean_text = cls.HTML_TAG_CLEANER.sub("", match).strip()
            if clean_text:
                snippets.append(clean_text)
        return snippets


class IntrospectionFetcher(BaseFetcher):
    """
    Introspects local runtime environments and symbol tables without assuming ecosystem.
    Useful for fallback introspection when internet access is unavailable or disabled.
    """

    async def fetch(self, query: str, context: Optional[Dict[str, Any]] = None) -> RAGResponse:
        def _introspect() -> str:
            import importlib
            import inspect

            try:
                parts = query.strip().split(".")
                mod = None
                sub_attrs = []
                for i in range(len(parts), 0, -1):
                    mod_path = ".".join(parts[:i])
                    try:
                        mod = importlib.import_module(mod_path)
                        sub_attrs = parts[i:]
                        break
                    except Exception:
                        continue
                if mod is None:
                    return f"Introspection unavailable: cannot import '{query}'"

                target = mod
                for attr in sub_attrs:
                    target = getattr(target, attr)

                doc = inspect.getdoc(target) or ""
                sig = ""
                try:
                    sig = str(inspect.signature(target))
                except Exception:
                    pass
                return f"Signature: {sig}\n\nDocumentation:\n{doc}"
            except Exception as e:
                return f"Introspection unavailable: {e}"

        content = await asyncio.to_thread(_introspect)
        return RAGResponse(
            source="LocalIntrospection",
            identifier=query,
            content=content,
            success=not content.startswith("Introspection unavailable"),
        )


class LiveDocFetcher(BaseFetcher):
    """
    Live documentation fetcher exposing a SYNCHRONOUS, string-returning
    interface over the shared default registry. This is the concrete fetcher
    the synthesis engine consumes (`SynthesisEngine.synthesize_micro_cell`):
    callers receive the raw documentation text directly rather than a
    RAGResponse envelope, so live grounding works from non-async contexts.
    """

    async def fetch(self, query: str, context: Optional[Dict[str, Any]] = None) -> RAGResponse:  # type: ignore[override]
        """Async contract honored for BaseFetcher compatibility."""
        return await default_registry.fetch(query, context=context)

    def fetch_text(self, query: str, domain_context: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None) -> str:
        """Synchronous fetch returning the documentation content as plain text."""
        response = fetch_docs_sync(query, domain_context=domain_context, context=context)
        if response is None:
            return ""
        if getattr(response, "success", False):
            return getattr(response, "content", "") or ""
        return ""

    # Convenience alias so callers may simply call `fetcher.fetch(query)`
    # synchronously and receive the documentation text.
    def fetch_sync(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        return self.fetch_text(query, context=context)


# ============================================================================
# Dynamic Registry & Factory
# ============================================================================

class FetcherRegistry:
    """Registry maintaining domain-specific and global fetchers."""

    def __init__(self):
        self._fetchers: Dict[str, BaseFetcher] = {}
        self._default_fetcher: BaseFetcher = IntrospectionFetcher()
        self._cache = TTLCache()

    def register(self, domain_key: str, fetcher: BaseFetcher) -> None:
        """Register a fetcher for a domain identifier (e.g. 'python', 'rust', 'npm')."""
        self._fetchers[domain_key.lower()] = fetcher

    def set_default(self, fetcher: BaseFetcher) -> None:
        self._default_fetcher = fetcher

    def resolve(self, domain_context: Optional[str] = None) -> BaseFetcher:
        """Resolve the appropriate fetcher based on context without hardcoded string branching."""
        if domain_context:
            domain_key = domain_context.strip().lower()
            if domain_key in self._fetchers:
                return self._fetchers[domain_key]
        return self._default_fetcher

    async def fetch(
        self,
        query: str,
        domain_context: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Cached dispatch fetcher."""
        cache_key = f"{domain_context or 'default'}:{query}"
        cached = await self._cache.get(cache_key)
        if cached:
            return cached

        fetcher = self.resolve(domain_context)
        response = await fetcher.fetch(query, context=context)

        if response.success:
            await self._cache.set(cache_key, response)

        return response


# ============================================================================
# Default Registry Configuration
# ============================================================================

def create_default_registry() -> FetcherRegistry:
    """Instantiate a registry populated with agnostic ecosystem parsers."""
    registry = FetcherRegistry()

    # PyPI / Python Provider
    def _pypi_extractor(raw_json: str) -> str:
        data = json.loads(raw_json)
        info = data.get("info", {})
        summary = info.get("summary", "")
        description = info.get("description", "")
        return f"{summary}\n\n{description[:2000]}"

    registry.register(
        "python",
        PackageRegistryFetcher(
            name="PyPI",
            url_template="https://pypi.org/pypi/{name}/json",
            extractor=_pypi_extractor,
        ),
    )

    # Crates.io / Rust Provider
    def _crates_extractor(raw_json: str) -> str:
        data = json.loads(raw_json)
        crate = data.get("crate", {})
        return crate.get("description", "")

    registry.register(
        "rust",
        PackageRegistryFetcher(
            name="CratesIo",
            url_template="https://crates.io/api/v1/crates/{name}",
            extractor=_crates_extractor,
            headers={"User-Agent": "GEVR-Synthesis-RAG/2.0 (crates-io-fetcher)"},
        ),
    )

    # Generic Web Search Provider
    registry.register("web", WebSearchFetcher())

    # Fallback Local Introspector
    registry.set_default(IntrospectionFetcher())

    return registry


# Module-level shared registry
default_registry = create_default_registry()


def fetch_docs_sync(
    query: str, domain_context: Optional[str] = None, context: Optional[Dict[str, Any]] = None
) -> RAGResponse:
    """Synchronous interface for environments lacking active async loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Run inside a dedicated worker thread so synchronous callers invoked from
        # within an active event loop never deadlock on the same loop thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            return _pool.submit(
                asyncio.run,
                default_registry.fetch(query, domain_context, context),
            ).result()
    else:
        return asyncio.run(default_registry.fetch(query, domain_context, context))
