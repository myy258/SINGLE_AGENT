# 联网搜索：可插拔后端（baidu HTML / Tavily API / Serper API），统一返回带 URL 的结构化结果
# Co-authored with CoCo

"""
原来只有一个 baidu_search，直接抓 https://www.baidu.com/s 的 HTML 并按
`div.result` / `div.c-abstract` 这些 class 名解析。两个问题：

1. 脆：百度随时改版，class 名一变就静默返回"未找到有效结果"；遇到反爬验证页
   同样是静默失败，模型会以为"网上真的没有"。
2. 不返回 URL：只提取标题和摘要，模型没法给出任何可核查的出处。这跟 system
   prompt 里"一切结论以真实工具返回为准"是直接冲突的——用户无法验证。

现在：
- 统一的结果结构 SearchResult(title, url, snippet)，一定带 URL。
- 后端可插拔（config.SEARCH_BACKEND），推荐用 Tavily/Serper 这类正规 API；
  baidu 仍保留作为"没有 API Key 时的兜底"，但会在返回里明确标注它的局限。
- 失败时区分"网络不通"/"被反爬拦"/"解析不出来"，把真实原因告诉模型，
  而不是一律说"没找到"。
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from config import (
    SEARCH_BACKEND,
    SEARCH_TIMEOUT,
    SERPER_API_KEY,
    TAVILY_API_KEY,
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchUnavailable(RuntimeError):
    """搜索这次做不了（网络/配置/被拦），message 里说明真实原因。"""


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


# ── 后端 1：百度 HTML 抓取（无需 Key，但脆且 URL 是跳转链接）──────────────
def _search_baidu(query: str, top_k: int) -> list[SearchResult]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise SearchUnavailable(
            "baidu 后端需要 beautifulsoup4，请 pip install beautifulsoup4，"
            "或把 config.SEARCH_BACKEND 换成 tavily/serper。"
        ) from exc

    try:
        resp = requests.get(
            "https://www.baidu.com/s",
            params={"wd": query, "rn": top_k},
            headers=_BROWSER_HEADERS,
            timeout=SEARCH_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise SearchUnavailable(
            "连接百度失败，当前网络环境可能限制了外网访问。"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise SearchUnavailable(f"百度搜索请求超时（{SEARCH_TIMEOUT}s）。") from exc

    resp.encoding = "utf-8"
    html = resp.text

    # 反爬验证页：明确告知，不要让模型误判为"网上没有这个信息"
    if "百度安全验证" in html or "wappass.baidu.com" in html:
        raise SearchUnavailable(
            "百度返回了安全验证页（触发反爬），这次搜索没有拿到结果。"
            "这不代表网上没有相关信息。建议配置 TAVILY_API_KEY 或 SERPER_API_KEY "
            "并把 config.SEARCH_BACKEND 改成 tavily/serper。"
        )

    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for item in soup.select("div.result, div.result-op"):
        title_tag = item.select_one("h3 a")
        if not title_tag:
            continue
        abstract_tag = item.select_one("div.c-abstract, span.content-right_8Zs40")
        results.append(
            SearchResult(
                title=title_tag.get_text(strip=True),
                # 百度给的是 www.baidu.com/link?url=... 跳转链接，不是最终目标 URL，
                # 但至少可点击可核查，比完全不给强。
                url=str(title_tag.get("href") or ""),
                snippet=abstract_tag.get_text(strip=True) if abstract_tag else "",
            )
        )
        if len(results) >= top_k:
            break

    if not results:
        raise SearchUnavailable(
            "百度页面解析不出任何结果条目——通常意味着百度改了页面结构，"
            "或者返回的是拦截页。请改用 tavily/serper 后端。"
        )
    return results


# ── 后端 2：Tavily Search API ─────────────────────────────────────────────
def _search_tavily(query: str, top_k: int) -> list[SearchResult]:
    if not TAVILY_API_KEY:
        raise SearchUnavailable(
            "SEARCH_BACKEND=tavily 但没有 TAVILY_API_KEY，请设置环境变量后重试。"
        )
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": top_k,
                "search_depth": "basic",
            },
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as exc:
        raise SearchUnavailable(f"Tavily API 调用失败：{exc}") from exc
    except ValueError as exc:
        raise SearchUnavailable("Tavily API 返回的不是合法 JSON。") from exc

    return [
        SearchResult(
            title=str(r.get("title") or ""),
            url=str(r.get("url") or ""),
            snippet=str(r.get("content") or ""),
        )
        for r in (payload.get("results") or [])[:top_k]
    ]


# ── 后端 3：Serper.dev（Google）────────────────────────────────────────────
def _search_serper(query: str, top_k: int) -> list[SearchResult]:
    if not SERPER_API_KEY:
        raise SearchUnavailable(
            "SEARCH_BACKEND=serper 但没有 SERPER_API_KEY，请设置环境变量后重试。"
        )
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": top_k},
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as exc:
        raise SearchUnavailable(f"Serper API 调用失败：{exc}") from exc
    except ValueError as exc:
        raise SearchUnavailable("Serper API 返回的不是合法 JSON。") from exc

    return [
        SearchResult(
            title=str(r.get("title") or ""),
            url=str(r.get("link") or ""),
            snippet=str(r.get("snippet") or ""),
        )
        for r in (payload.get("organic") or [])[:top_k]
    ]


_BACKENDS = {
    "baidu": _search_baidu,
    "tavily": _search_tavily,
    "serper": _search_serper,
}


def run_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """按 config.SEARCH_BACKEND 执行搜索。失败抛 SearchUnavailable（含真实原因）。"""
    backend = (SEARCH_BACKEND or "baidu").strip().lower()
    func = _BACKENDS.get(backend)
    if func is None:
        raise SearchUnavailable(
            f"未知的搜索后端「{backend}」。可选：{', '.join(sorted(_BACKENDS))}"
        )
    return func(query, max(1, min(top_k, 20)))


def format_results(results: list[SearchResult]) -> str:
    """格式化成给模型看的文本。URL 必须出现，便于它在回答里给出处。"""
    blocks = []
    for i, r in enumerate(results, 1):
        lines = [f"{i}. 【{r.title}】"]
        if r.url:
            lines.append(f"   来源: {r.url}")
        if r.snippet:
            lines.append(f"   摘要: {r.snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
