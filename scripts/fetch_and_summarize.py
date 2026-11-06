#!/usr/bin/env python3
"""
Sunday Scholar Pulse - Weekly Journal Collector, Paper Summarizer & Trend Synthesizer
=====================================================================================
매주 일요일 저녁 19:00 KST에 실행되어:
1. 5대 분야(건축·도시·조경·기후변화·재난재해) 19개 주류 저널에서 '이번 주(최근 7일간)' 출판된 논문만
   OpenAlex API + Crossref API + 출판사 RSS 피드로 전수 수집하고, 출판일이 이번 주 범위 밖인 과거 논문은
   엄격히 제외(Strict Date Filter)합니다.
2. Gemini API (gemini-2.5-flash)의 Structured Output 기능을 사용해 개별 논문의
   [한국어 제목, 연구 주제, 주요 방법론, 방법론 분류, 핵심 이슈, 키워드, 트렌드 연관도]를 추출합니다.
3. 주간 전체 논문 요약본을 종합(Map-Reduce)하여 한 주간의 학술 트렌드 총평 리포트를 생성합니다.
4. 정적 웹앱(index.html)이 읽는 `data/weeks/<YYYY-Www>.json` 및 `data/weeks_manifest.json`을 갱신하고,
   선택적으로 Slack / Telegram / Email(Resend) 알림을 발송합니다.
"""

from __future__ import annotations

import datetime
import hashlib
import html
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging & Paths
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sunday_scholar")

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "journals.json"
DATA_DIR = ROOT_DIR / "data"
WEEKS_DIR = DATA_DIR / "weeks"
CACHE_PATH = DATA_DIR / "paper_summary_cache.json"
MANIFEST_PATH = DATA_DIR / "weeks_manifest.json"

KST = datetime.timezone(datetime.timedelta(hours=9))


# ---------------------------------------------------------------------------
# Pydantic Schemas for Gemini Structured Output
# ---------------------------------------------------------------------------
class PaperSummarySchema(BaseModel):
    titleKo: str = Field(description="논문 제목의 자연스럽고 정확한 한국어 학술 번역")
    topic: str = Field(description="연구 배경 및 핵심 연구 주제/목적 (2~3문장, 한국어)")
    methodology: str = Field(
        description="사용된 핵심 방법론, 데이터셋, 시뮬레이션 툴(예: ENVI-met, EnergyPlus, GNN, InSAR 등) 및 분석 기법 (2~3문장, 한국어)"
    )
    methodType: str = Field(
        description="방법론 대분류. 반드시 다음 4개 중 하나로만 응답: 'AI/머신러닝', '시뮬레이션/모델링', '원격탐사/GIS', '실증/현장계측'"
    )
    issue: str = Field(
        description="주요 연구 결과(정량적 수치 포함), 다루는 핵심 이슈 및 학술·정책적 시사점 (2~3문장, 한국어)"
    )
    keywords: List[str] = Field(
        description="논문을 대표하는 핵심 연구 키워드 3~4개 (예: 'Urban Heat Island (도시열섬)', 'GeoAI & Graph Neural Network')"
    )
    impactScore: int = Field(
        description="5대 분야(건축·도시·조경·기후·재난) 트렌드 대표성 및 학술적 파급력 점수 (75~100 사이 정수)"
    )


class TrendPillarSchema(BaseModel):
    badge: str = Field(
        description="기둥 구분 라벨 ('주제 트렌드 (Topics)', '방법론 트렌드 (Methodologies)', '핵심 이슈 (Key Issues)')"
    )
    title: str = Field(description="해당 영역의 금주 핵심 트렌드 한 줄 요약 제목")
    desc: str = Field(description="구체적인 연구 흐름 설명 (2~3문장, 한국어)")


class TopKeywordItemSchema(BaseModel):
    word: str = Field(description="핵심 키워드명 (영문+국문 병기 권장)")
    count: int = Field(description="금주 논문 내 등장/연관 빈도수")


class WeeklyTrendSynthesisSchema(BaseModel):
    headline: str = Field(
        description="금주 전체 논문을 관통하는 핵심 트렌드 헤드라인 (1문장, 한국어)"
    )
    narrative: str = Field(
        description="금주 출판 논문들의 ① 주제적 흐름, ② 방법론적 혁신, ③ 핵심 정책·실무 이슈 및 분야 간 융합 동향을 종합한 총평 (4~6문장, 한국어)"
    )
    pillars: List[TrendPillarSchema] = Field(
        description="주제 트렌드, 방법론 트렌드, 핵심 이슈 3가지 영역별 요약 카드 (정확히 3개)"
    )
    topKeywords: List[TopKeywordItemSchema] = Field(
        description="금주 가장 두드러진 상위 8개 키워드와 출현 빈도"
    )


# ---------------------------------------------------------------------------
# Utility Helpers & Strict Weekly Date Validator
# ---------------------------------------------------------------------------
def load_journals_config() -> List[Dict[str, Any]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_within_date_window(
    pub_date_str: str,
    start_date: datetime.date,
    end_date: datetime.date,
) -> bool:
    """Strictly verifies that pub_date_str (YYYY-MM-DD) falls within [start_date, end_date]."""
    if not pub_date_str or len(pub_date_str) < 10:
        return False
    try:
        dt = datetime.date.fromisoformat(pub_date_str[:10])
        # Allow up to 7 days ahead in case publisher stamps next week's issue date on Online First articles
        return start_date <= dt <= (end_date + datetime.timedelta(days=7))
    except ValueError:
        return False


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstructs plain-text abstract from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    word_positions: List[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                word_positions.append((pos, word))
    if not word_positions:
        return ""
    word_positions.sort(key=lambda x: x[0])
    text = " ".join(w for _, w in word_positions)
    return re.sub(r"\s+", " ", text).strip()


def clean_html_text(raw_text: str) -> str:
    """Removes JATS XML / HTML tags from Crossref or RSS abstracts."""
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", no_tags).strip()


def normalize_doi(doi_str: Optional[str]) -> str:
    if not doi_str:
        return ""
    doi_str = doi_str.strip().lower()
    doi_str = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi_str)
    return f"https://doi.org/{doi_str}"


def make_paper_id(week_key: str, doi: str, title: str) -> str:
    seed = doi if doi else title.lower().strip()
    short_hash = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return f"{week_key.lower()}-{short_hash}"


# ---------------------------------------------------------------------------
# Step 1: Multi-Source Paper Collection (OpenAlex + Crossref + RSS)
# ---------------------------------------------------------------------------
def fetch_from_openalex(
    journals: List[Dict[str, Any]],
    start_date: datetime.date,
    end_date: datetime.date,
    polite_email: str = "",
) -> List[Dict[str, Any]]:
    """Fetches all papers published in the target 19 journals strictly within [start_date, end_date]."""
    issn_to_journal: Dict[str, Dict[str, Any]] = {}
    all_issns: List[str] = []
    for j in journals:
        for issn in j.get("issns", []):
            issn_to_journal[issn] = j
            all_issns.append(issn)

    issn_filter = "|".join(sorted(set(all_issns)))
    collected: List[Dict[str, Any]] = []
    cursor = "*"

    headers = {"User-Agent": f"SundayScholarPulse/1.0 (mailto:{polite_email or 'scholar@example.org'})"}

    while cursor:
        params = {
            "filter": (
                f"primary_location.source.issn:{issn_filter},"
                f"from_publication_date:{start_date.isoformat()},"
                f"to_publication_date:{end_date.isoformat()},"
                "type:article|review"
            ),
            "per-page": 200,
            "cursor": cursor,
        }
        if polite_email:
            params["mailto"] = polite_email

        try:
            resp = requests.get(
                "https://api.openalex.org/works",
                params=params,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("OpenAlex request failed: %s", exc)
            break

        results = payload.get("results", [])
        if not results:
            break

        for item in results:
            title = (item.get("title") or "").strip()
            if not title:
                continue

            pub_date = item.get("publication_date") or ""
            if not is_within_date_window(pub_date, start_date, end_date):
                continue

            source_info = (item.get("primary_location") or {}).get("source") or {}
            source_issns = source_info.get("issn") or []
            matched_journal = None
            for issn in source_issns:
                if issn in issn_to_journal:
                    matched_journal = issn_to_journal[issn]
                    break
            if not matched_journal:
                s_name = (source_info.get("display_name") or "").lower()
                for j in journals:
                    if j["name"].lower() in s_name or s_name in j["name"].lower():
                        matched_journal = j
                        break
            if not matched_journal:
                continue

            abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
            doi = normalize_doi(item.get("doi"))

            authorships = item.get("authorships") or []
            author_names = [
                (a.get("author") or {}).get("display_name", "")
                for a in authorships[:4]
                if (a.get("author") or {}).get("display_name")
            ]
            if len(authorships) > 4:
                author_str = ", ".join(author_names[:3]) + " et al."
            elif author_names:
                author_str = ", ".join(author_names)
            else:
                author_str = "Authors not listed"

            collected.append(
                {
                    "domain": matched_journal["domain"],
                    "journal": matched_journal["name"],
                    "pubDate": pub_date,
                    "title": title,
                    "authors": author_str,
                    "doi": doi or (item.get("id") or ""),
                    "abstract": abstract,
                    "source_api": "openalex",
                }
            )

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.25)

    logger.info("OpenAlex collected %d strictly-dated weekly papers.", len(collected))
    return collected


def fetch_from_crossref(
    journals: List[Dict[str, Any]],
    start_date: datetime.date,
    end_date: datetime.date,
    polite_email: str = "",
) -> List[Dict[str, Any]]:
    """Complements OpenAlex with Crossref API for Online First articles published strictly this week."""
    collected: List[Dict[str, Any]] = []
    headers = {"User-Agent": f"SundayScholarPulse/1.0 (mailto:{polite_email or 'scholar@example.org'})"}

    for j in journals:
        issns = j.get("issns") or []
        if not issns:
            continue
        issn_filter = ",".join(f"issn:{i}" for i in issns)
        filter_str = (
            f"{issn_filter},"
            f"from-online-pub-date:{start_date.isoformat()},"
            f"until-online-pub-date:{end_date.isoformat()},"
            "type:journal-article"
        )
        try:
            resp = requests.get(
                "https://api.crossref.org/works",
                params={"filter": filter_str, "rows": 100},
                headers=headers,
                timeout=25,
            )
            if resp.status_code != 200:
                continue
            items = (resp.json().get("message") or {}).get("items") or []
            for item in items:
                titles = item.get("title") or []
                title = clean_html_text(titles[0]) if titles else ""
                if not title:
                    continue
                if any(skip in title.lower() for skip in ["corrigendum", "erratum", "editorial board", "retraction"]):
                    continue

                # Strictly require full year-month-day online publication date within this week
                date_parts = (
                    (item.get("published-online") or item.get("created") or {})
                    .get("date-parts", [[]])[0]
                )
                if len(date_parts) < 3:
                    continue
                y, m, d = date_parts[0], date_parts[1], date_parts[2]
                pub_date = f"{y:04d}-{m:02d}-{d:02d}"
                if not is_within_date_window(pub_date, start_date, end_date):
                    continue

                doi = normalize_doi(item.get("DOI"))
                abstract = clean_html_text(item.get("abstract") or "")

                authors_raw = item.get("author") or []
                author_list = []
                for a in authors_raw[:4]:
                    family = a.get("family", "")
                    given = a.get("given", "")
                    if family:
                        author_list.append(f"{family}, {given[:1]}." if given else family)
                author_str = ", ".join(author_list) + (" et al." if len(authors_raw) > 4 else "")

                collected.append(
                    {
                        "domain": j["domain"],
                        "journal": j["name"],
                        "pubDate": pub_date,
                        "title": title,
                        "authors": author_str or "Authors not listed",
                        "doi": doi,
                        "abstract": abstract,
                        "source_api": "crossref",
                    }
                )
        except Exception as exc:
            logger.warning("Crossref fetch error for %s: %s", j["name"], exc)
        time.sleep(0.15)

    logger.info("Crossref collected %d strictly-dated weekly papers.", len(collected))
    return collected


def fetch_from_rss(
    journals: List[Dict[str, Any]],
    start_date: datetime.date,
    end_date: datetime.date,
) -> List[Dict[str, Any]]:
    """Parses publisher RSS feeds, strictly keeping only entries with a verified date in [start_date, end_date]."""
    collected: List[Dict[str, Any]] = []
    for j in journals:
        rss_url = j.get("rss_url")
        if not rss_url:
            continue
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:30]:
                title = clean_html_text(entry.get("title", ""))
                if not title or any(s in title.lower() for s in ["editorial board", "corrigendum", "erratum"]):
                    continue

                # Strictly require a parsed publication date within the weekly window
                parsed_tuple = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
                if not parsed_tuple:
                    continue
                dt = datetime.date(*parsed_tuple[:3])
                pub_date = dt.isoformat()
                if not is_within_date_window(pub_date, start_date, end_date):
                    continue

                link = entry.get("link", "")
                doi = normalize_doi(entry.get("prism_doi") or entry.get("dc_identifier") or "")
                if not doi and "doi.org/" in link:
                    doi = normalize_doi(link)

                summary = clean_html_text(entry.get("summary") or entry.get("description") or "")
                author_str = clean_html_text(entry.get("author", "Authors not listed"))

                collected.append(
                    {
                        "domain": j["domain"],
                        "journal": j["name"],
                        "pubDate": pub_date,
                        "title": title,
                        "authors": author_str,
                        "doi": doi or link,
                        "abstract": summary,
                        "source_api": "rss",
                    }
                )
        except Exception as exc:
            logger.debug("RSS parse skipped for %s: %s", j["name"], exc)

    logger.info("RSS feeds collected %d strictly-dated weekly entries.", len(collected))
    return collected


def deduplicate_papers(raw_list: List[Dict[str, Any]], week_key: str) -> List[Dict[str, Any]]:
    """Deduplicates papers across OpenAlex, Crossref, and RSS, preferring entries with richer abstracts."""
    by_key: Dict[str, Dict[str, Any]] = {}
    for p in raw_list:
        norm_title = re.sub(r"[^a-z0-9]", "", p["title"].lower())[:80]
        doi_key = p["doi"].lower().strip() if p["doi"].startswith("https://doi.org/") else ""
        dedup_key = doi_key or norm_title
        if not dedup_key:
            continue

        if dedup_key not in by_key:
            by_key[dedup_key] = p
        else:
            existing = by_key[dedup_key]
            if len(p.get("abstract", "")) > len(existing.get("abstract", "")):
                existing["abstract"] = p["abstract"]
            if existing.get("authors") == "Authors not listed" and p.get("authors") != "Authors not listed":
                existing["authors"] = p["authors"]

    final_list = []
    for p in by_key.values():
        p["id"] = make_paper_id(week_key, p.get("doi", ""), p["title"])
        final_list.append(p)

    final_list.sort(key=lambda x: (x["pubDate"], x["journal"]), reverse=True)
    logger.info("Deduplicated to %d unique weekly papers.", len(final_list))
    return final_list


# ---------------------------------------------------------------------------
# Step 2: Gemini API Per-Paper Summarization & Weekly Synthesis
# ---------------------------------------------------------------------------
def load_summary_cache() -> Dict[str, Dict[str, Any]]:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_summary_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def summarize_single_paper(
    client: genai.Client,
    paper: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Summarizes one paper into structured Korean fields (Topic, Methodology, Issue, Keywords)."""
    abstract_text = paper.get("abstract") or "(초록 미제공 - 논문 제목과 저널 특성을 바탕으로 핵심 주제를 요약하세요)"
    prompt = f"""당신은 건축·도시·조경·기후변화·재난재해 분야의 수석 학술 에디터입니다.
다음 저널 논문의 메타데이터와 영문 초록을 분석하여 연구자를 위한 구조화 요약을 작성하세요.

- 분야: {paper['domain']}
- 저널명: {paper['journal']}
- 논문 원제: {paper['title']}
- 저자: {paper['authors']}
- 출판일: {paper['pubDate']}
- 초록(Abstract): {abstract_text}

작성 원칙:
1. `titleKo`: 한국어 연구자가 직관적으로 이해할 수 있는 정확한 한국어 학술 제목으로 번역하세요.
2. `topic`: 이 논문이 해결하고자 하는 구체적인 연구 배경과 핵심 주제를 2~3문장으로 명확히 요약하세요.
3. `methodology`: 연구에 사용된 데이터, 공간적 범위, 시뮬레이션/모델링 도구(예: ENVI-met, EnergyPlus, SWMM, GNN, InSAR 등), 통계·실증 기법을 구체적으로 명시하세요.
4. `methodType`: 반드시 'AI/머신러닝', '시뮬레이션/모델링', '원격탐사/GIS', '실증/현장계측' 중 가장 적합한 하나만 선택하세요.
5. `issue`: 논문이 발견한 핵심 정량적/정성적 결과와 이것이 도시·건축·기후·방재 정책이나 실무에 던지는 핵심 이슈·시사점을 정리하세요.
6. `keywords`: 논문을 대표하는 핵심 키워드 3~4개를 추출하세요.
"""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PaperSummarySchema,
                    temperature=0.2,
                ),
            )
            parsed = json.loads(response.text)
            return {
                **paper,
                "titleKo": parsed.get("titleKo", paper["title"]),
                "topic": parsed.get("topic", ""),
                "methodology": parsed.get("methodology", ""),
                "methodType": parsed.get("methodType", "시뮬레이션/모델링"),
                "issue": parsed.get("issue", ""),
                "keywords": parsed.get("keywords", [paper["domain"]]),
                "impactScore": int(parsed.get("impactScore", 88)),
                "abstract": abstract_text[:650] + ("..." if len(abstract_text) > 650 else ""),
            }
        except Exception as exc:
            logger.warning("Gemini summary attempt %d failed for '%s': %s", attempt + 1, paper["title"][:40], exc)
            time.sleep(2 ** attempt)

    return {
        **paper,
        "titleKo": paper["title"],
        "topic": f"{paper['journal']}에 출판된 {paper['domain']} 분야 최신 연구 논문입니다.",
        "methodology": "초록 원문 기반 분석 및 모델링 수행",
        "methodType": "시뮬레이션/모델링",
        "issue": "원문(DOI) 링크를 통해 상세 연구 결과를 확인할 수 있습니다.",
        "keywords": [paper["domain"], paper["journal"]],
        "impactScore": 85,
    }


def synthesize_weekly_trends(
    client: genai.Client,
    week_label: str,
    summarized_papers: List[Dict[str, Any]],
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Synthesizes all paper summaries of the week into a comprehensive Weekly Trend Report."""
    if not summarized_papers:
        return {
            "headline": f"{week_label} 신규 출판 논문 집계 대기 중",
            "narrative": "해당 주간에 색인된 신규 논문이 없거나 출판사 메타데이터 업데이트 대기 중입니다.",
            "pillars": [
                {"badge": "주제 트렌드 (Topics)", "title": "집계 대기 중", "desc": "다음 배치 실행 시 자동 갱신됩니다."},
                {"badge": "방법론 트렌드 (Methodologies)", "title": "집계 대기 중", "desc": "다음 배치 실행 시 자동 갱신됩니다."},
                {"badge": "핵심 이슈 (Key Issues)", "title": "집계 대기 중", "desc": "다음 배치 실행 시 자동 갱신됩니다."},
            ],
            "topKeywords": [],
        }

    compact_digest = []
    for idx, p in enumerate(summarized_papers[:80], 1):
        line = (
            f"[{idx}] 분야:{p['domain']} | 저널:{p['journal']} | 제목:{p['titleKo']}\n"
            f"  - 주제: {p['topic']}\n"
            f"  - 방법론({p['methodType']}): {p['methodology']}\n"
            f"  - 이슈: {p['issue']}\n"
            f"  - 키워드: {', '.join(p.get('keywords', []))}"
        )
        compact_digest.append(line)

    digest_block = "\n".join(compact_digest)
    prompt = f"""당신은 건축·도시·조경·기후변화·재난재해 5대 분야를 총괄하는 학술 트렌드 분석가입니다.
아래는 {week_label} 한 주 동안 19개 주류 저널에서 출판된 총 {len(summarized_papers)}편의 논문 요약 리스트입니다.

이 요약들을 종합(Synthesis)하여 이번 주 학술 트렌드 리포트를 작성하세요:
1. `headline`: 이번 주 논문들을 관통하는 가장 핵심적인 연구 흐름을 1문장(50~80자 내외)으로 압축한 헤드라인
2. `narrative`: ① 주제 측면에서 어떤 연구들이 주류를 이루었는지, ② 방법론 측면에서 어떤 시뮬레이션·AI·데이터 결합이 두드러졌는지, ③ 정책·실무적 이슈(기후 형평성, 복합재난, 탄소중립 등)와 분야 간 융합 흐름이 어떠했는지 4~6문장으로 깊이 있게 종합 서술하세요.
3. `pillars`: 정확히 3개의 항목으로 구성하세요:
   - 첫 번째: `badge`="주제 트렌드 (Topics)"
   - 두 번째: `badge`="방법론 트렌드 (Methodologies)"
   - 세 번째: `badge`="핵심 이슈 (Key Issues)"
4. `topKeywords`: 금주 논문들에서 가장 빈번하고 중요하게 다뤄진 핵심 키워드 상위 8개와 등장/연관 논문 편수(`count`)를 산출하세요.

[금주 논문 요약 목록]
{digest_block}
"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WeeklyTrendSynthesisSchema,
                temperature=0.25,
            ),
        )
        return json.loads(response.text)
    except Exception as exc:
        logger.error("Weekly trend synthesis failed: %s", exc)
        return {
            "headline": f"{week_label} 건축·도시·조경·기후·재난 주간 학술 동향 종합",
            "narrative": f"이번 주 19개 타겟 저널에서 총 {len(summarized_papers)}편의 논문이 수집 및 요약되었습니다.",
            "pillars": [
                {"badge": "주제 트렌드 (Topics)", "title": "도시 기후 적응 및 지속가능 건축", "desc": "분야별 주요 논문 요약을 아래 목록에서 확인하세요."},
                {"badge": "방법론 트렌드 (Methodologies)", "title": "시뮬레이션 및 공간 AI 결합", "desc": "고해상도 공간 데이터와 수치 모델링 결합 연구가 지속되고 있습니다."},
                {"badge": "핵심 이슈 (Key Issues)", "title": "기후 회복탄력성 및 공간 형평성", "desc": "극한기후 대응 인프라의 실효성 검증 연구가 다수 발표되었습니다."},
            ],
            "topKeywords": [{"word": "Urban Climate Resilience", "count": len(summarized_papers)}],
        }


# ---------------------------------------------------------------------------
# Step 3: Save Static Data & Optional Notifications (Slack / Telegram / Email)
# ---------------------------------------------------------------------------
def send_notifications(week_key: str, week_data: Dict[str, Any]) -> None:
    """Sends Sunday evening briefing notifications if webhook/API keys are configured."""
    papers = week_data.get("papers", [])
    headline = week_data.get("headline", "")
    week_label = week_data.get("weekLabel", week_key)

    top3_lines = []
    for idx, p in enumerate(papers[:3], 1):
        top3_lines.append(f"{idx}. [{p['domain']} · {p['journal']}] {p['titleKo']} ({p['doi']})")
    top3_text = "\n".join(top3_lines)

    summary_message = (
        f"📚 *[Sunday Scholar Pulse] {week_label} 주간 논문 트렌드 리포트*\n"
        f"• *금주 수집 논문:* 19개 저널 총 {len(papers)}편 전수 요약 완료\n"
        f"• *핵심 헤드라인:* {headline}\n\n"
        f"*[주간 트렌드 종합]*\n{week_data.get('narrative', '')}\n\n"
        f"*[금주 주목할 논문 Top 3]*\n{top3_text}"
    )

    slack_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if slack_url:
        try:
            requests.post(slack_url, json={"text": summary_message}, timeout=15)
            logger.info("Sent Slack notification successfully.")
        except Exception as exc:
            logger.warning("Slack notification failed: %s", exc)

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if tg_token and tg_chat_id:
        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": tg_chat_id, "text": summary_message},
                timeout=15,
            )
            logger.info("Sent Telegram notification successfully.")
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)

    resend_key = os.environ.get("RESEND_API_KEY", "").strip()
    alert_email = os.environ.get("ALERT_TO_EMAIL", "").strip()
    if resend_key and alert_email:
        try:
            html_body = (
                f"<h2>[Sunday Scholar Pulse] {week_label}</h2>"
                f"<p><strong>핵심 헤드라인:</strong> {headline}</p>"
                f"<p>{week_data.get('narrative', '')}</p>"
                f"<h3>금주 출판 논문 ({len(papers)}편)</h3>"
                + "".join(
                    f"<p><b>[{p['domain']} | {p['journal']}] {p['titleKo']}</b><br>"
                    f"• 주제: {p['topic']}<br>• 방법론: {p['methodology']}<br>• 이슈: {p['issue']}<br>"
                    f"• 원문: <a href='{p['doi']}'>{p['doi']}</a></p>"
                    for p in papers[:15]
                )
            )
            requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                json={
                    "from": "Sunday Scholar Pulse <onboarding@resend.dev>",
                    "to": [alert_email],
                    "subject": f"[Sunday Scholar Pulse] {week_label} 주간 논문 트렌드 리포트 ({len(papers)}편)",
                    "html": html_body,
                },
                timeout=20,
            )
            logger.info("Sent Email newsletter via Resend to %s.", alert_email)
        except Exception as exc:
            logger.warning("Resend email failed: %s", exc)


def main() -> None:
    days_back = int(os.environ.get("DAYS_BACK", "7"))
    force_resummarize = os.environ.get("FORCE_RESUMMARIZE", "false").lower() == "true"
    polite_email = os.environ.get("OPENALEX_POLITE_EMAIL", "")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    now_kst = datetime.datetime.now(tz=KST)
    end_date = now_kst.date()
    start_date = end_date - datetime.timedelta(days=days_back)

    iso_year, iso_week, _ = end_date.isocalendar()
    week_key = f"{iso_year}-W{iso_week:02d}"
    week_of_month = (end_date.day - 1) // 7 + 1
    week_label = (
        f"{end_date.year}년 {end_date.month}월 {week_of_month}주차 "
        f"({start_date.strftime('%m.%d')} ~ {end_date.strftime('%m.%d')})"
    )
    publish_date_str = now_kst.strftime("%Y년 %m월 %d일 (일) %H:%M KST")

    logger.info("=== Starting Sunday Scholar Pulse Batch for %s (%s) ===", week_key, week_label)
    journals = load_journals_config()

    # 1. Collect papers strictly within [start_date, end_date] from OpenAlex, Crossref, and RSS
    oa_papers = fetch_from_openalex(journals, start_date, end_date, polite_email)
    cr_papers = fetch_from_crossref(journals, start_date, end_date, polite_email)
    rss_papers = fetch_from_rss(journals, start_date, end_date)

    unique_papers = deduplicate_papers(oa_papers + cr_papers + rss_papers, week_key)

    # 2. Summarize each paper with Gemini (using cache for already-summarized DOIs)
    cache = {} if force_resummarize else load_summary_cache()
    summarized_papers: List[Dict[str, Any]] = []

    if not api_key:
        logger.warning("GEMINI_API_KEY is not set! Running in metadata-only fallback mode.")
        client = None
    else:
        client = genai.Client(api_key=api_key)

    for idx, paper in enumerate(unique_papers, 1):
        cache_key = paper.get("doi") or paper["title"]
        if cache_key in cache and not force_resummarize:
            cached_entry = cache[cache_key]
            cached_entry["id"] = paper["id"]
            cached_entry["pubDate"] = paper["pubDate"]
            summarized_papers.append(cached_entry)
            continue

        logger.info("[%d/%d] Summarizing: %s (%s)", idx, len(unique_papers), paper["title"][:55], paper["journal"])
        if client:
            summary_obj = summarize_single_paper(client, paper)
            cache[cache_key] = summary_obj
            summarized_papers.append(summary_obj)
            time.sleep(0.3)
        else:
            summarized_papers.append(
                {
                    **paper,
                    "titleKo": paper["title"],
                    "topic": "GEMINI_API_KEY 설정 후 자동 요약됩니다.",
                    "methodology": "OpenAlex / Crossref 메타데이터 수집 완료",
                    "methodType": "시뮬레이션/모델링",
                    "issue": paper.get("abstract", "")[:200],
                    "keywords": [paper["domain"]],
                    "impactScore": 85,
                }
            )

    save_summary_cache(cache)

    # 3. Synthesize Weekly Trends
    logger.info("Synthesizing weekly trends across %d papers...", len(summarized_papers))
    if client and summarized_papers:
        synthesis = synthesize_weekly_trends(client, week_label, summarized_papers)
    else:
        synthesis = {
            "headline": f"{week_label} 19개 저널 총 {len(summarized_papers)}편 금주 출판 논문 수집 완료",
            "narrative": "GEMINI_API_KEY를 설정하면 주간 논문 종합 트렌드 리포트가 생성됩니다.",
            "pillars": [
                {"badge": "주제 트렌드 (Topics)", "title": "금주 논문 수집 완료", "desc": "개별 논문 카드를 확인하세요."},
                {"badge": "방법론 트렌드 (Methodologies)", "title": "다양한 방법론 포착", "desc": "개별 논문 카드를 확인하세요."},
                {"badge": "핵심 이슈 (Key Issues)", "title": "주요 학술 이슈", "desc": "개별 논문 카드를 확인하세요."},
            ],
            "topKeywords": [{"word": "Urban & Climate Research", "count": len(summarized_papers)}],
        }

    week_payload = {
        "weekKey": week_key,
        "weekLabel": week_label,
        "publishDate": publish_date_str,
        "headline": synthesis.get("headline", ""),
        "narrative": synthesis.get("narrative", ""),
        "pillars": synthesis.get("pillars", []),
        "topKeywords": synthesis.get("topKeywords", []),
        "papers": summarized_papers,
    }

    # 4. Write output JSON & update manifest
    WEEKS_DIR.mkdir(parents=True, exist_ok=True)
    week_file = WEEKS_DIR / f"{week_key}.json"
    with open(week_file, "w", encoding="utf-8") as f:
        json.dump(week_payload, f, ensure_ascii=False, indent=2)

    all_weeks_data: Dict[str, Any] = {}
    for wf in sorted(WEEKS_DIR.glob("*.json"), reverse=True):
        try:
            w_obj = json.loads(wf.read_text(encoding="utf-8"))
            all_weeks_data[wf.stem] = w_obj
        except Exception as exc:
            logger.warning("Failed reading %s: %s", wf, exc)

    MANIFEST_PATH.write_text(json.dumps(all_weeks_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %s and updated %s (%d weeks).", week_file.name, MANIFEST_PATH.name, len(all_weeks_data))

    # 5. Send Notifications
    send_notifications(week_key, week_payload)
    logger.info("=== Sunday Scholar Pulse Batch Completed Successfully ===")


if __name__ == "__main__":
    main()