import os
import re
import json
import time
import urllib.parse
from datetime import date
from pathlib import Path
import feedparser
import requests
import yaml
from google import genai

# ───────────────────────────────────────────
# 환경 변수 & 상수
# ───────────────────────────────────────────
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GITHUB_REPOSITORY  = os.environ.get("GITHUB_REPOSITORY", "")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL  = "gemini-3.1-flash-lite-preview"
TODAY  = date.today().isoformat()

if GITHUB_REPOSITORY:
    _owner, _repo = GITHUB_REPOSITORY.split("/", 1)
    PAGES_BASE = f"https://{_owner}.github.io/{_repo}"
else:
    PAGES_BASE = ""


# ───────────────────────────────────────────
# 설정
# ───────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────
def strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text).strip()

def strip_emoji(text: str) -> str:
    return re.sub(
        r'[\U0001F000-\U0001FFFF\U00002600-\U000027FF\U0000FE00-\U0000FE0F\u200d]',
        '', text
    ).strip()

def resolve_google_news_url(url: str) -> str:
    try:
        res = requests.get(url, allow_redirects=True, timeout=5)
        return res.url
    except Exception:
        return url

def build_google_news_url(keywords: str, lang: str, region: str) -> str:
    return f"https://news.google.com/rss/search?q={urllib.parse.quote(keywords)}&hl={lang}&gl={region}&ceid={region}:{lang}"


# ───────────────────────────────────────────
# 뉴스 수집
# ───────────────────────────────────────────
def fetch_from_rss(rss_url: str, per_source: int, seen_urls: set, resolve: bool = False) -> list:
    results = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            if len(results) >= per_source:
                break
            link = entry.get('link', '')
            if resolve:
                link = resolve_google_news_url(link)
            if not link or link in seen_urls:
                continue
            seen_urls.add(link)
            results.append({
                'title':   strip_html(entry.get('title', '')),
                'summary': strip_html(entry.get('description', entry.get('summary', ''))),
                'link':    link,
            })
    except Exception as e:
        print(f"    ⚠️ RSS 에러 ({rss_url.split('/')[2]}): {e}")
    return results

def fetch_news(interest: dict, lang: str, region: str, limit: int, seen_urls: set) -> list:
    custom_rss    = interest.get("custom_rss", [])
    total_sources = len(custom_rss) + 1
    per_source    = max(2, limit // total_sources)
    all_results   = []

    for rss_url in custom_rss:
        fetched = fetch_from_rss(rss_url, per_source, seen_urls)
        all_results.extend(fetched)
        if fetched:
            print(f"    └ {len(fetched)}개 ← {rss_url.split('/')[2]}")

    remaining = limit - len(all_results)
    if remaining > 0:
        url     = build_google_news_url(interest["keywords"], lang, region)
        fetched = fetch_from_rss(url, remaining, seen_urls, resolve=True)
        all_results.extend(fetched)
        if fetched:
            print(f"    └ {len(fetched)}개 ← Google News")

    return all_results


# ───────────────────────────────────────────
# 글로벌 랭킹 (동적 개수)
# ───────────────────────────────────────────
def global_rank_and_select(articles_by_cat: dict) -> tuple:
    flat = []
    for cat_name, articles in articles_by_cat.items():
        for a in articles:
            flat.append({**a, 'category': cat_name})

    if not flat:
        return articles_by_cat, flat

    numbered = "\n".join(
        f"[{i}] [{a['category']}] {a['title']}"
        for i, a in enumerate(flat)
    )

    prompt = f"""
너는 오늘의 AI/테크 뉴스 큐레이터야.
아래는 여러 카테고리에서 수집된 기사 목록이야.

오늘 진짜 읽을 가치 있는 기사만 골라줘.
- 고정 개수 없음. 오늘 이슈가 많으면 많이(최대 20개), 적으면 적게(최소 5개).
- 카테고리 쿼터 없음. 오늘 핫한 분야가 있으면 거기서 더 뽑아도 됨.
- 중복되거나 뻔한 기사는 과감히 제외.
- 중요한 기사를 앞에 배치해줘. 배열 순서 = 중요도 순서.

결과는 인덱스 번호만 JSON 배열로. 다른 말 없이 JSON만.
예시: [0, 3, 7, 12, 15]

[선별 기준]
1. 새 모델/제품 출시, 벤치마크 결과, 획기적 기술 발표
2. 인수합병, 대규모 투자, 핵심 인물 동향
3. 보안 취약점, 정책/규제 변화
4. 분석 기사, 인터뷰 (위 기준 해당 없으면 제외)

[기사 목록]
{numbered}
"""
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        raw      = re.sub(r'```[a-z]*', '', response.text.strip()).strip('`')
        indices  = json.loads(raw)
        indices  = [i for i in indices if isinstance(i, int) and 0 <= i < len(flat)]
        selected = [flat[i] for i in indices]
    except Exception as e:
        print(f"  ⚠️ 글로벌 랭킹 실패: {e} → 전체 반환")
        selected = flat

    result = {cat: [] for cat in articles_by_cat}
    for a in selected:
        result[a['category']].append(a)
    return result, selected


# ───────────────────────────────────────────
# 요약
# ───────────────────────────────────────────
def summarize(article: dict) -> str:
    prompt = f"""
너는 실리콘밸리 딥테크 트렌드에 빠삭하고 까칠한 동료 엔지니어야.
다음 영문 뉴스를 개발자 입장에서 핵심 팩트만 한국어로 딱 3줄 요약해.

[규칙]
1. "~습니다" 금지. "~했어.", "~상황이야." 등 반말 사용.
2. 벤치마크 점수, 아키텍처 변화, 보안 취약점 등 구체적 기술 팩트 위주.
3. 마크다운/이모지 없이 순수 텍스트 3줄.

제목: {article.get('title', '')}
내용: {article.get('summary', '')}
"""
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"요약 실패: {e}"


# ───────────────────────────────────────────
# 텔레그램 하이라이트
# ───────────────────────────────────────────
def generate_highlights(selected_by_cat: dict) -> str:
    all_titles = []
    for cat, articles in selected_by_cat.items():
        for a in articles:
            all_titles.append(f"[{strip_emoji(cat)}] {a['title']}")

    if not all_titles:
        return "(하이라이트 없음)"

    prompt = f"""
오늘의 AI/테크 뉴스 중 가장 임팩트 있는 3개만 골라 한 줄씩 한국어 반말로 요약해.
각 줄 앞에 카테고리 이름을 붙여. 이모지/마크다운 없이 텍스트 3줄만.

{chr(10).join(all_titles[:40])}
"""
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"하이라이트 생성 실패: {e}"


# ───────────────────────────────────────────
# HTML 리포트 생성 (Liquid Glass 디자인)
# ───────────────────────────────────────────
def generate_html(selected_by_cat: dict, summaries: dict, interests: list, ranked_flat: list = None) -> str:
    # ranked_flat: LLM 중요도 순으로 정렬된 전체 기사
    if ranked_flat is None:
        ranked_flat = []
        for cat, articles in selected_by_cat.items():
            for a in articles:
                ranked_flat.append({**a, 'category': cat})

    flat = [{**a, 'cat': strip_emoji(a.get('category', ''))} for a in ranked_flat]

    total_count   = len(flat)
    active_cats   = sum(1 for v in selected_by_cat.values() if v)
    total_sources = sum(len(i.get("custom_rss", [])) + 1 for i in interests)

    # 카드는 순수 유리 — 배경 orb가 움직이며 색 변화

    # 크기: 2위부터 완만하게 줄어듦
    SIZE_STEPS = [
        ("2rem",    "15px",   "600"),  # 1위 히어로
        ("1.35rem", "15px",   "600"),  # 2위
        ("1.26rem", "14.5px", "600"),  # 3위
        ("1.18rem", "14px",   "600"),  # 4위
        ("1.12rem", "14px",   "600"),  # 5위
        ("1.06rem", "13.5px", "600"),  # 6위
        ("1.01rem", "13px",   "500"),  # 7위
        ("0.97rem", "13px",   "500"),  # 8위+
    ]

    # 데코 패널 너비: 카드마다 약간씩 달라서 리듬감 줌
    DECO_WIDTHS = [220, 200, 240, 190, 230, 210, 200, 220]

    def get_size(idx):
        return SIZE_STEPS[min(idx, len(SIZE_STEPS)-1)]

    def get_deco_w(idx):
        return DECO_WIDTHS[idx % len(DECO_WIDTHS)]

    # 카드마다 약간 다른 border-radius로 딱딱함 없애기
    RADII = [20, 22, 18, 24, 20, 18, 22, 20]

    # ── 히어로 카드 (1위 풀너비)
    all_cards_html = ""
    if flat:
        a       = flat[0]
        link    = a.get("link","#")
        title   = a.get("title","")
        summary = summaries.get(link,"").replace("\n","<br>")
        domain  = link.split("/")[2] if "/" in link else link
        tsz, ssz, tw = get_size(0)
        all_cards_html += f"""
        <article class="card card-hero-full">
          <div class="hero-num">01</div>
          <div class="hero-body">
            <span class="badge">{a["cat"]}</span>
            <h2 style="font-size:{tsz};font-weight:{tw};color:#fff;line-height:1.2;letter-spacing:-0.025em;margin:14px 0 16px">
              <a href="{link}" target="_blank" rel="noopener">{title}</a>
            </h2>
            <p style="font-size:{ssz};color:var(--wh);line-height:1.75;margin-bottom:20px">{summary}</p>
            <div class="card-foot"><span class="src">{domain}</span><a class="read-btn" href="{link}" target="_blank" rel="noopener">읽기</a></div>
          </div>
        </article>"""

    # ── 2위~: 지그재그
    # fix: 항상 HTML 순서는 [text, deco]로 고정하고
    # flex-direction으로만 방향 제어 → row=텍스트 왼쪽, row-reverse=텍스트 오른쪽
    for i, a in enumerate(flat[1:], 1):
        link    = a.get("link","#")
        title   = a.get("title","")
        summary = summaries.get(link,"").replace("\n","<br>")
        domain  = link.split("/")[2] if "/" in link else link
        num     = f"0{i+1}" if i+1 < 10 else str(i+1)
        tsz, ssz, tw = get_size(i)
        dw      = get_deco_w(i)
        radius  = RADII[i % len(RADII)]

        # 홀수 i → row (텍스트 왼쪽), 짝수 i → row-reverse (텍스트 오른쪽)
        flex_dir     = "row" if i % 2 == 1 else "row-reverse"
        # deco 테두리: row일 때 deco가 오른쪽이므로 border-left, row-reverse면 border-right
        deco_border  = "border-left:1px solid var(--border);border-right:none" if i % 2 == 1 else "border-right:1px solid var(--border);border-left:none"

        all_cards_html += f"""
        <article class="card card-zz" style="flex-direction:{flex_dir};border-radius:{radius}px">
          <div class="zz-text">
            <h3 style="font-size:{tsz};font-weight:{tw};color:#d8ecf8;line-height:1.35;letter-spacing:-0.015em;margin-bottom:12px">
              <a href="{link}" target="_blank" rel="noopener">{title}</a>
            </h3>
            <p style="font-size:{ssz};color:var(--wh);line-height:1.75;margin-bottom:16px">{summary}</p>
            <div class="card-foot"><span class="src">{domain}</span><a class="read-btn" href="{link}" target="_blank" rel="noopener">읽기</a></div>
          </div>
          <div class="zz-deco" style="min-width:{dw}px;max-width:{dw}px;{deco_border}">
            <div class="dot-grid"></div>
            <div class="zz-num">{num}</div>
            <span class="badge" style="margin-top:10px;position:relative;z-index:1">{a["cat"]}</span>
          </div>
        </article>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Myrmidon — {TODAY}</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{
      --violet:#663af3;
      --violet-dim:rgba(102,58,243,0.15);
      --comet:#d8ecf8;
      --mist:#d1e4fa;
      --celestial:#b6d9fc;
      --azure:#c7d3ea;
      --whisper:#9da7ba;
      --ig:#81899b;
      --slate:#3f4959;
      --ghost:#fff;
      --border:rgba(186,215,247,0.14);
      --glass:rgba(255,255,255,0.04);
      --glass-strong:rgba(255,255,255,0.07);
      --fd:'Space Grotesk',system-ui,sans-serif;
      --fb:'Inter',system-ui,sans-serif;
      --fm:'IBM Plex Mono',monospace;
    }}

    html{{scroll-behavior:smooth}}

    /* ── 배경 ── */
    body{{
      background:#05060f;
      color:var(--comet);
      font-family:var(--fb);
      font-size:14px;
      line-height:1.5;
      min-height:100vh;
      overflow-x:hidden;
    }}
    .bg{{
      position:fixed;
      inset:0;
      z-index:0;
      overflow:hidden;
    }}
    .orb{{
      position:absolute;
      border-radius:50%;
      filter:blur(80px);
      opacity:0.55;
      animation:drift 20s ease-in-out infinite alternate;
    }}
    .orb{{
      position:absolute;
      border-radius:50%;
      filter:blur(90px);
      opacity:0.6;
      transition:none;
      will-change:transform;
    }}
    @keyframes drift{{
      0%{{transform:translate(0,0) scale(1)}}
      33%{{transform:translate(60px,-40px) scale(1.05)}}
      66%{{transform:translate(-30px,60px) scale(0.97)}}
      100%{{transform:translate(40px,20px) scale(1.03)}}
    }}

    /* ── 레이어 ── */
    .wrap{{position:relative;z-index:1}}

    /* ── 네비 ── */
    nav{{
      position:sticky;top:0;z-index:100;
      background:rgba(5,6,15,0.7);
      backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
      border-bottom:1px solid var(--border);
      padding:0 2.5rem;height:56px;
      display:flex;align-items:center;justify-content:space-between;
    }}
    .nav-logo{{
      font-family:var(--fd);font-size:18px;font-weight:600;
      color:var(--ghost);text-decoration:none;letter-spacing:-0.02em;
    }}
    .nav-logo span{{color:var(--violet)}}
    .nav-date{{font-family:var(--fm);font-size:11px;color:var(--ig);letter-spacing:0.05em}}
    .nav-cta{{
      font-family:var(--fb);font-size:13px;font-weight:500;color:var(--ghost);
      background:var(--violet);border-radius:999px;padding:7px 20px;
      text-decoration:none;
      box-shadow:rgba(102,58,243,0.5) 0px 0px 20px 0px;
      transition:box-shadow 0.2s,transform 0.2s;
    }}
    .nav-cta:hover{{box-shadow:rgba(102,58,243,0.8) 0px 0px 30px 0px;transform:translateY(-1px)}}

    /* ── 히어로 ── */
    .hero{{
      padding:6rem 2.5rem 5rem;
      text-align:center;
      border-bottom:1px solid var(--border);
    }}
    .hero-eyebrow{{
      display:inline-block;
      font-family:var(--fm);font-size:10px;color:var(--ig);
      letter-spacing:0.14em;text-transform:uppercase;
      border:1px solid var(--border);
      background:rgba(186,215,247,0.04);
      backdrop-filter:blur(8px);
      padding:5px 16px;border-radius:6px;margin-bottom:2rem;
    }}
    .hero h1{{
      font-family:var(--fd);
      font-size:clamp(2.6rem,7vw,4.2rem);
      font-weight:500;color:var(--ghost);
      line-height:1.1;letter-spacing:-0.03em;margin-bottom:1.25rem;
    }}
    .hero h1 .accent{{
      background:linear-gradient(120deg,var(--celestial) 0%,var(--violet) 60%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    }}
    .hero-sub{{font-size:15px;color:var(--azure);line-height:1.6;max-width:420px;margin:0 auto 3rem}}
    .stats{{
      display:inline-flex;
      border:1px solid var(--border);border-radius:16px;overflow:hidden;
      background:rgba(255,255,255,0.04);
      backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
      box-shadow:rgba(255,255,255,0.08) 0px 1px 0px inset,
                 rgba(102,58,243,0.1) 0px 0px 40px 0px;
    }}
    .stat{{padding:1.25rem 2.5rem;text-align:center}}
    .stat+.stat{{border-left:1px solid var(--border)}}
    .sn{{font-family:var(--fd);font-size:2rem;font-weight:600;color:var(--ghost);line-height:1;margin-bottom:5px}}
    .sl{{font-family:var(--fm);font-size:9px;color:var(--ig);letter-spacing:0.1em;text-transform:uppercase}}

    /* ── 콘텐츠 ── */
    .content{{
      max-width:1280px;margin:0 auto;
      padding:3rem 2.5rem 6rem;
    }}

    /* ── 공통 카드 ── */
    .card{{
      background:var(--glass);
      backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
      border:1px solid var(--border);
      border-radius:20px;
      box-shadow:
        rgba(255,255,255,0.07) 0px 1px 0px inset,
        rgba(186,215,247,0.04) 0px 0px 60px 0px inset,
        rgba(0,0,0,0.4) 0px 20px 40px 0px;
      transition:transform 0.3s cubic-bezier(0.175,0.885,0.32,1.1),
                 box-shadow 0.3s,border-color 0.3s;
      overflow:hidden;
    }}
    .card:hover{{
      transform:translateY(-5px) scale(1.005);
      border-color:rgba(186,215,247,0.28);
      box-shadow:
        rgba(255,255,255,0.1) 0px 1px 0px inset,
        rgba(102,58,243,0.12) 0px 0px 60px 0px inset,
        rgba(0,0,0,0.6) 0px 30px 60px 0px;
    }}
    .card-inner{{padding:28px;height:100%;display:flex;flex-direction:column;gap:14px}}
    .card-meta{{display:flex;align-items:center;justify-content:space-between}}
    .badge{{
      font-family:var(--fm);font-size:10px;color:var(--ig);
      letter-spacing:0.06em;text-transform:uppercase;
      border:1px solid var(--border);
      background:rgba(186,215,247,0.06);
      padding:3px 10px;border-radius:6px;
    }}
    .card-rank{{font-family:var(--fm);font-size:11px;color:var(--slate);letter-spacing:0.08em}}
    .card-title{{flex:0;}}
    .card-title a{{color:inherit;text-decoration:none;transition:color 0.2s}}
    .card-title a:hover{{color:var(--ghost)}}
    .card-summary{{color:var(--whisper);line-height:1.7;flex:1}}
    .card-foot{{
      display:flex;align-items:center;justify-content:space-between;
      padding-top:14px;border-top:1px solid var(--border);
      margin-top:auto;
    }}
    .src{{font-family:var(--fm);font-size:10px;color:var(--slate);letter-spacing:0.04em}}
    .read-btn{{
      font-family:var(--fb);font-size:12px;font-weight:500;color:var(--mist);
      background:rgba(186,214,247,0.06);border:1px solid var(--border);
      border-radius:999px;padding:4px 14px;text-decoration:none;
      transition:all 0.2s;
    }}
    .read-btn:hover{{
      background:rgba(102,58,243,0.2);border-color:rgba(102,58,243,0.5);
      color:var(--ghost);box-shadow:rgba(102,58,243,0.25) 0px 0px 16px 0px;
    }}

    /* ── 히어로 풀너비 카드 ── */
    .card-hero-full{{
      border-radius:24px;
      display:flex;
      align-items:center;
      gap:0;
      margin-bottom:3rem;
      overflow:hidden;
      min-height:260px;
    }}
    .hero-num{{
      font-family:var(--fd);
      font-size:clamp(5rem,12vw,9rem);
      font-weight:700;
      color:rgba(255,255,255,0.06);
      letter-spacing:-0.04em;
      line-height:1;
      padding:3rem 2.5rem;
      flex-shrink:0;
      user-select:none;
      min-width:180px;
      text-align:center;
      border-right:1px solid var(--border);
    }}
    .hero-body{{
      flex:1;
      padding:3rem;
    }}

    /* ── 지그재그 카드 ── */
    .card-zz{{
      display:flex;
      align-items:stretch;
      border-radius:20px;
      margin-bottom:1.5rem;
      min-height:180px;
      overflow:hidden;
    }}
    .zz-deco{{
      display:flex;
      flex-direction:column;
      align-items:center;
      justify-content:center;
      padding:2rem 2.5rem;
      min-width:200px;
      max-width:240px;
      flex-shrink:0;
      border-right:1px solid var(--border);
      position:relative;
      overflow:hidden;
    }}
    .card-zz[style*="row-reverse"] .zz-deco{{
      border-right:none;
      border-left:1px solid var(--border);
    }}
    .zz-num{{
      font-family:var(--fd);
      font-size:clamp(3rem,6vw,5rem);
      font-weight:700;
      color:rgba(255,255,255,0.07);
      letter-spacing:-0.04em;
      line-height:1;
      user-select:none;
    }}
    .dot-grid{{
      position:absolute;
      inset:0;
      background-image:radial-gradient(circle, rgba(186,215,247,0.15) 1px, transparent 1px);
      background-size:18px 18px;
      pointer-events:none;
      mask-image:radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
    }}
    .zz-text{{
      flex:1;
      padding:2rem 2.5rem;
      display:flex;
      flex-direction:column;
      justify-content:center;
    }}
    .zz-text h3 a, .hero-body h2 a{{
      color:inherit;
      text-decoration:none;
      transition:opacity 0.2s;
    }}
    .zz-text h3 a:hover, .hero-body h2 a:hover{{opacity:0.75}}

    /* ── 카드 공통 ── */

    /* ── 구분선 ── */
    .divider{{
      text-align:center;padding:3rem 2rem;
      border-top:1px solid var(--border);
    }}
    .divider p{{font-size:13px;color:var(--ig);line-height:1.8;max-width:460px;margin:0 auto}}
    .divider strong{{color:var(--comet);font-weight:500}}

    /* ── 푸터 ── */
    footer{{
      border-top:1px solid var(--border);padding:2rem 2.5rem;
      display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;
    }}
    .fl{{font-family:var(--fd);font-size:15px;font-weight:600;color:var(--ghost);letter-spacing:-0.02em}}
    .fl span{{color:var(--violet)}}
    .fm-text{{font-family:var(--fm);font-size:10px;color:var(--ig);letter-spacing:0.06em}}

    @media(max-width:700px){{
      .feature-row{{grid-template-columns:1fr}}
      .card-grid-wrap{{grid-template-columns:1fr}}
      nav{{padding:0 1.25rem}}
      .nav-date{{display:none}}
      .content{{padding:2rem 1.25rem 4rem}}
      .hero{{padding:4rem 1.25rem 3rem}}
      .stat{{padding:1rem 1.5rem}}
    }}
  </style>
</head>
<body>

<div class="bg">
  <canvas id="orb-canvas"></canvas>
<script>
(function(){{
  const canvas = document.getElementById('orb-canvas');
  const ctx    = canvas.getContext('2d');

  // 뷰포트 고정
  Object.assign(canvas.style, {{
    position:'fixed', inset:'0', width:'100%', height:'100%', zIndex:'0', pointerEvents:'none'
  }});

  // orb 정의: 색상 다양하게
  const COLORS = [
    [102, 58, 243],   // 보라
    [50,  100, 240],  // 파랑
    [20,  180, 100],  // 초록
    [220, 140,  20],  // 주황
    [200,  50, 180],  // 핑크
    [20,  160, 200],  // 청록
    [240,  80,  40],  // 빨강-주황
    [130, 200,  30],  // 연두
  ];

  const N = 6;
  const orbs = Array.from({{length: N}}, (_, i) => {{
    const [r,g,b] = COLORS[i % COLORS.length];
    const size = 380 + Math.random() * 280;
    return {{
      x:  Math.random() * window.innerWidth,
      y:  Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      r, g, b,
      size,
      // 각 orb마다 독립적인 위상
      phase: Math.random() * Math.PI * 2,
      freq:  0.0003 + Math.random() * 0.0004,
    }};
  }});

  function resize() {{
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }}
  window.addEventListener('resize', resize);
  resize();

  let t = 0;
  function draw() {{
    t++;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    orbs.forEach(o => {{
      // 부드럽게 랜덤 이동 (perlin-like sinusoidal drift)
      o.x += o.vx + Math.sin(t * o.freq + o.phase) * 1.2;
      o.y += o.vy + Math.cos(t * o.freq * 0.7 + o.phase) * 1.0;

      // 경계 반사
      if (o.x < -o.size/2) o.x = canvas.width + o.size/2;
      if (o.x > canvas.width  + o.size/2) o.x = -o.size/2;
      if (o.y < -o.size/2) o.y = canvas.height + o.size/2;
      if (o.y > canvas.height + o.size/2) o.y = -o.size/2;

      // 그리기
      const grad = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.size/2);
      grad.addColorStop(0,   `rgba(${{o.r}},${{o.g}},${{o.b}},0.55)`);
      grad.addColorStop(0.5, `rgba(${{o.r}},${{o.g}},${{o.b}},0.20)`);
      grad.addColorStop(1,   `rgba(${{o.r}},${{o.g}},${{o.b}},0)`);
      ctx.globalCompositeOperation = 'lighter';
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(o.x, o.y, o.size/2, 0, Math.PI*2);
      ctx.fill();
    }});

    requestAnimationFrame(draw);
  }}
  draw();
}})();
</script>
</div>

<div class="wrap">
  <nav>
    <a class="nav-logo" href="#">Myrm<span>i</span>don</a>
    <span class="nav-date">{TODAY}</span>
    <a class="nav-cta" href="#content">브리핑 보기</a>
  </nav>

  <div class="hero">
    <span class="hero-eyebrow">Daily Intelligence Briefing</span>
    <h1>오늘의 <span class="accent">AI & Tech</span><br>브리핑</h1>
    <p class="hero-sub">글로벌 소스 {total_sources}개에서 수집</p>
    <div class="stats">
      <div class="stat"><div class="sn">{total_count}</div><div class="sl">선별 기사</div></div>
      <div class="stat"><div class="sn">{active_cats}</div><div class="sl">카테고리</div></div>
      <div class="stat"><div class="sn">{total_sources}</div><div class="sl">뉴스 소스</div></div>
    </div>
  </div>

  <div class="content" id="content">
    {all_cards_html}
  </div>

  <div class="divider">
    <p>매일 KST 07:00, <strong>GitHub Actions</strong>가 자동으로 수집·선별·요약합니다.<br>서버 비용 0원 · API 비용 월 2000원 이하</p>
  </div>

  <footer>
    <span class="fl">Myrm<span>i</span>don</span>
    <span class="fm-text">GENERATED {TODAY} · POWERED BY GEMINI</span>
  </footer>
</div>

</body>
</html>"""


def save_html(html_content: str) -> Path:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    report_path = docs_dir / f"{TODAY}.html"
    report_path.write_text(html_content, encoding="utf-8")

    reports = sorted(docs_dir.glob("????-??-??.html"), reverse=True)
    links   = "\n".join(
        f'<li><a href="{r.name}">{r.stem}</a></li>' for r in reports
    )
    index_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Myrmidon Archive</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Mono:wght@400&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Space Grotesk', sans-serif; background: #05060f; color: #d8ecf8;
           max-width: 400px; margin: 3rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.5rem; font-weight: 600; color: #fff; margin-bottom: 1.5rem; letter-spacing: -0.02em; }}
    h1 span {{ color: #663af3; }}
    ul {{ list-style: none; padding: 0; }}
    li {{ margin-bottom: 0.6rem; }}
    a {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #9da7ba; text-decoration: none; letter-spacing: 0.04em; }}
    a:hover {{ color: #b6d9fc; }}
  </style>
</head>
<body>
  <h1>Myrm<span>i</span>don</h1>
  <ul>{links}</ul>
</body>
</html>"""
    (docs_dir / "index.html").write_text(index_html, encoding="utf-8")

    return report_path


# ───────────────────────────────────────────
# 텔레그램
# ───────────────────────────────────────────
def get_chat_id() -> int | None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        res = requests.get(url).json()
        if res.get("result"):
            return res["result"][-1]["message"]["chat"]["id"]
    except Exception as e:
        print(f"❌ Chat ID 가져오기 실패: {e}")
    return None

def send_message(chat_id: int, text: str):
    MAX_LEN = 4096
    chunks  = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chunk in chunks:
        res = requests.post(url, json={"chat_id": chat_id, "text": chunk})
        if res.status_code == 200:
            print("✅ 텔레그램 전송 완료!")
        else:
            print(f"❌ 전송 실패: {res.json()}")
        time.sleep(0.5)


# ───────────────────────────────────────────
# 메인
# ───────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Myrmidon 가동...")

    config    = load_config()
    interests = config["interests"]
    limit     = config.get("articles_per_interest", 15)
    lang      = config.get("language", "en")
    region    = config.get("region", "US")

    chat_id = get_chat_id()
    if not chat_id:
        print("❌ 텔레그램 봇에게 먼저 말 걸고 다시 실행해줘!")
        exit()

    # 1. 수집
    seen_urls       = set()
    articles_by_cat = {}
    for interest in interests:
        name      = interest["name"]
        src_count = len(interest.get("custom_rss", [])) + 1
        print(f"\n  📡 [{name}] 수집 중 ({src_count}개 소스)")
        articles = fetch_news(interest, lang, region, limit, seen_urls)
        articles_by_cat[name] = articles
        print(f"  ✅ {len(articles)}개 수집")

    # 2. 글로벌 랭킹
    print(f"\n  🧠 글로벌 랭킹 중... (동적 선별)")
    selected_by_cat, ranked_flat = global_rank_and_select(articles_by_cat)
    total_selected  = sum(len(v) for v in selected_by_cat.values())
    print(f"  🎯 총 {total_selected}개 선별 완료")
    for cat, arts in selected_by_cat.items():
        if arts:
            print(f"    └ {cat}: {len(arts)}개")

    # 3. 요약
    print(f"\n  ✍️  요약 생성 중...")
    summaries = {}
    for cat, articles in selected_by_cat.items():
        for article in articles:
            summaries[article['link']] = summarize(article)
            time.sleep(1)

    # 4. HTML 생성 & 저장
    print(f"\n  📄 리포트 생성 중...")
    html_content = generate_html(selected_by_cat, summaries, interests, ranked_flat)
    report_path  = save_html(html_content)
    report_url   = f"{PAGES_BASE}/{TODAY}.html" if PAGES_BASE else f"(로컬: {report_path})"
    print(f"  ✅ 저장 완료: {report_path}")

    # 5. 텔레그램
    print(f"\n  📱 텔레그램 전송 중...")
    highlights = generate_highlights(selected_by_cat)
    telegram_msg = (
        f"Myrmidon — {TODAY}\n"
        f"총 {total_selected}개 기사 선별\n\n"
        f"오늘의 하이라이트:\n{highlights}\n\n"
        f"풀 리포트: {report_url}"
    )
    send_message(chat_id, telegram_msg)
