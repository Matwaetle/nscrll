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
GITHUB_REPOSITORY  = os.environ.get("GITHUB_REPOSITORY", "")  # Actions에서 자동 주입

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL  = "gemini-3.1-flash-lite-preview"  # 가장 저렴한 모델로 비용 제어
TODAY  = date.today().isoformat()         # "2026-05-18"

# GitHub Pages URL 자동 구성
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

def resolve_google_news_url(url: str) -> str:
    try:
        res = requests.get(url, allow_redirects=True, timeout=5)
        return res.url
    except Exception:
        return url

def build_google_news_url(keywords: str, lang: str, region: str) -> str:
    return f"https://news.google.com/rss/search?q={urllib.parse.quote(keywords)}&hl={lang}&gl={region}&ceid={region}:{lang}"


# ───────────────────────────────────────────
# 뉴스 수집 - 소스별 골고루
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
# 글로벌 랭킹 - 카테고리 경계 없이 동적 분배
# ───────────────────────────────────────────
def global_rank_and_select(articles_by_cat: dict) -> dict:
    """
    모든 카테고리 기사를 한 번에 LLM에 넘겨서
    개수도 LLM이 동적으로 결정. 고정 쿼터 없음.
    """
    flat = []
    for cat_name, articles in articles_by_cat.items():
        for a in articles:
            flat.append({**a, 'category': cat_name})

    if not flat:
        return articles_by_cat

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
- 중복되거나 뻔한 분석 기사는 과감히 제외.

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
    return result


# ───────────────────────────────────────────
# 요약
# ───────────────────────────────────────────
def summarize(article: dict) -> str:
    prompt = f"""
너는 실리콘밸리 딥테크 트렌드에 빠삭하고 까칠한 동료 엔지니어 'NoScroll'이야.
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
# 텔레그램 하이라이트 (링크 전송용 3줄)
# ───────────────────────────────────────────
def generate_highlights(selected_by_cat: dict) -> str:
    all_titles = []
    for cat, articles in selected_by_cat.items():
        for a in articles:
            all_titles.append(f"[{cat}] {a['title']}")

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
# HTML 리포트 생성
# ───────────────────────────────────────────
def generate_html(selected_by_cat: dict, summaries: dict) -> str:
    """selected_by_cat: {cat_name: [article, ...]}, summaries: {link: summary_text}"""

    # 목차 HTML
    toc_items = ""
    for cat, articles in selected_by_cat.items():
        if not articles:
            continue
        anchor = re.sub(r'[^\w]', '-', cat)
        toc_items += f'<li><a href="#{anchor}">{cat} ({len(articles)})</a></li>\n'

    # 카테고리별 기사 HTML
    sections_html = ""
    for cat, articles in selected_by_cat.items():
        if not articles:
            continue
        anchor   = re.sub(r'[^\w]', '-', cat)
        cards_html = ""
        for i, a in enumerate(articles, 1):
            title   = a.get('title', '제목 없음')
            link    = a.get('link', '#')
            summary = summaries.get(link, '').replace('\n', '<br>')
            domain  = link.split('/')[2] if '/' in link else link
            cards_html += f"""
            <article class="card">
              <div class="card-num">{i}</div>
              <div class="card-body">
                <h3><a href="{link}" target="_blank" rel="noopener">{title}</a></h3>
                <p class="summary">{summary}</p>
                <span class="source">{domain}</span>
              </div>
            </article>"""

        sections_html += f"""
        <section id="{anchor}">
          <h2>{cat}</h2>
          {cards_html}
        </section>"""

    total_count = sum(len(v) for v in selected_by_cat.values())

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NoScroll — {TODAY}</title>
  <style>
    :root {{
      --bg: #0f1117;
      --surface: #1a1d27;
      --border: #2a2d3a;
      --accent: #4f8ef7;
      --text: #e2e8f0;
      --muted: #8892a4;
      --green: #34d399;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      line-height: 1.6;
      padding: 0 1rem 4rem;
      max-width: 860px;
      margin: 0 auto;
    }}
    header {{
      padding: 2.5rem 0 1.5rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 2rem;
    }}
    header h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--accent);
      letter-spacing: -0.5px;
    }}
    header .meta {{
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 0.3rem;
    }}
    nav.toc {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.2rem 1.5rem;
      margin-bottom: 2.5rem;
    }}
    nav.toc h2 {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--muted);
      margin-bottom: 0.8rem;
    }}
    nav.toc ul {{
      list-style: none;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    nav.toc a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 0.9rem;
      background: rgba(79,142,247,0.1);
      padding: 0.25rem 0.75rem;
      border-radius: 20px;
      border: 1px solid rgba(79,142,247,0.2);
      transition: background 0.2s;
    }}
    nav.toc a:hover {{ background: rgba(79,142,247,0.25); }}
    section {{ margin-bottom: 3rem; }}
    section h2 {{
      font-size: 1.1rem;
      font-weight: 600;
      padding-bottom: 0.6rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.2rem;
    }}
    .card {{
      display: flex;
      gap: 1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.2rem;
      margin-bottom: 0.9rem;
      transition: border-color 0.2s;
    }}
    .card:hover {{ border-color: var(--accent); }}
    .card-num {{
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--muted);
      min-width: 1.5rem;
      padding-top: 2px;
    }}
    .card-body {{ flex: 1; }}
    .card h3 {{
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      line-height: 1.4;
    }}
    .card h3 a {{
      color: var(--text);
      text-decoration: none;
    }}
    .card h3 a:hover {{ color: var(--accent); }}
    .summary {{
      font-size: 0.875rem;
      color: var(--muted);
      line-height: 1.65;
      margin-bottom: 0.5rem;
    }}
    .source {{
      font-size: 0.75rem;
      color: var(--green);
      opacity: 0.8;
    }}
    footer {{
      color: var(--muted);
      font-size: 0.8rem;
      text-align: center;
      padding-top: 2rem;
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <header>
    <h1>📰 NoScroll Daily</h1>
    <div class="meta">{TODAY} &nbsp;·&nbsp; 총 {total_count}개 기사</div>
  </header>

  <nav class="toc">
    <h2>목차</h2>
    <ul>{toc_items}</ul>
  </nav>

  {sections_html}

  <footer>Generated by NoScroll · {TODAY}</footer>
</body>
</html>"""


# ───────────────────────────────────────────
# HTML 저장 (날짜당 1개, 덮어쓰기)
# ───────────────────────────────────────────
def save_html(html_content: str) -> Path:
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    # 오늘 리포트 저장 (덮어쓰기)
    report_path = docs_dir / f"{TODAY}.html"
    report_path.write_text(html_content, encoding="utf-8")

    # index.html 업데이트
    reports = sorted(docs_dir.glob("????-??-??.html"), reverse=True)
    links   = "\n".join(
        f'<li><a href="{r.name}">{r.stem}</a></li>' for r in reports
    )
    index_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>NoScroll Archive</title>
  <style>
    body {{ font-family: sans-serif; background: #0f1117; color: #e2e8f0;
           max-width: 400px; margin: 3rem auto; padding: 0 1rem; }}
    h1 {{ color: #4f8ef7; margin-bottom: 1.5rem; }}
    ul {{ list-style: none; padding: 0; }}
    li {{ margin-bottom: 0.6rem; }}
    a {{ color: #4f8ef7; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>📰 NoScroll Archive</h1>
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
    print("🚀 NoScroll 가동...")

    config    = load_config()
    interests = config["interests"]
    limit     = config.get("articles_per_interest", 15)
    lang      = config.get("language", "en")
    region    = config.get("region", "US")

    chat_id = get_chat_id()
    if not chat_id:
        print("❌ 텔레그램 봇에게 먼저 말 걸고 다시 실행해줘!")
        exit()

    # ── 1. 모든 카테고리에서 수집 ──────────────────
    seen_urls       = set()
    articles_by_cat = {}

    for interest in interests:
        name = interest["name"]
        src_count = len(interest.get("custom_rss", [])) + 1
        print(f"\n  📡 [{name}] 수집 중 ({src_count}개 소스)")
        articles = fetch_news(interest, lang, region, limit, seen_urls)
        articles_by_cat[name] = articles
        print(f"  ✅ {len(articles)}개 수집")

    # ── 2. 글로벌 랭킹 (LLM 1회) ──────────────────
    print(f"\n  🧠 전체 기사 글로벌 랭킹 중... (동적 선별)")
    selected_by_cat = global_rank_and_select(articles_by_cat)
    total_selected  = sum(len(v) for v in selected_by_cat.values())
    print(f"  🎯 총 {total_selected}개 선별 완료")
    for cat, arts in selected_by_cat.items():
        if arts:
            print(f"    └ {cat}: {len(arts)}개")

    # ── 3. 요약 생성 ───────────────────────────────
    print(f"\n  ✍️  요약 생성 중...")
    summaries = {}
    for cat, articles in selected_by_cat.items():
        for article in articles:
            summaries[article['link']] = summarize(article)
            time.sleep(1)

    # ── 4. HTML 리포트 생성 & 저장 ─────────────────
    print(f"\n  📄 HTML 리포트 생성 중...")
    html_content = generate_html(selected_by_cat, summaries)
    report_path  = save_html(html_content)
    report_url   = f"{PAGES_BASE}/{TODAY}.html" if PAGES_BASE else f"(로컬: {report_path})"
    print(f"  ✅ 리포트 저장 완료: {report_path}")

    # ── 5. 텔레그램 하이라이트 생성 ───────────────
    print(f"\n  📱 텔레그램 메시지 생성 중...")
    highlights = generate_highlights(selected_by_cat)

    telegram_msg = (
        f"📰 NoScroll — {TODAY}\n"
        f"총 {total_selected}개 기사 선별\n\n"
        f"오늘의 하이라이트:\n{highlights}\n\n"
        f"🔗 풀 리포트: {report_url}"
    )

    # ── 6. 전송 ───────────────────────────────────
    send_message(chat_id, telegram_msg)
