import os
import re
import json
import time
import urllib.parse
import feedparser
import requests
import yaml
from google import genai

# ───────────────────────────────────────────
# 환경 변수
# ───────────────────────────────────────────
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
client = genai.Client(api_key=GEMINI_API_KEY)


# ───────────────────────────────────────────
# 설정 불러오기
# ───────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────
def strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text).strip()

def resolve_google_news_url(google_url: str) -> str:
    try:
        res = requests.get(google_url, allow_redirects=True, timeout=5)
        return res.url
    except Exception:
        return google_url

def build_google_news_url(keywords: str, lang: str, region: str) -> str:
    encoded = urllib.parse.quote(keywords)
    return f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl={region}&ceid={region}:{lang}"


# ───────────────────────────────────────────
# 뉴스 수집 - 소스별 골고루
# ───────────────────────────────────────────
def fetch_from_rss(rss_url: str, per_source: int, seen_urls: set, resolve: bool = False) -> list:
    """단일 RSS URL에서 per_source개 수집"""
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
                'source':  rss_url
            })
    except Exception as e:
        print(f"    ⚠️ RSS 파싱 에러 ({rss_url}): {e}")
    return results

def fetch_news(interest: dict, lang: str, region: str, total_limit: int, seen_urls: set) -> list:
    """
    모든 소스에서 골고루 수집.
    소스 수에 따라 per_source를 동적으로 계산해서
    한 소스가 전체 쿼터를 독점하지 않도록 함.
    """
    custom_rss_list = interest.get("custom_rss", [])
    total_sources   = len(custom_rss_list) + 1  # +1은 Google News
    per_source      = max(2, total_limit // total_sources)

    all_results = []

    # 1. Custom RSS 소스별 골고루 수집
    for rss_url in custom_rss_list:
        fetched = fetch_from_rss(rss_url, per_source, seen_urls)
        all_results.extend(fetched)
        if fetched:
            print(f"    └ {len(fetched)}개 ← {rss_url.split('/')[2]}")  # 도메인만 출력

    # 2. Google News로 나머지 채우기
    remaining = total_limit - len(all_results)
    if remaining > 0:
        url  = build_google_news_url(interest["keywords"], lang, region)
        fetched = fetch_from_rss(url, remaining, seen_urls, resolve=True)
        all_results.extend(fetched)
        if fetched:
            print(f"    └ {len(fetched)}개 ← Google News")

    return all_results


# ───────────────────────────────────────────
# 1단계: LLM으로 중요도 순위 매겨 top_k 선별
# ───────────────────────────────────────────
def rank_and_filter(articles: list, interest_name: str, top_k: int) -> list:
    """제목 목록을 LLM에 한 번에 던져서 top_k개 인덱스를 받아옴 (1회 호출)"""
    if len(articles) <= top_k:
        return articles

    numbered = "\n".join(
        f"[{i}] {a['title']}" for i, a in enumerate(articles)
    )

    prompt = f"""
너는 {interest_name} 분야의 AI/테크 뉴스 큐레이터야.
아래 기사 제목 목록에서 오늘 가장 중요한 기사 {top_k}개를 골라 인덱스 번호만 JSON 배열로 출력해.
다른 말은 일절 하지 말고 오직 JSON만. 예시: [0, 3, 7, 12]

[선별 기준 - 우선순위 순]
1. 새로운 모델/제품 출시, 벤치마크 결과, 획기적 기술 발표
2. 업계 판도를 바꿀 인수합병, 대규모 투자, 핵심 인물 동향
3. 보안 취약점, 정책 변화, 규제 이슈
4. 일반 분석 기사, 인터뷰, 의견

[기사 목록]
{numbered}
"""
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
        )
        raw     = response.text.strip()
        raw     = re.sub(r'```[a-z]*', '', raw).strip().strip('`')
        indices = json.loads(raw)
        indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(articles)]
        return [articles[i] for i in indices[:top_k]]
    except Exception as e:
        print(f"  ⚠️ 랭킹 실패, 앞에서 {top_k}개로 대체: {e}")
        return articles[:top_k]


# ───────────────────────────────────────────
# 2단계: 선별된 기사 개별 요약
# ───────────────────────────────────────────
def summarize(article: dict, interest_name: str) -> str:
    prompt = f"""
너는 실리콘밸리 딥테크 트렌드에 빠삭하고 까칠한 동료 엔지니어 'NoScroll'이야.
다음 영문 뉴스 제목과 요약을 읽고, 개발자 입장에서 가장 솔깃할 만한 핵심 팩트만 한국어로 딱 3줄 요약해 줘.

[절대 지켜야 할 출력 규칙]
1. 말투: "~습니다", "~전망입니다" 같은 뻔한 뉴스 톤 절대 금지. "~했어.", "~상황이야.", "~라고 해." 등 간결하고 쿨한 평어(반말) 사용.
2. 내용: 불필요한 서론이나 배경 설명은 쳐내고, 새로운 벤치마크 점수, 하드웨어 아키텍처 변화, 뚫린 보안 취약점 등 가장 자극적이고 구체적인 기술 팩트만 남길 것.
3. 형식: 마크다운 기호(-, *, [1])나 이모지(✨, 🚀)를 일절 쓰지 말고, 순수 텍스트로만 3줄을 엔터 쳐서 출력할 것.

[뉴스 데이터]
제목: {article.get('title', '')}
내용: {article.get('summary', '')}
"""
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"요약 실패 (에러: {e})"


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
            print("✅ 전송 완료!")
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
    top_k     = config.get("top_k", 10)
    lang      = config.get("language", "en")
    region    = config.get("region", "US")

    chat_id = get_chat_id()
    if not chat_id:
        print("❌ 텔레그램 봇에게 먼저 말 걸고 다시 실행해줘!")
        exit()

    final_message = "🤖 NoScroll 오늘의 뉴스 브리핑 🤖\n\n"
    seen_urls     = set()

    for interest in interests:
        name  = interest["name"]
        emoji = interest.get("emoji", "📰")

        # 1. 소스별 골고루 수집
        src_count = len(interest.get("custom_rss", [])) + 1
        print(f"\n  📡 [{name}] 수집 시작 ({src_count}개 소스, 목표 {limit}개)")
        articles = fetch_news(interest, lang, region, limit, seen_urls)
        print(f"  ✅ 총 {len(articles)}개 수집 완료")

        if not articles:
            print(f"  ⚠️  [{name}] 기사 없음, 스킵")
            continue

        # 2. LLM으로 top_k 선별 (1회 호출)
        print(f"  🧠 [{name}] 중요도 랭킹 중...")
        selected = rank_and_filter(articles, name, top_k)
        print(f"  🎯 {len(selected)}개 선별 완료")

        # 3. 선별된 기사만 요약
        final_message += f"{emoji} {name}\n"
        final_message += "━" * 20 + "\n"

        for idx, article in enumerate(selected, 1):
            summary = summarize(article, name)
            final_message += f"[{idx}] {article.get('title', '제목 없음')}\n"
            final_message += f"🔗 {article.get('link', '')}\n"
            final_message += f"{summary}\n\n"
            time.sleep(3)

        final_message += "\n"

    send_message(chat_id, final_message)
