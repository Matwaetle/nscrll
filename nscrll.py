import os
import re
import time
import urllib.parse
import feedparser
import requests
import yaml
from google import genai

# ───────────────────────────────────────────
# 환경 변수
# ───────────────────────────────────────────
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
client = genai.Client(api_key=GEMINI_API_KEY)


# ───────────────────────────────────────────
# 설정 불러오기
# ───────────────────────────────────────────
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ───────────────────────────────────────────
# Google News RSS (API 키 불필요)
# ───────────────────────────────────────────
def build_google_news_url(keywords: str, lang: str, region: str) -> str:
    encoded = urllib.parse.quote(keywords)
    return f"https://news.google.com/rss/search?q={encoded}&hl={lang}&gl={region}&ceid={region}:{lang}"

def strip_html(text: str) -> str:
    """HTML 태그 제거"""
    return re.sub(r'<[^>]+>', '', text).strip()

def resolve_google_news_url(google_url: str) -> str:
    """Google News 리다이렉트 URL → 실제 기사 URL 추출"""
    try:
        res = requests.get(google_url, allow_redirects=True, timeout=5)
        return res.url
    except Exception:
        return google_url  # 실패하면 원본 반환

# ───────────────────────────────────────────
# 글로벌 뉴스 수집 (Custom RSS + Google News US)
# ───────────────────────────────────────────
def fetch_news(interest: dict, lang: str, region: str, limit: int, seen_urls: set) -> list:
    results = []
    
    # 1. 해외 찐 개발자 커뮤니티(Hacker News, Reddit 등) 우선 수집
    for rss_url in interest.get("custom_rss", []):
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                if len(results) >= limit: break
                link = entry.get('link', '')
                if link in seen_urls: continue
                seen_urls.add(link)
                
                results.append({
                    'title': strip_html(entry.get('title', '')),
                    'summary': strip_html(entry.get('description', entry.get('summary', ''))),
                    'link': link
                })
        except Exception as e:
            print(f"⚠️ RSS 파싱 에러 ({rss_url}): {e}")

    # 2. 개수가 부족하면 구글 뉴스(영문)에서 최신 기사 긁어오기
    if len(results) < limit:
        url = build_google_news_url(interest["keywords"], lang, region)
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if len(results) >= limit: break
            link = resolve_google_news_url(entry.get('link', ''))
            if link in seen_urls: continue
            seen_urls.add(link)

            results.append({
                'title': strip_html(entry.get('title', '')),
                'summary': strip_html(entry.get('summary', '')),
                'link': link
            })

    return results

# ───────────────────────────────────────────
# Gemini 요약 (까칠한 시니어 개발자 톤앤매너)
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
            model='gemini-1.5-flash',
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
    # 텔레그램 메시지 최대 길이 4096자 제한 처리
    MAX_LEN = 4096
    chunks = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
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

    # 1. 설정 불러오기
    config = load_config()
    interests         = config["interests"]
    limit             = config.get("articles_per_interest", 2)
    lang              = config.get("language", "ko")
    region            = config.get("region", "KR")

    # 2. 텔레그램 Chat ID 확인
    chat_id = get_chat_id()
    if not chat_id:
        print("❌ 텔레그램 봇에게 먼저 말 걸고 다시 실행해줘!")
        exit()

    # 3. 분야별 뉴스 수집 & 요약
    final_message = "🤖 NoScroll 오늘의 뉴스 브리핑 🤖\n\n"
    seen_urls = set()  # 전체 분야 걸쳐 중복 URL 추적

    for interest in interests:
        name     = interest["name"]
        keywords = interest["keywords"]
        emoji    = interest.get("emoji", "📰")

        print(f"  📡 [{name}] 뉴스 수집 중...")
        articles = fetch_news(keywords, lang, region, limit, seen_urls)

        if not articles:
            print(f"  ⚠️  [{name}] 기사 없음, 스킵")
            continue

        final_message += f"{emoji} {name}\n"
        final_message += "━" * 20 + "\n"

        for idx, article in enumerate(articles, 1):
            summary = summarize(article, name)
            final_message += f"[{idx}] {article.get('title', '제목 없음')}\n"
            final_message += f"🔗 {article.get('link', '')}\n"
            final_message += f"✨ {summary}\n\n"
            time.sleep(3)  # API 호출 간격

        final_message += "\n"

    # 4. 전송
    send_message(chat_id, final_message)
