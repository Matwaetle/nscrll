import os
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

def fetch_news(keywords: str, lang: str, region: str, limit: int) -> list:
    url = build_google_news_url(keywords, lang, region)
    feed = feedparser.parse(url)
    return feed.entries[:limit]


# ───────────────────────────────────────────
# Gemini 요약
# ───────────────────────────────────────────
def summarize(article: dict, interest_name: str) -> str:
    prompt = f"""
너는 바쁜 엔지니어를 위한 'NoScroll AI 에이전트'야.
다음 영문 뉴스 제목과 요약을 읽고, [{interest_name}] 분야 관점에서 핵심만 한국어로 3줄 요약해.

[특별 지시사항]
- 새로운 모델 벤치마크, 하드웨어 동향, 보안 업데이트가 있으면 눈에 띄게 강조해.
- 서론 없이 바로 요약 내용만 출력해.

[뉴스 데이터]
제목: {article.get('title', '')}
내용: {article.get('summary', '')}
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
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

    for interest in interests:
        name     = interest["name"]
        keywords = interest["keywords"]
        emoji    = interest.get("emoji", "📰")

        print(f"  📡 [{name}] 뉴스 수집 중...")
        articles = fetch_news(keywords, lang, region, limit)

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
            time.sleep(1)  # API 호출 간격

        final_message += "\n"

    # 4. 전송
    send_message(chat_id, final_message)
