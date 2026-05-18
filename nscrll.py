import os
import feedparser
import requests
from google import genai

# 환경 변수에서 키 값을 불러오도록 수정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

client = genai.Client(api_key=GEMINI_API_KEY)



def get_telegram_chat_id():
    """봇에게 최근에 말을 건 사용자의 Chat ID를 자동으로 찾아옵니다."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        response = requests.get(url).json()
        if response.get("result"):
            # 가장 최근에 대화한 사람의 chat id 추출
            return response["result"][-1]["message"]["chat"]["id"]
    except Exception as e:
        print(f"❌ Chat ID 가져오기 실패: {e}")
    return None


def send_telegram_message(chat_id, text):
    """텔레그램 폰 앱으로 메시지를 전송합니다."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        # 마크다운 파싱 에러 방지를 위해 일단 주석 처리
        # "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)

    # 텔레그램 서버가 응답한 결과를 확인하는 디버깅 로직 추가
    if response.status_code == 200:
        print("✅ 폰으로 요약본 배달 완료! 텔레그램을 확인해 봐.")
    else:
        print(f"❌ 텔레그램 전송 실패! 원인: {response.json()}")


def get_latest_ai_news():
    RSS_URL = "https://techcrunch.com/category/artificial-intelligence/feed/"
    feed = feedparser.parse(RSS_URL)
    return feed.entries[:3]


def summarize_news_with_llm(news_item):
    prompt = f"""
    너는 바쁜 엔지니어를 위한 'NoScroll AI 에이전트'야.
    다음 제공되는 영문 뉴스 제목과 요약본을 읽고, 가장 중요한 핵심만 한국어로 3줄 요약해.

    [특별 지시사항]
    - 만약 이 기사가 새로운 LLM 모델의 벤치마크 점수, AI 하드웨어 시장 동향, 또는 보안 관련 업데이트를 다루고 있다면 그 부분을 눈에 띄게 강조해.
    - 불필요한 서론 없이 바로 요약 내용만 출력해.

    [뉴스 데이터]
    제목: {news_item['title']}
    내용: {news_item['summary']}
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro',
            contents=prompt,
        )
        return response.text.strip()
    except Exception:
        return "요약 실패"


if __name__ == "__main__":
    print("🚀 NoScroll 프로토타입 가동...")

    # 1. 텔레그램 Chat ID 확인
    chat_id = get_telegram_chat_id()
    if not chat_id:
        print("❌ 텔레그램 앱을 켜고, 네가 만든 봇에게 아무 대화나(예: '하이') 한 마디 걸고 다시 실행해 줘!")
        exit()

    # 2. 뉴스 수집 및 요약
    articles = get_latest_ai_news()

    # 폰으로 보낼 최종 메시지 조립
    final_message = "🤖 NoScroll 오늘의 AI 뉴스 요약 🤖\n\n"

    for idx, article in enumerate(articles, 1):
        summary = summarize_news_with_llm(article)

        final_message += f"[{idx}] {article['title']}\n"
        final_message += f"🔗 기사 원문 보기: {article['link']}\n"
        final_message += f"✨ {summary}\n"
        final_message += "—" * 15 + "\n\n"

    # 3. 폰으로 전송
    send_telegram_message(chat_id, final_message)
