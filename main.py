import os
import sys
import time
import logging
import httpx
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from starlette import status as Status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("NezzxSignals")

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")  # можно менять через env

if ANTHROPIC_API_KEY:
    logger.info("Anthropic API key загружен успешно.")
else:
    logger.warning("ANTHROPIC_API_KEY не найден в переменных окружения!")

DEFAULT_PROMPT_FILENAME = "master_prompt.txt"

# ==============================================================================
# ЗАГРУЗКА СИСТЕМНОГО ПРОМПТА
# ==============================================================================

def load_system_prompt() -> str:
    """
    Приоритет:
    1. Переменная окружения SYSTEM_PROMPT
    2. Файл master_prompt.txt рядом с main.py
    3. Встроенный резервный промпт
    """
    env_prompt = os.getenv("SYSTEM_PROMPT", "").strip()
    if env_prompt:
        logger.info("Системный промпт загружен из env SYSTEM_PROMPT.")
        return env_prompt

    prompt_path = os.path.join(os.path.dirname(__file__), DEFAULT_PROMPT_FILENAME)
    if os.path.exists(prompt_path):
        try:
            content = open(prompt_path, encoding="utf-8").read().strip()
            if content:
                logger.info(f"Системный промпт загружен из {DEFAULT_PROMPT_FILENAME}.")
                return content
        except Exception as e:
            logger.error(f"Ошибка чтения {DEFAULT_PROMPT_FILENAME}: {e}")

    logger.warning("Используется встроенный резервный промпт.")
    return """Ты — профессиональный трейдинговый ИИ-аналитик NEZZX SIGNALS.
Специализируешься на криптовалютных сигналах и техническом анализе.

ВАЖНО: Отвечай ТОЛЬКО в формате JSON. Никакого текста до или после JSON.

Когда пользователь просит сигнал или анализ по коину:
{
  "type": "signal",
  "coin": "BTC",
  "pair": "BTC/USDT",
  "direction": "LONG",
  "timeframe": "4H",
  "entry": "67200",
  "stop_loss": "65800",
  "take_profit_1": "69500",
  "take_profit_2": "72000",
  "take_profit_3": "75000",
  "confidence": 78,
  "current_price": "67450",
  "rsi": "52",
  "trend": "бычий",
  "reasons": [
    "EMA 20 выше EMA 50 — бычье пересечение",
    "RSI 52 — нейтрально, без перекупленности",
    "Поддержка на $66 800 держит уровень",
    "Объём растёт на зелёных свечах",
    "Паттерн бычьего флага на 4H"
  ],
  "summary": "Краткое объяснение почему именно этот сигнал"
}

Если вопрос общий (не о сигнале):
{
  "type": "text",
  "message": "Твой ответ здесь по-русски"
}

ПРАВИЛА:
- direction только LONG или SHORT
- confidence от 60 до 92 (реалистично)
- Используй реалистичные цены для монеты
- reasons: 4-5 конкретных технических причин по-русски
- Все цифры без знака $ в полях entry/stop_loss/take_profit
- Отвечай по-русски
- Только чистый JSON, без обёртки"""

# ==============================================================================
# PYDANTIC МОДЕЛИ
# ==============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

    @validator("message")
    def strip_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Сообщение не может быть пустым")
        return v

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., example="BTCUSDT")
    timeframe: str = Field("4h", example="4h")
    current_price: Optional[float] = Field(None, example=65000.0)
    notes: Optional[str] = Field("", example="Найди FVG и ближайший Order Block")
    mode: Optional[str] = Field("full", example="full")

    @validator("symbol")
    def sanitize_symbol(cls, v):
        v = v.strip().upper()
        if not v:
            raise ValueError("Символ не может быть пустым")
        return v

class DealEvaluationRequest(BaseModel):
    symbol: str = Field(..., example="BTCUSDT")
    entry_price: float = Field(..., example=64200.0)
    stop_loss: float = Field(..., example=63500.0)
    take_profit: float = Field(..., example=66500.0)
    direction: str = Field(..., example="LONG")
    rationale: str = Field("", example="Вход от бычьего Order Block после CHoCH")

class TermExplanationRequest(BaseModel):
    term: str = Field(..., example="Fair Value Gap")

class StandardResponse(BaseModel):
    status: str
    symbol: Optional[str] = None
    analysis: str
    timestamp: float

# ==============================================================================
# ЖИЗНЕННЫЙ ЦИКЛ
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NEZZX Signals Server запускается...")
    logger.info(f"Модель Claude: {CLAUDE_MODEL}")
    yield
    logger.info("NEZZX Signals Server останавливается...")

app = FastAPI(
    title="NEZZX Signals — Trading AI Backend",
    description="FastAPI backend + Claude Anthropic API для криптосигналов",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.3f}s"
    return response

# ==============================================================================
# ЯДРО: ЗАПРОС К CLAUDE API
# ==============================================================================

async def query_claude(
    user_message: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 1500,
    temperature: float = 0.7
) -> str:
    """
    Асинхронный запрос к Anthropic Claude API через httpx.
    Возвращает текст ответа или бросает HTTPException.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ANTHROPIC_API_KEY не установлен в Environment Variables на Render."
        )

    system = system_prompt or load_system_prompt()

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            logger.info(f"Запрос к Claude ({CLAUDE_MODEL}), токены: {max_tokens}")
            resp = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)

            if resp.status_code != 200:
                err_body = resp.text
                logger.error(f"Claude API вернул {resp.status_code}: {err_body}")
                raise HTTPException(
                    status_code=Status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Claude API ошибка {resp.status_code}: {err_body[:200]}"
                )

            data = resp.json()
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            if not text:
                raise HTTPException(
                    status_code=Status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Claude вернул пустой ответ."
                )

            logger.info(f"Claude ответил успешно. Длина: {len(text)} символов.")
            return text

    except httpx.TimeoutException:
        logger.error("Таймаут запроса к Claude API (90s).")
        raise HTTPException(
            status_code=Status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Превышено время ожидания ответа от Claude API."
        )
    except httpx.RequestError as e:
        logger.error(f"Ошибка сети при запросе к Claude: {e}")
        raise HTTPException(
            status_code=Status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Сетевая ошибка: {str(e)}"
        )

# ==============================================================================
# ЭНДПОИНТЫ
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Отдаёт index.html если он есть рядом с main.py"""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return HTMLResponse(
            content=open(index_path, encoding="utf-8").read(),
            status_code=200
        )
    return HTMLResponse(
        content="""
        <html>
        <body style="font-family:monospace;background:#000;color:#e0e0e0;padding:30px">
          <h2 style="color:#ff0000">🔴 NEZZX SIGNALS — Server Running</h2>
          <p>Файл <code>index.html</code> не найден.</p>
          <p>Положи его рядом с <code>main.py</code>.</p>
          <p><a href="/docs" style="color:#ff6666">→ API Документация</a></p>
        </body>
        </html>
        """,
        status_code=200
    )

@app.get("/api/health")
async def health_check():
    """Проверка состояния сервера"""
    return {
        "status": "healthy",
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "claude_model": CLAUDE_MODEL,
        "prompt_loaded": bool(load_system_prompt()),
        "timestamp": time.time()
    }

@app.post("/api/chat")
async def chat_endpoint(data: ChatRequest):
    """
    Основной чат-эндпоинт для фронтенда.
    Возвращает сырой текст Claude (JSON или текст).
    """
    raw_response = await query_claude(
        user_message=data.message,
        max_tokens=1200,
        temperature=0.7
    )
    return {
        "status": "success",
        "response": raw_response,
        "timestamp": time.time()
    }

@app.post("/api/analyze", response_model=StandardResponse)
async def analyze_market(data: AnalysisRequest):
    """Детальный технический анализ по символу"""
    prompt = f"Проведи технический и стратегический анализ фьючерсной пары {data.symbol}."
    prompt += f"\n- Таймфрейм: {data.timeframe}"
    if data.current_price:
        prompt += f"\n- Текущая цена: ${data.current_price}"
    if data.notes:
        prompt += f"\n- Вопросы пользователя: {data.notes}"
    if data.mode == "short":
        prompt += "\n\nОтвечай кратко, по сигнальному шаблону."

    result = await query_claude(prompt, max_tokens=1500)
    return StandardResponse(
        status="success",
        symbol=data.symbol,
        analysis=result,
        timestamp=time.time()
    )

@app.post("/api/evaluate-deal", response_model=StandardResponse)
async def evaluate_deal(data: DealEvaluationRequest):
    """Оценка сделки пользователя"""
    risk   = abs(data.entry_price - data.stop_loss)
    reward = abs(data.take_profit - data.entry_price)
    rr     = round(reward / risk, 2) if risk > 0 else 0

    prompt = f"""Оцени торговую сделку:
- Инструмент: {data.symbol} ({data.direction})
- Вход: {data.entry_price} | Стоп: {data.stop_loss} | Тейк: {data.take_profit}
- Risk/Reward: 1:{rr}
- Логика трейдера: {data.rationale}

Дай разбор: что правильно, слабые места, конкретные рекомендации."""

    result = await query_claude(prompt, max_tokens=1000)
    return StandardResponse(
        status="success",
        symbol=data.symbol,
        analysis=result,
        timestamp=time.time()
    )

@app.post("/api/explain-term", response_model=StandardResponse)
async def explain_term(data: TermExplanationRequest):
    """Объяснение торгового термина"""
    prompt = (
        f"Объясни термин '{data.term}': "
        f"определение, аналогию из жизни, как выглядит на графике, "
        f"как применять в торговле."
    )
    result = await query_claude(prompt, max_tokens=800)
    return StandardResponse(
        status="success",
        symbol=data.term,
        analysis=result,
        timestamp=time.time()
    )

# ==============================================================================
# ОБРАБОТЧИКИ ОШИБОК
# ==============================================================================

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail, "timestamp": time.time()}
    )

@app.exception_handler(Exception)
async def global_exc_handler(request: Request, exc: Exception):
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Внутренняя ошибка сервера.", "timestamp": time.time()}
    )

# ==============================================================================
# ЗАПУСК
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
