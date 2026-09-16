import os
import sys
import time
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
from google.generativeai.types import RequestOptions

# ==============================================================================
# ЛОГИРОВАНИЕ И КОНФИГУРАЦИЯ СЕРВЕРА
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TradingAI")

# Инициализация API ключа Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API успешно конфигурирован.")
else:
    logger.warning("ВНИМАНИЕ: GEMINI_API_KEY не найден в переменных окружения!")

# ==============================================================================
# ДИНАМИЧЕСКАЯ ЗАГРУЗКА МАСТЕР-ПРОМПТА ИЗ ФАЙЛА
# ==============================================================================

DEFAULT_PROMPT_FILENAME = "master_prompt.txt"

def load_system_prompt() -> str:
    """
    Загружает системный промпт из файла master_prompt.txt или переменной окружения SYSTEM_PROMPT.
    Если файл не найден, используется базовый аварийный промпт.
    """
    # 1. Проверяем переменную окружения
    env_prompt = os.getenv("SYSTEM_PROMPT")
    if env_prompt and env_prompt.strip():
        logger.info("Системный промпт успешно загружен из переменной окружения SYSTEM_PROMPT.")
        return env_prompt.strip()

    # 2. Проверяем файл master_prompt.txt в директории
    prompt_file_path = os.path.join(os.path.dirname(__file__), DEFAULT_PROMPT_FILENAME)
    if os.path.exists(prompt_file_path):
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    logger.info(f"Системный промпт успешно загружен из файла {DEFAULT_PROMPT_FILENAME}.")
                    return content
        except Exception as e:
            logger.error(f"Ошибка при чтении {DEFAULT_PROMPT_FILENAME}: {str(e)}")

    # 3. Резервный фолбэк промпт
    logger.warning(f"Файл {DEFAULT_PROMPT_FILENAME} не найден. Используется резервный промпт.")
    return """
    Ты — профессиональный ИИ-аналитик и наставник по фьючерсной торговле криптовалютами.
    Проводи глубокий технический анализ (HTF/LTF, S/R, Order Blocks, FVG, Volume Profile, Wyckoff).
    Всегда давай четкий план сделки: Вход, Стоп-лосс, Тейк-профиты (1-3), Risk/Reward и условия отмены сценария.
    Не давай финансовых гарантий и всегда предупреждай о рисках торговли с плечом.
    """

# ==============================================================================
# PYDANTIC МОДЕЛИ ЗАПРОСОВ И ОТВЕТОВ
# ==============================================================================

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., example="BINANCE:BTCUSDT", description="Торговая пара или тикер")
    timeframe: str = Field("1h", example="1h", description="Таймфрейм для анализа (15m, 1h, 4h, 1d)")
    current_price: Optional[float] = Field(None, example=65000.50, description="Текущая цена инструмента")
    notes: Optional[str] = Field("", example="Найди FVG и ближайший Order Block", description="Дополнительный контекст от пользователя")
    mode: Optional[str] = Field("full", example="full", description="Режим ответа: full (полный) или short (краткий)")

    @validator('symbol')
    def sanitize_symbol(cls, v):
        v = v.strip().upper()
        if not v:
            raise ValueError("Символ пары не может быть пустым")
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
# ЖИЗНЕННЫЙ ЦИКЛ ПРИЛОЖЕНИЯ FASTAPI
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trading AI Analytics Server запускается...")
    yield
    logger.info("Trading AI Analytics Server останавливается...")

app = FastAPI(
    title="Trading AI Analytics Platform Engine",
    description="Backend на FastAPI для взаимодействия с Gemini API и отдачи веб-интерфейса",
    version="2.1.0",
    lifespan=lifespan
)

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ==============================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ОБРАЩЕНИЯ К GEMINI API
# ==============================================================================

def query_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY не установлен в Environment Variables на Render."
        )

    system_instruction = load_system_prompt()
    models_to_try = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
    last_error = None

    for model_name in models_to_try:
        try:
            logger.info(f"Запрос к Gemini через модель: {model_name}")
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            response = model.generate_content(
                prompt,
                request_options=RequestOptions(timeout=60.0)
            )
            if response and response.text:
                return response.text
        except Exception as e:
            logger.warning(f"Ошибка модели {model_name}: {str(e)}")
            last_error = e
            continue

    raise HTTPException(
        status_code=Status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Не удалось получить ответ от Gemini API: {str(last_error)}"
    )

# ==============================================================================
# ЭНДПОИНТЫ
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(
        content="""
        <html>
            <body style="font-family: sans-serif; background: #0b0e14; color: #e2e8f0; padding: 20px;">
                <h2>🚀 Trading AI Server is Running</h2>
                <p>Файл <code>index.html</code> не найден в корне проекта.</p>
            </body>
        </html>
        """,
        status_code=200
    )

@app.get("/api/health")
async def health_check():
    prompt_loaded = bool(load_system_prompt())
    return {
        "status": "healthy",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "prompt_file_active": prompt_loaded,
        "timestamp": time.time()
    }

@app.post("/api/analyze", response_model=StandardResponse)
async def analyze_market(data: AnalysisRequest):
    prompt = f"Проведи технический и стратегический анализ фьючерсной пары {data.symbol}."
    prompt += f"\n- Таймфрейм: {data.timeframe}"
    if data.current_price:
        prompt += f"\n- Текущая цена: ${data.current_price}"
    if data.notes:
        prompt += f"\n- Вопросы и пожелания пользователя: {data.notes}"

    if data.mode == "short":
        prompt += "\n\nПредоставь ответ строго по краткому шаблону."

    analysis_result = query_gemini(prompt)

    return StandardResponse(
        status="success",
        symbol=data.symbol,
        analysis=analysis_result,
        timestamp=time.time()
    )

@app.post("/api/evaluate-deal", response_model=StandardResponse)
async def evaluate_deal(data: DealEvaluationRequest):
    risk = abs(data.entry_price - data.stop_loss)
    reward = abs(data.take_profit - data.entry_price)
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    prompt = f"""
Оцени торговую сделку пользователя:
- Инструмент: {data.symbol} ({data.direction})
- Вход: {data.entry_price} | Стоп: {data.stop_loss} | Тейк: {data.take_profit} (R:R 1:{rr_ratio})
- Логика пользователя: {data.rationale}

Дай разбор: что сделано правильно, слабые места и конкретные рекомендации.
"""
    analysis_result = query_gemini(prompt)

    return StandardResponse(
        status="success",
        symbol=data.symbol,
        analysis=analysis_result,
        timestamp=time.time()
    )

@app.post("/api/explain-term", response_model=StandardResponse)
async def explain_term(data: TermExplanationRequest):
    prompt = f"Объясни термин '{data.term}': определение, аналогию из жизни, вид на графике и применение в торговле."
    analysis_result = query_gemini(prompt)

    return StandardResponse(
        status="success",
        symbol=data.term,
        analysis=analysis_result,
        timestamp=time.time()
    )

# ==============================================================================
# ОБРАБОТКА ИСКЛЮЧЕНИЙ
# ==============================================================================

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail, "timestamp": time.time()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Системная ошибка: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=Status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "detail": "Внутренняя ошибка сервера.", "timestamp": time.time()}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
