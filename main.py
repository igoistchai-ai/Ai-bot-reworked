import os
import sys
import time
import logging
import re
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

API_KEY      = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://riskradarai.ru/v1")
AI_MODEL     = os.getenv("AI_MODEL", "claude-fable-5")

if API_KEY:
    logger.info(f"API Key загружен. Модель: {AI_MODEL}")
else:
    logger.warning("API_KEY не найден!")

DEFAULT_PROMPT_FILENAME = "master_prompt.txt"

# ==============================================================================
# СИСТЕМНЫЙ ПРОМПТ
# ==============================================================================

def load_system_prompt() -> str:
    env_prompt = os.getenv("SYSTEM_PROMPT", "").strip()
    if env_prompt:
        return env_prompt

    prompt_path = os.path.join(os.path.dirname(__file__), DEFAULT_PROMPT_FILENAME)
    if os.path.exists(prompt_path):
        try:
            content = open(prompt_path, encoding="utf-8").read().strip()
            if content:
                return content
        except Exception as e:
            logger.error(f"Ошибка чтения промпта: {e}")

    return """Ты — профессиональный трейдинговый ИИ-аналитик NEZZX SIGNALS.
Специализируешься на криптовалютных сигналах и техническом анализе.

Тебе передаются РЕАЛЬНЫЕ рыночные данные с Binance (цена, свечи 1D/4H/1H, RSI, объёмы, уровни).
Используй эти данные для точного анализа. НЕ проси пользователя предоставить данные — они уже есть.

ВАЖНО: Отвечай ТОЛЬКО в формате JSON. Никакого текста до или после. Никаких markdown блоков.

Когда пользователь просит сигнал или анализ:
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
  "summary": "Краткое объяснение сигнала"
}

Если вопрос общий:
{
  "type": "text",
  "message": "Твой ответ по-русски"
}

ПРАВИЛА:
- direction только LONG или SHORT
- confidence от 60 до 92
- Цены берёшь из реальных данных которые тебе переданы
- reasons: 4-5 технических причин основанных на реальных данных
- Все цифры без знака $ в полях entry/stop_loss/take_profit
- Только чистый JSON без обёртки
- Отвечай по-русски"""

# ==============================================================================
# BINANCE — ПОЛУЧЕНИЕ РЫНОЧНЫХ ДАННЫХ
# ==============================================================================

KNOWN_COINS = [
    "BTC","ETH","SOL","LTC","BNB","XRP","ADA","DOGE",
    "AVAX","DOT","LINK","MATIC","UNI","ATOM","FIL",
    "NEAR","APT","ARB","OP","INJ","SUI","TRX","TON",
    "PEPE","WIF","BONK","JUP","SEI","TIA","PYTH","JTO",
    "MANTA","ALT","PIXEL","PORTAL","DYM","STRK","ZETA"
]

def extract_symbol(text: str) -> Optional[str]:
    """Извлекает символ монеты из текста пользователя"""
    text_up = text.upper()
    for coin in KNOWN_COINS:
        pattern = r'\b' + coin + r'\b'
        if re.search(pattern, text_up):
            return coin
    return None

def calculate_rsi(closes: list, period: int = 14) -> float:
    """Считает RSI по списку цен закрытия"""
    if len(closes) < period + 1:
        return 50.0
    deltas    = [closes[i+1] - closes[i] for i in range(len(closes)-1)]
    gains     = [d for d in deltas if d > 0]
    losses    = [-d for d in deltas if d < 0]
    avg_gain  = sum(gains[-period:]) / period if gains else 0
    avg_loss  = sum(losses[-period:]) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calculate_ema(closes: list, period: int) -> float:
    """Считает EMA"""
    if len(closes) < period:
        return closes[-1] if closes else 0
    k   = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 4)

async def get_market_data(symbol: str) -> dict:
    """
    Получает реальные данные с Binance:
    цена, свечи 1D/4H/1H, RSI, EMA, объёмы, уровни
    """
    clean = symbol.upper().replace("/","").replace("-","")
    if not clean.endswith("USDT"):
        clean += "USDT"

    base = "https://api.binance.com/api/v3"

    async with httpx.AsyncClient(timeout=20.0) as client:
        # 24h тикер
        ticker_r = await client.get(
            f"{base}/ticker/24hr",
            params={"symbol": clean}
        )
        ticker_r.raise_for_status()
        ticker = ticker_r.json()

        # Свечи по таймфреймам
        candles = {}
        for tf in ["1d", "4h", "1h"]:
            r = await client.get(
                f"{base}/klines",
                params={"symbol": clean, "interval": tf, "limit": 50}
            )
            r.raise_for_status()
            raw = r.json()
            candles[tf] = [{
                "open":   float(c[1]),
                "high":   float(c[2]),
                "low":    float(c[3]),
                "close":  float(c[4]),
                "volume": float(c[5])
            } for c in raw]

    # RSI
    closes_1h = [c["close"] for c in candles["1h"]]
    closes_4h = [c["close"] for c in candles["4h"]]
    closes_1d = [c["close"] for c in candles["1d"]]
    rsi_1h    = calculate_rsi(closes_1h)
    rsi_4h    = calculate_rsi(closes_4h)

    # EMA
    ema20_1h = calculate_ema(closes_1h, 20)
    ema50_1h = calculate_ema(closes_1h, 50)
    ema20_4h = calculate_ema(closes_4h, 20)
    ema50_4h = calculate_ema(closes_4h, 50)

    # Уровни поддержки/сопротивления
    highs_1d = [c["high"] for c in candles["1d"][-20:]]
    lows_1d  = [c["low"]  for c in candles["1d"][-20:]]
    highs_4h = [c["high"] for c in candles["4h"][-20:]]
    lows_4h  = [c["low"]  for c in candles["4h"][-20:]]

    current_price = float(ticker["lastPrice"])

    return {
        "symbol":       clean,
        "price":        current_price,
        "change_24h":   float(ticker["priceChangePercent"]),
        "volume_24h":   float(ticker["quoteVolume"]),
        "high_24h":     float(ticker["highPrice"]),
        "low_24h":      float(ticker["lowPrice"]),
        "rsi_1h":       rsi_1h,
        "rsi_4h":       rsi_4h,
        "ema20_1h":     ema20_1h,
        "ema50_1h":     ema50_1h,
        "ema20_4h":     ema20_4h,
        "ema50_4h":     ema50_4h,
        "resistance_1d": round(max(highs_1d), 4),
        "support_1d":    round(min(lows_1d), 4),
        "resistance_4h": round(max(highs_4h), 4),
        "support_4h":    round(min(lows_4h), 4),
        "candles_1d":   candles["1d"][-10:],
        "candles_4h":   candles["4h"][-20:],
        "candles_1h":   candles["1h"][-24:],
    }

def format_market_context(md: dict) -> str:
    """Форматирует рыночные данные в текст для промпта"""
    candles_1d_str = "\n".join([
        f"  O:{c['open']} H:{c['high']} L:{c['low']} C:{c['close']} V:{c['volume']:.0f}"
        for c in md["candles_1d"]
    ])
    candles_4h_str = "\n".join([
        f"  O:{c['open']} H:{c['high']} L:{c['low']} C:{c['close']}"
        for c in md["candles_4h"]
    ])
    candles_1h_str = "\n".join([
        f"  O:{c['open']} H:{c['high']} L:{c['low']} C:{c['close']}"
        for c in md["candles_1h"]
    ])

    return f"""
=== РЕАЛЬНЫЕ РЫНОЧНЫЕ ДАННЫЕ ({md['symbol']}) ===
Текущая цена:      ${md['price']}
Изменение 24h:     {md['change_24h']}%
Объём 24h:         ${md['volume_24h']:,.0f}
Максимум 24h:      ${md['high_24h']}
Минимум 24h:       ${md['low_24h']}

--- ИНДИКАТОРЫ ---
RSI 1H:            {md['rsi_1h']}
RSI 4H:            {md['rsi_4h']}
EMA20 1H:          {md['ema20_1h']}
EMA50 1H:          {md['ema50_1h']}
EMA20 4H:          {md['ema20_4h']}
EMA50 4H:          {md['ema50_4h']}

--- УРОВНИ ---
Сопротивление 1D:  ${md['resistance_1d']}
Поддержка 1D:      ${md['support_1d']}
Сопротивление 4H:  ${md['resistance_4h']}
Поддержка 4H:      ${md['support_4h']}

--- СВЕЧИ 1D (последние 10) ---
{candles_1d_str}

--- СВЕЧИ 4H (последние 20) ---
{candles_4h_str}

--- СВЕЧИ 1H (последние 24) ---
{candles_1h_str}
=== КОНЕЦ ДАННЫХ ===

Проанализируй эти реальные данные и сформируй точный сигнал.
"""

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
    logger.info(f"Модель: {AI_MODEL} | Base: {API_BASE_URL}")
    yield
    logger.info("NEZZX Signals Server останавливается...")

app = FastAPI(
    title="NEZZX Signals",
    version="3.1.0",
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
async def process_time(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time()-start:.3f}s"
    return response

# ==============================================================================
# ЗАПРОС К AI
# ==============================================================================

async def query_ai(user_message: str, max_tokens: int = 1200) -> str:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API_KEY не установлен в Environment Variables."
        )

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model":      AI_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": load_system_prompt()},
            {"role": "user",   "content": user_message}
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            logger.info(f"Запрос к {AI_MODEL}")
            resp = await client.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            if resp.status_code != 200:
                logger.error(f"AI API {resp.status_code}: {resp.text}")
                raise HTTPException(
                    status_code=503,
                    detail=f"AI ошибка {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if not text:
                raise HTTPException(status_code=503, detail="AI вернул пустой ответ.")
            logger.info(f"Ответ получен, длина: {len(text)} символов")
            return text

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Таймаут запроса к AI.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Сетевая ошибка: {str(e)}")

# ==============================================================================
# ЭНДПОИНТЫ
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
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
          <h2 style="color:#ff0000">NEZZX SIGNALS — Running</h2>
          <p>index.html не найден.</p>
          <a href="/docs" style="color:#ff6666">API Docs</a>
        </body>
        </html>
        """,
        status_code=200
    )

@app.get("/api/health")
async def health_check():
    return {
        "status":      "healthy",
        "api_key_set": bool(API_KEY),
        "model":       AI_MODEL,
        "base_url":    API_BASE_URL,
        "timestamp":   time.time()
    }

@app.post("/api/chat")
async def chat_endpoint(data: ChatRequest):
    user_msg       = data.message
    market_context = ""
    found_symbol   = None

    # Ищем монету в запросе
    symbol = extract_symbol(user_msg)

    if symbol:
        try:
            logger.info(f"Получаем данные Binance для {symbol}...")
            md             = await get_market_data(symbol)
            market_context = format_market_context(md)
            found_symbol   = md["symbol"]
            logger.info(f"Данные получены: {found_symbol} @ ${md['price']}")
        except Exception as e:
            logger.warning(f"Не удалось получить данные для {symbol}: {e}")
            market_context = f"\n[Binance данные недоступны для {symbol}, делай анализ на основе общих знаний]\n"

    # Собираем финальный промпт
    full_prompt = f"{market_context}\nЗапрос пользователя: {user_msg}"

    raw_response = await query_ai(
        user_message=full_prompt,
        max_tokens=1200
    )

    return {
        "status":    "success",
        "response":  raw_response,
        "symbol":    found_symbol,
        "timestamp": time.time()
    }

# ==============================================================================
# ОБРАБОТЧИКИ ОШИБОК
# ==============================================================================

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status":"error","detail":exc.detail,"timestamp":time.time()}
    )

@app.exception_handler(Exception)
async def global_exc_handler(request: Request, exc: Exception):
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status":"error","detail":"Внутренняя ошибка сервера.","timestamp":time.time()}
    )

# ==============================================================================
# ЗАПУСК
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
