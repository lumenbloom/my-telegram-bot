# bot.py
import os
import asyncio
from collections import defaultdict

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from openai import AsyncOpenAI

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager

# ──────────────────────── 配置 ────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
LLM_API_KEY    = os.environ["LLM_API_KEY"]
LLM_BASE_URL   = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "deepseek-chat")
BOT_PERSONALITY = os.environ.get("BOT_PERSONALITY", "你是一个聪明又有趣的助手，请说中文。")

PORT           = int(os.environ.get("PORT", 10000))  # Render 要求 10000
WEBHOOK_PATH   = "/webhook"
RENDER_EXTERNAL_HOSTNAME = os.environ["RENDER_EXTERNAL_HOSTNAME"]
WEBHOOK_URL    = f"https://{RENDER_EXTERNAL_HOSTNAME}{WEBHOOK_PATH}"

# ──────────────────────── 初始化 ────────────────────────
user_history = defaultdict(list)
client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=60.0)

tg_app = None

# ──────────────────────── Handlers ────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("你好！我是由大模型驱动的机器人，随便聊～")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text: return

    history = user_history[user_id]
    history.append({"role": "user", "content": text})
    if len(history) > 24: history = history[-24:]

    messages = [{"role": "system", "content": BOT_PERSONALITY}] + history

    reply = await ask_llm(messages)

    history.append({"role": "assistant", "content": reply})

    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i:i+4000])

# ──────────────────────── 调用 LLM ────────────────────────
async def ask_llm(messages):
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            stream=False
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"抱歉，大模型出错了：{str(e)[:200]}"

# ──────────────────────── 初始化 Telegram Bot ────────────────────────
async def init():
    global tg_app
    tg_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await tg_app.initialize()
    await tg_app.bot.set_webhook(WEBHOOK_URL)
    print(f"✅ Webhook set to: {WEBHOOK_URL}")

# ──────────────────────── FastAPI APP Setup ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init()
    yield

app = FastAPI(lifespan=lifespan)

@app.post(WEBHOOK_PATH)
async def webhook(request: Request):
    json_data = await request.json()
    update = Update.de_json(json_data, tg_app.bot)
    await tg_app.process_update(update)
    return PlainTextResponse("OK")

# ──────────────────────── Main Entry Point ────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting bot on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)