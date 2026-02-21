# bot.py
import os
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
import httpx

# ──────────────────────── 环境变量配置 ────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
LLM_API_KEY    = os.environ["LLM_API_KEY"]
LLM_BASE_URL   = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "deepseek-chat")
BOT_PERSONALITY = os.environ.get("BOT_PERSONALITY", "你是一个聪明又有趣的助手，请说中文。")

# 🎛️ 新增：基于 token 的上下文控制
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "8000"))  # 默认8000 tokens
LLM_TEMPERATURE    = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS     = int(os.environ.get("LLM_MAX_TOKENS", "2000"))

PORT               = int(os.environ.get("PORT", 10000))
WEBHOOK_PATH       = "/webhook"
RENDER_EXTERNAL_HOSTNAME = os.environ["RENDER_EXTERNAL_HOSTNAME"]
WEBHOOK_URL        = f"https://{RENDER_EXTERNAL_HOSTNAME}{WEBHOOK_PATH}"

# ──────────────────────── 初始化 ────────────────────────
user_history = defaultdict(list)
client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=60.0)
tg_app = None

# ──────────────────────── 简单的 token 估算函数 ────────────────────────
def estimate_tokens(messages):
    """估算消息列表的 token 数（简单按字符估算）"""
    text = "".join([msg["content"] for msg in messages])
    # 粗略估算：1个 token ≈ 4个中文字符 或 1个英文单词
    # 这里按 1 token = 1.5 字符 来估算，你可以根据模型调整
    return len(text) // 1.5

# ──────────────────────── Handlers ────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("你好！我是由大模型驱动的机器人，随便聊～")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_history[user_id].clear()
    await update.message.reply_text("✅ 已重置对话历史")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text: return

    history = user_history[user_id]
    history.append({"role": "user", "content": text})
    
    # 🔁 控制历史上下文 token 长度
    while True:
        system_msg = {"role": "system", "content": BOT_PERSONALITY}
        full_context = [system_msg] + history
        tokens = estimate_tokens(full_context)
        
        if tokens <= MAX_CONTEXT_TOKENS:
            messages = full_context
            break
        elif len(history) > 1:
            # 移除最早的一条用户+助手对话
            if len(history) >= 2 and history[0]["role"] == "user" and history[1]["role"] == "assistant":
                history = history[2:]
            else:
                history = history[1:]
        else:
            # 如果只剩下一条消息还超长，截断它
            history[-1]["content"] = history[-1]["content"][-1000:]  # 保留最后1000字符
            break

    reply = await ask_llm(messages)
    history.append({"role": "assistant", "content": reply})

    # 📝 分段发送长消息
    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i:i+4000])

# ──────────────────────── 调用 LLM ────────────────────────
async def ask_llm(messages):
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
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
    tg_app.add_handler(CommandHandler("reset", reset))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await tg_app.initialize()
    await tg_app.bot.set_webhook(WEBHOOK_URL)
    print(f"✅ Webhook set to: {WEBHOOK_URL}")
    print(f"🔧 配置: 最大上下文={MAX_CONTEXT_TOKENS} tokens, 温度={LLM_TEMPERATURE}, 最大输出={LLM_MAX_TOKENS}")

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