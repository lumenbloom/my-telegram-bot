# 文件名: bot.py
# Python 3.10+ 

import asyncio
import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─── 环境变量（在Render上设置，不要硬编码） ────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["8322901183:AAFQVJQXgGEMII6VnBFEewF_vtrOQSJ_gGg"]   # 从 @BotFather 拿
LLM_API_KEY    = os.environ["sk-b4ac8baabb0b4c8ea65c4702d6d8a3f1"]      # 大模型 key
LLM_BASE_URL   = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")  # 默认DeepSeek，改成你的
MODEL_NAME     = os.environ.get("MODEL_NAME", "deepseek-chat")                  # 默认模型

PORT = int(os.environ.get("PORT", 8080))  # Render 默认端口 10000，但我们用 8080
WEBHOOK_PATH = "/webhook"                 # Webhook 路径，随便定，但要匹配下面

# ─── LLM 调用函数（同之前） ───────────────────────────────────────────
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=90.0,
)

async def ask_llm(messages):
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            stream=False,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"抱歉，大模型出错了：{str(e)[:200]}"

# ─── 对话历史（简单内存版） ──────
from collections import defaultdict
user_history = defaultdict(list)

# ─── 命令处理器 ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("你好！我是由大模型驱动的机器人，随便聊～")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text:
        return

    history = user_history[user_id]
    history.append({"role": "user", "content": text})

    if len(history) > 24:
        history = history[-24:]

    messages = [
        {"role": "system", "content": "你是一个幽默、聪明、有点毒舌的助手。用中文回复。"},
    ] + history

    reply = await ask_llm(messages)

    history.append({"role": "assistant", "content": reply})

    if len(reply) > 4000:
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i : i + 4000])
    else:
        await update.message.reply_text(reply)

# ─── 主函数：Webhook 模式 ──────────────────────────────────────────
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # 启动 Webhook
    await app.initialize()
    webhook_url = f"https://{os.environ['RENDER_EXTERNAL_HOSTNAME']}{WEBHOOK_PATH}"  # Render 会自动提供域名
    await app.bot.set_webhook(webhook_url)  # 告诉 Telegram 你的 webhook url

    # 用内置 webserver 跑（Render 需要这个）
    from telegram.ext import Application
    import uvicorn

    async def webhook_handler(request):
        json_string = await request.body()
        update = Update.de_json(json_string.decode("utf-8"), app.bot)
        await app.process_update(update)
        return "OK", 200

    # 简单 ASGI app
    async def asgi_app(scope, receive, send):
        if scope["type"] == "http":
            if scope["path"] == WEBHOOK_PATH and scope["method"] == "POST":
                request_body = await receive()
                body = request_body.get("body", b"")
                response = await webhook_handler({"body": lambda: body})
                await send({"type": "http.response.start", "status": 200})
                await send({"type": "http.response.body", "body": b"OK"})
            else:
                await send({"type": "http.response.start", "status": 404})
                await send({"type": "http.response.body", "body": b"Not Found"})
        else:
            raise NotImplementedError()

    print(f"Bot 已启动 Webhook 模式 on port {PORT}...")
    uvicorn.run(asgi_app, host="0.0.0.0", port=PORT, log_level="info")

if __name__ == "__main__":
    asyncio.run(main())