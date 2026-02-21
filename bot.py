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
import asyncio
import time
import tiktoken

# ──────────────────────── 环境变量配置 ────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
LLM_API_KEY    = os.environ["LLM_API_KEY"]
LLM_BASE_URL   = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "deepseek-chat")
BOT_PERSONALITY = os.environ.get("BOT_PERSONALITY", "你是一个聪明又有趣的助手，请说中文。")

# 🎛️ 可控参数
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "4000"))
LLM_TEMPERATURE    = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS     = int(os.environ.get("LLM_MAX_TOKENS", "500"))  # 从2000改为500
MAX_HISTORY_ROUNDS = int(os.environ.get("MAX_HISTORY_ROUNDS", "10"))  # 新增：最大历史轮数
CONTEXT_TIMEOUT    = int(os.environ.get("CONTEXT_TIMEOUT", "10"))  # 新增：上下文超时时间（分钟）

PORT               = int(os.environ.get("PORT", 10000))
WEBHOOK_PATH       = "/webhook"
RENDER_EXTERNAL_HOSTNAME = os.environ["RENDER_EXTERNAL_HOSTNAME"]
WEBHOOK_URL        = f"https://{RENDER_EXTERNAL_HOSTNAME}{WEBHOOK_PATH}"

# ──────────────────────── 初始化 ────────────────────────
# 修改 user_history 结构，包含消息历史和最后访问时间
user_history = defaultdict(lambda: {"history": [], "last_access": time.time()})
client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=60.0)
tg_app = None

# ──────────────────────── Token 估算函数 ────────────────────────
# 使用 tiktoken 进行更准确的 token 计算
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 可根据实际模型调整

def estimate_tokens(messages):
    """使用 tiktoken 准确计算 token 数量"""
    try:
        text = "".join([msg["content"] for msg in messages])
        return len(encoding.encode(text))
    except Exception:
        # 备用方案：字符数估算
        text = "".join([msg["content"] for msg in messages])
        return len(text) // 1.5

# ──────────────────────── 清理过期上下文 ────────────────────────
def cleanup_expired_context():
    """清理过期的用户上下文"""
    current_time = time.time()
    expired_users = []
    
    for user_id, user_data in user_history.items():
        # 检查时间超时
        if current_time - user_data["last_access"] > CONTEXT_TIMEOUT * 60:
            expired_users.append(user_id)
    
    # 清理过期用户
    for user_id in expired_users:
        del user_history[user_id]
        print(f"[清理] 用户 {user_id} 的上下文已过期并被清理")

# ──────────────────────── Handlers ────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("你好！我是由大模型驱动的机器人，随便聊～")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_history:
        user_history[user_id]["history"].clear()
        user_history[user_id]["last_access"] = time.time()
    await update.message.reply_text("✅ 已重置对话历史")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()
    if not text: return

    # 更新最后访问时间
    if user_id not in user_history:
        user_history[user_id] = {"history": [], "last_access": time.time()}
    else:
        user_history[user_id]["last_access"] = time.time()
    
    # 清理过期上下文
    cleanup_expired_context()
    
    history = user_history[user_id]["history"]
    history.append({"role": "user", "content": text})
    
    # 🔁 控制上下文长度和轮数
    while True:
        system_msg = {"role": "system", "content": BOT_PERSONALITY}
        full_context = [system_msg] + history
        tokens = estimate_tokens(full_context)
        
        # 检查 token 数量
        if tokens <= MAX_CONTEXT_TOKENS:
            messages = full_context
            break
        # 检查历史轮数
        elif len(history) > MAX_HISTORY_ROUNDS * 2:  # user + assistant 为一轮
            history = history[2:]  # 移除最前面的一轮对话
        # 剪裁历史记录
        elif len(history) > 1:
            if len(history) >= 2 and history[0]["role"] == "user" and history[1]["role"] == "assistant":
                history = history[2:]
            else:
                history = history[1:]
        else:
            # 最后手段：裁剪单条消息内容
            history[-1]["content"] = history[-1]["content"][-500:]  # 减少到500字符
            break

    # 🔄 根据 STREAM_SWITCH 决定调用方式
    if STREAM_SWITCH:
        await handle_stream_response(update, messages, history)
    else:
        await handle_normal_response(update, messages, history)

async def handle_normal_response(update, messages, history):
    """普通模式：等待完整回复后一次性发送"""
    reply = await ask_llm(messages, stream=False)
    if reply:
        history.append({"role": "assistant", "content": reply})
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i+4000], disable_web_page_preview=True)

async def handle_stream_response(update, messages, history):
    """流式模式：边接收边发送"""
    assistant_reply = ""
    message_obj = None
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True  # ✅ 开启流式传输
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assistant_reply += content
                
                # 发送流式内容（Telegram 消息不能太频繁）
                if not message_obj:
                    message_obj = await update.message.reply_text(content or "...")
                else:
                    # 编辑现有消息（注意频率限制）
                    try:
                        if len(assistant_reply) % 20 == 0 or len(assistant_reply) < 200:  # 控制更新频率
                            await message_obj.edit_text(assistant_reply[:4000] or "...", disable_web_page_preview=True)
                    except Exception:
                        pass  # 忽略编辑错误
        
        # 最终整理并保存历史
        if assistant_reply:
            history.append({"role": "assistant", "content": assistant_reply})
            try:
                await message_obj.edit_text(assistant_reply[:4000] or "...", disable_web_page_preview=True)
            except:
                pass
                
    except Exception as e:
        error_msg = f"❌ 流式传输出错: {str(e)[:200]}"
        if not message_obj:
            await update.message.reply_text(error_msg)
        else:
            try:
                await message_obj.edit_text(error_msg)
            except:
                await update.message.reply_text(error_msg)

# ──────────────────────── 调用 LLM ────────────────────────
async def ask_llm(messages, stream=False):
    """统一的 LLM 调用接口"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=stream
        )
        
        if stream:
            # 流式响应在外面处理
            return resp
        else:
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
    print(f"🔧 配置:")
    print(f"   - 最大上下文: {MAX_CONTEXT_TOKENS} tokens")
    print(f"   - 温度: {LLM_TEMPERATURE}")
    print(f"   - 最大输出: {LLM_MAX_TOKENS}")
    print(f"   - 最大历史轮数: {MAX_HISTORY_ROUNDS}")
    print(f"   - 上下文超时: {CONTEXT_TIMEOUT} 分钟")
    print(f"   - 流式传输: {'✅' if STREAM_SWITCH else '❌'}")

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