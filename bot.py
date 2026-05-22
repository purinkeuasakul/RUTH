"""
=============================================================
  Telegram Translation Bot — Russian <-> Thai
  Framework : aiogram 3.x (async)
  Translator: deep-translator (Google Translate, ฟรี)
=============================================================

วิธีติดตั้ง:
    pip install -r requirements.txt

วิธีรัน:
    python bot.py

การตั้งค่า:
    แก้ไขไฟล์ .env แล้วใส่ BOT_TOKEN ที่ได้จาก @BotFather
=============================================================
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from deep_translator import GoogleTranslator
from groq import Groq
from dotenv import load_dotenv
# ─────────────────────────────────────────────
#  ⚙️  CONFIG
# ─────────────────────────────────────────────

load_dotenv()  # โหลดค่าจากไฟล์ .env

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("❌ ไม่พบ BOT_TOKEN — กรุณาใส่ค่าในไฟล์ .env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ตั้งค่า Groq ถ้ามี API Key
if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE":
    _groq_client = Groq(api_key=GROQ_API_KEY)
    USE_AI = True
else:
    _groq_client = None
    USE_AI = False

# Thread pool สำหรับรัน sync translation โดยไม่บล็อก event loop
_executor = ThreadPoolExecutor(max_workers=4)

# ─────────────────────────────────────────────
#  📝  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),                          # แสดงใน terminal
        logging.FileHandler("bot.log", encoding="utf-8"), # บันทึกลงไฟล์
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  🔄  TRANSLATION ENGINE
# ─────────────────────────────────────────────

def _sync_translate(text: str, source: str, target: str) -> str:
    """แปลข้อความ — ใช้ Groq (llama) ถ้ามี API Key, ไม่งั้นใช้ Google Translate"""
    if USE_AI:
        lang_name = {"ru": "Russian", "th": "Thai"}
        src = lang_name.get(source, source)
        tgt = lang_name.get(target, target)
        response = _groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate from {src} to {tgt}. Return ONLY the translated text, nothing else.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    else:
        return GoogleTranslator(source=source, target=target).translate(text)


async def translate_async(text: str, source: str, target: str) -> str:
    """รัน sync translation ใน thread pool (non-blocking)"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _sync_translate, text, source, target)


def _is_cyrillic(text: str) -> bool:
    """
    ตรวจสอบว่าข้อความเป็น Cyrillic (รัสเซีย/สลาฟ) หรือไม่
    โดยนับสัดส่วนตัวอักษร Cyrillic ในข้อความ
    วิธีนี้แม่นยำกว่า langdetect มากสำหรับข้อความสั้น
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    cyrillic_count = sum(1 for c in letters if "\u0400" <= c <= "\u04FF")
    return (cyrillic_count / len(letters)) > 0.5


async def detect_and_translate(text: str) -> tuple[str, str, str]:
    """
    ตรวจจับภาษาอัตโนมัติ แล้วแปลตามกฎ:
      Cyrillic (รัสเซีย ฯลฯ) → ไทย  |  ภาษาอื่นทั้งหมด → รัสเซีย

    คืนค่า: (detected_lang, target_lang, translated_text)
    """
    if _is_cyrillic(text):
        source, target, detected = "ru", "th", "ru"
    else:
        source, target, detected = "auto", "ru", "th"

    translated = await translate_async(text, source, target)
    return detected, target, translated

# ─────────────────────────────────────────────
#  🤖  BOT & DISPATCHER
# ─────────────────────────────────────────────

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()

LANG_LABEL = {
    "ru":   "🇷🇺",
    "th":   "🇹🇭",
    "en":   "🇬🇧",
    "auto": "🌐",
}

# ─────────────────────────────────────────────
#  📨  HANDLERS
# ─────────────────────────────────────────────

@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 สวัสดี! ฉันคือบอตแปลภาษา\n\n"
        "🇷🇺 รัสเซีย ↔ 🇹🇭 ไทย\n\n"
        "แค่ส่งข้อความมาเลย ฉันจะตรวจจับภาษาและแปลให้อัตโนมัติ\n\n"
        "📌 กฎการแปล:\n"
        "  • ข้อความรัสเซีย → แปลเป็นไทย\n"
        "  • ข้อความไทย (หรือภาษาอื่น) → แปลเป็นรัสเซีย\n\n"
        "พิมพ์ /help เพื่อดูคำสั่งทั้งหมด"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "ℹ️ <b>วิธีใช้งาน</b>\n\n"
        "พิมพ์ข้อความที่ต้องการแปลแล้วส่งมาได้เลย\n\n"
        "<b>คำสั่ง:</b>\n"
        "  /start — เริ่มต้นใช้งาน\n"
        "  /help  — แสดงวิธีใช้"
    )


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    """รับข้อความ ตรวจจับภาษา และแปล (รองรับทั้ง DM และ Group)"""
    # ข้ามข้อความจากบอท (ป้องกันบอทแปลข้อความตัวเอง)
    if message.from_user.is_bot:
        return

    text = message.text.strip()
    if not text:
        return

    # ข้ามข้อความที่เป็น command (/start, /help ฯลฯ)
    if text.startswith("/"):
        return

    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name or "unknown"
    logger.info(f"[{user_id}/@{username}] {text[:80]}")

    await bot.send_chat_action(message.chat.id, "typing")

    try:
        detected_lang, target_lang, translated = await detect_and_translate(text)

        source_label = LANG_LABEL.get(detected_lang, f"🌐 {detected_lang}")
        target_label = LANG_LABEL.get(target_lang, f"🌐 {target_lang}")

        # ใน Group แสดงชื่อผู้ส่งด้วย เพื่อให้รู้ว่าใครพูด
        if message.chat.type in ("group", "supergroup"):
            header = f"👤 {message.from_user.first_name}  {source_label} → {target_label}"
        else:
            header = f"{source_label} → {target_label}"

        await message.reply(
            f"{header}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"{translated}"
        )

    except Exception as exc:
        logger.exception(f"Translation failed for user {user_id}: {exc}")
        await message.reply("❌ เกิดข้อผิดพลาดในการแปล กรุณาลองใหม่อีกครั้ง")


# ─────────────────────────────────────────────
#  🚀  MAIN — Graceful startup & shutdown
# ─────────────────────────────────────────────

async def main() -> None:
    engine = "Groq (llama-3.3-70b)" if USE_AI else "Google Translate (free)"
    logger.info(f"🤖 Bot is starting... Translation engine: {engine}")
    try:
        await dp.start_polling(
            bot,
            allowed_updates=["message"],  # รับทั้ง DM และ Group messages
        )
    finally:
        # ปิด thread pool และ bot session อย่างสะอาด
        _executor.shutdown(wait=False)
        await bot.session.close()
        logger.info("🛑 Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
