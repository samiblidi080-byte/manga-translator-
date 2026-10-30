
import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from redis import Redis
from rq import Queue

from config import settings
from pipeline import process_pdf

logging.basicConfig(level=logging.INFO)

bot = Bot(token=settings.BOT_TOKEN)
dp = Dispatcher()

redis_conn = Redis.from_url(
    settings.REDIS_URL,
    decode_responses=False
)

queue = Queue(
    "manga",
    connection=redis_conn,
    default_timeout=settings.JOB_TIMEOUT
)


@dp.message(F.document)
async def receive_pdf(message):

    document = message.document

    if not document:
        return

    filename = (document.file_name or "").lower()

    if not filename.endswith(".pdf"):
        return

    max_size = settings.MAX_FILE_MB * 1024 * 1024

    if document.file_size and document.file_size > max_size:
        return

    queue.enqueue(
        process_pdf,
        document.file_id,
        message.chat.id,
        document.file_name or "manga.pdf",
        job_timeout=settings.JOB_TIMEOUT,
        result_ttl=0,
        failure_ttl=86400
    )


async def main():

    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types()
    )


if __name__ == "__main__":
    asyncio.run(main())

