

import logging

from redis import Redis
from rq import Queue, Worker

from config import settings

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    connection = Redis.from_url(
        settings.REDIS_URL
    )

    queues = [
        Queue("manga", connection=connection)
    ]

    worker = Worker(
        queues,
        connection=connection
    )

    worker.work()
