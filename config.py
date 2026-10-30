
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    BOT_TOKEN: str = os.environ["BOT_TOKEN"]
    REDIS_URL: str = os.environ["REDIS_URL"]
    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
    GOOGLE_CREDENTIALS: str = os.environ["GOOGLE_CREDENTIALS"]

    OPENAI_MODEL: str = os.getenv(
        "OPENAI_MODEL", "gpt-4.1-mini"
    )

    DPI: int = int(os.getenv("DPI", "180"))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", "100"))
    MAX_FILE_MB: int = int(os.getenv("MAX_FILE_MB", "19"))
    JOB_TIMEOUT: int = int(os.getenv("JOB_TIMEOUT", "7200"))

    FONT_PATH: str = os.getenv(
        "FONT_PATH",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )

settings = Settings()

