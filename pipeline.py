
import os
import re
import json
import base64
import asyncio
import tempfile
import logging
import shutil

import cv2
import fitz
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

from aiogram import Bot
from google.cloud import vision
from google.oauth2 import service_account
from openai import OpenAI

from config import settings

log = logging.getLogger(__name__)

ai = OpenAI(api_key=settings.OPENAI_API_KEY)


def get_vision_client():
    info = json.loads(settings.GOOGLE_CREDENTIALS)
    credentials = service_account.Credentials.from_service_account_info(info)
    return vision.ImageAnnotatorClient(credentials=credentials)


def render_pdf(pdf_path, directory):

    document = fitz.open(pdf_path)

    if document.page_count > settings.MAX_PAGES:
        raise ValueError("PDF page limit exceeded")

    pages = []

    scale = settings.DPI / 72
    matrix = fitz.Matrix(scale, scale)

    for index, page in enumerate(document):

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        path = os.path.join(
            directory,
            f"page_{index:04}.png"
        )

        pix.save(path)
        pages.append(path)

    document.close()

    return pages


def extract_text(image_path):

    client = get_vision_client()

    with open(image_path, "rb") as file:
        content = file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(
        image=image
    )

    if response.error.message:
        raise RuntimeError(response.error.message)

    results = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:

                words = []
                points = []

                for word in paragraph.words:

                    word_text = "".join(
                        symbol.text
                        for symbol in word.symbols
                    )

                    words.append(word_text)

                    points.extend([
                        (vertex.x, vertex.y)
                        for vertex in word.bounding_box.vertices
                    ])

                if not words or not points:
                    continue

                x0 = min(p[0] for p in points)
                y0 = min(p[1] for p in points)
                x1 = max(p[0] for p in points)
                y1 = max(p[1] for p in points)

                text = " ".join(words).strip()

                if text:
                    results.append({
                        "text": text,
                        "box": [x0, y0, x1, y1]
                    })

    return results


def translate_text(items):

    if not items:
        return []

    payload = [
        {"id": i, "text": item["text"]}
        for i, item in enumerate(items)
    ]

    response = ai.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional manga localization translator. "
                    "Translate every source segment into natural Arabic. "
                    "Preserve character names, context, tone and dialogue. "
                    "Translate sound effects when appropriate. "
                    "Return all segments without omission."
                )
            },
            {
                "role": "user",
                "content": (
                    'Return JSON: {"translations":['
                    '{"id":0,"translation":"Arabic text"}]}\\n'
                    + json.dumps(payload, ensure_ascii=False)
                )
            }
        ]
    )

    data = json.loads(
        response.choices[0].message.content
    )

    mapping = {
        int(item["id"]): item["translation"]
        for item in data.get("translations", [])
    }

    return [
        mapping.get(i, items[i]["text"])
        for i in range(len(items))
    ]


def proofread(items, translations):

    if not items:
        return translations

    payload = [
        {
            "id": i,
            "source": items[i]["text"],
            "arabic": translations[i]
        }
        for i in range(len(items))
    ]

    response = ai.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior Arabic proofreader. "
                    "Check each translation against its original. "
                    "Correct spelling, grammar and mistranslations. "
                    "Keep the original meaning and all segment IDs."
                )
            },
            {
                "role": "user",
                "content": (
                    'Return JSON: {"translations":['
                    '{"id":0,"translation":"corrected Arabic"}]}\\n'
                    + json.dumps(payload, ensure_ascii=False)
                )
            }
        ]
    )

    data = json.loads(
        response.choices[0].message.content
    )

    mapping = {
        int(item["id"]): item["translation"]
        for item in data.get("translations", [])
    }

    return [
        mapping.get(i, translations[i])
        for i in range(len(items))
    ]


def wrap_arabic(text, font, max_width):

    words = text.split()
    lines = []
    current = ""

    for word in words:

        candidate = (
            current + " " + word
        ).strip()

        bbox = font.getbbox(candidate)

        width = bbox[2] - bbox[0]

        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return "\n".join(lines)


def clean_and_letter(image_path, items, translations, output_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError("Image loading failed")

    height, width = image.shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    boxes = []

    for item in items:

        x0, y0, x1, y1 = item["box"]

        padding = max(3, int((y1-y0) * 0.15))

        x0 = max(0, int(x0) - padding)
        y0 = max(0, int(y0) - padding)
        x1 = min(width, int(x1) + padding)
        y1 = min(height, int(y1) + padding)

        if x1 <= x0 or y1 <= y0:
            continue

        cv2.rectangle(
            mask,
            (x0, y0),
            (x1, y1),
            255,
            -1
        )

        boxes.append((x0, y0, x1, y1))

    cleaned = cv2.inpaint(
        image,
        mask,
        5,
        cv2.INPAINT_TELEA
    ) if boxes else image

    rgb = cv2.cvtColor(
        cleaned,
        cv2.COLOR_BGR2RGB
    )

    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)

    for box, original_translation in zip(boxes, translations):

        x0, y0, x1, y1 = box

        text = arabic_reshaper.reshape(
            str(original_translation)
        )

        text = get_display(text)

        max_width = max(20, x1 - x0)
        max_height = max(20, y1 - y0)

        font_size = min(38, max(10, int(max_height * 0.75)))

        while font_size >= 9:

            font = ImageFont.truetype(
                settings.FONT_PATH,
                font_size
            )

            wrapped = wrap_arabic(
                text,
                font,
                max_width
            )

            bbox = draw.multiline_textbbox(
                (0, 0),
                wrapped,
                font=font,
                align="center",
                spacing=1
            )

            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            if (
                text_width <= max_width
                and text_height <= max_height * 1.6
            ):
                break

            font_size -= 1

        if font_size < 9:
            log.warning("Text does not fit in its original box")
            continue

        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2

        draw.multiline_text(
            (center_x, center_y),
            wrapped,
            font=font,
            fill=(15, 15, 15),
            anchor="mm",
            align="center",
            spacing=1,
            stroke_width=1,
            stroke_fill=(255, 255, 255)
        )

    canvas.save(output_path, "PNG", optimize=True)


def create_pdf(images, output_path):

    output = fitz.open()

    for image_path in images:

        image = Image.open(image_path)
        width, height = image.size

        page_width = width * 72 / settings.DPI
        page_height = height * 72 / settings.DPI

        page = output.new_page(
            width=page_width,
            height=page_height
        )

        page.insert_image(
            page.rect,
            filename=image_path
        )

        image.close()

    output.save(
        output_path,
        garbage=4,
        deflate=True
    )

    output.close()


async def download_file(file_id, destination):

    bot = Bot(token=settings.BOT_TOKEN)

    try:
        telegram_file = await bot.get_file(file_id)

        await bot.download_file(
            telegram_file.file_path,
            destination=destination
        )

    finally:
        await bot.session.close()


async def send_result(chat_id, output_path, filename):

    bot = Bot(token=settings.BOT_TOKEN)

    try:
        with open(output_path, "rb") as file:

            await bot.send_document(
                chat_id=chat_id,
                document=file,
                filename=filename
            )

    finally:
        await bot.session.close()


def process_pdf(file_id, chat_id, original_name):

    directory = tempfile.mkdtemp(prefix="manga_")

    try:

        source = os.path.join(directory, "source.pdf")

        asyncio.run(
            download_file(file_id, source)
        )

        pages = render_pdf(source, directory)

        translated_pages = []

        for index, page_path in enumerate(pages):

            log.info("Processing page %s", index + 1)

            detected = extract_text(page_path)

            translations = translate_text(detected)

            translations = proofread(
                detected,
                translations
            )

            output = os.path.join(
                directory,
                f"arabic_{index:04}.png"
            )

            clean_and_letter(
                page_path,
                detected,
                translations,
                output
            )

            translated_pages.append(output)

        final_pdf = os.path.join(
            directory,
            "Arabic_Manga.pdf"
        )

        create_pdf(
            translated_pages,
            final_pdf
        )

        safe_name = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            os.path.splitext(original_name)[0]
        )[:50]

        asyncio.run(
            send_result(
                chat_id,
                final_pdf,
                f"{safe_name}_Arabic.pdf"
            )
        )

        log.info("Job completed successfully")

    except Exception:

        log.exception(
            "PDF processing failed for chat %s",
            chat_id
        )

        raise

    finally:

        shutil.rmtree(
            directory,
            ignore_errors=True
        )

