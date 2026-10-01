from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _source_stem(source: str) -> str:
    """
    Reduces a (possibly Windows) path to its lower-case file stem, so that joins ignore extension and directory.

    :param source: Source path of a slide, possibly in Windows notation.
    :return: The lowercased file stem used as the join key.
    """
    normalized = str(source).replace("\\", "/")
    return Path(normalized).stem.lower()


def _combine_text(element: dict) -> str:
    """
    Joins title, content and image description of a slide entry into one text.

    Title and content live on the top level (``metadata`` as fallback for old files), the
    image description only in ``metadata``.

    :param element: One page entry of the slide JSON.
    :return: Title, content and image description joined into one text.
    """
    meta = element.get("metadata", {})
    parts: list[str] = []
    title = element.get("title") or meta.get("title")
    content = element.get("content") or meta.get("content")
    for value in (title, content, meta.get("image_description")):
        if value and str(value).strip():
            parts.append(str(value).strip())
    return "\n".join(parts)


CHAPTER_PREFIX = re.compile(r"^kapitel\s*\d+\s*[:.\-–]\s*", re.IGNORECASE)


def _origin(meta: dict) -> dict:
    """
    Extracts the content-free origin data of a slide from its ``metadata``.

    The chapter name is collapsed to one line and stripped of an embedded "Kapitel 3:"
    prefix, which would otherwise be duplicated in the citation.

    :param meta: The ``metadata`` block of one slide entry.
    :return: Chapter name, chapter number and source file of that slide.
    """
    chapter = CHAPTER_PREFIX.sub("", " ".join(str(meta.get("chapter") or "").split()))
    number = meta.get("chapter_num")
    return {
        "chapter": chapter,
        "chapter_num": int(number) if isinstance(number, (int, float)) else None,
        "source": str(meta.get("source") or ""),
    }


class SlideIndex:
    """
    Lookup of slide texts and origin data by ``(file stem, page number)`` from the merged chunks JSON.

    Origin data (chapter, chapter number, source file) carries no slide content and is the
    only information the chat may cite below an answer.
    """

    def __init__(self, json_path: Path) -> None:
        """
        Loads the slide JSON into memory.

        :param json_path: Path of the merged chunks JSON.
        """
        self.json_path = Path(json_path)
        self._by_source_page: dict[tuple[str, int], str] = {}
        self._meta_by_source_page: dict[tuple[str, int], dict] = {}
        self._chapter_names: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        """
        Parses the JSON; origin data is stored independently of the text so that slides without text stay citable.

        Chapter names are collected per chapter number because only the title slide of a
        chapter carries the name.

        :raises FileNotFoundError: If the JSON does not exist.
        """
        if not self.json_path.is_file():
            raise FileNotFoundError(
                f"Slide content JSON not found: {self.json_path}. "
                "Please check SLIDE_CONTENT_JSON in the .env."
            )
        data = json.loads(self.json_path.read_text(encoding="utf-8"))

        n_slides = 0
        for element in data:
            meta = element.get("metadata", {})
            if meta.get("type") != "slide":
                continue
            source = meta.get("source")
            page = meta.get("slide_number")
            if not source or page is None:
                continue
            key = (_source_stem(source), int(page))
            origin = _origin(meta)
            self._meta_by_source_page[key] = origin
            number, name = origin["chapter_num"], origin["chapter"]
            if number is not None and name and number not in self._chapter_names:
                self._chapter_names[number] = name
            text = _combine_text(element)
            if text:
                self._by_source_page[key] = text
                n_slides += 1

        logger.info("SlideIndex loaded: %s slides from %s", n_slides, self.json_path.name)

    def get_text(self, source: str, page: int) -> str:
        """
        Returns the slide text for ``(source, page)``.

        :param source: Source file of the slide.
        :param page: Page number within that source.
        :return: The slide text, or an empty string if the page is unknown.
        """
        return self._by_source_page.get((_source_stem(source), int(page)), "")

    def get_meta(self, source: str, page: int) -> dict:
        """
        Returns the origin data for ``(source, page)``, never the slide content; empty fields if unknown.

        :param source: Source file of the slide.
        :param page: Page number within that source.
        :return: ``{chapter, chapter_num, source}`` for that slide.
        """
        key = (_source_stem(source), int(page))
        if key not in self._meta_by_source_page:
            return {"chapter": "", "chapter_num": None, "source": str(source)}
        meta = dict(self._meta_by_source_page[key])
        if not meta["chapter"] and meta["chapter_num"] is not None:
            meta["chapter"] = self._chapter_names.get(meta["chapter_num"], "")
        return meta
