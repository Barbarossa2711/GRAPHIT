from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_PDF_DIR = _HERE.parents[1] / "BDT_Folien"
_EXISTING_JSON = _HERE / "BDT26-merged-chunks.json"
_OUT_JSON = _HERE / "BDT26-chunks-from-pdf.json"
_GRAPHIC_AREA_THRESHOLD = 0.12
_ANNOTATE_DPI = 120


def _norm_title(title: str) -> str:
    """
    Normalises a title for comparison: whitespace, lower case, trailing "(1)" or "(Beispiel)" removed.

    :param title: Title to normalise.
    :return: Normalised title used for comparison.
    """
    s = re.sub(r"\s+", " ", (title or "").strip()).lower()
    s = re.sub(r"\s*\((\d+|beispiel)\)\s*$", "", s)
    return s


def _titles_match(a: str, b: str) -> bool:
    """
    Compares two normalised titles exactly or by a clear common prefix.

    :param a: First title.
    :param b: Second title.
    :return: ``True`` if both titles are considered equal.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    return len(a) >= 8 and len(b) >= 8 and (a.startswith(b[:24]) or b.startswith(a[:24]))


def _source_stem(source: str) -> str:
    """
    Reduces a source path to its lower-case file stem.

    :param source: Source path of a slide.
    :return: The lowercased file stem used as the join key.
    """
    return Path(str(source).replace("\\", "/")).stem.lower()


def load_reuse_index(*existing_jsons: Path) -> dict[str, list[dict]]:
    """
    Builds a list of existing annotations per source stem, so reruns need no new vision calls.

    Several sources are merged; later ones extend earlier ones.

    :param existing_jsons: Slide JSON files to reuse annotations from; missing files are skipped.
    :return: Annotations with normalised title per source stem.
    """
    idx: dict[str, list[dict]] = defaultdict(list)
    for existing_json in existing_jsons:
        if not existing_json or not existing_json.is_file():
            continue
        data = json.loads(existing_json.read_text(encoding="utf-8"))
        for el in data:
            m = el.get("metadata", {})
            if m.get("type") != "slide" or not m.get("source"):
                continue
            idx[_source_stem(m["source"])].append(
                {
                    "ntitle": _norm_title(el.get("title") or m.get("title", "")),
                    "image_description": m.get("image_description"),
                    "section": m.get("section"),
                    "chapter": m.get("chapter"),
                    "slide_type": m.get("slide_type", ""),
                }
            )
    return idx


def find_reuse(reuse_idx: dict[str, list[dict]], stem: str, ntitle: str) -> dict | None:
    """
    Finds the existing annotation of a page by title match.

    :param reuse_idx: Index of existing annotations.
    :param stem: Source stem of the page.
    :param ntitle: Normalised title of the page.
    :return: The reusable annotation, or ``None`` if there is none.
    """
    for entry in reuse_idx.get(stem, []):
        if _titles_match(ntitle, entry["ntitle"]):
            return entry
    return None


_WATERMARK_RE = re.compile(r"erstellt mit (chat)?gpt", re.IGNORECASE)


def extract_page(page: pymupdf.Page) -> tuple[str, str, float]:
    """
    Extracts title, content and raster image area fraction of a PDF page.

    The title is the topmost text block that is not a watermark such as "Erstellt mit GPT".

    :param page: Page of the PDF to read.
    :return: ``(title, content, graphic_area_fraction)`` of that page.
    """
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    blocks.sort(key=lambda b: (round(b[1]), b[0]))
    title_blocks = [b for b in blocks if not _WATERMARK_RE.search(b[4])]
    title = title_blocks[0][4].strip() if title_blocks else ""
    rest = [b for b in blocks if b[4].strip() != title]
    content = "\n".join(b[4].strip() for b in rest) if rest else ""

    parea = page.rect.width * page.rect.height or 1.0
    iarea = sum(
        (info["bbox"][2] - info["bbox"][0]) * (info["bbox"][3] - info["bbox"][1])
        for info in page.get_image_info()
    )
    return title, content, iarea / parea


def render_page_png(page: pymupdf.Page, dpi: int = _ANNOTATE_DPI) -> bytes:
    """
    Renders a PDF page to PNG for the image annotation.

    :param page: Page to render.
    :param dpi: Resolution of the rendering.
    :return: The page as PNG bytes.
    """
    return page.get_pixmap(dpi=dpi).tobytes("png")


_ANNOTATE_PROMPT = (
    "Dies ist eine Folie aus einer Universitätsvorlesung zu Big-Data-Technologien. "
    "Beschreibe eine enthaltene Grafik/Abbildung (Diagramm, Schaubild, Architektur, Beziehungen) "
    "sachlich, strukturiert und auf Deutsch, sodass ihr Informationsgehalt auch ohne das Bild "
    "verständlich ist. Beschreibe nur die bildliche Darstellung, nicht den reinen Fließtext. "
    "Enthält die Folie KEINE nennenswerte Grafik (nur Titel und Textaufzählungen), antworte "
    "ausschließlich mit dem Wort: KEINE. "
    "Andernfalls gib ausschließlich die Bildbeschreibung zurück."
)
_NO_GRAPHIC_SENTINEL = "KEINE"


def annotate_image(client, png_bytes: bytes, model: str = "gpt-4o") -> str:
    """
    Has the vision model describe a slide image.

    :param client: OpenAI-compatible client.
    :param png_bytes: The rendered page.
    :param model: Vision model to use.
    :return: The description text.
    """
    b64 = base64.b64encode(png_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _ANNOTATE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def build(dry_run: bool, annotate: bool, fresh: list[str] | None = None) -> None:
    """
    Builds the slide content JSON from the PDFs in ``BDT_Folien/``, one entry per PDF page.

    ``slide_number`` equals the PDF page and thus the graph ``pageNumber``, so no alignment
    is needed. Existing image descriptions are reused by title match; only slides with a
    graphic (raster area of at least ``_GRAPHIC_AREA_THRESHOLD``) and without reusable
    description are sent to the vision model. For ``fresh`` sources nothing is reused and
    every non-question slide is annotated, so vector diagrams missed by the raster
    heuristic are covered; the model answers ``KEINE`` if there is no graphic.

    :param dry_run: ``True`` reports only and writes nothing.
    :param annotate: ``True`` describes graphic-heavy pages via the vision model.
    :param fresh: Source name substrings whose annotations are regenerated instead of reused.
    """
    fresh = [f.lower() for f in (fresh or [])]
    reuse_idx = load_reuse_index(_EXISTING_JSON, _OUT_JSON)
    pdfs = sorted(_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {_PDF_DIR}")

    client = None
    if annotate and not dry_run:
        import dotenv
        from openai import OpenAI
        dotenv.load_dotenv(_HERE.parents[0] / ".env")
        client = OpenAI()

    out: list[dict] = []
    slide_counter = 0
    stats = defaultdict(lambda: {"pages": 0, "questions": 0, "reused": 0, "new_annot": 0, "no_graphic": 0})

    for pdf in pdfs:
        stem = pdf.stem.lower()
        chapter_num = int(re.match(r"(\d+)", pdf.name).group(1)) if re.match(r"(\d+)", pdf.name) else None
        doc = pymupdf.open(str(pdf))
        for pno in range(doc.page_count):
            page = doc[pno]
            slide_number = pno + 1
            title, content, garea = extract_page(page)
            ntitle = _norm_title(title)
            is_question = ntitle.startswith("wiederholungsfrage")
            has_graphic = garea >= _GRAPHIC_AREA_THRESHOLD
            force_fresh = any(f in stem for f in fresh)
            reuse = None if force_fresh else find_reuse(reuse_idx, stem, ntitle)
            should_annotate = (not is_question) and (has_graphic or force_fresh)

            image_description = None
            if reuse and reuse.get("image_description"):
                image_description = reuse["image_description"]
                stats[stem]["reused"] += 1
            elif should_annotate:
                stats[stem]["new_annot"] += 1
                if client is not None:
                    try:
                        desc = annotate_image(client, render_page_png(page))
                        if desc and desc.strip().rstrip(".").upper() != _NO_GRAPHIC_SENTINEL:
                            image_description = desc
                    except Exception as exc:
                        logger.warning("Annotation failed %s p.%s: %s", pdf.name, slide_number, exc)
            else:
                stats[stem]["no_graphic"] += 1

            stats[stem]["pages"] += 1
            if is_question:
                stats[stem]["questions"] += 1

            slide_counter += 1
            slide_id = f"slide_{slide_counter:04d}"
            meta = {
                "type": "question" if is_question else "slide",
                "slide_type": (reuse or {}).get("slide_type", ""),
                "slide_number": slide_number,
                "chapter": (reuse or {}).get("chapter"),
                "chapter_num": chapter_num,
                "section": (reuse or {}).get("section"),
                "image_path": f"..\\data\\BDT26-slides\\kap{chapter_num}\\images\\Folie{slide_number}.PNG",
                "skip_annotation": image_description is None,
                "source": pdf.name,
                "image_description": image_description,
            }
            out.append({"metadata": meta, "title": title, "content": content, "slide_id": slide_id})
        doc.close()

    print("\n=== Build statistics per source ===")
    tot = defaultdict(int)
    for stem in sorted(stats):
        s = stats[stem]
        for k, v in s.items():
            tot[k] += v
        print(f"  {stem}: pages={s['pages']} | questions={s['questions']} | "
              f"annotation reused={s['reused']} | new needed={s['new_annot']} | "
              f"no graphic={s['no_graphic']}")
    print(f"\nTOTAL: pages={tot['pages']} | questions={tot['questions']} | "
          f"reused={tot['reused']} | new vision calls={tot['new_annot']}")

    if dry_run:
        print("\n(dry-run: no vision calls, nothing written)")
        return
    if annotate and client is None:
        print("\n(no OpenAI client, new annotations skipped)")

    _OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten: {_OUT_JSON}  ({len(out)} entries)")


def main() -> None:
    """
    Builds the slide content JSON from the lecture PDFs.

    Usage: ``python -m Multiagent.BDT2026_data.build_chunks [--dry-run] [--no-annotate] [--fresh 07]``.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Builds the slide chunk JSON from the BDT_Folien PDFs.")
    parser.add_argument("--dry-run", action="store_true", help="Only analysis/cost estimate.")
    parser.add_argument("--no-annotate", action="store_true", help="Do not create new vision annotations.")
    parser.add_argument("--fresh", default="", help="Comma-separated source substrings (e.g. '07') "
                        "whose annotations are regenerated instead of reused.")
    args = parser.parse_args()

    fresh = [f.strip() for f in args.fresh.split(",") if f.strip()]
    build(dry_run=args.dry_run, annotate=not args.no_annotate, fresh=fresh)


if __name__ == "__main__":
    main()
