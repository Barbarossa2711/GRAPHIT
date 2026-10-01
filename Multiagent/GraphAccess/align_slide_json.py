from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

from .config import load_config
from .slide_index import _source_stem

logger = logging.getLogger(__name__)


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
    Compares two normalised titles fuzzily, since the graph title is often only the first line of the multi-line JSON title.

    :param a: First title.
    :param b: Second title.
    :return: ``True`` if both titles match exactly or by a clear prefix.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    return len(a) >= 8 and len(b) >= 8 and (a.startswith(b[:24]) or b.startswith(a[:24]))


def _lcs_pairs(graph_titles: list[str], json_titles: list[str]) -> list[tuple[int, int]]:
    """
    Aligns two title sequences by longest common subsequence, which is robust against inserted or deleted slides.

    :param graph_titles: Slide titles taken from the graph.
    :param json_titles: Slide titles taken from the JSON.
    :return: Aligned index pairs ``(graph_idx, json_idx)``.
    """
    n, m = len(graph_titles), len(json_titles)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if _titles_match(graph_titles[i], json_titles[j]):
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if _titles_match(graph_titles[i], json_titles[j]):
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def fetch_graph_slides(config) -> dict[str, list[dict]]:
    """
    Fetches all :Slide nodes, grouped by source stem and sorted by page number.

    :param config: Neo4j connection settings.
    :return: All :Slide nodes grouped by source stem and sorted by page.
    """
    driver = GraphDatabase.driver(config.uri, auth=config.auth)
    try:
        with driver.session(database=config.database) as session:
            rows = session.run(
                "MATCH (sl:Slide) "
                "RETURN sl.source AS source, sl.pageNumber AS page, sl.title AS title "
                "ORDER BY sl.source, sl.pageNumber"
            ).data()
    finally:
        driver.close()

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["source"] is None or r["page"] is None:
            continue
        by_source[_source_stem(r["source"])].append(
            {"page": int(r["page"]), "title": r["title"] or ""}
        )
    return by_source


def align(json_path: Path, config) -> tuple[list, dict]:
    """
    Rewrites the slide JSON so that every slide chunk carries the page number of its matching graph :Slide.

    The graph is the source of truth. Unmatched chunks get ``slide_number = None`` and are
    skipped by :class:`SlideIndex`; the original number is kept as ``original_slide_number``.

    :param json_path: Slide JSON to align.
    :param config: Neo4j connection settings.
    :return: The pair ``(aligned data, statistics)``.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    graph_by_source = fetch_graph_slides(config)

    json_by_source: dict[str, list[dict]] = defaultdict(list)
    for element in data:
        meta = element.get("metadata", {})
        if meta.get("type") != "slide":
            continue
        source = meta.get("source")
        if not source:
            continue
        json_by_source[_source_stem(source)].append(element)

    stats = {"aligned": 0, "unmatched_json": 0, "sources": {}}

    for stem, json_elems in json_by_source.items():
        graph_slides = graph_by_source.get(stem, [])
        g_titles = [_norm_title(g["title"]) for g in graph_slides]
        j_titles = [_norm_title(e.get("title") or e["metadata"].get("title", "")) for e in json_elems]

        pairs = _lcs_pairs(g_titles, j_titles)
        matched_json_idx = set()

        for gi, ji in pairs:
            meta = json_elems[ji]["metadata"]
            if "original_slide_number" not in meta:
                meta["original_slide_number"] = meta.get("slide_number")
            new_page = graph_slides[gi]["page"]
            meta["slide_number"] = new_page
            json_elems[ji]["slide_number"] = new_page
            matched_json_idx.add(ji)

        unmatched = 0
        for ji, elem in enumerate(json_elems):
            if ji in matched_json_idx:
                continue
            meta = elem["metadata"]
            if "original_slide_number" not in meta:
                meta["original_slide_number"] = meta.get("slide_number")
            meta["slide_number"] = None
            unmatched += 1

        stats["aligned"] += len(pairs)
        stats["unmatched_json"] += unmatched
        stats["sources"][stem] = {
            "graph_slides": len(graph_slides),
            "json_slides": len(json_elems),
            "aligned": len(pairs),
            "unmatched_json": unmatched,
        }

    return data, stats


def main() -> None:
    """
    Aligns the slide JSON with the graph and writes it back, in place with a backup by default.

    Usage: ``python -m Multiagent.GraphAccess.align_slide_json [--json IN] [--out OUT] [--dry-run]``.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="JSON an Neo4j-:Slide-Referenzen ausrichten.")
    parser.add_argument("--json", default=None, help="Path to the input JSON (default: from .env/config).")
    parser.add_argument("--out", default=None, help="Output path (default: input in place, with .backup).")
    parser.add_argument("--dry-run", action="store_true", help="Only statistics, write nothing.")
    args = parser.parse_args()

    config = load_config()
    json_path = Path(args.json) if args.json else config.slide_content_json

    new_data, stats = align(json_path, config)

    print("\n=== Alignment statistics (JSON chunk -> graph pageNumber) ===")
    for stem, s in sorted(stats["sources"].items()):
        cov = 100 * s["aligned"] // max(s["graph_slides"], 1)
        print(f"  {stem}: graph={s['graph_slides']} json={s['json_slides']} "
              f"| aligned={s['aligned']} ({cov}% of graph slides) "
              f"| JSON without graph counterpart={s['unmatched_json']}")
    print(f"\nTotal aligned: {stats['aligned']} | "
          f"JSON chunks without match (slide_number=null): {stats['unmatched_json']}")

    if args.dry_run:
        print("\n(dry-run: nothing written)")
        return

    out_path = Path(args.out) if args.out else json_path
    if out_path == json_path:
        backup = json_path.with_suffix(json_path.suffix + ".backup")
        if not backup.exists():
            shutil.copy2(json_path, backup)
            print(f"\nBackup created: {backup}")
    out_path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aligned JSON written: {out_path}")


if __name__ == "__main__":
    main()
