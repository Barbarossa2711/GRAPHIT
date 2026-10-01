from __future__ import annotations

import glob
import os
import re
import sys

_PATTERN = (
    r'\.u\s*=\s*\w+\s*=>\s*""\s*\+\s*\w+\s*\+\s*"\."\s*\+\s*\(?(\{[^}]*\})\)?',
    r'return\s*""\s*\+\s*chunkId\s*\+\s*"\."\s*\+\s*(\{[^}]*\})\[chunkId\]',
)
_ENTRY = r'"?([\w.-]+)"?\s*:\s*"([0-9a-f]+)"'


def main(static_dir: str) -> int:
    """
    Checks that every chunk referenced by the remoteEntry of a prebuilt JupyterLab extension exists.

    ``jupyter labextension list`` reports "enabled OK" even if chunk files are missing, which
    only shows up in the browser as 404. Supports the chunk table formats of rspack and
    webpack; an unrecognised table is an error, since the check would guarantee nothing.
    Usage: ``python3 check_bundle.py <static-dir>``.

    :param static_dir: The extension's ``static`` directory.
    :return: Exit code, 0 if the bundle is complete.
    """
    hits = glob.glob(os.path.join(static_dir, "remoteEntry.*.js"))
    if not hits:
        print(f"ERROR: no remoteEntry in {static_dir}", file=sys.stderr)
        return 1
    src = open(hits[0], encoding="utf-8").read()

    for pattern in _PATTERN:
        m = re.search(pattern, src, re.S)
        if m:
            break
    else:
        print(
            "ERROR: chunk table not found in remoteEntry, probably a new "
            "bundler or a changed output format. Adjust the pattern in this file.",
            file=sys.stderr,
        )
        return 1

    table = dict(re.findall(_ENTRY, m.group(1)))
    if not table:
        print("ERROR: chunk table found, but empty.", file=sys.stderr)
        return 1

    expected = {f"{k}.{v}.js" for k, v in table.items()}
    missing = sorted(f for f in expected if not os.path.isfile(os.path.join(static_dir, f)))
    if missing:
        print(
            "ERROR: incomplete wheel, these chunks are missing:\n  "
            + "\n  ".join(missing)
            + "\n\nUsually caused by a build over an output folder that was not emptied "
            "(persistent rspack cache). Delete hard in the frontend project and rebuild:\n"
            "  rm -rf graphit_jupyter/labextension lib tsconfig.tsbuildinfo "
            "node_modules/.cache dist\n"
            "  python -m build --wheel",
            file=sys.stderr,
        )
        return 1

    present = {f for f in os.listdir(static_dir) if f.endswith(".js")}
    orphaned = sorted(present - expected - {"style.js"} - set(os.path.basename(hits[0]).split()))
    orphaned = [f for f in orphaned if not f.startswith("remoteEntry.")]
    if orphaned:
        print(f"WARNING: unreferenced chunk files in the bundle: {orphaned}")

    print(f"OK: remoteEntry and {len(table)} chunks consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
