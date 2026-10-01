# Build context is Docker/ (see docker-compose.yml), so the wheel in GRAPHIT_Frontend_extension/ is reachable.

FROM quay.io/jupyter/minimal-notebook:2026-08-17

# The wheel requires JupyterLab >= 4.6; jupyterhub must match the hub version exactly.
RUN pip install --no-cache-dir \
        "jupyterlab>=4.6,<5" \
        "jupyterhub==5.5.1"

COPY --chown=${NB_UID}:${NB_GID} GRAPHIT_Frontend_extension/graphit_jupyter-0.1.0-py3-none-any.whl /tmp/

RUN pip install --no-cache-dir /tmp/graphit_jupyter-0.1.0-py3-none-any.whl \
 && rm /tmp/graphit_jupyter-0.1.0-py3-none-any.whl

# Fail the build instead of showing an empty sidebar; sed strips the ANSI colour codes.
RUN jupyter labextension list 2>&1 \
    | sed -r 's/\x1b\[[0-9;]*m//g' \
    | grep -q "graphit-jupyter v0.1.0 enabled OK"

# "labextension list" only checks registration, not whether all chunks are present.
COPY --chown=${NB_UID}:${NB_GID} singleuser/check_bundle.py /tmp/check_bundle.py
RUN python3 /tmp/check_bundle.py /opt/conda/share/jupyter/labextensions/graphit-jupyter/static \
 && rm /tmp/check_bundle.py

# Seeds the extension settings before every server start. Written as heredoc instead of a
# .sh file in the repo, since a Windows checkout would give it CRLF line endings.
USER root
RUN mkdir -p /usr/local/bin/before-notebook.d \
 && cat <<'EOF' > /usr/local/bin/before-notebook.d/10-graphit-settings.sh
#!/bin/bash
# Never fails: the file is sourced, a non-zero exit would kill the spawn.
python3 - <<'PY' || echo "GRAPHIT: seeding skipped (not fatal)" >&2
import json, os, pathlib

user = os.environ.get("JUPYTERHUB_USER", "").strip()
base = os.environ.get("GRAPHIT_BASE_URL", "").strip()
if not user:
    raise SystemExit("no JUPYTERHUB_USER")

d = pathlib.Path.home() / ".jupyter/lab/user-settings/graphit-jupyter"
d.mkdir(parents=True, exist_ok=True)
f = d / "core.jupyterlab-settings"

try:
    cfg = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        cfg = {}
except Exception:
    cfg = {}

cfg["studentId"] = user
if base:
    cfg["baseUrl"] = base

def flag(text):
    """
    Parses a boolean switch from an environment variable.

    :param text: Raw value of the variable.
    :return: True for true/1/yes/on, False for false/0/no/off, otherwise None.
    """
    t = text.strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    return None

switches = {
    "showDiagnostics": ("GRAPHIT_SHOW_DIAGNOSTICS", flag),
    "mockMode": ("GRAPHIT_MOCK_MODE", flag),
    "streaming": ("GRAPHIT_STREAMING", flag),
    "persistChatHistory": ("GRAPHIT_PERSIST_CHAT_HISTORY", flag),
    "chatModel": ("GRAPHIT_CHAT_MODEL", lambda t: t.strip() or None),
    "reviewSessionSize": ("GRAPHIT_REVIEW_SESSION_SIZE",
                          lambda t: int(t) if t.strip().isdigit() else None),
}
for key, (var, parse) in switches.items():
    raw = os.environ.get(var, "")
    value = parse(raw) if raw.strip() else None
    if value is None:
        cfg.pop(key, None)
    else:
        cfg[key] = value

# Must match the properties of schema/core.json (additionalProperties: false).
allowed = {"baseUrl", "studentId", "mockMode", "streaming", "chatModel",
           "reviewSessionSize", "persistChatHistory", "showDiagnostics",
           "timeouts", "mock"}
cfg = {k: v for k, v in cfg.items() if k in allowed}

f.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"GRAPHIT: studentId={user} baseUrl={cfg.get('baseUrl', '<default>')}")
PY
EOF
RUN chmod +x /usr/local/bin/before-notebook.d/10-graphit-settings.sh

USER ${NB_UID}
