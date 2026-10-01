#!/bin/bash
# Starts Neo4j and writes a dump on shutdown, the only moment the Community edition's
# database is reliably stopped. Each dump gets its own directory because
# "neo4j-admin database load" requires the file name neo4j.dump.
# GRAPHIT_DUMP_AUS=1 disables it, GRAPHIT_DUMP_BEHALTEN sets how many dumps are kept.
set -uo pipefail

AUTO_DIR="${GRAPHIT_DUMP_DIR:-/dumps/auto}"
KEEP="${GRAPHIT_DUMP_BEHALTEN:-3}"
DATABASE="${GRAPHIT_DUMP_DB:-neo4j}"

log() { echo "[graphit-dump] $*"; }

as_neo4j() {
  if [ "$(id -u)" != "0" ]; then
    "$@"
  elif command -v gosu >/dev/null 2>&1; then
    gosu neo4j "$@"
  elif command -v su-exec >/dev/null 2>&1; then
    su-exec neo4j "$@"
  else
    su neo4j -s /bin/bash -c "$(printf '%q ' "$@")"
  fi
}

take_dump() {
  if [ "${GRAPHIT_DUMP_AUS:-0}" = "1" ]; then
    log "skipped (GRAPHIT_DUMP_AUS=1)"
    return 0
  fi

  local target="$AUTO_DIR/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$target" || { log "ERROR: cannot create $target"; return 1; }
  # chown often fails silently on Windows bind mounts, hence chmod as well.
  chown neo4j "$target" 2>/dev/null || true
  chmod 0777 "$target" 2>/dev/null || true

  log "writing dump of '$DATABASE' to $target"
  local output
  if output=$(as_neo4j neo4j-admin database dump "$DATABASE" \
                 --to-path="$target" --overwrite-destination=true 2>&1); then
    log "dump done: $(ls -lh "$target"/*.dump 2>/dev/null | awk '{print $5, $9}')"
  else
    log "ERROR during dump, message from neo4j-admin:"
    printf '%s\n' "$output" | tail -n 15 | sed 's/^/[graphit-dump]   /'
    log "removing the empty directory again"
    rm -rf "$target"
    return 1
  fi

  local -a dumps
  mapfile -t dumps < <(ls -1d "$AUTO_DIR"/*/ 2>/dev/null | sort -r)
  if [ "${#dumps[@]}" -gt "$KEEP" ]; then
    local old
    for old in "${dumps[@]:$KEEP}"; do
      log "removing old dump: $old"
      rm -rf "$old"
    done
  fi
  log "existing dumps: ${#dumps[@]} (keeping $KEEP)"
}

on_signal() {
  # Ignore further signals, otherwise a second "docker stop" interrupts the dump.
  trap '' TERM INT
  log "signal received, shutting down Neo4j"
  if [ -n "${NEO4J_PID:-}" ] && kill -0 "$NEO4J_PID" 2>/dev/null; then
    kill -TERM "$NEO4J_PID" 2>/dev/null
    wait "$NEO4J_PID"
  fi
  log "Neo4j stopped, starting the dump"
  take_dump || true
  exit 0
}

trap on_signal TERM INT

if [ "$#" -eq 0 ]; then
  set -- neo4j
fi

/startup/docker-entrypoint.sh "$@" &
NEO4J_PID=$!
log "Neo4j started (PID $NEO4J_PID), dump on shutdown is active"

wait "$NEO4J_PID"
exit $?
