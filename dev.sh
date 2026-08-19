#!/usr/bin/env bash
# Start the analysis server and the web dev server together.
#
#   ./dev.sh                  both, waits for the server before vite starts
#   ./dev.sh --no-server      web only (same as: cd web && npm run dev)
#   ./dev.sh -- -v            everything after -- goes to terrarium-server
#
# Ctrl-C stops both. The server needs the `server` extra installed in
# analysis/.venv — see README.md. The PowerShell twin is dev.ps1.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
no_server=0
port=8765
server_args=()

while [ $# -gt 0 ]; do
  case "$1" in
    --no-server) no_server=1; shift ;;
    --port) port="$2"; shift 2 ;;
    --) shift; server_args=("$@"); break ;;
    -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1 (pass server options after --)" >&2; exit 2 ;;
  esac
done

server_pid=""
cleanup() {
  if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
    # Signal the group if we can; uv forwards it to the python child either way.
    kill -- "-$server_pid" 2>/dev/null || kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; }

if [ "$no_server" -eq 0 ]; then
  if port_open; then
    echo "analysis server already listening on $port — leaving it alone"
  else
    command -v uv >/dev/null 2>&1 || {
      echo "uv is not on PATH. Install it, or run with --no-server for the web app alone." >&2
      exit 1
    }
    # Own process group, so cleanup can take the whole tree down.
    set -m
    ( cd "$root/analysis" && exec uv run --extra server terrarium-server \
        --port "$port" "${server_args[@]+"${server_args[@]}"}" ) &
    server_pid=$!
    set +m

    # The browser probes the server exactly once at startup, and cold-loading
    # the models takes a while — so hold vite back until the port answers.
    echo "waiting for the analysis server on $port ..."
    deadline=$(( SECONDS + 180 ))
    while ! port_open; do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "terrarium-server exited before it came up" >&2
        exit 1
      fi
      if [ "$SECONDS" -gt "$deadline" ]; then
        echo "server still not up after 180s — starting vite anyway" >&2
        break
      fi
      sleep 0.5
    done
  fi
fi

cd "$root/web"
npm run dev
