#!/bin/sh
# Select the process to run based on KB_PROCESS (api | bot). Defaults to api.
set -e

case "${KB_PROCESS:-api}" in
  bot)
    exec uv run python -m kb.channels.telegram
    ;;
  *)
    exec uv run uvicorn kb.api.app:app --host 0.0.0.0 --port 8000
    ;;
esac
