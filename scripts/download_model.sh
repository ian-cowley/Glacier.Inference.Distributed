#!/usr/bin/env bash
# Glacier High-Speed Model Downloader (Bash / .NET 10)
# Cross-platform Bash script running the pure C# multi-stream model downloader.
# Zero Python dependencies. Works on Linux and macOS.
#
# Examples:
#   ./scripts/download_model.sh qwen2-57b
#   ./scripts/download_model.sh mixtral-8x22b
#   ./scripts/download_model.sh https://huggingface.co/path/to/model.gguf

set -euo pipefail

MODEL_OR_URL="${1:-}"
DEST="${2:-models}"
STREAMS="${3:-8}"

if [ -z "$MODEL_OR_URL" ]; then
    echo "Usage: ./scripts/download_model.sh <qwen2-57b|mixtral-8x22b|url> [dest_dir] [streams]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI_PROJ="$ROOT_DIR/src/Glacier.Inference.Distributed.Cli/Glacier.Inference.Distributed.Cli.csproj"

dotnet run --project "$CLI_PROJ" -c Release -- download "$MODEL_OR_URL" --dest "$DEST" --streams "$STREAMS"
