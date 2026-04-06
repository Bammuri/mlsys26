#!/bin/bash
# Hook: After editing kernel.cu files, remind about build verification
# Triggered by PostToolUse on Edit/Write tools

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filePath // empty')

# Only trigger for kernel.cu or binding.py files in solution directories
if [[ "$FILE_PATH" == *"kernel.cu"* ]] || [[ "$FILE_PATH" == *"binding.py"* ]]; then
    if [[ "$FILE_PATH" == *"solution/cuda/"* ]]; then
        echo "Kernel file modified: $FILE_PATH — run /bench to verify on Modal B200"
    fi
fi

exit 0
