#!/usr/bin/env bash
# request.sh OUT_DIR DOCUMENT N_PREDICT: one chat request to one server
# with the prompt cache off, the server's own timings printed. The prompt
# is "Summarize this document in two sentences." and DOCUMENT's text; the
# request and the answer are kept in OUT_DIR. Needs curl, jq and bc.
#
#   SERVER  the server (http://127.0.0.1:8095)
set -euo pipefail
out=${1:?usage: request.sh OUT_DIR DOCUMENT N_PREDICT}
doc=${2:?usage: request.sh OUT_DIR DOCUMENT N_PREDICT}
n_predict=${3:?usage: request.sh OUT_DIR DOCUMENT N_PREDICT}
server=${SERVER:-http://127.0.0.1:8095}
mkdir -p "$out"
jq -n --rawfile doc "$doc" --argjson n "$n_predict" \
   '{messages: [{role: "user", content: ("Summarize this document in two sentences.\n\n" + $doc)}],
     max_tokens: $n, stream: false, cache_prompt: false}' > "$out/body.json"
t0=$(date +%s.%N)
curl -sS -f --max-time 3000 "$server/v1/chat/completions" -H 'Content-Type: application/json' \
     -d @"$out/body.json" > "$out/answer.json"
t1=$(date +%s.%N)
echo "timings: $(jq -c '.timings | {prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_second}' "$out/answer.json") wall $(echo "$t1 - $t0" | bc) s"
echo "finish: $(jq -c '.choices[0].finish_reason' "$out/answer.json")"
echo "text: $(jq -c '.choices[0].message | {content, reasoning_content}' "$out/answer.json" | cut -c1-300)"
