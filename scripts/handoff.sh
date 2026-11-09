#!/usr/bin/env bash
# handoff.sh OUT_DIR DOCUMENT N_PREDICT SLOT_NAME: one chat request through
# the split layout (scripts/split.sh, docs/q4_0.md), every step's answer
# kept in OUT_DIR and the timings printed:
#
#   1. the chat template applied and the prompt tokenized on the GPU server
#   2. all but the last prompt token processed there (n_predict 0)
#   3. its slot saved to SLOT_NAME in the shared slot directory
#   4. the slot restored on the decode server
#   5. the chat sent to the decode server with cache_prompt: it evaluates
#      the one remaining prompt token and generates N_PREDICT tokens
#
# The last token stays with the decode server because a restored state is
# reused only when the new prompt extends it (docs/cuda.md). The prompt is
# "Summarize this document in two sentences." and DOCUMENT's text.
# Needs curl, jq and bc.
#
#   PREFILL  the GPU server (http://127.0.0.1:8091)
#   DECODE   the decode server (http://127.0.0.1:8092)
set -euo pipefail
out=${1:?usage: handoff.sh OUT_DIR DOCUMENT N_PREDICT SLOT_NAME}
doc=${2:?usage: handoff.sh OUT_DIR DOCUMENT N_PREDICT SLOT_NAME}
n_predict=${3:?usage: handoff.sh OUT_DIR DOCUMENT N_PREDICT SLOT_NAME}
name=${4:?usage: handoff.sh OUT_DIR DOCUMENT N_PREDICT SLOT_NAME}
prefill=${PREFILL:-http://127.0.0.1:8091}
decode=${DECODE:-http://127.0.0.1:8092}
json='Content-Type: application/json'
mkdir -p "$out"
now() { date +%s.%N; }
post() { curl -sS -f --max-time 3000 "$1" -H "$json" -d "$2"; }

jq -n --rawfile doc "$doc" --argjson n "$n_predict" \
   '{messages: [{role: "user", content: ("Summarize this document in two sentences.\n\n" + $doc)}],
     max_tokens: $n, stream: false}' > "$out/body.json"
post "$prefill/apply-template" @"$out/body.json" > "$out/template.json"
jq '{content: .prompt, add_special: true, parse_special: true}' "$out/template.json" > "$out/tokenize.json"
post "$prefill/tokenize" @"$out/tokenize.json" > "$out/tokens.json"
tokens=$(jq '.tokens | length' "$out/tokens.json")
jq '{prompt: .tokens[0:-1], n_predict: 0, id_slot: 0, cache_prompt: false}' "$out/tokens.json" > "$out/prefill.json"

t0=$(now)
post "$prefill/completion" @"$out/prefill.json" > "$out/prefill-out.json"
t1=$(now)
post "$prefill/slots/0?action=save" "{\"filename\":\"$name\"}" > "$out/save.json"
t2=$(now)
post "$decode/slots/0?action=restore" "{\"filename\":\"$name\"}" > "$out/restore.json"
t3=$(now)
jq '. + {cache_prompt: true, id_slot: 0}' "$out/body.json" > "$out/chat.json"
post "$decode/v1/chat/completions" @"$out/chat.json" > "$out/chat-out.json"
t4=$(now)

echo "prompt tokens: $tokens"
echo "prefill: $(jq -c '.timings | {prompt_n, prompt_ms, prompt_per_second}' "$out/prefill-out.json") wall $(echo "$t1 - $t0" | bc) s"
echo "save: $(jq -c '{n_saved, n_written}' "$out/save.json") wall $(echo "$t2 - $t1" | bc) s"
echo "restore: $(jq -c '{n_restored, n_read}' "$out/restore.json") wall $(echo "$t3 - $t2" | bc) s"
echo "decode: $(jq -c '.timings | {cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms, predicted_per_second}' "$out/chat-out.json") wall $(echo "$t4 - $t3" | bc) s"
echo "finish: $(jq -c '.choices[0].finish_reason' "$out/chat-out.json")"
echo "text: $(jq -c '.choices[0].message | {content, reasoning_content}' "$out/chat-out.json" | cut -c1-400)"
