#!/usr/bin/env bash
set -euo pipefail

f="${1:?usage: emufix.sh <file.nro>}"

python3 - "$f" <<'PY'
import sys
path = sys.argv[1]
data = bytearray(open(path, "rb").read())

nop = bytes([0x1f, 0x20, 0x03, 0xd5])
targets = {
    "mrs gcspr_el0": bytes([0x21, 0x25, 0x3b, 0xd5]),
    "chkfeat x16":   bytes([0x1f, 0x25, 0x03, 0xd5]),
}

total = 0
for name, pat in targets.items():
    count = 0
    i = 0
    while True:
        j = data.find(pat, i)
        if j < 0:
            break
        data[j:j+4] = nop
        i = j + 4
        count += 1
    total += count

open(path, "wb").write(data)
print("emufix: fixed %d values" % total)
PY
