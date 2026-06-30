"""Estimate prompt size for each record in a UFO request.log line-stream."""
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/test/request.log")
text = path.read_text(encoding="utf-8")
print(f"Raw file size: {len(text):>12} chars ({len(text)/1024/1024:.1f} MB)")

# Try parsing as one big JSON; if not, as JSON lines; if not, find embedded JSON blobs.
records = []
try:
    obj = json.loads(text)
    records = obj if isinstance(obj, list) else [obj]
except json.JSONDecodeError:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass

print(f"Parsed {len(records)} top-level records")

# Count base64 image strings (huge data: URLs)
img_count = len(re.findall(r"data:image/(?:png|jpeg);base64,", text))
print(f"Embedded base64 image strings: {img_count}")

# Show the message structure of the first record if we have one
if records:
    r0 = records[0]
    print("First record top-level keys:", list(r0.keys())[:20])
