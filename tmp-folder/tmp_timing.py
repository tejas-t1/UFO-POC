"""Print per-step timing breakdown from a UFO response.log."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/test/response.log")
lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"{len(lines)} response records from {path}\n")

hdr = f"{'idx':>3}  {'agent':<10} {'rnd/step':<10} {'action':<30} {'data':>7} {'llm':>8} {'act':>7} {'total':>8}"
print(hdr)
print("-" * len(hdr))

total_llm = 0.0
total_other = 0.0
for i, raw in enumerate(lines):
    try:
        e = json.loads(raw)
    except json.JSONDecodeError as ex:
        print(f"{i:>3}  <unparseable: {ex}>")
        continue
    et = e.get("execution_times", {}) or {}

    def _f(v):
        if isinstance(v, list):
            return sum(_f(x) for x in v)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    data = _f(et.get("DATA_COLLECTION"))
    llm = _f(et.get("LLM_INTERACTION"))
    act = _f(et.get("ACTION_EXECUTION"))
    total = _f(e.get("total_time"))
    agent = e.get("agent_type", "")
    rnd_step = f"r{e.get('round_num','?')}s{e.get('round_step','?')}"
    action = (e.get("action_type") or e.get("function_call") or "-")[:30]
    print(f"{i:>3}  {agent:<10} {rnd_step:<10} {action:<30} {data:>6.1f}s {llm:>7.1f}s {act:>6.1f}s {total:>7.1f}s")
    total_llm += llm
    total_other += (total - llm)

print()
print(f"Sum LLM time:     {total_llm:>7.1f}s")
print(f"Sum non-LLM time: {total_other:>7.1f}s")
print(f"Grand total:      {total_llm + total_other:>7.1f}s")
