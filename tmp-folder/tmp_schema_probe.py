"""Probe whether Ollama (via its OpenAI shim) honors strict JSON-schema response_format.

If a model accepts `response_format={"type":"json_schema", ...}` and returns the
required keys, UFO can be made to enforce its HostAgentResponse / AppAgentResponse
schemas directly at the model layer -> no more 'Field required' Pydantic errors.
"""

import base64
import json
import time

import httpx
from openai import OpenAI

auth = httpx.BasicAuth("inteluser", "intel@200$")
hc = httpx.Client(verify=False, auth=auth, timeout=600, trust_env=True)
c = OpenAI(
    base_url="https://10.223.160.85/v1",
    api_key="ollama",
    http_client=hc,
    timeout=600,
    max_retries=0,
)

with open(r"logs\test\action_step0.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

schema = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "thought": {"type": "string"},
    },
    "required": ["observation", "thought"],
    "additionalProperties": False,
}

MODELS = ("qwen2-max-ctx:latest", "qwen2.5vl:7b", "llava:7b")

for model in MODELS:
    print(f"\n========== {model} ==========")
    try:
        t0 = time.time()
        r = c.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Look at the screenshot and produce the required JSON.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + img_b64},
                        },
                    ],
                }
            ],
            max_tokens=400,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "obs_thought",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        dt = time.time() - t0
        content = r.choices[0].message.content or ""
        print(f"elapsed {dt:.1f}s  prompt_tokens={r.usage.prompt_tokens}")
        print("raw:", content[:400])
        try:
            parsed = json.loads(content)
            print("keys:", sorted(parsed.keys()))
            if set(parsed) >= {"observation", "thought"}:
                print("schema OK")
            else:
                print("MISSING KEYS")
        except Exception as e:
            print("not valid JSON:", e)
    except Exception as e:
        print(f"REQUEST FAILED: {type(e).__name__}: {e}")
