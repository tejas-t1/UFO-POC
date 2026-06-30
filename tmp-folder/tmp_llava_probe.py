import base64, time, httpx
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
    raw = f.read()
img = base64.b64encode(raw).decode()
print(f"image bytes: {len(raw)}  base64 chars: {len(img)}")

t0 = time.time()
r = c.chat.completions.create(
    model="qwen2-max-ctx:latest",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Look at this Windows screenshot. Reply ONLY with valid JSON "
                        'of the form {"observation":"...","thought":"..."} '
                        "and nothing else."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + img},
                },
            ],
        }
    ],
    max_tokens=400,
    response_format={"type": "json_object"},
)
print(f"elapsed {time.time()-t0:.1f}s")
print("---- content ----")
print(r.choices[0].message.content)
print("---- usage ----")
print(r.usage)
