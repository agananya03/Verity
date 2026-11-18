"""Time single judge-style calls directly (no RAGAS). Usage: python judge_latency_test.py MODEL_ID [MODEL_ID ...]"""
import os
import sys
import time

try:
    from dotenv import load_dotenv

    for _p in ("backend/.env", ".env", "../backend/.env", "../.env"):
        if os.path.exists(_p):
            load_dotenv(_p)
            print("Loaded env from", _p)
            break
except Exception:
    pass

from google import genai

key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not key:
    sys.exit("No GOOGLE_API_KEY / GEMINI_API_KEY found in the environment or .env")

client = genai.Client(api_key=key, http_options={"timeout": 180000})

print("Gemma models available to this key:")
try:
    for m in client.models.list():
        if "gemma" in m.name.lower():
            print("  ", m.name)
except Exception as e:  # noqa: BLE001
    print("  could not list models:", repr(e)[:200])

context = (
    "Apple Inc. fiscal 2023 results. Net sales decreased 3% or $11.0 billion during 2023 "
    "compared to 2022. The weakness in foreign currencies relative to the U.S. dollar accounted "
    "for more than the entire year-over-year decrease in total net sales. "
) * 40
prompt = (
    "You are a strict fact checker. Context:\n"
    + context
    + '\n\nStatement: "Apple net sales fell 3% in fiscal 2023." '
    'Reply ONLY with JSON: {"verdict": 1 or 0, "reason": "<short>"}'
)

for name in sys.argv[1:]:
    for label, cfg in (("plain", None), ("json_mode", {"response_mime_type": "application/json"})):
        t0 = time.time()
        try:
            r = client.models.generate_content(model=name, contents=prompt, config=cfg)
            dt = time.time() - t0
            usage = getattr(r, "usage_metadata", None)
            toks = getattr(usage, "total_token_count", "?") if usage else "?"
            print(f"[{name} | {label}] OK in {dt:.1f}s, tokens={toks}, output head: {(r.text or '')[:150]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[{name} | {label}] FAILED after {time.time() - t0:.1f}s: {repr(e)[:300]}")