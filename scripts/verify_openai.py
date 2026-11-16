"""Run this once to verify the OpenAI key and the vision path work end to end.

Usage: python scripts/verify_openai.py [recording.mp4]

Checks the text path (Stage 1's model) and, when a recording is given, the local
decode path plus the vision path (Stage 2's model) by sending one real frame. A key
that works for text can still fail on vision if the configured model is not
vision-capable, so the second check is the one worth running before a real audit.
"""

import base64
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from reckon.extractor import EXTRACTION_MODEL  # noqa: E402
from reckon.frames import extract_frame, frame_to_jpeg_bytes  # noqa: E402
from reckon.gateway import make_gateway_client  # noqa: E402
from reckon.keyframes import detect_change_events  # noqa: E402
from reckon.parser import OPENAI_MODEL  # noqa: E402


def main() -> None:
    recording_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        client = make_gateway_client()
    except ValueError as exc:
        raise SystemExit(f"{exc} — copy .env.example to .env and fill it in") from exc

    reply = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    print(f"Text OK    — {OPENAI_MODEL} replied {reply.choices[0].message.content.strip()!r}")

    if recording_path is None:
        print("\nPass a recording path to also verify the decode and vision paths.")
        return

    events = detect_change_events(recording_path)
    print(f"Decode OK  — {len(events)} change event(s) found in {recording_path.name}")

    probe_ms = events[0] if events else 0
    jpeg_bytes = frame_to_jpeg_bytes(extract_frame(recording_path, probe_ms), max_dimension=1536)
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    print(f"Frame OK   — decoded frame at {probe_ms}ms ({len(jpeg_bytes)} bytes JPEG)")

    vision_reply = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "In five words or fewer, what is on this screen?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
    )
    print(f"Vision OK  — {EXTRACTION_MODEL} replied "
          f"{vision_reply.choices[0].message.content.strip()!r}")


if __name__ == "__main__":
    main()
