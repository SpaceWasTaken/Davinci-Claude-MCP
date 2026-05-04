"""
AI video analysis helpers — frame extraction and OpenAI GPT Vision wrapper.
Used by the video_ai compound tool in server.py.

Requires: pip install openai
          OPENAI_API_KEY environment variable set
          ffmpeg installed (brew install ffmpeg on macOS)
"""

import base64
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Union

FRAMES_DIR = Path.home() / "Documents" / "resolve-ai" / "frames"
INDEX_PATH = Path.home() / "Documents" / "resolve-ai" / "clip_index.json"

DEFAULT_MODEL = "gpt-4o-mini"

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_ffmpeg_path: str | None = None


def _find_ffmpeg() -> str:
    global _ffmpeg_path
    if _ffmpeg_path:
        return _ffmpeg_path
    candidates = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    for path in candidates:
        if os.path.exists(path):
            _ffmpeg_path = path
            return path
    try:
        result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            _ffmpeg_path = result.stdout.strip()
            return _ffmpeg_path
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found. Install with: brew install ffmpeg\n"
        "Expected at /opt/homebrew/bin/ffmpeg (Apple Silicon) or /usr/local/bin/ffmpeg (Intel)."
    )


def _find_ffprobe() -> str:
    ffmpeg = _find_ffmpeg()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if os.path.exists(ffprobe):
        return ffprobe
    return "ffprobe"


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Add it to your shell profile so Claude Desktop inherits it."
        )
    return OpenAI(api_key=api_key)


def get_video_duration(file_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path!r}: {result.stderr.strip()}")
    output = result.stdout.strip()
    if not output or output == "N/A":
        raise RuntimeError(f"ffprobe could not determine duration for {file_path!r}")
    return float(output)


def extract_frame(file_path: str, output_path: str, position: float = 0.5) -> str:
    """
    Extract a single JPEG frame at `position` (0.0–1.0 fraction of duration).
    Caps output at 1280×720, preserving aspect ratio.
    Returns output_path on success.
    """
    ffmpeg = _find_ffmpeg()
    duration = get_video_duration(file_path)
    seek_secs = max(0.0, min(duration * position, max(0.0, duration - 0.1)))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{seek_secs:.3f}",
        "-i", file_path,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease",
        "-vframes", "1",
        "-q:v", "3",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed: {result.stderr.strip()}")
    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg ran but output file was not created")
    return output_path


def extract_frames_multi(
    file_path: str,
    output_dir: str,
    num_frames: int = 10,
    stem: str = "frame",
) -> list[str]:
    """Extract `num_frames` evenly-spaced frames. Returns list of file paths."""
    if num_frames < 2:
        positions = [0.5]
    else:
        positions = [i / (num_frames - 1) for i in range(num_frames)]
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, pos in enumerate(positions):
        out = os.path.join(output_dir, f"{stem}_f{i:03d}.jpg")
        paths.append(extract_frame(file_path, out, pos))
    return paths


def analyze_with_claude(
    image_paths: Union[str, list[str]],
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send one or more images to GPT Vision. Returns raw response text."""
    client = _openai_client()
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    content: list[dict] = []
    for path in image_paths:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        media_type = _MEDIA_TYPES.get(Path(path).suffix.lower(), "image/jpeg")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{data}",
                "detail": "low",
            },
        })
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


def query_claude_text(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a text-only prompt to GPT (no images). Returns raw response text."""
    client = _openai_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def parse_json_response(text: str) -> dict:
    """Extract and parse a JSON object or array from a GPT response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def load_index() -> dict:
    """Load the clip description index from disk. Returns empty dict if missing."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {}


def save_index(index: dict) -> None:
    """Persist the clip description index to disk."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
