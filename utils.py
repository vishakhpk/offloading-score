"""Utility helpers for action parsing, image processing, and file I/O."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


def is_click_action(action: str) -> bool:
    return "click" in action.lower()


def is_keyboard_action(action: str) -> bool:
    a = action.lower()
    return "key_press" in a or "keypress" in a or "key." in a


def is_scroll_action(action: str) -> bool:
    return "scroll" in action.lower()


# %% Keyboard Input
def get_key_input(action: str) -> str:
    """Parse the key input from the action."""
    if "(" in action and ")" in action:
        kin = action.split("(")[1].split(")")[0]
    else:
        kin = action
    kin = kin.replace("'", "").strip()

    if kin == "Key.space":
        return " "
    elif kin == "Key.shift":
        return ""  # upper/lower case already applied to characters
    elif kin == "Key.backspace":
        return kin
    elif kin.startswith("Key."):  # shift/ctrl/alt/cmd
        return kin + "+"
    else:
        return kin


def _is_special_token(token: str) -> bool:
    """Return True if the token represents a special key/hotkey (e.g., Key.enter, cmd+v)."""
    if not token:
        return False
    if token.startswith("Key."):
        return True
    if "+" in token:
        return True
    special_names = {"enter", "esc", "tab"}
    return token.lower() in special_names


def compose_key_input(input_list: list[str]) -> str:
    """Compose the key input from the actions, ensuring special keys are separated."""
    segments: list[dict] = []
    current_text: list[str] = []

    def flush_text():
        nonlocal current_text
        if not current_text:
            return
        text_value = "".join(current_text)
        if segments and segments[-1]["type"] == "text":
            segments[-1]["value"] += text_value
        else:
            segments.append({"type": "text", "value": text_value})
        current_text = []

    def remove_last_character():
        """Remove the last character from current text or previous text segments."""
        nonlocal current_text
        if current_text:
            current_text.pop()
            return
        for idx in range(len(segments) - 1, -1, -1):
            if segments[idx]["type"] == "text" and segments[idx]["value"]:
                segments[idx]["value"] = segments[idx]["value"][:-1]
                if not segments[idx]["value"]:
                    segments.pop(idx)
                return
            elif segments[idx]["type"] == "text":
                segments.pop(idx)

    for token in input_list:
        if token == "Key.backspace":
            remove_last_character()
            continue

        if _is_special_token(token):
            flush_text()
            segments.append({"type": "special", "value": token})
        else:
            current_text.append(token)

    flush_text()

    final_values = [segment["value"] for segment in segments if segment["value"]]
    if not final_values:
        return ""
    return " ".join(final_values)


# %% LLM
import os
import openai
from openai import OpenAI

def call_openai(prompt: str, content = None, model_name: str = "gpt-4o") -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # print(f"[MODEL]Calling {model_name} with prompt: {prompt} and content: {content}")
    try:
        temp = 1 if model_name.startswith("gpt-5") else 0.0
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            temperature=temp,
        )
        print(f"[MODEL]Response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling {model_name}: {type(e)}")
        print(f"Error: {e}")
        return ""

# import anthropic
# from anthropic import Anthropic

def call_claude(prompt: str, *, content = None, model_name: str = "claude-sonnet-4-20250514") -> str:
    # print(f"[MODEL]Calling {model_name} with prompt: {prompt} and content: {content}")
    client = Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            system=prompt,
            messages=[{"role": "user", "content": content}],
            temperature=0
        )
        print(f"[MODEL]Response: {response.content[0].text}")
        return response.content[0].text

    except Exception as e:
        print(f"Error calling {model_name}: {type(e)}")
        print(f"Error: {e}")
        return ""
        
import re

def call_gemini(prompt: str, *, content = None, model_name: str = "gemini-1.5-pro") -> str:
    # print(f"[MODEL]Calling {model_name} with prompt: {prompt} and content: {content}")
    try:
        import google.generativeai as genai
    except ImportError:
        print("Error: google-generativeai package not installed. Install with: pip install google-generativeai")
        return ""
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return ""
    
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=prompt,
        )
        response = model.generate_content(
            content or "",
            generation_config={"temperature": 0.0, "max_output_tokens": 4096},
        )
        result_text = response.text
        print(f"[MODEL]Response: {result_text}")
        return result_text
    except Exception as e:
        print(f"Error calling {model_name}: {type(e)}")
        print(f"Error: {e}")
        return ""


# ---------------------------------------------------------------------------
# Image helpers

MIME_MAP = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def resolve_image_path(path: str | None, fallback_dir: Path | None = None) -> Path | None:
    """
    Resolve an image file path.
    
    Args:
        path: Image path string (can be None)
        fallback_dir: Optional directory to check if the path doesn't exist
        
    Returns:
        Resolved Path if found, None otherwise
    """
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if fallback_dir is not None:
        fallback = fallback_dir / candidate.name
        if fallback.exists():
            return fallback
    return None


def load_image(path: str | None, fallback_dir: Path | None = None) -> Image.Image | None:
    """
    Load an image file as a PIL Image object.
    
    Args:
        path: Image path string (can be None)
        fallback_dir: Optional directory to check if the path doesn't exist
        
    Returns:
        PIL Image in RGB mode, or None if loading fails
    """
    resolved = resolve_image_path(path, fallback_dir)
    if resolved is None:
        return None
    try:
        with Image.open(resolved) as img:
            return img.convert("RGB")
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"[ERROR] Unable to open {resolved}: {exc}")
        return None


def encode_image_for_llm(path: Path) -> dict[str, Any] | None:
    """
    Encode an image file as a base64 data URL for LLM API calls.
    
    Args:
        path: Path to the image file
        
    Returns:
        Dictionary with OpenAI-compatible image_url format, or None if encoding fails
    """
    try:
        data = path.read_bytes()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Unable to read image {path}: {exc}")
        return None
    mime = MIME_MAP.get(path.suffix.lower(), "image/png")
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def download_and_encode_image(url: str) -> dict[str, Any] | None:
    """
    Download an image from a URL and encode it for LLM API calls.
    
    Args:
        url: HTTP/HTTPS URL to the image
        
    Returns:
        Dictionary with OpenAI-compatible image_url format, or None if download/encoding fails
    """
    try:
        import requests
        from urllib.parse import urlparse
        
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Determine MIME type from Content-Type header or URL extension
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            mime = content_type
        else:
            # Fallback to extension-based detection
            parsed = urlparse(url)
            ext = Path(parsed.path).suffix.lower()
            mime = MIME_MAP.get(ext, "image/png")
        
        # Read image data
        data = response.content
        encoded = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Unable to download/encode image from {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# JSON I/O helpers

def load_json(path: Path) -> Any:
    """
    Load JSON data from a file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    with path.open() as fh:
        return json.load(fh)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """
    Save data as JSON to a file.
    
    Args:
        data: Data to serialize as JSON
        path: Path to write the JSON file
        indent: Number of spaces for indentation (default: 2)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=indent)
        fh.write("\n")


def extract_segments(data: Any) -> list[dict[str, Any]]:
    """
    Extract segments from various data structures.
    
    Args:
        data: Can be a list of segments, a dict with "segments" key, or a single sequence node
        
    Returns:
        List of segment dictionaries
    """
    if isinstance(data, list):
        # Direct list of segments
        return data
    elif isinstance(data, dict):
        # Check for segments key
        if "segments" in data:
            return data["segments"]
        # Check if it's a single sequence node
        elif data.get("node_type") == "sequence":
            return [data]
    return []


def save_segments(
    segments: list[dict[str, Any]],
    *,
    output_file: Path | None,
    output_dir: Path | None,
) -> None:
    """
    Save segments both as a single JSON file and as individual files in a directory.
    
    Args:
        segments: List of segment dictionaries to save
        output_file: Optional path to save all segments as a single JSON file
        output_dir: Optional directory path to save each segment as a separate JSON file (0.json, 1.json, etc.)
    """
    if output_file is None and output_dir is None:
        return
    if output_file is not None:
        save_json(segments, output_file)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, segment in enumerate(segments):
            save_json(segment, output_dir / f"{idx}.json")


# ---------------------------------------------------------------------------
# Statistics helpers

def calculate_percentile(sorted_data: list[float], percentile: float) -> float:
    """
    Calculate percentile from sorted data.
    
    Args:
        sorted_data: Sorted list of numeric values
        percentile: Percentile value (0.0-1.0)
        
    Returns:
        Percentile value
    """
    if not sorted_data:
        return 0.0
    index = int(len(sorted_data) * percentile)
    return sorted_data[min(index, len(sorted_data) - 1)]


def calculate_median(sorted_data: list[float]) -> float:
    """
    Calculate median from sorted data.
    
    Args:
        sorted_data: Sorted list of numeric values
        
    Returns:
        Median value
    """
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string with seconds and minutes
    """
    return f"{seconds:.2f}s ({seconds/60:.2f} min)"

