import os
from pathlib import Path

SUPPORTED_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".mp3",
    ".wav",
    ".m4a",
    ".m4v",
}

PRIORITY_EXTENSIONS = [".mp4", ".mkv", ".mov", ".webm", ".avi", ".mp3", ".wav", ".m4a", ".m4v"]


def find_media_file(data_dir=None):
    """Return the first supported media file in a directory, preferring common video formats."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    candidates = []
    for path in sorted(data_path.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS:
            candidates.append(path)

    if not candidates:
        return None

    preferred_names = ["lecture.mp4", "lecture.mkv", "lecture.mov", "lecture.webm"]
    for preferred_name in preferred_names:
        for candidate in candidates:
            if candidate.name.lower() == preferred_name.lower():
                return candidate

    for ext in PRIORITY_EXTENSIONS:
        for candidate in candidates:
            if candidate.suffix.lower() == ext:
                return candidate

    return candidates[0]
