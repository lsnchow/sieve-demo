"""Fetch public demo sources from Wikimedia Commons and cut flat raw_input clips."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT_DIR = REPO_ROOT / "raw_input"
CACHE_DIR = REPO_ROOT / ".cache" / "source_videos"
SOURCE_MAP_PATH = RAW_INPUT_DIR / "source_map.csv"

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Sieve2DatasetDemo/1.0 (local demo fetch)"

CLIPS = [
    {
        "source_title": "File:Horse and cart first person.webm",
        "filename": "high_value_pov_horse_cart.mp4",
        "demo_category": "high_value_pov",
        "clip_start_sec": 18,
        "clip_duration_sec": 18,
    },
    {
        "source_title": 'File:Whitewater kayaking through "Mini Gorge" on the Whitewater River, North Carolina, USA.webm',
        "filename": "high_value_pov_whitewater.mp4",
        "demo_category": "high_value_pov",
        "clip_start_sec": 6,
        "clip_duration_sec": 18,
    },
    {
        "source_title": "File:First person view from a snowmobile driven by the FBI Hostage Rescue Team in Yellowstone National Park.webm",
        "filename": "high_value_pov_snowmobile.mp4",
        "demo_category": "high_value_pov",
        "clip_start_sec": 0,
        "clip_duration_sec": 20,
    },
    {
        "source_title": "File:Airplane inside while taking off.webm",
        "filename": "pov_no_action_airplane.mp4",
        "demo_category": "pov_no_action",
        "clip_start_sec": 28,
        "clip_duration_sec": 18,
    },
    {
        "source_title": "File:Action Cam Footage from U.S. Spacewalk -41.webm",
        "filename": "pov_no_action_spacewalk.mp4",
        "demo_category": "pov_no_action",
        "clip_start_sec": 45,
        "clip_duration_sec": 20,
    },
    {
        "source_title": "File:Village Dance.webm",
        "filename": "third_person_village_dance.mp4",
        "demo_category": "third_person",
        "clip_start_sec": 0,
        "clip_duration_sec": 11,
    },
    {
        "source_title": "File:Tap Dance Technique.webm",
        "filename": "low_quality_tap_dance.mp4",
        "demo_category": "low_quality",
        "clip_start_sec": 10,
        "clip_duration_sec": 20,
    },
    {
        "source_title": "File:Trail Camera.webm",
        "filename": "static_or_duplicate_trail_camera.mp4",
        "demo_category": "static_or_duplicate",
        "clip_start_sec": 0,
        "clip_duration_sec": 10,
    },
    {
        "source_title": "File:Krvavec šestseda.webm",
        "filename": "ambiguous_chairlift.mp4",
        "demo_category": "ambiguous",
        "clip_start_sec": 0,
        "clip_duration_sec": 20,
    },
    {
        "source_title": "File:C'était pas pire….webm",
        "filename": "ambiguous_first_person_walk.mp4",
        "demo_category": "ambiguous",
        "clip_start_sec": 25,
        "clip_duration_sec": 20,
    },
]


def wiki_page_url(title: str) -> str:
    return f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}"


def fetch_wikimedia_metadata(title: str) -> dict:
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "format": "json",
    }
    url = f"{WIKIMEDIA_API}?{urlencode(params)}"
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT})) as response:
        data = json.load(response)

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    imageinfo = (page.get("imageinfo") or [{}])[0]
    ext = imageinfo.get("extmetadata", {})
    return {
        "download_url": imageinfo.get("url", ""),
        "source_url": wiki_page_url(title),
        "license": ext.get("LicenseShortName", {}).get("value", ""),
        "attribution": (
            ext.get("AttributionRequired", {}).get("value")
            or ext.get("Artist", {}).get("value")
            or ext.get("Credit", {}).get("value", "")
        ),
    }


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT})) as response:
        dest.write_bytes(response.read())


def build_clip(source_path: Path, output_path: Path, start_sec: int, duration_sec: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(source_path),
        "-t",
        str(duration_sec),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    RAW_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip in CLIPS:
        meta = fetch_wikimedia_metadata(clip["source_title"])
        cache_name = quote(clip["source_title"].replace("File:", ""), safe="._-")
        source_path = CACHE_DIR / cache_name
        download_file(meta["download_url"], source_path)

        output_path = RAW_INPUT_DIR / clip["filename"]
        build_clip(
            source_path=source_path,
            output_path=output_path,
            start_sec=clip["clip_start_sec"],
            duration_sec=clip["clip_duration_sec"],
        )

        rows.append({
            "filename": clip["filename"],
            "source_title": clip["source_title"],
            "source_url": meta["source_url"],
            "license": meta["license"],
            "attribution": meta["attribution"],
            "demo_category": clip["demo_category"],
            "clip_start_sec": clip["clip_start_sec"],
            "clip_duration_sec": clip["clip_duration_sec"],
        })

    with open(SOURCE_MAP_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "source_title",
                "source_url",
                "license",
                "attribution",
                "demo_category",
                "clip_start_sec",
                "clip_duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips to {RAW_INPUT_DIR}")
    print(f"Wrote provenance map to {SOURCE_MAP_PATH}")


if __name__ == "__main__":
    main()
