#!/usr/bin/env python3
"""
Underwater Image Collector
===========================
Bulk downloads underwater background images for model training using 
Pexels and Pixabay APIs. Supports resuming interrupted downloads and deduplication
(by unique URL / content MD5 hash).

SETUP
-----
1) pip install requests tqdm

2) Get API keys (both are free and take seconds):
   - Pexels:  https://www.pexels.com/api/         -> Click "Get Started" to obtain key
   - Pixabay: https://pixabay.com/api/docs/        -> Create an account to obtain key

3) Set environment variables:
   export PEXELS_API_KEY="xxxxxxx"
   export PIXABAY_API_KEY="xxxxxxx"

   (Windows PowerShell: $env:PEXELS_API_KEY="xxxxxxx")

USAGE
-----
python underwater_downloader.py --target 2000 --out ./underwater_dataset

The script rotates through both API sources until reaching the target limit.
If interrupted, rerun the same command; it skips already-downloaded files and resumes.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------
# Diverse search queries: Searching with a single term yields pagination limits
# after a few hundred results. Using varied sub-theme queries helps reach higher
# targets like 2000 and increases dataset diversity. Terms likely to return
# marine life ("marine life", "tropical fish", etc.) were intentionally omitted
# to prioritize empty/landscape-oriented underwater scenes.
# -----------------------------------------------------------------------
SEARCH_QUERIES = [
    "underwater background",
    "underwater empty ocean",
    "underwater blue water texture",
    "underwater cave no fish",
    "underwater light rays",
    "underwater sand floor",
    "underwater rocks",
    "deep sea water background",
    "ocean floor empty",
    "underwater bubbles background",
    "underwater seascape no animals",
    "underwater cavern light",
    "shipwreck underwater empty",
    "underwater blue gradient",
    "underwater kelp forest empty",
]

# If any of these keywords appear in the photo description or tags
# (Pixabay 'tags' field, Pexels 'alt' field), the image is filtered out.
# This helps eliminate a large portion of results containing sea creatures or people.
CREATURE_BLOCKLIST = [
    "fish", "shark", "turtle", "dolphin", "whale", "jellyfish", "octopus",
    "squid", "crab", "seal", "ray", "eel", "seahorse", "diver", "person",
    "man", "woman", "animal", "creature", "school of", "clownfish", "coral fish",
    "stingray", "manta", "sea lion", "penguin", "otter", "snorkel", "swimmer",
]

STATE_FILE = "download_state.json"


def is_creature_free(text: str) -> bool:
    """Returns True if the description/tags text contains no creature-related keywords."""
    if not text:
        return True
    lowered = text.lower()
    return not any(word in lowered for word in CREATURE_BLOCKLIST)


def load_state(out_dir: Path) -> dict:
    state_path = out_dir / STATE_FILE
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return {"downloaded_hashes": [], "downloaded_ids": [], "count": 0}


def save_state(out_dir: Path, state: dict):
    with open(out_dir / STATE_FILE, "w") as f:
        json.dump(state, f)


def file_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


# -----------------------------------------------------------------------
# PEXELS
# -----------------------------------------------------------------------
def fetch_pexels(query: str, page: int, api_key: str) -> list:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": 80, "page": page, "orientation": "landscape"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code == 429:
        print("  [Pexels] Rate limit reached, sleeping for 20 seconds...")
        time.sleep(20)
        return fetch_pexels(query, page, api_key)
    r.raise_for_status()
    data = r.json()
    results = []
    for photo in data.get("photos", []):
        alt_text = photo.get("alt", "") or ""
        if not is_creature_free(alt_text):
            continue
        results.append(
            {
                "id": f"pexels_{photo['id']}",
                "url": photo["src"]["large2x"],
            }
        )
    return results


# -----------------------------------------------------------------------
# PIXABAY
# -----------------------------------------------------------------------
def fetch_pixabay(query: str, page: int, api_key: str) -> list:
    url = "https://pixabay.com/api/"
    params = {
        "key": api_key,
        "q": query,
        "image_type": "photo",
        "orientation": "horizontal",
        "per_page": 200,
        "page": page,
        "safesearch": "true",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 429:
        print("  [Pixabay] Rate limit reached, sleeping for 20 seconds...")
        time.sleep(20)
        return fetch_pixabay(query, page, api_key)
    r.raise_for_status()
    data = r.json()
    results = []
    for hit in data.get("hits", []):
        tags_text = hit.get("tags", "") or ""
        if not is_creature_free(tags_text):
            continue
        results.append(
            {
                "id": f"pixabay_{hit['id']}",
                "url": hit["largeImageURL"],
            }
        )
    return results


def download_image(url: str, timeout=30) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def main():
    parser = argparse.ArgumentParser(description="Underwater Image Collector")
    parser.add_argument("--target", type=int, default=2000, help="Target number of images to download")
    parser.add_argument("--out", type=str, default="./underwater_dataset", help="Output directory")
    parser.add_argument("--min-size-kb", type=int, default=20, help="Skip files smaller than this size (in KB)")
    args = parser.parse_args()

    pexels_key = os.environ.get("PEXELS_API_KEY")
    pixabay_key = os.environ.get("PIXABAY_API_KEY")

    if not pexels_key and not pixabay_key:
        print("ERROR: At least one environment variable (PEXELS_API_KEY or PIXABAY_API_KEY) is required.")
        print("Get free API keys at:")
        print("  Pexels : https://www.pexels.com/api/")
        print("  Pixabay: https://pixabay.com/api/docs/")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(out_dir)

    downloaded_hashes = set(state["downloaded_hashes"])
    downloaded_ids = set(state["downloaded_ids"])
    count = state["count"]

    pbar = tqdm(total=args.target, initial=count, desc="Downloaded images")

    sources = []
    if pexels_key:
        sources.append(("pexels", fetch_pexels, pexels_key))
    if pixabay_key:
        sources.append(("pixabay", fetch_pixabay, pixabay_key))

    query_idx = 0
    page_by_query = {}  # (source_name, query) -> page number

    try:
        while count < args.target:
            source_name, fetch_fn, api_key = sources[query_idx % len(sources)]
            query = SEARCH_QUERIES[query_idx % len(SEARCH_QUERIES)]
            key = (source_name, query)
            page = page_by_query.get(key, 1)

            try:
                items = fetch_fn(query, page, api_key)
            except Exception as e:
                print(f"  [{source_name}] Error ({query}, page {page}): {e}")
                items = []

            if not items:
                # If no items returned for this query/source/page, move to the next combination
                query_idx += 1
                if query_idx > len(SEARCH_QUERIES) * len(sources) * 5:
                    print("Sources exhausted. No more unique images found.")
                    break
                continue

            page_by_query[key] = page + 1

            for item in items:
                if count >= args.target:
                    break
                if item["id"] in downloaded_ids:
                    continue
                try:
                    content = download_image(item["url"])
                except Exception:
                    continue

                if len(content) < args.min_size_kb * 1024:
                    continue

                h = file_hash(content)
                if h in downloaded_hashes:
                    continue

                ext = ".jpg"
                filename = f"underwater_{count:05d}_{item['id']}{ext}"
                with open(out_dir / filename, "wb") as f:
                    f.write(content)

                downloaded_hashes.add(h)
                downloaded_ids.add(item["id"])
                count += 1
                pbar.update(1)

                # Save state every 25 images to avoid progress loss on sudden termination
                if count % 25 == 0:
                    state = {
                        "downloaded_hashes": list(downloaded_hashes),
                        "downloaded_ids": list(downloaded_ids),
                        "count": count,
                    }
                    save_state(out_dir, state)

            query_idx += 1
            time.sleep(0.3)  # Be gentle on rate limits

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving state...")

    finally:
        state = {
            "downloaded_hashes": list(downloaded_hashes),
            "downloaded_ids": list(downloaded_ids),
            "count": count,
        }
        save_state(out_dir, state)
        pbar.close()
        print(f"\nTotal images downloaded: {count} / {args.target}")
        print(f"Directory: {out_dir.resolve()}")
        if count < args.target:
            print("Target not fully reached. Rerun the same command to resume.")


if __name__ == "__main__":
    main()