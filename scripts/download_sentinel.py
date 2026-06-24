"""
Download Sentinel-2 L2A imagery for a given bounding box and date range
via the Copernicus Data Space STAC API.

Usage:
    python scripts/download_sentinel.py \
        --bbox 12.3 41.8 12.6 42.1 \
        --start 2023-06-01 \
        --end 2023-08-31 \
        --output data/raw/imagery

Requires a free Copernicus Data Space account:
    https://dataspace.copernicus.eu/
"""

import argparse
from pathlib import Path

import requests
from pystac_client import Client
from rich.console import Console
from rich.progress import track

console = Console()

STAC_API = "https://catalogue.dataspace.copernicus.eu/stac"
COLLECTION = "SENTINEL-2"


def get_token(username: str, password: str) -> str:
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_items(bbox: list[float], start: str, end: str, max_cloud: int = 20) -> list:
    client = Client.open(STAC_API)
    results = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=50,
    )
    items = list(results.items())
    console.print(f"Found [bold]{len(items)}[/bold] scenes with <{max_cloud}% cloud cover")
    return items


def download_item(item, output_dir: Path, token: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for band_name, asset in item.assets.items():
        if not asset.href.endswith(".tif"):
            continue
        out_path = output_dir / f"{item.id}_{band_name}.tif"
        if out_path.exists():
            continue
        resp = requests.get(
            asset.href,
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
        )
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), required=True)
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/raw/imagery")
    parser.add_argument("--max-cloud", type=int, default=20)
    parser.add_argument("--username", help="Copernicus username (or set COPERNICUS_USER env var)")
    parser.add_argument("--password", help="Copernicus password (or set COPERNICUS_PASS env var)")
    args = parser.parse_args()

    import os
    username = args.username or os.environ.get("COPERNICUS_USER")
    password = args.password or os.environ.get("COPERNICUS_PASS")
    if not username or not password:
        raise ValueError("Provide --username/--password or set COPERNICUS_USER/COPERNICUS_PASS env vars")

    token = get_token(username, password)
    items = search_items(args.bbox, args.start, args.end, args.max_cloud)
    output_dir = Path(args.output)

    for item in track(items, description="Downloading scenes..."):
        download_item(item, output_dir / item.id, token)

    console.print(f"[green]Done.[/green] Files saved to {output_dir}")


if __name__ == "__main__":
    main()
