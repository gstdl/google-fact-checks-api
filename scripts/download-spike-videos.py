#!/usr/bin/env python3
"""Find and download fact-checked social media videos for the AI-detection spike test.

Reviews are read from the snapshot Parquet, filtered by claim text, date window,
and retrieval keyword, then visited oldest first. For each review the script
reads the fact-check article, collects the social media posts it cites (directly
or through perma.cc), and downloads the first one that still has a video. It
stops after --n reviews have a downloaded video.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import polars as pl
import requests
import yt_dlp
from warcio.archiveiterator import ArchiveIterator

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = (
    REPO_ROOT / "data" / "google_fact_check_tool_id_snapshots"
    / "20260822T144631.895801Z-5ef18cc447bf" / "flattened.parquet"
)
OUTPUT_ROOT = REPO_ROOT / "tmp" / "spike-videos"

# 2024 Indonesian presidential campaign period.
DEFAULT_START = "2023-11-28"
DEFAULT_END = "2024-02-10"
# Snapshot retrieval keywords tied to the 2024 presidential election.
DEFAULT_TOPIC_KEYWORDS = (
    "pemilu", "kpu", "presiden", "jokowi", "prabowo", "gibran", "anies", "ganjar", "megawati",
)

# Post URLs (not profile URLs) on the platforms in scope, matched anywhere in the
# article HTML, including inside web.archive.org links.
POST_PATTERNS = {
    "tiktok": re.compile(
        r"https?://(?:www\.|m\.)?tiktok\.com/@[\w.-]+/video/\d+|https?://v[mt]\.tiktok\.com/[\w-]+/?"
    ),
    "instagram": re.compile(r"https?://(?:www\.)?instagram\.com/(?:p|reels?|tv)/[\w-]+/?"),
    "facebook": re.compile(
        r"https?://(?:www\.|web\.|m\.)?facebook\.com/"
        r"(?:[\w.-]+/videos/(?:[\w.-]+/)?\d+|watch/?\?v=\d+|reel/\d+)"
        r"|https?://fb\.watch/[\w-]+/?"
    ),
    "x": re.compile(r"https?://(?:www\.|mobile\.)?(?:twitter|x)\.com/\w+/status/\d+"),
}
PERMA_PATTERN = re.compile(r"https?://perma\.cc/[A-Z0-9]{4}-[A-Z0-9]{4}")
PERMA_REPLAY_ORIGIN = "https://rejouer.perma.cc"
PERMA_WARC_SOURCE = re.compile(r'srcQuery\.append\("source", "([^"]+)"\)')
AI_MENTION = re.compile(r"\b(?:ai|deepfake|kecerdasan buatan|artificial intelligence)\b", re.IGNORECASE)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)
# Behind a TLS-inspecting proxy, trust the CA bundle in SSL_CERT_FILE instead of certifi.
TLS_VERIFY = os.environ.get("SSL_CERT_FILE") or True

MANIFEST_SCHEMA = {
    "sample_id": pl.Utf8,
    "review_url": pl.Utf8,
    "claim_date": pl.Utf8,
    "claim_text": pl.Utf8,
    "review_title": pl.Utf8,
    "publisher_site": pl.Utf8,
    "textual_rating": pl.Utf8,
    "matched_keywords": pl.List(pl.Utf8),
    "ai_mentioned_in_review": pl.Boolean,
    "platform": pl.Utf8,
    "source_url": pl.Utf8,
    "archive_url": pl.Utf8,
    "downloaded_from": pl.Utf8,
    "video_path": pl.Utf8,
    "sha256": pl.Utf8,
    "duration": pl.Float64,
    "width": pl.Int64,
    "height": pl.Int64,
    "upload_date": pl.Utf8,
    "subtitle_files": pl.List(pl.Utf8),
}


def http_get(url: str, timeout: int = 60) -> requests.Response:
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, verify=TLS_VERIFY)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_reviews(keyword: str, start: str, end: str, topic_keywords: list[str]) -> pl.DataFrame:
    """One row per review URL, oldest first."""
    claim_date = pl.coalesce("claim_date", "review_date").str.slice(0, 10)
    return (
        pl.read_parquet(SNAPSHOT_PATH)
        .filter(
            pl.col("claim_text").str.to_lowercase().str.contains(keyword.lower(), literal=True)
            & claim_date.is_between(pl.lit(start), pl.lit(end))
        )
        .group_by("review_url")
        .agg(
            claim_date.first().alias("claim_date"),
            pl.col("claim_text").first(),
            pl.col("review_title").first(),
            pl.col("publisher_site").first(),
            pl.col("textual_rating").first(),
            pl.col("keyword").unique().sort().alias("matched_keywords"),
        )
        .filter(pl.col("matched_keywords").list.set_intersection(topic_keywords).list.len() > 0)
        .sort("claim_date", "review_url")
    )


def fetch_article(review_url: str, claim_date: str) -> str | None:
    """Article HTML from the live site, or from the Wayback Machine when the site refuses."""
    for url in (review_url, f"https://web.archive.org/web/{claim_date.replace('-', '')}id_/{review_url}"):
        try:
            response = http_get(url)
        except requests.RequestException as error:
            print(f"    article unavailable at {url}: {error}")
            continue
        if response.status_code == 200:
            return html.unescape(response.text.replace("\\/", "/"))
        print(f"    article unavailable at {url}: HTTP {response.status_code}")
    return None


def perma_original(archive_url: str) -> str | None:
    """Original URL recorded by a perma.cc capture (its `Link: <...>; rel=original` header)."""
    try:
        response = requests.head(archive_url, headers={"User-Agent": USER_AGENT}, timeout=60, verify=TLS_VERIFY)
    except requests.RequestException:
        return None
    return response.links.get("original", {}).get("url")


def match_post(url: str) -> tuple[str, str] | None:
    """(platform, post URL) when url starts with a post URL on a platform in scope."""
    for platform, pattern in POST_PATTERNS.items():
        if match := pattern.match(url):
            return platform, match.group(0)
    return None


def cited_posts(article: str) -> list[dict[str, str | None]]:
    """Social media posts cited by the article, in order of first mention.

    A post cited both directly and through perma.cc keeps its first position and
    gains the capture as archive_url.
    """
    matches = [(m.start(), m.group(0)) for pattern in POST_PATTERNS.values() for m in pattern.finditer(article)]
    matches += [(m.start(), m.group(0)) for m in PERMA_PATTERN.finditer(article)]
    posts: dict[str, dict[str, str | None]] = {}
    for _, url in sorted(set(matches)):
        archive_url = url if PERMA_PATTERN.fullmatch(url) else None
        post = match_post((perma_original(url) or "") if archive_url else url)
        if post is None:
            continue
        platform, source_url = post
        entry = posts.setdefault(source_url, {"platform": platform, "source_url": source_url, "archive_url": None})
        entry["archive_url"] = entry["archive_url"] or archive_url
    return list(posts.values())


class SilentLogger:
    def debug(self, message: str) -> None: ...
    def info(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...


def download_live(source_url: str, sample_dir: Path) -> dict[str, object]:
    options = {
        "outtmpl": str(sample_dir / "video.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["all"],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Failures are reported by download_post; keep yt-dlp from printing them again.
        "logger": SilentLogger(),
    }
    if TLS_VERIFY is not True:
        options["compat_opts"] = {"no-certifi"}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(source_url, download=True)
    subtitles = info.get("requested_subtitles") or {}
    return {
        "downloaded_from": "live",
        "video_path": Path(info["requested_downloads"][0]["filepath"]),
        "upload_date": info.get("upload_date"),
        "subtitle_files": sorted(
            str(Path(subtitle["filepath"]).relative_to(sample_dir))
            for subtitle in subtitles.values()
            if subtitle.get("filepath")
        ),
    }


def download_archived(archive_url: str, sample_dir: Path) -> dict[str, object]:
    """Keep the perma.cc WARC and extract the first complete video response from it."""
    page = http_get(archive_url)
    page.raise_for_status()
    match = PERMA_WARC_SOURCE.search(page.text)
    if match is None:
        raise LookupError(f"no WARC source on {archive_url}")
    # The page embeds a JavaScript string with \u escapes; it is a signed, expiring path.
    warc_source = json.loads(f'"{match.group(1)}"')
    warc = http_get(f"{PERMA_REPLAY_ORIGIN}/{warc_source}", timeout=120)
    warc.raise_for_status()
    warc_path = sample_dir / "archive.warc.gz"
    warc_path.write_bytes(warc.content)
    with warc_path.open("rb") as handle:
        for record in ArchiveIterator(handle):
            if (
                record.rec_type != "response"
                or record.http_headers is None
                or not (record.http_headers.get_header("Content-Type") or "").startswith("video/")
            ):
                continue
            body = record.content_stream().read()
            if is_complete_mp4(record.http_headers, body):
                video_path = sample_dir / "video.mp4"
                video_path.write_bytes(body)
                return {"downloaded_from": "archive", "video_path": video_path, "upload_date": None, "subtitle_files": []}
    raise LookupError(f"no complete video response in {archive_url}")


def is_complete_mp4(headers, body: bytes) -> bool:
    """A whole MP4 file: a 200 response, or a 206 whose range spans the entire file.

    TikTok also serves captions labelled video/mp4, so the body must start with an
    MP4 `ftyp` box; callers pass only video/* responses, since AVIF images share it.
    """
    if body[4:8] != b"ftyp":
        return False
    status = headers.get_statuscode()
    if status == "200":
        return True
    content_range = re.fullmatch(r"bytes 0-(\d+)/(\d+)", headers.get_header("Content-Range") or "")
    return status == "206" and content_range is not None and int(content_range[1]) + 1 == int(content_range[2])


def download_post(post: dict[str, str | None], sample_dir: Path) -> dict[str, object] | None:
    """Download the post live, else from its perma.cc capture; None when neither has a video."""
    attempts = [("live", lambda: download_live(post["source_url"], sample_dir))]
    if post["archive_url"] is not None:
        attempts.append(("archive", lambda: download_archived(post["archive_url"], sample_dir)))
    for name, attempt in attempts:
        sample_dir.mkdir(parents=True)
        try:
            return attempt()
        except (yt_dlp.utils.DownloadError, requests.RequestException, LookupError) as error:
            print(f"    {name} failed: {str(error).splitlines()[0][:160]}")
            shutil.rmtree(sample_dir)
    return None


def video_properties(video_path: Path) -> dict[str, object]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration",
            "-of", "json", str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    properties = json.loads(probe.stdout)
    return {
        "duration": float(properties["format"]["duration"]),
        "width": properties["streams"][0]["width"],
        "height": properties["streams"][0]["height"],
    }


def collect(reviews: pl.DataFrame, n: int, run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for review in reviews.iter_rows(named=True):
        print(f"[{len(rows)}/{n}] {review['claim_date']} {review['review_url']}", flush=True)
        article = fetch_article(review["review_url"], review["claim_date"])
        if article is None:
            continue
        for post in cited_posts(article):
            sample_id = f"{len(rows) + 1:02d}-{post['platform']}"
            print(f"  trying {post['source_url']}" + (f" (archive {post['archive_url']})" if post["archive_url"] else ""))
            downloaded = download_post(post, run_dir / sample_id)
            if downloaded is None:
                continue
            video_path = downloaded.pop("video_path")
            rows.append({
                **review,
                **post,
                **downloaded,
                **video_properties(video_path),
                "sample_id": sample_id,
                "ai_mentioned_in_review": bool(AI_MENTION.search(f"{review['review_title']} {review['claim_text']}")),
                "video_path": str(video_path.relative_to(run_dir)),
                "sha256": file_sha256(video_path),
            })
            print(f"  saved {sample_id} ({downloaded['downloaded_from']})", flush=True)
            break
        if len(rows) == n:
            return rows
    raise RuntimeError(f"Only {len(rows)} of {n} reviews had a downloadable video; widen the filters.")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=positive_int, default=5, help="Reviews to collect (default: 5).")
    parser.add_argument("--keyword", default="video", help="Substring required in claim_text (default: video).")
    parser.add_argument("--start", default=DEFAULT_START, help=f"First claim date, YYYY-MM-DD (default: {DEFAULT_START}).")
    parser.add_argument("--end", default=DEFAULT_END, help=f"Last claim date, YYYY-MM-DD (default: {DEFAULT_END}).")
    parser.add_argument(
        "--topic-keywords", nargs="+", default=list(DEFAULT_TOPIC_KEYWORDS),
        help="Snapshot retrieval keywords; a review must match at least one (default: election keywords).",
    )
    arguments = parser.parse_args()

    reviews = select_reviews(arguments.keyword, arguments.start, arguments.end, arguments.topic_keywords)
    print(f"{reviews.height} candidate reviews in {SNAPSHOT_PATH.relative_to(REPO_ROOT)}", flush=True)

    # Each run gets its own folder; it appears only when all n samples succeed.
    run_id = str(int(time.time()))
    run_dir = OUTPUT_ROOT / run_id
    staging_dir = OUTPUT_ROOT / f".{run_id}.partial"
    staging_dir.mkdir(parents=True)
    try:
        rows = collect(reviews, arguments.n, staging_dir)
        manifest = pl.DataFrame(rows, schema=MANIFEST_SCHEMA, strict=True)
        manifest.write_parquet(staging_dir / "manifest.parquet", compression="zstd")
        os.replace(staging_dir, run_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    print(f"Wrote {manifest.height} samples to {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
