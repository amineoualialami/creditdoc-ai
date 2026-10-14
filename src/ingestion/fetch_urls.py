"""
Fetch URLs listed in data/corpus/urls.txt, extract main content with trafilatura,
save as markdown in data/corpus/.

urls.txt format (one URL per line, # for comments):
    # ACPR
    https://acpr.banque-france.fr/.../assurance-emprunteur.html
    # Service public
    https://www.service-public.fr/particuliers/vosdroits/F2439

Usage:
    uv run python src\ingestion\fetch_urls.py
"""
import re
import sys
from pathlib import Path
from urllib.parse import urlparse
import trafilatura

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data"
URLS_FILE = CORPUS_DIR / "urls.txt"


def slugify(url: str) -> str:
    """Build a readable filename from a URL."""
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "").split(".")[0]
    path = re.sub(r"[^a-zA-Z0-9]+", "-", parsed.path).strip("-")
    path = path[:60] if path else "index"
    return f"{host}_{path}.md"


def fetch_and_save(url: str) -> bool:
    """Fetch URL, extract clean main content, save as markdown."""
    print(f"  Fetching: {url}")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"  ERROR: could not fetch")
        return False

    content = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not content:
        print(f"  ERROR: no extractable content")
        return False

    filename = slugify(url)
    out_path = CORPUS_DIR / filename
    header = f"<!-- source: {url} -->\n\n"
    out_path.write_text(header + content, encoding="utf-8")
    print(f"  Saved: {filename} ({len(content)} chars)")
    return True


def main():
    if not URLS_FILE.exists():
        print(f"ERROR: {URLS_FILE} not found. Create it with one URL per line.")
        sys.exit(1)

    urls = []
    for line in URLS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)

    print(f"Found {len(urls)} URLs to fetch.\n")

    ok, failed = 0, 0
    for url in urls:
        if fetch_and_save(url):
            ok += 1
        else:
            failed += 1

    print(f"\n=== Done ===")
    print(f"Succeeded: {ok}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    main()