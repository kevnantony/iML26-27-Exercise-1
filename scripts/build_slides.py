"""Build the slides: fill our results into presentation/slides_template.html and export a PDF.

Usage::

    uv run python scripts/build_slides.py            # slides.html + presentation.pdf (PDF needs Google Chrome)
    uv run python scripts/build_slides.py --no-pdf   # slides.html only

Present with presentation/slides.html: arrow keys to step, N for speaker notes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

# Path constants for folders and Google Chrome binary.
PRESENTATION = Path("presentation")
RESULTS = Path("results")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def build_html() -> tuple[Path, int]:
    """Replace the __PLACEHOLDERS__ in the template with numbers from the results files."""
    # Step 1: Read results generated from make_figures.py and analyse_task2.py.
    task1 = json.loads((RESULTS / "slide_data.json").read_text())
    task2 = json.loads((RESULTS / "task2_slide.json").read_text())

    # Step 2: Prepare specific Task 1 subsets needed by the slide visualisations.
    task1_for_slides = {
        # Scatter plots focus on Titanic and Phoneme to demonstrate easy vs hard scenarios.
        "points": [p for p in task1["points"] if p["dataset"] in ("titanic", "phoneme")],
        "summary": task1["summary"],
        "grid": task1["grid"],
    }

    # Step 3: Extract meta-tree parameters for dynamic rendering.
    tree = task2["tree"]
    # Node sample counts: [datasets not interpretable, datasets interpretable].
    counts = tree["value"]
    # Identify child node indices for the root split and the secondary split.
    root_left, root_right = tree["left"][0], tree["right"][0]
    second_left, second_right = tree["left"][root_right], tree["right"][root_right]

    # Step 4: Map template placeholders to concrete values.
    replacements = {
        "__DATA__": json.dumps(task1_for_slides),
        "__DATA2__": json.dumps({k: task2[k] for k in ("rho", "p", "points")}),
        "__T0__": f"{tree['threshold'][0]:.2f}",
        "__T2__": f"{tree['threshold'][root_right]:.2f}",
        "__L1__": f"{counts[root_left][1]} / {sum(counts[root_left])} datasets",
        "__L3__": f"{counts[second_left][0]} / {sum(counts[second_left])} datasets",
        "__L4__": f"{counts[second_right][1]} of {sum(counts[second_right])} simple",
        "__LOO__": f"{100 * task2['loo']:.0f}%",
        "__BASE__": f"{100 * task2['base']:.0f}%",
    }

    # Step 5: Read the raw slide template HTML file.
    html = (PRESENTATION / "slides_template.html").read_text()

    # Step 6: Perform string replacement for all placeholders.
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    # Step 7: Write the rendered HTML file.
    out = PRESENTATION / "slides.html"
    out.write_text(html)

    # Count the number of slide sections in the output HTML.
    n_slides = len(re.findall(r'<section class="slide"', html))
    return out, n_slides


def export_pdf(html: Path, n_slides: int) -> Path:
    """Screenshot every slide (fully built) with headless Chrome and join the images into one PDF."""
    pages = []

    # Use a temporary directory to store intermediate slide screenshots.
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n_slides):
            png = Path(tmp) / f"{i:02d}.png"
            # Append URL hash '#p=i' so JavaScript renders slide index i immediately.
            url = f"{html.resolve().as_uri()}#p={i}"

            # Run headless Google Chrome to render and capture high-res screenshot of the slide.
            subprocess.run(
                [
                    CHROME,
                    "--headless=new",
                    "--hide-scrollbars",
                    "--window-size=1600,900",
                    "--virtual-time-budget=6000",
                    f"--screenshot={png}",
                    url,
                ],
                check=True,
                capture_output=True,
            )

            # Open image and convert to RGB format.
            pages.append(Image.open(png).convert("RGB"))

        # Combine all slide page images into one unified presentation PDF.
        out = PRESENTATION / "presentation.pdf"
        pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150)

    return out


def main() -> None:
    # Set up command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true", help="only build slides.html")
    args = parser.parse_args()

    # Build the HTML slide deck.
    html, n_slides = build_html()
    print(f"wrote {html} ({n_slides} slides)")

    # Unless --no-pdf is passed, render PDF via headless Chrome.
    if not args.no_pdf:
        print(f"wrote {export_pdf(html, n_slides)}")


if __name__ == "__main__":
    main()
