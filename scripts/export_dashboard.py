"""Export a standalone, explicitly dated dashboard snapshot for offline review."""

import argparse
import json
from pathlib import Path

from fireworkshack.dashboard import build_status


def export_dashboard(root: Path, output: Path) -> Path:
    state = build_status(root)
    html = (root / "web/index.html").read_text()
    css = (root / "web/styles.css").read_text()
    javascript = (root / "web/app.js").read_text()
    payload = json.dumps(state, ensure_ascii=True).replace("<", "\\u003c")
    html = html.replace(
        '<link rel="stylesheet" href="/styles.css">', f"<style>{css}</style>"
    )
    html = html.replace('<script src="/app.js" defer></script>', "")
    html = html.replace(
        "</body>",
        f"<script>window.FIREWORKSHACK_SNAPSHOT = {payload};</script>\n<script>{javascript}</script>\n</body>",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/dashboard.html"))
    args = parser.parse_args()
    print(export_dashboard(Path(__file__).resolve().parents[1], args.output))
