"""Render a result card (SVG always, PNG when resvg-py is installed).

The card states the proven result in the tool's own words and tells the reader
how to check it on their own machine. The SVG is a plain string, so it needs no
dependencies; the PNG is rendered from that SVG by ``resvg-py`` (a pure wheel),
which is an optional dependency: without it the SVG is still written and the PNG
is skipped with a note.
"""

from __future__ import annotations

import html
import shlex
from pathlib import Path

W, H = 1200, 630  # a standard social-card size
BG = "#0f172a"  # slate-900
FG = "#f8fafc"  # slate-50
MUTED = "#94a3b8"  # slate-400
ACCENT = "#34d399"  # emerald-400 (AA on the dark background)
BAR_BG = "#1e293b"  # slate-800


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def card_svg(headline_title: str, ratio: float, before: float, after: float, unit: str, loss: str, check_cmd: str, machine: str) -> str:
    """Build the card SVG string. ``headline_title`` is the short name, e.g. the script stem."""
    loss_ok = loss in ("identical", "within tolerance")
    loss_label = "loss unchanged" if loss_ok else f"loss {loss}"
    # bar widths: candidate relative to baseline, baseline capped at the longer bar
    maxv = max(before, after) or 1.0
    bx, bw = 80, 760
    b_base = bw * (before / maxv)
    b_cand = bw * (after / maxv)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<text x="80" y="110" fill="{MUTED}" font-size="30">train-doctor</text>',
        f'<text x="80" y="200" fill="{FG}" font-size="76" font-weight="700">{_esc(headline_title)}</text>',
        f'<text x="80" y="285" fill="{ACCENT}" font-size="104" font-weight="800">{ratio:.2f}x faster</text>',
        # baseline bar
        f'<rect x="{bx}" y="340" width="{bw:.0f}" height="46" rx="8" fill="{BAR_BG}"/>',
        f'<rect x="{bx}" y="340" width="{b_base:.0f}" height="46" rx="8" fill="{MUTED}"/>',
        f'<text x="{bx + bw + 20}" y="373" fill="{MUTED}" font-size="30">before {before:,.0f} {_esc(unit)}</text>',
        # candidate bar
        f'<rect x="{bx}" y="404" width="{bw:.0f}" height="46" rx="8" fill="{BAR_BG}"/>',
        f'<rect x="{bx}" y="404" width="{b_cand:.0f}" height="46" rx="8" fill="{ACCENT}"/>',
        f'<text x="{bx + bw + 20}" y="437" fill="{FG}" font-size="30">after {after:,.0f} {_esc(unit)}</text>',
        f'<text x="80" y="512" fill="{FG}" font-size="34">{_esc(loss_label)}  ·  {_esc(machine)}</text>',
        f'<rect x="80" y="545" width="{W - 160}" height="56" rx="10" fill="{BAR_BG}"/>',
        f'<text x="104" y="582" fill="{MUTED}" font-size="27" font-family="monospace">check yours: {_esc(check_cmd)}</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def render_card(label: str, compare_result: dict, machine: dict, argv: list[str], out_dir: Path) -> dict:
    """Write ``<label>.card.svg`` (always) and ``<label>.card.png`` (if resvg-py is available)."""
    ra = compare_result["ratio"]
    unit = "samples/s" if compare_result["metric"] == "samples_per_s" else "steps/s"
    tp = compare_result["throughput"]
    before, after = tp["baseline"]["median"], tp["candidate"]["median"]
    loss = (compare_result.get("loss_check") or {}).get("status", "unchecked")
    gpu = machine.get("gpu_backend")
    chip = {"apple": "Apple GPU (MPS)", "nvidia": "NVIDIA GPU", "none": machine.get("cpu", "CPU")}.get(gpu, machine.get("cpu", ""))
    mach = f"{chip}"
    # a clean call to action: "python <script>", not the resolved interpreter path
    script = next((a for a in argv if a.endswith(".py")), None)
    shown = ["python", script] if script else argv
    check = "uvx train-doctor fix -- " + shlex.join(shown)
    if len(check) > 82:
        check = check[:79] + "..."
    svg = card_svg(label, ra["point"], before, after, unit, loss, check, mach)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{label}.card.svg"
    svg_path.write_text(svg)
    out: dict = {"svg": str(svg_path)}
    try:
        import resvg_py

        png = resvg_py.svg_to_bytes(svg_string=svg)
        png_path = out_dir / f"{label}.card.png"
        png_path.write_bytes(bytes(png))
        out["png"] = str(png_path)
    except ImportError:
        out["png"] = None
        out["png_note"] = "install the card renderer for PNG output: pip install 'train-doctor[card]' (or uvx --with resvg-py)"
    return out
