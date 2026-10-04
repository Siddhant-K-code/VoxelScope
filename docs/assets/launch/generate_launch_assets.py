#!/usr/bin/env python3
"""Generate the VoxelScope public launch visuals from committed evidence."""

from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

try:
    import imageio.v3 as iio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:
    raise SystemExit(
        "Asset dependencies are isolated from the product. Run:\n"
        "uv run --with Pillow --with imageio --with numpy python "
        "docs/assets/launch/generate_launch_assets.py"
    ) from error


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent

WIDTH = 1600
HEIGHT = 900
GIF_WIDTH = 1200
GIF_HEIGHT = 720

BACKGROUND = "#0B1320"
PANEL = "#111D2E"
PANEL_ALT = "#16263A"
TEXT = "#F4F7FA"
MUTED = "#9FB0C3"
ORANGE = "#FF9F1C"
TEAL = "#2EC4B6"
BLUE = "#69A7FF"
GRID = "#2A3A50"

SOURCE_PATHS = (
    "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/result-summary.json",
    "research/gbm-evidence-communication-qwen3-8b-q8-study-v1/failure-analysis.json",
    "research/gbm-evidence-communication-qwen3-8b-q8-study-v4/result-summary.json",
    "research/evidence-communication-v4-lexical-audit-v1/audit.json",
    "research/evidence-communication-v5-discourse-planner-contract-v1/design-contract.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("DejaVuSans-Bold.ttf", "Arial Bold.ttf") if bold else ("DejaVuSans.ttf", "Arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    font_value: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
    width: int,
    spacing: int = 8,
) -> int:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font_value)[2] > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    x, y = xy
    line_height = draw.textbbox((0, 0), "Ag", font=font_value)[3]
    for line in lines:
        draw.text((x, y), line, font=font_value, fill=fill)
        y += line_height + spacing
    return y


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str = GRID,
    width: int = 2,
    radius: int = 24,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def save_hero_png(path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((80, 64), "THE DETERMINISTIC TRUST BOUNDARY", font=font(24, bold=True), fill=ORANGE)
    draw.text((80, 108), "Structure fixed the interface.", font=font(58, bold=True), fill=TEXT)
    draw.text(
        (80, 176), "Exact optimization removed the model.", font=font(58, bold=True), fill=TEXT
    )
    draw.text(
        (80, 258),
        "Code owns evidence identity, semantics, gates, prose, custody, replay, and publication.",
        font=font(25),
        fill=MUTED,
    )

    nodes = (
        (80, 370, 290, 510, "Evidence\nidentity", "hash-bound"),
        (330, 370, 540, 510, "Typed\nsemantics", "facts + caveats"),
        (580, 370, 790, 510, "Eligibility\ngate", "before inference"),
        (830, 370, 1040, 510, "Exact\noptimizer", "complete search"),
        (1080, 370, 1290, 510, "Code\nrenderer", "canonical prose"),
        (1330, 370, 1520, 510, "Closed\nreceipt", "atomic custody"),
    )
    for index, (x1, y1, x2, y2, title, detail) in enumerate(nodes):
        color = ORANGE if index == 2 else TEAL if index >= 3 else BLUE
        rounded_box(draw, (x1, y1, x2, y2), fill=PANEL, outline=color, width=3)
        title_y = y1 + 27
        for line in title.splitlines():
            draw.text((x1 + 24, title_y), line, font=font(27, bold=True), fill=TEXT)
            title_y += 35
        draw.text((x1 + 24, y2 - 38), detail, font=font(18), fill=MUTED)
        if index < len(nodes) - 1:
            draw.line((x2 + 10, 440, nodes[index + 1][0] - 10, 440), fill=GRID, width=4)
            draw.polygon(
                (
                    (nodes[index + 1][0] - 20, 432),
                    (nodes[index + 1][0] - 10, 440),
                    (nodes[index + 1][0] - 20, 448),
                ),
                fill=GRID,
            )

    rounded_box(draw, (580, 575, 1040, 718), fill=PANEL_ALT, outline=ORANGE, width=4)
    draw.text((612, 603), "MODEL PATH", font=font(22, bold=True), fill=ORANGE)
    draw.text(
        (612, 642), "Ineligible under the declared objective", font=font(22, bold=True), fill=TEXT
    )
    draw.text(
        (612, 681),
        "No prompt, API call, model action, or network action",
        font=font(18),
        fill=MUTED,
    )
    draw.line((685, 510, 685, 575), fill=ORANGE, width=3)

    rounded_box(draw, (1080, 575, 1520, 718), fill=PANEL_ALT, outline=TEAL, width=3)
    draw.text((1112, 603), "OFFLINE REPLAY", font=font(22, bold=True), fill=TEAL)
    draw.text((1112, 642), "Rebuild the same exact winner", font=font(27, bold=True), fill=TEXT)
    draw.text((1112, 681), "No model or network required", font=font(18), fill=MUTED)
    draw.line((1425, 510, 1425, 575), fill=TEAL, width=3)

    draw.text(
        (80, 820),
        "A model should earn its place against a strong deterministic baseline.",
        font=font(28, bold=True),
        fill=TEXT,
    )
    image.save(path, optimize=True)


def save_evolution_png(path: Path, evidence: dict[str, Any]) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((80, 60), "V3  ->  V4  ->  V5", font=font(24, bold=True), fill=ORANGE)
    draw.text(
        (80, 104),
        "The interface changed. The evidence types changed.",
        font=font(50, bold=True),
        fill=TEXT,
    )
    draw.text(
        (80, 170),
        "Contract evolution, not model-performance ordering.",
        font=font(24),
        fill=MUTED,
    )

    cards = (
        (
            80,
            "v3",
            "OBSERVED MODEL EVIDENCE",
            "Model constructs typed claims",
            (
                "0 / 18 accepted",
                "4 / 18 invalid outer outputs",
                "0 / 9 semantic variance",
                "repeatable construction/verifier mismatch",
            ),
            BLUE,
        ),
        (
            560,
            "v4",
            "OBSERVED MODEL EVIDENCE",
            "Model selects 15 code-owned skeletons",
            (
                "270 / 270 tasks matched",
                "0 / 18 invalid outputs",
                "108 lexical exclusions",
                "3 / 9 semantic variance",
            ),
            ORANGE,
        ),
        (
            1040,
            "v5",
            "CONTRACT + CONTROL EVIDENCE",
            "No model role in publication",
            (
                "finite feasible plan set",
                "exact lexicographic optimum",
                "model actions: 0",
                "network actions: 0",
            ),
            TEAL,
        ),
    )
    for x, version, evidence_type, role, values, color in cards:
        rounded_box(draw, (x, 250, x + 400, 690), fill=PANEL, outline=color, width=4)
        draw.text((x + 30, 280), version, font=font(54, bold=True), fill=color)
        draw.text((x + 30, 352), evidence_type, font=font(17, bold=True), fill=MUTED)
        y = draw_wrapped(
            draw,
            (x + 30, 398),
            role,
            font_value=font(25, bold=True),
            fill=TEXT,
            width=340,
            spacing=5,
        )
        y += 24
        for value in values:
            draw.ellipse((x + 32, y + 8, x + 42, y + 18), fill=color)
            y = draw_wrapped(
                draw,
                (x + 58, y),
                value,
                font_value=font(20),
                fill=TEXT,
                width=300,
                spacing=3,
            )
            y += 13

    draw.line((480, 470, 540, 470), fill=GRID, width=4)
    draw.polygon(((530, 462), (540, 470), (530, 478)), fill=GRID)
    draw.line((960, 470, 1020, 470), fill=GRID, width=4)
    draw.polygon(((1010, 462), (1020, 470), (1010, 478)), fill=GRID)

    v3_fact = evidence["v3"]["emitted_fact_coverage"]
    v3_caveat = evidence["v3"]["emitted_caveat_coverage"]
    v4_fact = evidence["v4"]["emitted_fact_coverage"]
    v4_caveat = evidence["v4"]["emitted_caveat_coverage"]
    rounded_box(draw, (80, 740, 1520, 838), fill=PANEL_ALT, outline=GRID)
    draw.text((112, 760), "ONLY MATCHING V3/V4 ENDPOINTS", font=font(18, bold=True), fill=MUTED)
    draw.text(
        (112, 795),
        f"Emitted fact coverage  v3: {v3_fact}  |  v4: {v4_fact}",
        font=font(22, bold=True),
        fill=TEXT,
    )
    draw.text(
        (790, 795),
        f"Emitted caveat coverage  v3: {v3_caveat}  |  v4: {v4_caveat}",
        font=font(22, bold=True),
        fill=TEXT,
    )
    image.save(path, optimize=True)


def svg_text(
    x: int,
    y: int,
    value: str,
    *,
    size: int,
    fill: str = TEXT,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">'
        f"{escape(value)}</text>"
    )


def svg_box(x: int, y: int, width: int, height: int, outline: str, fill: str = PANEL) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="24" '
        f'fill="{fill}" stroke="{outline}" stroke-width="3"/>'
    )


def save_hero_svg(path: Path) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        'viewBox="0 0 1600 900" role="img" aria-labelledby="hero-title hero-desc">',
        '<title id="hero-title">VoxelScope deterministic trust boundary</title>',
        '<desc id="hero-desc">Evidence identity and typed semantics enter a pre-inference '
        "eligibility gate. The model path is ineligible. An exact optimizer, code-owned "
        "renderer, closed receipt, and offline replay complete publication.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        svg_text(80, 86, "THE DETERMINISTIC TRUST BOUNDARY", size=24, fill=ORANGE, weight=700),
        svg_text(80, 158, "Structure fixed the interface.", size=58, weight=700),
        svg_text(80, 226, "Exact optimization removed the model.", size=58, weight=700),
        svg_text(
            80,
            282,
            "Code owns identity, semantics, gates, prose, custody, replay, and publication.",
            size=25,
            fill=MUTED,
        ),
    ]
    nodes = (
        (80, "Evidence", "identity", "hash-bound", BLUE),
        (330, "Typed", "semantics", "facts + caveats", BLUE),
        (580, "Eligibility", "gate", "before inference", ORANGE),
        (830, "Exact", "optimizer", "complete search", TEAL),
        (1080, "Code", "renderer", "canonical prose", TEAL),
        (1330, "Closed", "receipt", "atomic custody", TEAL),
    )
    for index, (x, first, second, detail, color) in enumerate(nodes):
        parts.extend(
            (
                svg_box(x, 370, 190 if index == 5 else 210, 140, color),
                svg_text(x + 24, 414, first, size=27, weight=700),
                svg_text(x + 24, 449, second, size=27, weight=700),
                svg_text(x + 24, 486, detail, size=18, fill=MUTED),
            )
        )
        if index < len(nodes) - 1:
            end = nodes[index + 1][0] - 10
            start = x + (190 if index == 5 else 210) + 10
            parts.append(
                f'<path d="M {start} 440 H {end}" stroke="{GRID}" stroke-width="4" '
                'marker-end="url(#arrow)"/>'
            )
    parts.insert(
        4,
        f'<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        f'orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{GRID}"/></marker></defs>',
    )
    parts.extend(
        (
            svg_box(580, 575, 460, 143, ORANGE, PANEL_ALT),
            svg_text(612, 621, "MODEL PATH", size=22, fill=ORANGE, weight=700),
            svg_text(612, 662, "Ineligible under the declared objective", size=22, weight=700),
            svg_text(612, 696, "No prompt, API, model, or network action", size=18, fill=MUTED),
            f'<path d="M685 510 V575" stroke="{ORANGE}" stroke-width="3"/>',
            svg_box(1080, 575, 440, 143, TEAL, PANEL_ALT),
            svg_text(1112, 621, "OFFLINE REPLAY", size=22, fill=TEAL, weight=700),
            svg_text(1112, 662, "Rebuild the same exact winner", size=27, weight=700),
            svg_text(1112, 696, "No model or network required", size=18, fill=MUTED),
            f'<path d="M1425 510 V575" stroke="{TEAL}" stroke-width="3"/>',
            svg_text(
                80,
                844,
                "A model should earn its place against a strong deterministic baseline.",
                size=28,
                weight=700,
            ),
            "</svg>",
        )
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def save_evolution_svg(path: Path, evidence: dict[str, Any]) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        'viewBox="0 0 1600 900" role="img" aria-labelledby="evolution-title evolution-desc">',
        '<title id="evolution-title">VoxelScope v3 to v5 evidence communication evolution</title>',
        '<desc id="evolution-desc">Version 3 let the model construct typed claims and exposed '
        "a repeatable interface mismatch. Version 4 used code-owned skeletons and removed "
        "invalid outputs in the observed study, while lexical exclusions and semantic variance "
        "remained. Version 5 used exact deterministic optimization with no model role.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        svg_text(80, 84, "V3  ->  V4  ->  V5", size=24, fill=ORANGE, weight=700),
        svg_text(
            80, 154, "The interface changed. The evidence types changed.", size=50, weight=700
        ),
        svg_text(
            80,
            196,
            "Contract evolution, not model-performance ordering.",
            size=24,
            fill=MUTED,
        ),
    ]
    cards = (
        (
            80,
            "v3",
            "OBSERVED MODEL EVIDENCE",
            "Model constructs typed claims",
            (
                "0 / 18 accepted",
                "4 / 18 invalid outer outputs",
                "0 / 9 semantic variance",
                "repeatable construction/verifier mismatch",
            ),
            BLUE,
        ),
        (
            560,
            "v4",
            "OBSERVED MODEL EVIDENCE",
            "Model selects 15 code-owned skeletons",
            (
                "270 / 270 tasks matched",
                "0 / 18 invalid outputs",
                "108 lexical exclusions",
                "3 / 9 semantic variance",
            ),
            ORANGE,
        ),
        (
            1040,
            "v5",
            "CONTRACT + CONTROL EVIDENCE",
            "No model role in publication",
            (
                "finite feasible plan set",
                "exact lexicographic optimum",
                "model actions: 0",
                "network actions: 0",
            ),
            TEAL,
        ),
    )
    for x, version, kind, role, values, color in cards:
        parts.extend(
            (
                svg_box(x, 250, 400, 440, color),
                svg_text(x + 30, 330, version, size=54, fill=color, weight=700),
                svg_text(x + 30, 374, kind, size=17, fill=MUTED, weight=700),
                svg_text(x + 30, 424, role, size=24, weight=700),
            )
        )
        y = 486
        for value in values:
            parts.append(f'<circle cx="{x + 37}" cy="{y - 7}" r="5" fill="{color}"/>')
            parts.append(svg_text(x + 58, y, value, size=20))
            y += 54
    parts.extend(
        (
            f'<path d="M480 470 H540" stroke="{GRID}" stroke-width="4"/>',
            f'<path d="M530 462 L540 470 L530 478" fill="{GRID}"/>',
            f'<path d="M960 470 H1020" stroke="{GRID}" stroke-width="4"/>',
            f'<path d="M1010 462 L1020 470 L1010 478" fill="{GRID}"/>',
            svg_box(80, 740, 1440, 98, GRID, PANEL_ALT),
            svg_text(112, 785, "ONLY MATCHING V3/V4 ENDPOINTS", size=18, fill=MUTED, weight=700),
            svg_text(
                112,
                820,
                f"Emitted fact coverage  v3: {evidence['v3']['emitted_fact_coverage']}  |  "
                f"v4: {evidence['v4']['emitted_fact_coverage']}",
                size=21,
                weight=700,
            ),
            svg_text(
                790,
                820,
                f"Emitted caveat coverage  v3: {evidence['v3']['emitted_caveat_coverage']}  |  "
                f"v4: {evidence['v4']['emitted_caveat_coverage']}",
                size=21,
                weight=700,
            ),
            "</svg>",
        )
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def terminal_frame(
    lines: list[tuple[str, str]],
    *,
    annotation: tuple[tuple[int, int, int, int], str] | None = None,
    annotation_alpha: float = 1.0,
    hero: str | None = None,
) -> Image.Image:
    image = Image.new("RGB", (GIF_WIDTH, GIF_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    rounded_box(draw, (28, 24, GIF_WIDTH - 28, GIF_HEIGHT - 24), fill="#0A0F18", outline=GRID)
    draw.rounded_rectangle(
        (28, 24, GIF_WIDTH - 28, 76),
        radius=20,
        fill=PANEL,
        outline=GRID,
        width=2,
    )
    draw.rectangle((28, 55, GIF_WIDTH - 28, 76), fill=PANEL)
    for x in (58, 86, 114):
        draw.ellipse((x, 43, x + 13, 56), fill="#526173")
    draw.text(
        (148, 39), "VoxelScope v5 | deterministic control", font=font(20, bold=True), fill=TEXT
    )
    draw.text((950, 40), "OFFLINE", font=font(18, bold=True), fill=TEAL)

    y = 105
    terminal_font = font(18)
    for value, color in lines:
        draw.text((58, y), value, font=terminal_font, fill=color)
        y += 30

    if annotation is not None:
        box, label = annotation
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        rgba = (255, 159, 28, int(255 * annotation_alpha))
        overlay_draw.rounded_rectangle(box, radius=14, outline=rgba, width=5)
        label_x = box[2] - 4
        label_y = box[1] - 44
        overlay_draw.line(
            (box[2], box[1] + 12, label_x + 24, label_y + 28),
            fill=rgba,
            width=5,
        )
        overlay_draw.text(
            (label_x + 30, label_y),
            label,
            font=font(26, bold=True),
            fill=rgba,
            stroke_width=1,
            stroke_fill=rgba,
        )
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    if hero is not None:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 190))
        image = Image.alpha_composite(image.convert("RGBA"), overlay)
        hero_draw = ImageDraw.Draw(image)
        hero_font = font(64, bold=True)
        for index, line in enumerate(hero.splitlines()):
            bbox = hero_draw.textbbox((0, 0), line, font=hero_font)
            hero_draw.text(
                ((GIF_WIDTH - (bbox[2] - bbox[0])) // 2, 262 + index * 82),
                line,
                font=hero_font,
                fill=ORANGE,
            )
        image = image.convert("RGB")
    return image


def append_typing(
    frames: list[np.ndarray[Any, Any]],
    durations: list[int],
    settled: list[tuple[str, str]],
    line: str,
    *,
    chunk_size: int = 9,
) -> None:
    for end in range(chunk_size, len(line) + chunk_size, chunk_size):
        current = line[: min(end, len(line))]
        frames.append(np.asarray(terminal_frame(settled + [(current, TEXT)])))
        durations.append(100)
    settled.append((line, TEXT))


def save_demo_gif(path: Path) -> tuple[int, list[int]]:
    frames: list[np.ndarray[Any, Any]] = []
    durations: list[int] = []
    lines: list[tuple[str, str]] = [
        ("Safe synthetic control. Dependencies already present.", MUTED),
        ("No model process. No model API. No network path.", MUTED),
        ("", TEXT),
    ]
    frames.append(np.asarray(terminal_frame(lines)))
    durations.append(900)

    append_typing(
        frames,
        durations,
        lines,
        "$ uv run python -m voxelscope.evidence_communication_v5_cli fixture-compile \\",
    )
    append_typing(
        frames,
        durations,
        lines,
        "    --repository-root . --run-id demo --output build/v5-control",
    )
    lines.extend(
        (
            ("model_eligibility=model_inference_forbidden", ORANGE),
            ("model_actions=0  network_actions=0", TEAL),
            ("v5_publication_status=closed", TEAL),
            ("", TEXT),
        )
    )
    frames.append(np.asarray(terminal_frame(lines)))
    durations.append(650)
    compile_box = (48, 288, 520, 352)
    for alpha in (0.5, 1.0):
        frames.append(
            np.asarray(
                terminal_frame(
                    lines,
                    annotation=(compile_box, "stopped before inference"),
                    annotation_alpha=alpha,
                )
            )
        )
        durations.append(100)
    frames.append(
        np.asarray(terminal_frame(lines, annotation=(compile_box, "stopped before inference")))
    )
    durations.append(900)

    append_typing(
        frames,
        durations,
        lines,
        "$ jq -r .terminal_state build/v5-control/receipt.json",
    )
    lines.extend((("closed", TEAL), ("", TEXT)))
    frames.append(np.asarray(terminal_frame(lines)))
    durations.append(650)
    receipt_box = (48, 396, 180, 430)
    for alpha in (0.5, 1.0):
        frames.append(
            np.asarray(
                terminal_frame(
                    lines,
                    annotation=(receipt_box, "receipt closes custody"),
                    annotation_alpha=alpha,
                )
            )
        )
        durations.append(100)
    frames.append(
        np.asarray(terminal_frame(lines, annotation=(receipt_box, "receipt closes custody")))
    )
    durations.append(900)

    append_typing(
        frames,
        durations,
        lines,
        "$ uv run python -m voxelscope.evidence_communication_v5_cli replay \\",
    )
    append_typing(
        frames,
        durations,
        lines,
        "    --repository-root . --bundle build/v5-control",
    )
    lines.extend(
        (
            ("v5_replay=verified  v5_publication_status=closed", TEAL),
            ("model_actions=0  network_actions=0", TEAL),
        )
    )
    frames.append(np.asarray(terminal_frame(lines)))
    durations.append(700)
    replay_box = (48, 516, 690, 590)
    for alpha in (0.5, 1.0):
        frames.append(
            np.asarray(
                terminal_frame(
                    lines,
                    annotation=(replay_box, "exact offline reconstruction"),
                    annotation_alpha=alpha,
                )
            )
        )
        durations.append(100)
    frames.append(
        np.asarray(terminal_frame(lines, annotation=(replay_box, "exact offline reconstruction")))
    )
    durations.append(1100)
    frames.append(np.asarray(terminal_frame(lines, hero="Exact replay.\nNo model. No network.")))
    durations.append(1800)

    iio.imwrite(path, frames, duration=durations, loop=0)
    actual_durations: list[int] = []
    with Image.open(path) as gif:
        for frame_index in range(gif.n_frames):
            gif.seek(frame_index)
            actual_durations.append(int(gif.info["duration"]))
        return gif.n_frames, actual_durations


def evidence_values() -> dict[str, dict[str, str]]:
    v3 = load_json(SOURCE_PATHS[0])["metrics"]
    v4 = load_json(SOURCE_PATHS[2])["endpoint_results"]
    return {
        "v3": {
            "emitted_fact_coverage": (
                f"{v3['emitted_fact_coverage']['numerator']} / "
                f"{v3['emitted_fact_coverage']['denominator']}"
            ),
            "emitted_caveat_coverage": (
                f"{v3['emitted_caveat_coverage']['numerator']} / "
                f"{v3['emitted_caveat_coverage']['denominator']}"
            ),
        },
        "v4": {
            "emitted_fact_coverage": (
                f"{v4['emitted_fact_coverage']['numerator']} / "
                f"{v4['emitted_fact_coverage']['denominator']}"
            ),
            "emitted_caveat_coverage": (
                f"{v4['emitted_caveat_coverage']['numerator']} / "
                f"{v4['emitted_caveat_coverage']['denominator']}"
            ),
        },
    }


def write_provenance(frame_count: int, durations: list[int]) -> None:
    outputs: list[dict[str, Any]] = []
    for name in (
        "architecture-trust-boundary.svg",
        "architecture-trust-boundary.png",
        "v3-v4-v5-evolution.svg",
        "v3-v4-v5-evolution.png",
        "v5-offline-workflow.gif",
    ):
        path = OUTPUT_DIR / name
        record: dict[str, Any] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix in {".png", ".gif"}:
            with Image.open(path) as image:
                record["dimensions"] = list(image.size)
                record["format"] = image.format
        else:
            record["dimensions"] = [WIDTH, HEIGHT]
            record["format"] = "SVG"
        if path.suffix == ".gif":
            record["frame_count"] = frame_count
            record["durations_ms"] = durations
            record["loop"] = 0
            record["assembly"] = "imageio.v3.imwrite"
        outputs.append(record)

    provenance = {
        "schema_version": "voxelscope/public-launch-assets/v1",
        "generator": "docs/assets/launch/generate_launch_assets.py",
        "regeneration_command": (
            "uv run --with Pillow --with imageio --with numpy python "
            "docs/assets/launch/generate_launch_assets.py"
        ),
        "source_kind": "committed synthetic study, audit, contract, and control evidence",
        "model_execution": False,
        "network_execution": False,
        "desktop_capture": False,
        "gif_method": "deterministic scripted terminal animation assembled with imageio",
        "annotation_color": ORANGE,
        "inputs": [{"path": path, "sha256": sha256_file(ROOT / path)} for path in SOURCE_PATHS],
        "outputs": outputs,
    }
    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = evidence_values()
    save_hero_svg(OUTPUT_DIR / "architecture-trust-boundary.svg")
    save_hero_png(OUTPUT_DIR / "architecture-trust-boundary.png")
    save_evolution_svg(OUTPUT_DIR / "v3-v4-v5-evolution.svg", evidence)
    save_evolution_png(OUTPUT_DIR / "v3-v4-v5-evolution.png", evidence)
    frame_count, durations = save_demo_gif(OUTPUT_DIR / "v5-offline-workflow.gif")
    write_provenance(frame_count, durations)


if __name__ == "__main__":
    main()
