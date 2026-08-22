#!/usr/bin/env python3
"""Build a sorted grid movie from per-protein ChimeraX cartoon movies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def fit_image(frame: np.ndarray, box_w: int, box_h: int) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", (box_w, box_h), "white")
    out.paste(img, ((box_w - img.width) // 2, (box_h - img.height) // 2))
    return out


def color_for_rank(rank: int, total: int) -> tuple[int, int, int]:
    # Green to orange-red, with a slightly muted middle so labels stay legible.
    t = (rank - 1) / max(1, total - 1)
    green = np.array([45, 126, 81])
    amber = np.array([236, 150, 34])
    red = np.array([203, 69, 42])
    if t < 0.55:
        u = t / 0.55
        c = green * (1 - u) + amber * u
    else:
        u = (t - 0.55) / 0.45
        c = amber * (1 - u) + red * u
    return tuple(int(x) for x in c)


def read_order(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["rank"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-csv", type=Path, default=ROOT / "runs/partial_benchmark/chimerax_cartoon_accuracy_grid_order.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/partial_benchmark/chimerax_cartoon_accuracy_grid.mp4")
    ap.add_argument("--preview", type=Path, default=ROOT / "runs/partial_benchmark/chimerax_cartoon_accuracy_grid_preview.png")
    ap.add_argument("--frames", type=int, default=94)
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    rows = read_order(args.order_csv)
    readers = []
    for row in rows:
        movie = Path(row["movie"])
        if not movie.is_absolute():
            movie = ROOT / movie
        readers.append(imageio.get_reader(str(movie)))

    width, height = 1920, 2160
    cols, rows_n = 4, 5
    header_h = 164
    tile_w = width // cols
    tile_h = (height - header_h) // rows_n
    video_h = 285
    label_h = tile_h - video_h
    pad = 12

    title_font = font(48, bold=True)
    subtitle_font = font(22)
    small_font = font(15)
    rank_font = font(30, bold=True)
    metric_font = font(19)

    def draw_frame(frame_idx: int) -> Image.Image:
        canvas = Image.new("RGB", (width, height), (248, 250, 252))
        draw = ImageDraw.Draw(canvas)
        draw.text((38, 26), "Decoded folding simulations ordered by structural error", fill=(22, 30, 42), font=title_font)
        draw.text(
            (40, 86),
            "Most accurate ESMFold predictions are upper-left; failure cases move toward lower-right. Cartoon geometry follows decoded C-alpha trajectories.",
            fill=(67, 79, 96),
            font=subtitle_font,
        )

        grad_x0, grad_y0, grad_w, grad_h = width - 560, 34, 360, 16
        for i in range(grad_w):
            t = i / max(1, grad_w - 1)
            if t < 0.55:
                u = t / 0.55
                a, b = np.array([45, 126, 81]), np.array([236, 150, 34])
            else:
                u = (t - 0.55) / 0.45
                a, b = np.array([236, 150, 34]), np.array([203, 69, 42])
            c = tuple(int(x) for x in (a * (1 - u) + b * u))
            draw.line((grad_x0 + i, grad_y0, grad_x0 + i, grad_y0 + grad_h), fill=c)
        draw.text((grad_x0, grad_y0 + 24), "accurate", fill=(45, 126, 81), font=subtitle_font)
        draw.text((grad_x0 + grad_w - 100, grad_y0 + 24), "inaccurate", fill=(203, 69, 42), font=subtitle_font)

        for i, row in enumerate(rows):
            r, c = divmod(i, cols)
            x = c * tile_w
            y = header_h + r * tile_h
            rank = int(row["rank"])
            border = color_for_rank(rank, len(rows))
            draw.rounded_rectangle(
                (x + 9, y + 9, x + tile_w - 9, y + tile_h - 9),
                radius=12,
                fill=(255, 255, 255),
                outline=border,
                width=4,
            )

            try:
                frame = readers[i].get_data(frame_idx)
            except IndexError:
                frame = readers[i].get_data(-1)
            fitted = fit_image(frame, tile_w - 2 * pad - 8, video_h - pad)
            canvas.paste(fitted, (x + pad + 4, y + pad))

            label_y = y + video_h + 18
            draw.text((x + 22, label_y), f"{rank:02d}. {row['name']}", fill=(22, 30, 42), font=rank_font)
            metric = f"distance MAE {float(row['mae']):.2f} A    pLDDT {float(row['plddt']):.3f}"
            draw.text((x + 23, label_y + 43), metric, fill=(62, 73, 91), font=metric_font)
            draw.text((x + 23, y + tile_h - 30), "blue helix + beta strand  gray loop/other", fill=(102, 112, 128), font=small_font)

        return canvas

    args.out.parent.mkdir(parents=True, exist_ok=True)
    preview = draw_frame(0)
    preview.save(args.preview)

    with imageio.get_writer(str(args.out), fps=args.fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for idx in range(args.frames):
            writer.append_data(np.asarray(draw_frame(idx)))

    for reader in readers:
        reader.close()

    print(args.out)
    print(args.preview)


if __name__ == "__main__":
    main()
