"""Helpers for blind judging of icon descriptions: contact sheets of glyphs, the text
of one description as a judge sees it, and a binomial CI. Used by the model pilot
(scripts/v2_pilot.py), the contrast pass's group check and the QA pass (v2_qa.py).
"""

from __future__ import annotations

import io
import math

from . import describe_v2 as dv2


def _label_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def contact_sheet(items, path, px=256, cols=5):
    """[(label, icon name)] -> one PNG: each glyph at `px` under its red label."""
    from PIL import Image, ImageDraw
    rows = math.ceil(len(items) / cols)
    pad = 40
    sheet = Image.new("RGB", (min(cols, len(items)) * px, rows * (px + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _label_font(28)
    for i, (gid, icon) in enumerate(items):
        x, y = (i % cols) * px, (i // cols) * (px + pad)
        draw.text((x + 10, y + 6), gid, fill=(200, 0, 0), font=font)
        img = Image.open(io.BytesIO(dv2.glyph_png(icon))).convert("RGB")
        if img.width != px:
            img = img.resize((px, px), Image.LANCZOS)
        sheet.paste(img, (x, y + pad))
        draw.rectangle([x, y, x + px - 1, y + px + pad - 1], outline=(200, 200, 200))
    sheet.save(path)


def describe_slot(d):
    """A v2 description's depiction, readings, ambiguity and scores, as a judge sees it."""
    dep = d["depiction"]
    mods = "; ".join(dep.get("modifiers") or []) or "none"
    reads = "; ".join(f"{r['meaning']} ({r['strength']})" for r in d["readings"])
    sc = d["scores"]
    return (f"- drawn: {dep['primary_object']} | modifiers: {mods} | {dep.get('composition', '')}\n"
            f"  - readings: {reads}\n"
            f"  - ambiguity: {d.get('ambiguity') or '—'}\n"
            f"  - discard: {d['discard']} | recognizability {sc['recognizability']}, "
            f"task_fit {sc['task_fit']}, distinctiveness {sc['distinctiveness']}")


def text_slot(d):
    """A v2 description's examples and near-misses, as a judge sees them."""
    ex = d["example_tasks"]
    pick = ex[::2][:8]
    lines = ["  - examples: " + " / ".join(
        (f"[{e['parent']}] " if e.get("parent") else "") + e["task"] for e in pick)]
    lines.append("  - ES/RU: " + " / ".join(d["example_tasks_es"][:1] + d["example_tasks_ru"][:1]))
    lines.append("  - near-misses: " + " / ".join(
        f"{n['task']} → better: {n['better']} ({n['trap']})" for n in d["near_misses"][:4]))
    return "\n".join(lines)


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion k / n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
