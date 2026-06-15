from pathlib import Path
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FONT_PATH = (
    "/Library/Fonts/MaterialSymbolsOutlined-Regular.ttf"
)

CODEPOINTS_FILE = "data/codepoints.txt"

OUTPUT_DIR = "data/icons"

IMAGE_SIZE = 128
FONT_SIZE = 96

# --------------------------------------------------

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

font = ImageFont.truetype(
    FONT_PATH,
    FONT_SIZE
)

with open(CODEPOINTS_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

count = 0

for line in lines:
    line = line.strip()

    if not line:
        continue

    try:
        name, codepoint_hex = line.split()
    except ValueError:
        print(f"Skipping malformed line: {line}")
        continue

    codepoint = int(codepoint_hex, 16)
    glyph = chr(codepoint)

    img = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        "white"
    )

    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox(
        (0, 0),
        glyph,
        font=font
    )

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x = (IMAGE_SIZE - width) / 2
    y = (IMAGE_SIZE - height) / 2

    draw.text(
        (x, y),
        glyph,
        fill="black",
        font=font
    )

    filename = output_dir / f"{name}.png"

    img.save(filename)

    count += 1

print(f"Generated {count} icons")
print(f"Output directory: {output_dir.resolve()}")
