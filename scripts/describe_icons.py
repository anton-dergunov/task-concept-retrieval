import json
import time
from pathlib import Path

from google import genai
from google.genai import types

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

API_KEY = "AIzaSyCWbHC2pvaz548x4aMDL5fRalKBBMOevNM"

MODEL = "gemini-3.1-flash-lite"

ICONS_DIR = Path("data/icons")

OUTPUT_DIR = Path("data/icon_descriptions")

REQUEST_DELAY_SECONDS = 5

# --------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

client = genai.Client(api_key=API_KEY)

PROMPT = r"""
You are helping build an icon recommendation system for a personal productivity application.

CONTEXT

The user manages thousands of tasks in Org Mode.

Each task may be assigned a Material Symbol icon.

The icon should act as a QUICK VISUAL REPRESENTATION of the task.

The goal is:

When scanning a long TODO list, the user should immediately understand the rough purpose of a task from the icon alone.

IMPORTANT

Analyze ONLY the visual appearance of the icon.

Ignore any possible official Material Symbol name.

Do NOT guess based on filenames.

The icon image is the only source of truth.

TASK TITLE STYLE

Task titles are concise action-oriented titles such as:

Research vector database alternatives
Review quarterly budget
Plan summer travel itinerary
Compare cloud GPU providers
Organize photography equipment
Investigate backup strategy
Analyze model evaluation metrics
Prepare conference presentation
Design note-taking workflow

EVALUATION CRITERIA

An icon is useful if:

- it has a clear visual metaphor
- users would likely interpret it consistently
- it could represent a category of tasks

An icon should be discarded if:

- it is primarily a UI/navigation glyph
- it is too generic
- it has no stable semantic meaning
- it would not help distinguish tasks in a TODO list

OUTPUT

Return ONLY valid JSON.

Schema:

{
  "icon_usefulness": integer,
  "discard": boolean,

  "visual_concepts": [
    string
  ],

  "task_intents": [
    string
  ],

  "example_tasks": [
    string
  ],

  "poor_matches": [
    string
  ],

  "reasoning": string
}

Rules:

- icon_usefulness must be between 0 and 10
- visual_concepts: 5-15 items
- task_intents: 5-15 items
- example_tasks: 10-20 items
- poor_matches: 5-10 items
- example_tasks must look like realistic task titles
- poor_matches must be realistic task titles that should NOT use this icon
- reasoning should be short
- return JSON only
"""

REQUIRED_FIELDS = {
    "icon_usefulness",
    "discard",
    "visual_concepts",
    "task_intents",
    "example_tasks",
    "poor_matches",
    "reasoning",
}


def validate_json(data):
    if not isinstance(data, dict):
        return False

    if REQUIRED_FIELDS - set(data.keys()):
        return False

    if not isinstance(data["icon_usefulness"], int):
        return False

    if not isinstance(data["discard"], bool):
        return False

    list_fields = [
        "visual_concepts",
        "task_intents",
        "example_tasks",
        "poor_matches",
    ]

    for field in list_fields:
        if not isinstance(data[field], list):
            return False

    return True


def output_is_valid(path):
    if not path.exists():
        return False

    try:
        with open(path) as f:
            data = json.load(f)

        return validate_json(data)

    except Exception:
        return False


def process_icon(icon_path):

    output_path = OUTPUT_DIR / f"{icon_path.stem}.json"

    if output_is_valid(output_path):
        print(f"SKIP {icon_path.name}")
        return

    print(f"PROCESS {icon_path.name}")

    image_bytes = icon_path.read_bytes()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    text = response.text

    data = json.loads(text)

    if not validate_json(data):
        raise RuntimeError(
            f"Invalid schema for {icon_path.name}"
        )

    with open(output_path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    time.sleep(REQUEST_DELAY_SECONDS)


def main():

    icons = sorted(ICONS_DIR.glob("*.png"))

    print(f"Found {len(icons)} icons")

    for icon_path in icons:

        try:
            process_icon(icon_path)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(
                f"ERROR {icon_path.name}: {e}"
            )


if __name__ == "__main__":
    main()
