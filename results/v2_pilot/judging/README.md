# Blind judging: v2 icon-description pilot

Each glyph (g01…g50) was described by 5 anonymous model configs, shown as slots A–E in
an order shuffled per glyph. Look at the glyph on the contact sheet, then judge each
slot. Judge what is DRAWN and how a person would read it; you are not told the icon's
official name, and it does not matter: a reading most people would share is correct
even if the font names the glyph differently.

## Task 1: glyph reading (all 50 glyphs; packets `batch_*.md` with `sheet_*.png`)

For each glyph and slot, a verdict on `drawn` + `readings` together:

- `correct`: the depiction is right and the primary reading is what most people would
  take it to mean (or a fair reading of a genuinely ambiguous glyph).
- `partly`: the main object is right but a meaningful modifier is missed or misread, or
  the primary reading is weak while a secondary one is right.
- `wrong`: the main object is misidentified (e.g. a bathroom scale read as a chat
  bubble), or the readings would send the icon to the wrong tasks.
- `none`: the slot has no output.

Ambiguity, discard and scores are context; do not judge them, but mention in the note
if they are clearly off (e.g. discard=true for a clear everyday object).

Write one JSON line per (glyph, slot) to `results/v2_pilot/judgments.jsonl`:
`{"glyph": "g07", "slot": "C", "verdict": "partly", "note": "misses the slash (disabled)"}`.
Notes are short; required for `partly` and `wrong`.

## Task 2: text quality (10 glyphs; `text_packet.md`)

For each glyph and slot, rate 1–5 (5 = best), using the selection principles: the icon
should show the task's central subject; properties and generic verbs don't count.

- `realism`: do the examples read like real people's task titles (varied, concrete,
  natural lengths), not templates?
- `fit`: is this icon really the best icon for the examples (central subject), given
  what the glyph shows?
- `near_miss`: are the near-misses genuinely hard (they touch the concept) and is the
  `better` icon right?

Write to `results/v2_pilot/text_judgments.jsonl`:
`{"glyph": "g07", "slot": "C", "realism": 4, "fit": 3, "near_miss": 4, "note": "…"}`.

Do NOT open `results/v2_pilot/blind_key.json` or `results/v2_pilot/raw/`.
