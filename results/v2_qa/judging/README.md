# Blind QA: glyph readings

Each glyph (q001 …) has two descriptions, slots A and B, from two different
description sets; which is which is shuffled per glyph. Look at the glyph on its
contact sheet (`sheet_NN.png`, labelled in red), then judge each slot. Judge what is
DRAWN and how a person would read it. You are not told the icon's official name and
it does not matter: a reading most people would share is correct even if the font
names the glyph differently. Small details (a corner badge, a slash, a fill, the
number of bars) change the meaning; look closely.

For each glyph and slot, a verdict on `shows` + `means` together:

- `correct`: the depiction is right and the main meaning is what most people would
  take it to mean (or a fair reading of a genuinely ambiguous glyph).
- `partly`: the main object is right but a meaningful modifier is missed or misread,
  or the main meaning is weak while a secondary one is right.
- `wrong`: the main object is misidentified, or the meanings would send the icon to the
  wrong tasks.

For `partly` and `wrong`, `fix` says in one or two sentences what is actually drawn
and what it would mean to most people. `note` is optional otherwise.

Write one JSON line per (glyph, slot) to the judgments file named in your
instructions (`results/v2_qa/judgments_<n>.jsonl`; append, one line at a time):
`{"glyph": "q007", "slot": "B", "verdict": "partly", "fix": "a bell with a small slash: notifications off", "note": ""}`

Do NOT open `results/v2_qa/blind_key.json`, `sample.json`, or any file under `data/`.
