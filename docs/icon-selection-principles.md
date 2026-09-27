# Icon selection principles

The single definition of a *good icon for a task*. It is used verbatim, or summarized,
by:

- the human labeller (the labelling tool);
- the LLM judge (fine-tuning-plan §6);
- the icon-description prompt (icon-descriptions-v2 §3–4);
- the instruction prompt of any LLM-based model we fine-tune (fine-tuning-plan §5).

## Purpose

The icon is a **glanceable pictorial reminder of what a task is about**. When scanning a
long task list, a person should recall a task's subject from its icon faster than by
reading the title. The icon does not summarize the whole title. It **cues the one thing
the task is about**, so that together with a quick glance at the title the task is
recognized at once. A wrong or vague icon costs attention, so **no icon is better than a
misleading one** (abstention).

## What the icon should show, in order of priority

1. **The central object or subject**: the thing the task is about, usually the head noun
   of the title. *"Buy a small waterproof rechargeable headlamp for trips"* → a **headlamp**.
2. **The context, domain or kind of activity** it belongs to: the purpose, place or project.
   Headlamp → **light**, or **trips / hiking**.
3. **The action**, only when the verb is the point of the task (call, pay, fix, read,
   sign, ship). Generic verbs (review, check, update, organize, buy) are rarely what should
   be shown.
4. **Never the properties.** Adjectives and attributes (waterproof, rechargeable, small,
   urgent), quantities, dates, deadlines, and people's names are not shown: a rain icon
   for "waterproof" is wrong.

Further rules:

- **Specific beats generic,** as long as it stays readable: headlamp > flashlight > light
  bulb > shopping bag.
- **Literal beats metaphorical** when a literal icon exists; use a metaphor (a lightbulb
  for an idea) only when nothing literal fits.
- **Recognizability beats correctness:** an icon that is technically right but that
  nobody would read correctly at a glance is not a good icon.
- **Ranked answers:** rank 1 shows the central subject; rank 2 its context or activity;
  lower ranks are partial fits. If only property-level or generic icons fit, answer
  **"no good icon"**.

## Parent headings (project / section context)

Tasks usually sit under a parent heading (a project or area, e.g. *"Music catalog
support"*). The parent is context, not the answer:

- **If the title is specific,** the title wins; the parent only disambiguates.
- **If the title is generic,** the parent's topic becomes the subject. *"Evaluate bids
  against my catalog rules"* under *"Music catalog support"* → a **music library / album
  collection**; then rules or processing.
- **Siblings should not all get the same icon.** Prefer what distinguishes the task from
  the other tasks of the same project, as long as the icon still reads as belonging to it.

## Worked examples

| Task (parent) | 1st choice | 2nd choice | Not |
|---|---|---|---|
| Buy a small waterproof rechargeable headlamp for trips | headlamp | light / flashlight, or trips / hiking | rain (waterproof), battery (rechargeable), shopping cart (generic verb) |
| Evaluate bids against my catalog rules (*Music catalog support*) | music library / album collection | rules / checklist / processing | a generic "evaluate" chart |
| Verify that all devices including tablet and phones are using git sync correctly | git + sync (synchronization of a repository) | devices (tablet / phone) syncing | a check mark (generic "verify") |

## Language

The principles are language-independent: the same icon for *"Comprar una linterna frontal
para excursiones"* and *"Купить налобный фонарик для походов"*.
