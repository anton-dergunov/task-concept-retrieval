# Icon labelling tool

A small web app for ranking icons for each task, on a phone, tablet or desktop. It is a
single-file server that needs only **Python 3.8+ and nothing else**, so it runs as-is on the
Synology NAS. Labels save on every tap, so you can close the page and continue on another device.

## How labelling works

- One task at a time: its title and body, with long bodies collapsed behind "more". Translated
  items (Public-*) show the title in all languages at once, so one label covers every version.
- 30 candidate icons: 24 from several matchers (M3, M1, M2, B1) and 6 random ones, in a
  shuffled order. The order is fixed per task, so every device shows the same grid.
  **More icons** at the end of the grid adds 70 more: the matchers' next-best icons and
  another 14 random ones. Every icon is a distinct glyph; font aliases that draw the same
  picture are collapsed.
- **What to choose** follows [docs/icon-selection-principles.md](../docs/icon-selection-principles.md):
  rank 1 shows the task's central object or subject, rank 2 its context or activity, and
  properties (waterproof, urgent, small) are never the answer. When a task has a
  parent heading (project, section, or list name), it follows the title in brackets.
- **Tap** icons in order of preference: the badge shows the rank (1, 2, 3, …). **Tap again**
  to deselect; the remaining ranks renumber.
- **Search** looks for icons by their descriptions (never by names). Icons you pick from
  search stay in the grid after you clear the search.
- **No good icon** records an explicit negative, which the abstention gate needs.
  **Next** with nothing selected records "skip" (unsure). **⚑** (top right) flags a task
  for exclusion: garbled, too short to label, or leaking; tap it again to unflag.
- **Personal-synth** tasks are publication candidates: flag anything that looks like a real
  task of yours, a recognisable detail, or garbage. `python scripts/datasets/synth_personal.py
  publish` then writes only the unflagged ones to the public dataset.
- The dataset picker is in the top left. Tasks come in a fixed pseudo-random order, so
  whatever you have labelled so far is always a uniform random sample.
- Keys (desktop): ← / → to move between tasks, `/` to search, Esc to clear the search.
- Deep link: `#ds=personal&pos=125`.

## Reviewing the icon set

A second page, `/curate` (also **⋯ → Review the icon set**), shows every distinct glyph at
once (3,901: aliases are collapsed) so the ones that cannot label a task can be removed by
hand. It includes the glyphs the v1 descriptions discarded, with nothing telling them apart,
so the result also measures how good the model's `discard` is.

- Look-alike glyphs sit together (ordered by image similarity, never by name), in numbered
  groups.
- **Tap** marks a glyph for removal; **tap again** undoes it. **Remove all** in a group
  header marks the whole group (with Undo); then tap the ones to keep.
- **Marked only** shows just the marked glyphs, for a final check. **S / M / L** changes the
  tile size.
- Every tap is saved. Without a connection the changes wait on the device and are sent when
  it returns; the header shows `✓ saved` or the number unsaved.
- **What to remove:** a glyph that can never be the subject of a task
  ([icon-selection-principles](../docs/icon-selection-principles.md)):
  - it shows only a property, quantity or setting (`1.5x`, a megapixel count, signal bars);
  - it is interface chrome with no subject (chevrons, drag handles, layout toggles);
  - it is only letters or digits;
  - it cannot be read at a glance.

  When in doubt, keep it: weak icons are down-weighted and the matcher can abstain.

Build it once on the Mac (a few minutes: it embeds and clusters the glyphs), then deploy
as usual:

```bash
python scripts/build_curation_bundle.py
```

The log is `labels/icon_curation.jsonl`: one line per change, `{icon, removed, bulk, ts,
client}`, where the latest line per icon wins and `bulk` marks a **Remove all**.
`python scripts/import_labels.py` copies it to `data/labels/` and prints the agreement with
the model's `discard`.

To use the laptop instead of the NAS, with the tablet on the same network:

```bash
python3 labeller/server.py --host 0.0.0.0 --token <secret>
```

and open `http://<laptop-name>.local:8766/curate?token=<secret>` on the tablet.

## Install on the home screen

Over HTTPS (Tailscale serve), use the browser's *Add to Home Screen* / *Install app*: the
tool then opens full-screen like an app. Icons you've seen stay cached on the device, and
the next three tasks' icons are fetched ahead, so moving on is instant. Over plain HTTP
this caching and installing are unavailable, but everything else works.

## Build the bundle (on the Mac)

```bash
python scripts/build_label_bundle.py            # all datasets present locally
python scripts/build_label_bundle.py --datasets public_short   # just one
```

This writes `labeller/bundle/`, which is gitignored and contains the private tasks.
Rebuilding keeps the grid of every existing task (use `--refresh` to recompute them) and only
computes grids for new tasks.

## Run locally

```bash
python3 labeller/server.py            # http://127.0.0.1:8766/
```

## Deploy on the Synology NAS (via Tailscale)

1. Copy the folder:
   ```bash
   rsync -a --exclude labels/ labeller/ nas:~/labeller/
   ```
   If rsync is disabled on DSM, use `scp -r labeller nas:~/` instead. Never overwrite
   `labels/` on the NAS: it is the source of truth.
2. Start the server on the NAS:
   ```bash
   cd ~/labeller && nohup python3 server.py --port 8766 > server.log 2>&1 &
   ```
   To survive reboots, add it in DSM → Control Panel → Task Scheduler → Create →
   Triggered Task → Boot-up, running as your user:
   `cd /var/services/homes/<you>/labeller && python3 server.py --port 8766`.
3. Expose it on your tailnet with HTTPS. HTTPS requires MagicDNS and HTTPS certificates to
   be enabled in the Tailscale admin console. On DSM this needs `sudo`. If another service
   already owns `https://<nas-name>.<tailnet>.ts.net/` (port 443), give the labeller its own
   HTTPS port instead of the root:
   ```bash
   sudo tailscale serve status                                  # see what is already served
   sudo tailscale serve --bg --https=8443 http://127.0.0.1:8766
   ```
   Open `https://<nas-name>.<tailnet>.ts.net:8443/` on any of your devices. Do not open
   `…ts.net:8766` with `https://`: that port is the server's own plain-HTTP socket.
   Older Tailscale packages may not support `serve`. In that case, bind to the NAS's
   tailnet address and use plain HTTP:
   ```bash
   python3 server.py --host "$(tailscale ip -4)" --port 8766 --token <secret>
   ```
   Open `http://<nas-tailscale-ip>:8766/?token=<secret>` once per device. The token is
   stored in a cookie.
4. Update the tasks later: rebuild the bundle on the Mac, rsync again (keeping `labels/`),
   and restart the server.

## Get the labels back

```bash
rsync -a nas:~/labeller/labels/ labeller/labels/
python scripts/import_labels.py
```

Public datasets' labels go to `data/labels/` (committed); personal labels go to
`data/private/labels/` (gitignored).

## Label format

`labels/<dataset>.jsonl` is an append-only event log; the **latest record per `task_id`
wins**. Each record contains:

| Field | Meaning |
|---|---|
| `status` | `labelled`, `none` (no good icon), `skip`, `flagged`, or `cleared` (selection undone) |
| `ranking` | Selected icons, best first |
| `expanded` | Whether "More icons" was opened |
| `shown` | The candidates in the order displayed: 30, or ~100 when `expanded`. Used to measure position bias (click position vs. rank) |
| `prov` | For each shown or selected icon: which matcher proposed it and at what rank, `random`, or `search`. Used to measure pool bias and per-method recall-of-pool |
| `search_added` | Selected icons that came from search rather than the grid |
| `queries` | Search strings used |
| `ms_spent` | Time spent on the task, in milliseconds |
| `ts` | Timestamp of the event |
| `client` | Device type |
