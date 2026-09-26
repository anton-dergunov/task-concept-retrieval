# Task datasets: sources and licenses

All files share one JSONL schema: `{"id", "dataset", "title", "body", "lang", "meta"}`.
`title` and `body` may contain org-mode markup. `meta` holds provenance and must never be
shown to labellers or used as a matching feature.

| File | Contents | Source and license |
|---|---|---|
| `realistic.jsonl` | Tasks parsed from `data/eval/realistic/` | Generated, non-personal sample notes from [productivity-system](https://github.com/anton-dergunov/productivity-system); same author as this repo |
| `public_short.jsonl` | Short real-world to-do titles, EN/ES/RU | MS-LaTTE and MASSIVE; see below |
| `public_expanded.jsonl` | LLM-expanded versions of other MS-LaTTE / MASSIVE items, EN/ES/RU | Derived from MS-LaTTE and MASSIVE; see below |
| `personal_synth.jsonl` | Synthetic tasks generated from abstracted topics of a private task list, after automated privacy audits and manual review | Original to this repo; see `design/anonymization.md` |

## MS-LaTTE

Task titles sampled from **MS-LaTTE** (Microsoft), distributed in
[github.com/microsoft/MS-LaTTE](https://github.com/microsoft/MS-LaTTE) under the MIT License,
Copyright (c) Microsoft Corporation (license text: `LICENSE-MS-LaTTE.txt`). The paper describes the data as released under a
permissive Community Data License Agreement. Titles are used verbatim, except that the first
letter is capitalized; `meta.source_id` is the original `ID`.

> Sujay Kumar Jauhar, Nirupama Chandrasekaran, Michael Gamon, Ryen W. White. 2022.
> *MS-LaTTE: A Dataset of Where and When To-do Tasks are Completed.* In Proceedings of
> LREC 2022. https://aclanthology.org/2022.lrec-1.577/

## MASSIVE

Utterances from **MASSIVE 1.1** (Amazon), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Changes made:** `calendar_set`
and `lists_createoradd` utterances in en-US, es-ES and ru-RU were rewritten by an LLM
(Gemini) into to-do titles, removing the assistant framing and date/time details. The
original utterances are kept in `meta.utt` and the rewrites in `meta.parallel`;
`meta.source_id` is the MASSIVE `id`.

> Jack FitzGerald, Christopher Hench, Charith Peris, et al. 2022. *MASSIVE: A 1M-Example
> Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse
> Languages.* arXiv:2204.08582.
