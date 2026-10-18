# Sub-project 3: the manuscript repository `nwa4587-paper`

Date: 2026-10-18. Status: approved in conversation 2026-10-18; builder-check questions answered below.

## Goal

Move the Computers and Geosciences revision of the NWA 4587 paper out of the
20 GB Dropbox folder `NWA4587_LPSC26` into its own private git repository,
`brendonhall/nwa4587-paper`, at `~/Projects/science/nwa4587-paper`. The new
repository builds the PDF with `latexmk`, takes every figure and every
reproducible number from a tagged release of `nwa4587-karak-paper`, and fails
the build when a number has no source.

## Why

- Dropbox and git sync the same tree. Dropbox writes conflicted copies into the
  working tree and resyncs 20 GB on every checkout.
- The folder mixes the manuscript with the poster, notebooks, the retired
  pipeline, and the data (the 2.2 GB HDF5, the raw maps, the TIMA raster under
  the TESCAN agreement). Nobody can be given the manuscript without the data.
- The paper's figures and numbers are now produced by `nwa4587-karak-paper`
  (sub-project 2). The manuscript should consume them, not regenerate them
  with scripts that depend on the retired pipeline. Typed numbers caused the
  lamellae error (128 grains, p 0.008 in the text; 272 grains, p 0.80 from the
  data).

## Decisions made with the user (2026-10-18)

- Scope: the current revision only. The poster, the LPSC abstract, the
  notebooks, the `.planning` directory, the web folder and the old pipeline
  stay in the archive repository.
- Editing: the user is the only editor for now. The build is `latexmk`
  through the Makefile on the user's machine.
- Figures and numbers: committed copies pulled by a script from a tagged
  release of `nwa4587-karak-paper`. The paper builds offline.
- History: `git filter-repo` keeps the `paper/` history (about 31 of the 173
  commits touch `paper/`; tag `review-r1`). The data never entered git, so no
  secret or large blob needs removing (verified: 188 tracked files, no blob
  over 20 MB, no `data/` path in any commit).
- Migration source: the Dropbox checkout, not GitHub. The checkout's `main`
  (`e261741`, 2026-08-06) has 11 commits that GitHub lacks, and they hold the
  revision (629 changed lines in `main.tex`, the corrected bibliography, the
  new Fig 1). GitHub `main` (`9d93da5`) has 2 commits the checkout lacks, in
  `tools/` and `pyproject.toml` only. The clone reconciles them before the
  filter (Section 1).
- First import: from the local checkout of `nwa4587-karak-paper`, because that
  repository has no release yet (its PRs 1 to 6 await the user's merge).
  `FIGURES.lock` records the commit. The release workflow and the `v0.1.0` tag
  are added to `nwa4587-karak-paper` after the merges, and a later import
  switches to the tag.
- Tables: Tables 1 to 4 stay hand-written in `main.tex` with their editorial
  columns (Stage, units, Diagnostic Elements, the unresolved-pyroxene row);
  every number cell becomes a `\nwa` macro. The generated `.tex` tables are
  imported for reference but not `\input`.
- Numbers the reproduction repository does not produce (run-to-run
  variability, the NWA 4587-3 transfer run, timing and baseline results from
  `sensitivity/`) stay typed, each with a `% source:` comment naming the JSON
  file or the archive commit. The macro rule binds only the numbers
  `nwa4587-karak-paper` reports.
- Macros: every reproducible number in `main.tex` becomes a `\nwa...` macro
  from `numbers.tex`. The build fails on an undefined macro.
- Archive: tag the old repository's final state and mark the GitHub repository
  `brendonhall/NWA4587_LPSC26` archived (read-only). The Dropbox folder stays
  as it is.
- Location and visibility: `~/Projects/science/nwa4587-paper`, private GitHub
  repository until the paper is accepted.
- Contents that move with the manuscript: `main.tex`, `references.bib` (19
  entries), the Elsevier CAS files (`cas-dc.cls`, `cas-common.sty`,
  `cas-model2-names.bst`, the `cageo-template/` folder), the `Makefile`,
  `figures/` (25 PDFs today; replaced by the imported set), the whole
  `sensitivity/` folder (6 scripts, 3 result JSON files), `CHECKLIST.md`,
  `COLLABORATION.md`, `thumbnails/`. Left behind: `email_izawa_prerelease.md`,
  `colormap_roundtrip_analysis.py` (ported into `nwa4587-karak-paper`).
- CI: GitHub Actions builds the PDF on every push and pull request in a TeX
  Live container, fails on an undefined macro or reference, and uploads
  `main.pdf` as an artifact.
- Figure names: the paper's numbering wins. `main.tex` changes its eight
  `\includegraphics` lines once to the names `nwa4587-karak-paper` writes.

## Section 1: repository layout and migration

```
nwa4587-paper/
  main.tex                 # the manuscript (moved up from paper/)
  references.bib
  cas-dc.cls, cas-common.sty, cas-model2-names.bst
  cageo-template/          # the publisher's template, unchanged
  Makefile                 # pdf, reviewpdf, clean, rebuild, view, import
  figures/                 # committed PDFs from nwa4587-karak-paper
  numbers.tex              # committed macros from nwa4587-karak-paper
  numbers.json             # the same numbers, for scripts and review
  FIGURES.lock             # the release tag and commit the figures came from
  sensitivity/             # scripts and result JSON, frozen (see below)
  tools/import_results.py  # the import script (Section 2)
  CHECKLIST.md, COLLABORATION.md, thumbnails/
  README.md, LICENSE (manuscript: all rights reserved until acceptance;
                      tools/: MIT)
  .github/workflows/build.yml
  .gitignore               # latexmk products, main.pdf, diff-*.tex/pdf
```

The layout is flat: the manuscript sits at the repository root, not under
`paper/`. The Makefile's `reviewpdf` target reads the old text with
`git show $(FROM):paper/main.tex`; after the move the path is
`$(FROM):main.tex`. `git filter-repo --subdirectory-filter paper` rewrites
the history so that `review-r1` and every earlier commit already see the files
at the root, so one path form serves every tag.

Migration steps (one session, on the user's machine, with the user's consent
before the GitHub repository is created and before the old repository is
archived):

1. `git clone --no-local /home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26`
   into a scratch directory (a read of the Dropbox checkout; never operate in
   it). In the clone: `git fetch origin`, merge `origin/main` (the 2 GitHub
   commits, `tools/` and `pyproject.toml` only; no conflict expected), and
   push the merged `main` to GitHub so both sides agree. This is the only
   write to the old GitHub repository besides the archive tag.
2. `git filter-repo --subdirectory-filter paper --tag-rename '':''` in a
   second clone of the reconciled history. Check: `git log --oneline | wc -l` is at most 173, `git tag` shows
   `review-r1`, `git ls-files` lists no path outside the list above, and no
   data file appears in `git log --all --stat`.
3. Remove `email_izawa_prerelease.md` and `colormap_roundtrip_analysis.py`
   in a commit (they stay in the history; the archive keeps them too).
4. Move the clone to `~/Projects/science/nwa4587-paper`, create the private
   GitHub repository with `gh repo create brendonhall/nwa4587-paper --private`,
   push `main` and the tags.
5. Verify the build before anything else changes: `make pdf` must produce a
   PDF identical in page count to the Dropbox `main.pdf`.
6. Tag the old repository `archive-2026-10` at the reconciled `main` (the
   merge of `e261741` and `9d93da5`) from the first clone, push the tag, and
   archive the GitHub repository with `gh repo archive`. The Dropbox folder is
   not touched.

The sensitivity scripts read the retired `data/eds_pipeline.h5` with h5py and
are not runnable from the new repository. They move as a frozen record of how
the three result files were made, with a `sensitivity/README.md` that says so
and names the data file and the commit they ran at. Making them runnable
against `paperflow.h5` is a later opt-in in `nwa4587-karak-paper`, as its spec
says.

## Section 2: figures and numbers import

`tools/import_results.py` (Python 3.12, standard library only, run with
`uv run --no-project python tools/import_results.py` or plain `python3`):

- `make import TAG=v0.1.0` downloads the release assets of
  `brendonhall/nwa4587-karak-paper` at that tag with `gh release download`,
  or copies from a local checkout when `--from PATH` is given (the first
  import uses `--from`). The asset list, the one release contract for
  `nwa4587-karak-paper`: the eight figure PDFs, the four generated tables
  (`table1_params.tex`, `table2_phases.tex`, `table3_correspondence.tex`,
  `table4_concordance.tex`), `numbers.tex`, `numbers.json` and
  `docs/acceptance-2026.md`. Sub-project 2 publishes exactly this list on its
  release (a `release` workflow added after PRs 1 to 6 merge).
- It writes `figures/fig1_workflow.pdf`, `figures/fig2_fingerprints.pdf`,
  `figures/fig3_phase_comparison.pdf`, `figures/fig4_agreement_map.pdf`,
  `figures/fig5_pyroxene_tima.pdf`, `figures/fig6_xenotime_callout.pdf`,
  `figures/fig7_pyroxene_refinement.pdf`, `figures/fig8_lamellae.pdf`,
  `numbers.tex` and `numbers.json`, and `FIGURES.lock` with the tag, the
  commit, the download time and the SHA-256 of each file.
- It refuses to run with a dirty `figures/` or `numbers.tex` unless `--force`.
- The imported files are committed. The commit message names the tag.

The exact file names are the ones `nwa4587-karak-paper` writes today; the
import script holds the list in one place so a rename on either side is a
one-line change.

## Section 3: numbers as macros

`main.tex` gains `\input{numbers}` in the preamble. Every number the
reproduction repository reports replaces its typed value: the pixel counts,
phase percentages, the pyroxene counts and GMM means, the lamellae grain
counts and statistics, the xenotime counts, the TIMA accuracies and included
pixel count, the colormap error figures. The macro names come from
`numbers.tex` (`\nwaTablesNMineralPixels`, `\nwaLamellaeSpacingMedianUm`, and
so on). A typed number may stay only when it is not a product of the pipeline
(instrument settings, literature values, sample mass). Each such number gets
a `% source:` comment on its line.

A guard macro set in the preamble makes an undefined `\nwa...` command a hard
error rather than LaTeX's default warning: the Makefile runs `latexmk` with
`-interaction=nonstopmode -halt-on-error` and then greps the log for
`Undefined control sequence` and `undefined references`; either fails the
build. CI runs the same target.

Tables 1 to 4 stay hand-written in `main.tex` and keep their editorial
columns; every number cell becomes a macro. The generated tables land under
`tables/` for cross-checking and are not `\input`. One consequence to write
into the text: the regenerated run assigns label 13 to Pigeonite and 14 to
Augite, the reverse of the published file; the manuscript refers to phases by
name, so no label number appears in the text or the tables.

The known differences recorded in the acceptance document of sub-project 2
are written into the text where they apply: 15 phases with pixels, the
corrected lamellae numbers, the TIMA metrics from the hires labels. The
lamellae spacing-direction bias is NOT resolved by this sub-project; the text
keeps the verbatim method's numbers and the user decides separately whether a
corrected analysis replaces them (see the memory note and the acceptance
document).

## Section 4: build, CI, licensing, delivery

- `make pdf` runs `latexmk -pdf -interaction=nonstopmode -halt-on-error
  main.tex` and the log check. `make reviewpdf FROM=review-r1` builds the
  latexdiff against the tag as today, with the corrected path.
- `.github/workflows/build.yml`: on push and pull request, `ubuntu-latest`,
  the `xu-cheng/latex-action` or a `texlive/texlive:latest` container, runs
  `make pdf`, uploads `main.pdf` as an artifact named by the commit. No
  secrets; the repository is private, so the artifact is private.
- Licensing: the manuscript text and figures carry no open license until the
  paper is accepted (the journal decides); `tools/` is MIT. `README.md` states
  both and the provenance of `figures/` (CC BY 4.0 from
  `nwa4587-karak-paper`, authors Hall and Izawa).
- Delivery in three PRs after the migration push: PR 1 layout, Makefile,
  CI, import script, imported figures, tables and numbers, `FIGURES.lock`;
  PR 2 the macro replacement in `main.tex` and in the four tables (one commit
  per section of the paper, so the diff is reviewable), with the `% source:`
  comments on the typed numbers; PR 3 README, licensing, sensitivity README,
  archive of the old repository. A fourth, small PR in `nwa4587-karak-paper`
  adds the release workflow once its PRs 1 to 6 are merged. The user merges.
- Acceptance: `make pdf` succeeds locally and in CI; the PDF has the same
  section structure as the Dropbox `main.pdf`; `grep -c '\\nwa' main.tex`
  equals the number of replaced values listed in the PR 2 body; a `latexdiff`
  against `review-r1` shows only the number and figure-name changes plus the
  text edits the user made.

## Out of scope

The poster, the LPSC abstract, the review response letter (not yet written),
Overleaf, co-author git workflow, the Zenodo deposit (sub-project 5), the
karak release (sub-project 4), the lamellae bias decision, making the
sensitivity scripts runnable.

## Calls made by the author of this spec (check these while reading)

- The asset list in Section 2 is the single release contract (figures,
  tables, numbers, acceptance document).
- The archive tag is `archive-2026-10` on the reconciled `main`.
- The sensitivity scripts move frozen, with a README naming the data file and
  commit they ran at.
- The flat layout (manuscript at the repository root) and the `reviewpdf`
  path fix.
- The manuscript license stays all rights reserved until acceptance; `tools/`
  is MIT.

## Open items for the user

- The paper title and journal reference for the citation files (also pending
  in `nwa4587-karak-paper`).
- The lamellae spacing-direction bias: keep the verbatim numbers or correct
  the analysis before PR 2 replaces the lamellae values.
