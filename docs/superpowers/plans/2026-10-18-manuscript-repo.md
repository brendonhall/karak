# Manuscript Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A private repository `brendonhall/nwa4587-paper` at `~/Projects/science/nwa4587-paper` that holds the Computers and Geosciences revision of the NWA 4587 paper, builds the PDF with `latexmk`, imports every figure, table and number from `nwa4587-karak-paper`, and fails the build when a pipeline number has no source.

**Architecture:** The manuscript's git history is filtered out of the old `NWA4587_LPSC26` repository into a flat repository (manuscript at the root). A standard-library Python script imports the release assets of `nwa4587-karak-paper` and records their origin in a lock file. `main.tex` reads `numbers.tex` (generated) and `derived.tex` (hand-written `\fpeval` combinations of generated macros), and a check script proves every `\nwa` macro the text uses is defined and every pipeline number in a known list is a macro. CI builds the PDF in a TeX Live container.

**Tech Stack:** LaTeX (Elsevier `cas-dc` class, `siunitx`, `xfp`), `latexmk`, `latexdiff`, Python 3.12 standard library for `tools/`, pytest for the tools, `git filter-repo` (via `uvx --from git-filter-repo`), GitHub Actions with the `texlive/texlive:latest` container, Docker for the local build when `latexmk` is absent.

**Spec:** `docs/superpowers/specs/2026-10-18-manuscript-repo-design.md` (karak repository). The number-to-macro mapping that Tasks 6 to 8 work from: `docs/superpowers/plans/2026-10-18-manuscript-repo-number-map.md` (same directory as this plan; 281 pipeline numbers: 149 match the regenerated macros, 84 differ, 48 have no macro; 43 non-pipeline numbers).

## Global Constraints

- The Dropbox checkout `/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26` is read-only: never `cd` into it, never run `git` commands that write there, never run `uv` or `python` with it as the working directory. `git clone --no-local <path>` reads it. The only writes to the OLD GitHub repository `brendonhall/NWA4587_LPSC26` are the reconciled `main` push (Task 1) and the archive tag (Task 10); both need the user's consent in the session.
- Creating the GitHub repository `brendonhall/nwa4587-paper` (private) and archiving the old repository each need the user's explicit yes in the session.
- Every change after the migration push goes through a PR from an up-to-date `main`; never commit to `main`. Commit messages end with the attribution lines the session reminder gives. The user merges. Prose (README, PR bodies, comments) follows the user's style: short sentences, active voice, no em dashes, concrete numbers.
- The flat layout: `main.tex`, `references.bib`, `cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst`, `cageo-template/`, `Makefile`, `figures/`, `tables/`, `numbers.tex`, `numbers.json`, `derived.tex`, `FIGURES.lock`, `sensitivity/`, `tools/`, `docs/`, `CHECKLIST.md`, `COLLABORATION.md`, `thumbnails/`, `README.md`, `LICENSE`, `.github/workflows/build.yml`, `.gitignore`.
- The import asset list (the release contract with `nwa4587-karak-paper`): `figures/fig1_workflow.pdf`, `fig2_fingerprints.pdf`, `fig3_phase_comparison.pdf`, `fig4_agreement_map.pdf`, `fig5_pyroxene_tima.pdf`, `fig6_xenotime_callout.pdf`, `fig7_pyroxene_refinement.pdf`, `fig8_lamellae.pdf`; `tables/table1_params.tex`, `table2_phases.tex`, `table3_correspondence.tex`, `table4_concordance.tex`; `numbers.tex`; `numbers.json`; `docs/acceptance-2026.md`. The first import is `--from /home/brendon/Projects/science/nwa4587-karak-paper` (its `results/` and `docs/`), because that repository has no release yet.
- Macro rule: every number that `numbers.tex` supplies (Table A of the number map) becomes a `\nwa...` macro or a `derived.tex` macro built from them. Numbers in Table B stay typed with a `% source:` comment on their line. An undefined macro fails the build.
- The implementer never rewrites scientific prose. Where a regenerated value changes a sentence's meaning (the number map's DIFFERS rows), the implementer inserts the macro and a `\TODO{was: <old value>; <one line on what changed>}` marker beside it; the author resolves the markers. The preamble already defines `\TODO` (line 16).
- `main.pdf` and every `latexmk` product are git-ignored; `data/` never exists in this repository.

## Review Focus

1. A `\nwa` macro used in `main.tex` but absent from `numbers.tex` and `derived.tex` must fail `make pdf` and `make check` with the macro's name, not produce a PDF with a blank. Pinned in Task 5 (`test_check_reports_undefined_macro`) and by the Makefile log grep.
2. An import from a directory that lacks one asset (for example no `tables/table4_concordance.tex`) must fail before writing anything, naming the missing file. Pinned in Task 4 (`test_import_missing_asset_writes_nothing`).
3. An import onto a dirty `figures/` or `numbers.tex` without `--force` must refuse and name the dirty paths. Pinned in Task 4 (`test_import_refuses_dirty_without_force`).
4. `make reviewpdf FROM=review-r1` must find `main.tex` at the root of the tag after the filter (the old Makefile reads `paper/main.tex`). Pinned in Task 2 (the `git show review-r1:main.tex` check) and Task 3 (`reviewpdf` build).
5. A derived macro whose input macro is missing must fail like an undefined macro, not evaluate `\fpeval` on garbage. Pinned in Task 5 (`test_check_reports_derived_input_missing`).

## File structure

```
nwa4587-paper/                      # new repository
  main.tex, references.bib, cas-dc.cls, cas-common.sty, cas-model2-names.bst, cageo-template/
  Makefile                          # Task 2: pdf, reviewpdf, clean, rebuild, view, import, check; Docker fallback
  .gitignore, .github/workflows/build.yml, LICENSE, README.md
  figures/, tables/, numbers.tex, numbers.json, FIGURES.lock   # Task 4 import
  derived.tex                       # Task 5: \fpeval macros over numbers.tex
  tools/import_results.py           # Task 4
  tools/check_numbers.py            # Task 5
  tools/tests/test_import_results.py, tools/tests/test_check_numbers.py
  tools/pyproject.toml              # pytest only, for `uv run --project tools pytest`
  docs/number-map.md                # Task 6: the mapping, moved from the karak plan directory
  docs/acceptance-2026.md           # imported
  sensitivity/README.md             # Task 9
  CHECKLIST.md, COLLABORATION.md, thumbnails/
```

Numbers the generator formats unhelpfully (fractions that the text prints as percentages, e-notation p values, counts printed as millions) get a `derived.tex` macro per case, named `\nwaD...` (D for derived), so a reader can tell generated from derived. `derived.tex` is hand-written and committed; its inputs are `\nwa...` macros only.

---

# Part 1: migration (local, one session; PR 1 branch `feat/layout-build-import`)

### Task 1: reconcile the two histories and filter the manuscript out

**Files:**
- Create (outside git): a scratch directory `~/Projects/science/.migrate-nwa4587/` with two clones.

**Interfaces:**
- Produces: `~/Projects/science/nwa4587-paper` with the filtered history on `main`, the tag `review-r1`, the manuscript files at the root, pushed to the new private GitHub repository.

- [ ] **Step 1: Consent**

Say: "Task 1 pushes the reconciled `main` of the old repository `brendonhall/NWA4587_LPSC26` to GitHub (11 local commits plus a merge of 2 GitHub commits) and creates the private GitHub repository `brendonhall/nwa4587-paper`. May I do both now?" Stop until the answer is yes.

- [ ] **Step 2: Clone from the Dropbox path and reconcile**

```bash
mkdir -p ~/Projects/science/.migrate-nwa4587 && cd ~/Projects/science/.migrate-nwa4587
git clone --no-local /home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26 full
cd full && git remote set-url origin https://github.com/brendonhall/NWA4587_LPSC26.git && git fetch origin
git log --oneline HEAD..origin/main        # expected: 2 commits, tools/ and pyproject.toml only
git log --oneline origin/main..HEAD | wc -l   # expected: 11
git merge --no-edit origin/main            # expected: no conflict
git push origin main
```
If the merge conflicts, stop and report; do not resolve by hand.

- [ ] **Step 3: Filter the manuscript history**

```bash
cd ~/Projects/science/.migrate-nwa4587 && git clone --no-local full paper && cd paper
uvx --from git-filter-repo git-filter-repo --subdirectory-filter paper --tag-rename '':''
git log --oneline | wc -l          # expected: about 31
git tag                            # expected: review-r1
git ls-files | grep -v -E '^(main.tex|references.bib|cas-|cageo-template/|Makefile|figures/|sensitivity/|CHECKLIST.md|COLLABORATION.md|thumbnails|email_izawa_prerelease.md|colormap_roundtrip_analysis.py|.gitignore)' ; echo "(expected: no output)"
git log --all --stat | grep -E 'data/|\.h5|Phases\.png' ; echo "(expected: no output)"
git show review-r1:main.tex | head -3        # expected: the documentclass line
```

- [ ] **Step 4: Remove the two files that stay in the archive**

```bash
git rm -q email_izawa_prerelease.md colormap_roundtrip_analysis.py
git commit -q -m "chore: drop the email draft and the colormap script (ported to nwa4587-karak-paper)"
```
(with the attribution lines)

- [ ] **Step 5: Create the GitHub repository and push**

```bash
mv ~/Projects/science/.migrate-nwa4587/paper ~/Projects/science/nwa4587-paper && cd ~/Projects/science/nwa4587-paper
git remote remove origin 2>/dev/null; gh repo create brendonhall/nwa4587-paper --private --description "NWA 4587 automated-mineralogy paper: manuscript source" --source . --remote origin
git push -u origin main && git push origin --tags
gh repo view brendonhall/nwa4587-paper --json visibility -q .visibility   # expected: PRIVATE
```

- [ ] **Step 6: Record the migration**

Write `docs/migration-2026-10.md` (create `docs/`): the source commits (`e261741` local, `9d93da5` GitHub, the merge commit SHA), the filter command, the commit count before and after, the two removed files, the date. Commit as `docs: record the migration from NWA4587_LPSC26` on `main` (this is the last commit on `main` before the branch workflow starts; say so in the message body).

### Task 2: flat layout, Makefile, Docker fallback, ignore list

**Files:**
- Modify: `Makefile`, `.gitignore`
- Create: `LICENSE`, `README.md` (stub), `tools/pyproject.toml`

**Interfaces:**
- Produces: `make pdf` (local `latexmk`, else Docker `texlive/texlive:latest`), `make reviewpdf FROM=<ref>`, `make clean`, `make check` (Task 5 fills it), `make import` (Task 4 fills it).

- [ ] **Step 1: Branch**

```bash
git switch -c feat/layout-build-import
```

- [ ] **Step 2: Makefile**

Change `git show $(FROM):paper/$(MAIN).tex` to `git show $(FROM):$(MAIN).tex`. Add at the top:

```make
# latexmk from PATH when present, else the TeX Live container (same image as CI).
DOCKER_TEX = docker run --rm -u $$(id -u):$$(id -g) -v "$(CURDIR)":/work -w /work texlive/texlive:latest
ifeq ($(shell command -v latexmk 2>/dev/null),)
LATEXMK = $(DOCKER_TEX) latexmk
else
LATEXMK = latexmk
endif
ifeq ($(shell command -v latexdiff 2>/dev/null),)
LATEXDIFF = $(DOCKER_TEX) latexdiff
else
LATEXDIFF = latexdiff
endif
```
and use `$(LATEXDIFF)` in the `reviewpdf` recipe. The process substitution `<(git show ...)` does not cross the container boundary: write the old text to `$(DIFFJOB)-old.tex` first (`git show $(FROM):$(MAIN).tex > $(DIFFJOB)-old.tex`) and pass the file; add `main-diff-old.tex` to the clean list and `.gitignore`.
Add after the `pdf` recipe a log check that fails the target on an undefined macro or reference:

```make
pdf:
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN).tex
	@! grep -E 'Undefined control sequence|There were undefined references|Citation .* undefined' $(MAIN).log
```
Add `check` and `import` as phony targets that call `python3 tools/check_numbers.py` and `python3 tools/import_results.py $(IMPORT_ARGS)` (Tasks 4 and 5 implement the scripts; until then the targets print "not implemented" and exit 1).

- [ ] **Step 3: `.gitignore`**

Replace the old list with: `main.pdf`, `main-diff.*`, `*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`, `*.fls`, `*.log`, `*.out`, `*.abs`, `*.spl`, `*.synctex.gz`, `*.toc`, `*.lof`, `*.lot`, `__pycache__/`, `.pytest_cache/`, `tools/.venv/`.

- [ ] **Step 4: LICENSE and README stub**

`LICENSE`: first paragraph "The manuscript text, `main.tex`, and the files under `cageo-template/` are not open source. All rights reserved by the authors until the paper is accepted; the journal's terms then apply." Second paragraph: "The scripts under `tools/` are released under the MIT License:" followed by the MIT text with `Copyright (c) 2026 Brendon Hall`. Third paragraph: "The files under `figures/`, `tables/`, `numbers.tex` and `numbers.json` are generated by `nwa4587-karak-paper` and carry its CC BY 4.0 license." `README.md` stub: title, one paragraph (what the repository holds, where the figures come from), "Status: migration in progress". Task 9 writes the full README.

- [ ] **Step 5: `tools/pyproject.toml`**

```toml
[project]
name = "nwa4587-paper-tools"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []
[dependency-groups]
dev = ["pytest>=8.0"]
[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 6: Build check**

Run: `make pdf` (Docker pulls the image the first time; several minutes). Expected: `main.pdf` exists; `pdfinfo main.pdf | grep Pages` equals the page count of `/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/paper/main.pdf` (read it with `pdfinfo` from the Dropbox path; `pdfinfo` is in `poppler`; if absent, compare `pdftotext | wc -l` or the `%%Pages` of `strings`). Record both counts in the commit body. Run `make reviewpdf FROM=review-r1`; expected: `main-diff.pdf` exists (Review Focus 4).

- [ ] **Step 7: Commit, push, open PR 1 (draft)**

```bash
git add Makefile .gitignore LICENSE README.md tools/pyproject.toml
git commit -m "chore: flat layout build, Docker fallback, licensing stub"
git push -u origin HEAD
gh pr create --draft --title "feat: layout, build, CI and the results import" --body "Part 1 of the plan."
```

### Task 3: CI builds the PDF

**Files:**
- Create: `.github/workflows/build.yml`

- [ ] **Step 1: Workflow**

```yaml
name: build
on:
  push:
    branches: [main]
  pull_request:
jobs:
  pdf:
    runs-on: ubuntu-latest
    container: texlive/texlive:latest
    steps:
      - uses: actions/checkout@v4
      - run: make pdf
      - run: make check
      - uses: actions/upload-artifact@v4
        with:
          name: main-${{ github.sha }}
          path: main.pdf
```
`make check` exits 1 until Task 5; add `continue-on-error: true` to that step now and remove it in Task 5.

- [ ] **Step 2: Commit, push, watch**

```bash
git add .github/workflows/build.yml && git commit -m "ci: build the PDF in the TeX Live container" && git push
gh pr checks --watch
```
Expected: the `pdf` job passes and the artifact lists `main.pdf`. If the container lacks `make` or `git` (the `reviewpdf` target needs `git`), add `apt-get update && apt-get install -y make git` as the first step and note it in the commit body.

### Task 4: the import script and the first import

**Files:**
- Create: `tools/import_results.py`, `tools/tests/test_import_results.py`, `FIGURES.lock`, `figures/*` (replaced), `tables/*`, `numbers.tex`, `numbers.json`, `docs/acceptance-2026.md`
- Modify: `Makefile` (`import` target), `main.tex` (eight `\includegraphics` lines)

**Interfaces:**
- Produces: `python3 tools/import_results.py (--from PATH | --tag TAG) [--force] [--repo brendonhall/nwa4587-karak-paper]`; module-level `ASSETS: dict[str, str]` mapping the destination path in this repository to the source path under the reproduction repository's checkout (`results/figures/fig1_workflow.pdf` and so on; `docs/acceptance-2026.md` for the acceptance document); functions `plan(src: Path) -> list[tuple[Path, Path]]`, `dirty(dests: list[Path]) -> list[Path]` (paths with uncommitted changes per `git status --porcelain`), `copy_all(pairs, force: bool) -> None`, `write_lock(path: Path, source: dict) -> None` (JSON: `tag` or `commit`, `source` path or repo, `imported_at` UTC ISO, `files: {dest: sha256}`), `main(argv) -> int` (exit 2 on a missing asset, 3 on dirty without `--force`, 0 on success).
- `--tag TAG` runs `gh release download TAG --repo REPO --dir <tmp>` and then treats `<tmp>` as `src` with the release's flat layout (every asset at the top level); `--from PATH` reads the checkout layout. Both paths go through `ASSETS`.

- [ ] **Step 1: Tests**

`tools/tests/test_import_results.py` with a fixture that builds a fake source tree holding every asset (one-line files) and a fake destination git repository (`git init`, one commit):

```python
def test_plan_lists_every_asset(fake_src):
    pairs = imp.plan(fake_src)
    assert {d.as_posix() for d, _ in pairs} == set(imp.ASSETS)

def test_import_missing_asset_writes_nothing(fake_src, dest_repo):
    (fake_src / "results/tables/table4_concordance.tex").unlink()
    rc = imp.main(["--from", str(fake_src)])          # cwd = dest_repo
    assert rc == 2
    assert not (dest_repo / "figures").exists() and not (dest_repo / "FIGURES.lock").exists()

def test_import_refuses_dirty_without_force(fake_src, dest_repo):
    assert imp.main(["--from", str(fake_src)]) == 0
    (dest_repo / "numbers.tex").write_text("edited")
    rc = imp.main(["--from", str(fake_src)])
    assert rc == 3
    assert (dest_repo / "numbers.tex").read_text() == "edited"
    assert imp.main(["--from", str(fake_src), "--force"]) == 0

def test_lock_records_sha256_and_source(fake_src, dest_repo):
    imp.main(["--from", str(fake_src)])
    lock = json.loads((dest_repo / "FIGURES.lock").read_text())
    assert set(lock["files"]) == set(imp.ASSETS)
    assert lock["source"]["path"] == str(fake_src) and len(lock["source"]["commit"]) == 40
```
The fixture makes `fake_src` a git repository with one commit so `--from` can record its `HEAD`.

- [ ] **Step 2: Run to verify failure**

Run: `cd tools && uv run --group dev pytest -q`. Expected: ImportError on `import_results`.

- [ ] **Step 3: Implement `tools/import_results.py`**

Standard library only (`argparse`, `hashlib`, `json`, `pathlib`, `shutil`, `subprocess`, `datetime`). The missing-asset check runs over the whole plan before any copy. The error message for exit 2 names every missing source path; for exit 3 every dirty destination.

- [ ] **Step 4: Run to verify pass**

Run: `cd tools && uv run --group dev pytest -q`. Expected: 4 passed.

- [ ] **Step 5: Makefile target and the first import**

`import: ; python3 tools/import_results.py $(IMPORT_ARGS)` with `IMPORT_ARGS ?= --from ../nwa4587-karak-paper`. Then:

```bash
git rm -q -r figures && make import IMPORT_ARGS="--from /home/brendon/Projects/science/nwa4587-karak-paper"
ls figures tables; cat FIGURES.lock | head -8
```
Expected: 8 PDFs, 4 tables, `numbers.tex` (875 macros), `numbers.json`, `docs/acceptance-2026.md`, `FIGURES.lock` with the commit of the reproduction checkout.

- [ ] **Step 6: Figure names in `main.tex`**

Change the eight `\includegraphics` lines (241, 585, 646, 657, 669, 706, 770, 783 in the migrated file; confirm with `grep -n includegraphics main.tex`) to the new names: `fig1_workflow`, `fig5_fingerprints` to `fig2_fingerprints`, `fig3_phase_comparison`, `fig4_agreement_map`, `fig_pyroxene_tima` to `fig5_pyroxene_tima`, `fig7_xenotime_callout` to `fig6_xenotime_callout`, `fig6_pyroxene_refinement` to `fig7_pyroxene_refinement`, `fig8_lamellae`. Run `make pdf`; expected: builds; `grep -c includegraphics main.tex` is 8.

- [ ] **Step 7: Commit**

```bash
git add tools/import_results.py tools/tests Makefile FIGURES.lock figures tables numbers.tex numbers.json docs/acceptance-2026.md main.tex
git commit -m "feat: import figures, tables and numbers from nwa4587-karak-paper"
git push && gh pr checks --watch
```
Then `gh pr ready` on PR 1 and fill its body: what changed, the build and import commands run, the page-count comparison, rulings (Docker fallback; first import from a local checkout), follow-ups (the release workflow).

---

# Part 2: numbers as macros (PR 2, branch `feat/numbers-as-macros` from `main` after PR 1 merges, or stacked on `feat/layout-build-import` if not)

### Task 5: `\input{numbers}`, `derived.tex`, the check script

**Files:**
- Create: `derived.tex`, `tools/check_numbers.py`, `tools/tests/test_check_numbers.py`
- Modify: `main.tex` (preamble), `Makefile` (`check` target), `.github/workflows/build.yml` (drop `continue-on-error`)

**Interfaces:**
- Produces: `tools/check_numbers.py [--tex main.tex] [--defs numbers.tex derived.tex]` exit 0 when every `\nwa[A-Za-z]+` used in `main.tex` is defined in one of the definition files and every `\nwaD...` in `derived.tex` only references defined `\nwa...` macros; exit 1 listing the undefined names and where they are used. Functions `used_macros(tex: str) -> set[str]`, `defined_macros(defs: str) -> set[str]`, `derived_inputs(defs: str) -> dict[str, set[str]]`, `main(argv) -> int`.
- `derived.tex` conventions: `\usepackage{xfp}` is loaded in `main.tex`; each derived macro is one `\newcommand{\nwaD<Name>}{\fpeval{...}}` or a `siunitx` wrapper, with a comment naming the text site it serves. The initial set (from the number map's format notes): `\nwaDTimaSplitAccuracyPct` = `\fpeval{round(\nwaTimaSplitOverallAccuracy*100,1)}`; `\nwaDTimaSplitDiscordantPct` = `\fpeval{round(100-\nwaTimaSplitOverallAccuracy*100,1)}`; `\nwaDTimaCombinedAccuracyPct`; `\nwaDTimaSplitNIncludedMillions` = `\fpeval{round(\nwaTimaSplitNIncluded/1e6,1)}`; `\nwaDTablesNMineralPixelsMillions`; `\nwaDPyroxeneNPixelsTwoXMillions`; `\nwaDPyroxeneNPixelsOneXMillions` (sum of the two 1x counts over 1e6, 1 dp); `\nwaDColormapJetMaePct` and `\nwaDColormapJetMaxAePct` (times 100, 2 and 1 dp); `\nwaDTablesParamsPCAVarianceKeptPct` (times 100, 0 dp); `\nwaDLamellaeNGrainsExcluded` = total minus analyzed; `\nwaDLamellaeNearNyquistPct`; a `\pthresh{<macro>}` command that prints `p < 0.001` when the value is under 0.001 and `p = <3 dp>` otherwise (`\ifdim` on `\fpeval`); a `\pctsmall{<macro>}` command that prints `<0.01` when the value is under 0.005 and the value otherwise. Every derived macro name appears in Table A of the number map beside the row it serves (Task 6 updates the map).

- [ ] **Step 1: Tests**

```python
def test_used_and_defined(tmp_path):
    tex = r"Text \nwaTablesNMineralPixels and \num{\nwaDFoo} and \pthresh{\nwaBar}."
    assert chk.used_macros(tex) == {"nwaTablesNMineralPixels", "nwaDFoo", "nwaBar"}
    defs = "\\newcommand{\\nwaTablesNMineralPixels}{1}\n\\newcommand{\\nwaBar}{0.5}\n"
    assert chk.defined_macros(defs) == {"nwaTablesNMineralPixels", "nwaBar"}

def test_check_reports_undefined_macro(tmp_path, capsys):
    (tmp_path / "main.tex").write_text(r"\nwaMissing")
    (tmp_path / "numbers.tex").write_text("")
    (tmp_path / "derived.tex").write_text("")
    rc = chk.main(["--tex", str(tmp_path / "main.tex"), "--defs", str(tmp_path / "numbers.tex"), str(tmp_path / "derived.tex")])
    assert rc == 1 and "nwaMissing" in capsys.readouterr().out

def test_check_reports_derived_input_missing(tmp_path, capsys):
    (tmp_path / "main.tex").write_text(r"\nwaDTwice")
    (tmp_path / "numbers.tex").write_text("")
    (tmp_path / "derived.tex").write_text(r"\newcommand{\nwaDTwice}{\fpeval{\nwaBase*2}}")
    rc = chk.main([...same args...])
    assert rc == 1 and "nwaBase" in capsys.readouterr().out

def test_check_passes_on_the_repository(repo_root):
    assert chk.main(["--tex", str(repo_root / "main.tex"), "--defs", str(repo_root / "numbers.tex"), str(repo_root / "derived.tex")]) == 0
```
The last test runs against the real files and passes only after Task 8; mark it `@pytest.mark.repo` and run it in `make check`, not in the unit run, until then.

- [ ] **Step 2: Run to verify failure**; **Step 3: Implement**; **Step 4: Run to verify pass** (3 unit tests).

- [ ] **Step 5: Preamble and derived.tex**

In `main.tex` after line 16 (`\TODO` definition) add `\usepackage{xfp}` (if `cas-dc` does not already load it; check the log for a clash) and `\input{numbers}` then `\input{derived}`. Create `derived.tex` with the initial set above. `make pdf` must still build (no macro is used yet). `make check` passes (nothing used). Remove `continue-on-error` from the workflow.

- [ ] **Step 6: Commit**

`feat: numbers.tex and derived.tex in the preamble; check script`.

### Task 6: Table 1 and the Methods section

**Files:**
- Create: `docs/number-map.md` (moved from the karak plan directory; updated as Tasks 6 to 8 proceed with a "done" column)
- Modify: `main.tex` lines 264 to 551 (Methods) and 1075 to 1101 (Table 1)

**Interfaces:**
- Consumes: the number map's "Methods" and "Table params" sections; `\nwaTablesParams...` macros (`DownsampleFactor`, `PCAVarianceKept`, `Normalisation`, `RarePhaseSubsample`, `MinimumClustersPerTile`, and the others the map lists).

- [ ] **Step 1: Replace** each MATCH and DIFFERS row in those sections with its macro (`\num{}` for counts, `$\nwaTablesParamsDownsampleFactor\times$` for the factor, `\nwaDTablesParamsPCAVarianceKeptPct\%` for the PCA variance, the `Normalisation` token through a two-entry display mapping `\nwaDNormalisationName` in `derived.tex`). DIFFERS rows get the `\TODO{was: ...}` marker. NO MACRO rows and Table B rows get `% source: <file or reference>` at the end of the line. Mark each row "done" in `docs/number-map.md`.

- [ ] **Step 2: Build and check**

Run: `make pdf && make check`. Expected: both pass; `grep -c '\\nwa' main.tex` equals the number of rows marked done (report the count).

- [ ] **Step 3: Commit** `feat: Methods and Table 1 numbers as macros` (one commit; the body lists the count of macros and TODO markers).

### Task 7: Results sections 3.1 to 3.4, Tables 2 to 4, the Fig 6 and Fig 8 captions

Same step shape as Task 6, one commit per subsection: 3.1 Pipeline Output with Table 2 (`feat: results 3.1 and Table 2 as macros`); 3.2 TIMA with Tables 3 and 4 (`feat: results 3.2 and Tables 3, 4 as macros`); 3.3 Rare phases; 3.4 Multi-resolution refinement with the lamellae paragraph and both captions (`feat: results 3.3, 3.4 and the lamellae numbers as macros`).

Specific rulings for this task:
- Table 2's "Pyroxene (unres.)" row (92 px) has no macro because the regenerated run has no unresolved pyroxene: keep the row, replace the count with `\TODO{row absent in the regenerated run; 92 px in the published file}`, and leave the author to drop or keep it.
- "16 phases" (lines 438 and 557) becomes `\nwaTablesNPhasesWithPixels` with a `\TODO{was 16: the regenerated run has 15 phases with pixels}`.
- The lamellae paragraph: every statistic becomes a macro (`\nwaLamellaeNGrainsTotal`, `...NGrainsAnalyzed`, `\nwaDLamellaeNGrainsExcluded`, `\nwaLamellaeSpacingMedianUm`, `...SpacingQTwoFiveUm`, `...SpacingQSevenFiveUm`, `...SpacingMaxUm`, `...OrientationCircularMeanDeg`, `...OrientationRayleighRBar`, `\pthresh{\nwaLamellaeOrientationRayleighPValue}`, `...GrainSizeVsSpacingSpearmanRho`, `\pthresh{...SpearmanP}`); the sentence that claims a statistically significant preferred direction gets `\TODO{the regenerated Rayleigh p is 0.80; the claim no longer holds; see docs/acceptance-2026.md on the spacing-direction bias}`. The implementer does not rewrite the sentence.
- Pixel counts in tables use `\num{\nwa...}` with `\sisetup{group-separator={,}, group-minimum-digits=4}` in the preamble (Task 5 adds the `\sisetup`).
- Percentages under 0.005 use `\pctsmall{}`.

Each subsection commit: `make pdf && make check` pass; the body lists macros and TODO counts.

### Task 8: Abstract, Introduction, Sensitivity, Discussion, Conclusions, Software and Data Availability

Same step shape. Sensitivity (3.5) and the transfer run (3.6) are Table B: `% source: sensitivity/<file>.json` comments only, no macros; where the map says "unknown" write `% source: unknown (old pipeline run; see docs/number-map.md)`. The Abstract, Introduction, Discussion and Conclusions repeat headline numbers (84.1 %, 99.1 %, 15 or 16 phases, 12.5 million pixels, the lamellae values): macros plus TODO markers where the value changed. Final commit: `feat: remaining sections as macros; number map complete`. Then:

- [ ] **Step: Totals**

Run: `make pdf && make check && (cd tools && uv run --group dev pytest -q -m repo)`. Expected: pass. Report `grep -c '\\nwa' main.tex`, `grep -c 'TODO{' main.tex`, and `grep -c '% source:' main.tex` in the PR 2 body, with the three counts from `docs/number-map.md` (149 match, 84 differ, 48 no macro, 43 non-pipeline) so a reviewer can reconcile them.

- [ ] **Step: Open PR 2**

`gh pr create --title "feat: every pipeline number in main.tex is a macro" --body-file <scratch body>`: what changed, the counts, the TODO markers as the author's work list grouped by section, the derived macros, how tested (`make pdf`, `make check`, pytest), rulings (no prose rewritten; the unresolved-pyroxene row; the lamellae claim). `gh pr checks --watch`.

---

# Part 3: documentation and archive (PR 3, branch `feat/readme-archive`)

### Task 9: README, sensitivity README, docs

**Files:**
- Modify: `README.md`; Create: `sensitivity/README.md`

- [ ] **Step 1: `sensitivity/README.md`**: the seven JSON files, the script that made each, the data file they read (`data/eds_pipeline.h5` of the archived repository, not present here), the commit they ran at (`git log -1 --format=%h -- sensitivity/` in the archive clone), and that they are a frozen record; making them runnable is a later opt-in in `nwa4587-karak-paper`.
- [ ] **Step 2: `README.md`** sections: what this repository is; layout; build (`make pdf`, Docker fallback, `make reviewpdf FROM=review-r1`); the import (`make import IMPORT_ARGS=...`, `FIGURES.lock`, the asset list, the release contract); numbers as macros (`numbers.tex`, `derived.tex`, `make check`, the `\TODO` convention, `docs/number-map.md`); the archive repository and the migration record; licensing; how to cite (the paper title and journal reference are pending; say so). Prose rules apply.
- [ ] **Step 3**: `make pdf && make check`; commit `docs: README and the sensitivity record`; push; open PR 3 (draft until Task 10).

### Task 10: archive the old repository

- [ ] **Step 1: Consent**: "Task 10 tags the old repository `brendonhall/NWA4587_LPSC26` at its reconciled `main` as `archive-2026-10` and marks it archived on GitHub (read-only). The Dropbox folder is not touched. May I?" Stop until yes.
- [ ] **Step 2**: in `~/Projects/science/.migrate-nwa4587/full`: `git tag -a archive-2026-10 -m "Archived 2026-10: the manuscript moved to brendonhall/nwa4587-paper" && git push origin archive-2026-10 && gh repo archive brendonhall/NWA4587_LPSC26 --yes`. Expected: `gh repo view brendonhall/NWA4587_LPSC26 --json isArchived -q .isArchived` prints `true`.
- [ ] **Step 3**: append the tag and the archive date to `docs/migration-2026-10.md`; commit `docs: archive tag recorded`; `gh pr ready`; body; `gh pr checks --watch`. Remove `~/Projects/science/.migrate-nwa4587` only after the user confirms PR 3 is merged (say so in the PR body).

---

# Part 4: the release workflow in `nwa4587-karak-paper` (PR 7 there, branch `ci/release-assets`; runs after that repository's PRs 1 to 6 are merged)

### Task 11: release assets

**Files (in `/home/brendon/Projects/science/nwa4587-karak-paper`):**
- Create: `.github/workflows/release.yml`; Modify: `README.md` (a "Releases" section)

- [ ] **Step 1**: workflow on `push: tags: ['v*']`: checkout, setup-uv, `uv sync`, `uv run paper synth smoke` is NOT used (the release carries real-data outputs, which CI cannot produce); instead the workflow fails unless the tag's commit contains the committed acceptance files, and it uploads the committed `expected/numbers.json` and `docs/acceptance-2026.md` plus the figure PDFs and tables the user attaches by hand. Ruling: the real figures are produced locally; the release step is `gh release create vX.Y.Z results/figures/*.pdf results/tables/*.tex results/numbers.tex results/numbers.json docs/acceptance-2026.md --notes-file docs/acceptance-2026.md`, documented in the README; the workflow only verifies that a published release has exactly the asset list of this plan's Global Constraints (a `check-release` job triggered on `release: published` that lists assets with `gh release view --json assets` and fails on a missing name).
- [ ] **Step 2**: README section with the command; commit `ci: verify release assets; document the release command`; PR; checks.
- [ ] **Step 3** (local, after the user cuts `v0.1.0`): in `nwa4587-paper`, `make import IMPORT_ARGS="--tag v0.1.0"`, confirm `FIGURES.lock` carries the tag and the file hashes equal the previous import's, commit `chore: import from release v0.1.0`, PR, merge by the user.

---

## Self-review notes

- Spec coverage: Section 1 (layout, migration, reconciliation, archive) is Tasks 1, 2, 10; Section 2 (import, lock, asset list, release) is Tasks 4 and 11; Section 3 (macros, guard, tables hand-written, typed numbers with sources, TODO markers for changed claims) is Tasks 5 to 8; Section 4 (build, CI, licensing, delivery in three PRs plus one) is Tasks 2, 3, 9 and the PR boundaries.
- Deviations from the spec, with reason: `derived.tex` is added because 84 of 281 numbers need a different format or an arithmetic combination of macros (percent from fraction, millions, excluded = total minus analyzed, p thresholds); the spec's "every number becomes a macro" is kept by making derived macros depend only on generated ones, and the check script verifies that. The spec said three result JSON files; there are seven, all frozen. The release workflow cannot regenerate real-data figures in CI, so it verifies a release's asset list instead and the release command is documented.
- Type consistency: `import_results.ASSETS`, `plan`, `dirty`, `copy_all`, `write_lock`, `main` and `check_numbers.used_macros`, `defined_macros`, `derived_inputs`, `main` are the names Tasks 4, 5 and the Makefile use. Derived macro names start with `\nwaD`; helper commands are `\pthresh` and `\pctsmall`.
- Review Focus: 1 and 5 in Task 5, 2 and 3 in Task 4, 4 in Tasks 2 and 3.
- Known soft spots: `cas-dc` may already load `xfp` or clash with `siunitx` `\sisetup` placement (check the log); the first `make pdf` pulls a 4 GB image; the macro names in Tasks 6 to 8 are the ones `numbers.tex` holds today (verified against the current file for the names this plan spells out), and the number map lists the rest.
