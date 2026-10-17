# AGENTS.md - molecular-dynamics-playground

## Push rule (standing, from 2026-10-17)
- Every change an agent produces in this repo must be committed AND pushed to origin.
- Before pushing: `git fetch origin` and check `git rev-list --left-right --count main...origin/main`.
  If origin has diverged, rebase or merge cleanly; never force-push. If it cannot be done without force, stop and report.
- Never stage the user's own uncommitted edits (e.g. scripts/md_visuals/common.py, other renderers) unless they are already
  committed. If pushed code only works with uncommitted working-tree edits, report it and propose a fix instead of committing them.
- Never write to the user's C: drive; staging/scratch goes under D:\CRYSTAL_PHD\PROGRESS_REPORT\202610xx\_archive.
