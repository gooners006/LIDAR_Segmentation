---
name: commit
description: >
  Commit local changes and push git commits to the configured remote. Use this
  skill whenever the user asks to commit, push, or save changes to git — even
  if they just say "commit this" or "push it" without elaborating.
---

# Commit & Push

Commit current changes and push to the remote. Follow this sequence exactly.

## Step 1: Gather context

Run these in parallel:
- `git status --short --branch` to see changes and ahead/behind state
- `git diff --name-status` and `git diff --cached --name-status` to see unstaged and staged changes
- `git ls-files --others --exclude-standard` to list untracked files
- `git log --oneline -5` to see recent commit style

If there are no changes and no unpushed commits, tell the user and stop.

If the user asked only to push:
- If there are unpushed commits, skip to Step 6.
- If there are local changes but no unpushed commits, ask whether to commit them first.
- If there are neither, stop.

Note: `git status --short --branch` only shows ahead/behind when an upstream is configured. If no upstream exists, treat the branch as "no upstream configured" and handle during Step 6.

## Step 2: Safety scan

Scan staged, unstaged, and untracked candidate files for:
- **Sensitive filenames:** `.env`, `.env.*`, `credentials.*`, `*secret*`, `*.pem`, `*.key`, `token*`
- **Secret-like content** (including but not limited to): `sk-`, `ghp_`, `BEGIN PRIVATE KEY`, `AWS_ACCESS_KEY_ID`, `password=`, `api_key=`, `token=`, `secret=`
- **Large artifacts:** model checkpoints (`.pth`, `.pt`, `.ckpt`), datasets (`.npy`, `.npz`), generated outputs (`.ply`), notebook checkpoints

Include already-staged files in the scan — do not skip them. Do not attempt to scan binary or very large files as text; flag them by path, extension, and size instead.

If risky files are found, warn the user and list them. Do NOT stage or commit them unless the user explicitly approves. Do not claim the scan proves the commit is secret-free.

## Step 3: Stage changes

Show the list of candidate files grouped by status (staged, modified, deleted, untracked).

If files are partially staged (appear in both staged and unstaged), warn and ask whether to commit only the staged portions or include unstaged changes too.

If there are already staged changes and the user said "commit this," ask whether to include unstaged changes or commit only what's staged.

Stage only approved paths explicitly:

```
git add -- <path1> <path2> ...
```

Quote paths with spaces or special characters. Do not use `git add -A`.

## Step 4: Draft the commit message

Write in imperative mood, descriptive style. No type prefixes, no emojis, no Co-Authored-By line.

- **Simple changes:** use a one-line summary.
- **Multi-file or non-obvious changes:** use a one-line summary + blank line + body. Bullets are preferred but short prose is acceptable.

```
Fix ground plane normal check rejecting valid planes
```

```
Add PCN model, training script, and dataset documentation

- Implement encoder-decoder with chunked Chamfer Distance loss
- Add ShapeNet training pipeline with depth-based partial generation
- Document both datasets (SemanticKITTI, ShapeNetCore v2)
```

Rules:
- Summary line under 72 characters
- Body uses dashes (`-`) as bullet points
- Focus on the "why" and "what", not the "how"

Show the draft message to the user. Wait for approval or edits before committing.

## Step 5: Commit

Before committing, verify something is staged with `git diff --cached --name-status`. If nothing is staged, stop.

Write the commit message to a temporary file outside the repository (OS temp directory). Commit with:

```
git commit -F <temp-file>
```

Delete the temp file after the commit attempt, whether it succeeds or fails.

## Step 6: Push

If the user only asked to commit (not push, save, or ship), ask whether to push before proceeding.

Before pushing, check the current branch and remote state:

```
git branch --show-current
git remote -v
```

- If on `main` or `master`, warn the user and require explicit confirmation.
- If no upstream is set, show the proposed remote/branch and ask before running `git push -u origin <current-branch>`. If multiple remotes exist, ask which to use.
- If the branch is behind the remote, stop and report the issue — do not pull or rebase automatically.
- If push is rejected, stop and report the reason.

After a successful push:
1. Run `git status --short --branch` to confirm clean state.
2. Report the commit hash, branch, and remote pushed to.
