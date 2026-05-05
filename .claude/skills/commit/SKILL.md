---
name: commit
description: >
  Commit and push changes to GitHub. Use this skill whenever the user asks to
  commit, push, save changes to git, or ship code — even if they just say
  "commit this" or "push it" without elaborating.
---

# Commit & Push

Commit all current changes and push to the remote. Follow this sequence exactly.

## Step 1: Gather context

Run these in parallel:
- `git status` (never use `-uall`)
- `git diff` and `git diff --cached` to see all staged and unstaged changes
- `git log --oneline -5` to see recent commit style

If there are no changes (no untracked files, no modifications), tell the user and stop.

## Step 2: Check for sensitive files

Before staging, scan untracked and modified files for sensitive content:
- `.env`, `.env.*`, `credentials.*`, `*secret*`, `*.pem`, `*.key`, `token*`
- Any file that looks like it contains API keys, passwords, or tokens

If found, warn the user and list the files. Do NOT stage them. Ask the user to confirm before proceeding.

## Step 3: Stage all changes

Run `git add -A` to stage everything (minus any sensitive files the user excluded).

## Step 4: Draft the commit message

Write in imperative mood, descriptive style. No type prefixes, no emojis, no Co-Authored-By line.

Always use a one-line summary + blank line + bulleted body, regardless of change size:

```
Fix ground plane normal check rejecting valid planes

- Relax normal Z-component threshold from 0.8 to 0.5
- Tilted but valid ground surfaces were being filtered out
```

```
Add PCN model, training script, and dataset documentation

- Implement encoder-decoder with chunked Chamfer Distance loss
- Add ShapeNet training pipeline with depth-based partial generation
  via Open3D RaycastingScene
- Document both datasets (SemanticKITTI, ShapeNetCore v2)
- Add PCN architecture docs (encoder, decoder, loss)
```

Rules:
- Summary line under 72 characters
- Body uses dashes (`-`) as bullet points, one item per line
- Each bullet should be a concise, self-contained point
- Focus on the "why" and "what", not the "how"
- Use semicolons or commas to separate independent changes in the summary when there are 2-3 small ones

Show the draft message to the user. Wait for approval or edits before committing.

## Step 5: Commit and push

After the user approves the message, ask which platform they are on (Windows or Mac) before committing. This determines the shell syntax for multiline commit messages.

**Windows (PowerShell)** — use a here-string:
```powershell
git commit -m @'
Summary line here

- Bullet one
- Bullet two
'@
```
The closing `'@` must be at column 0 on its own line.

**Mac (Bash)** — use a HEREDOC:
```bash
git commit -m "$(cat <<'EOF'
Summary line here

- Bullet one
- Bullet two
EOF
)"
```

Then:

1. Push to remote: `git push`
   - If no upstream is set, use `git push -u origin <current-branch>`

2. Run `git status` to confirm clean state.

3. Report the commit hash and confirm the push succeeded.
