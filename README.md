# ours_research_program

Symbolic regression for mathematical inference.

This repository is the shared workspace for our research group. We use a feature-branch workflow on top of `main`: every contributor branches off `main` for the solution they are working on, pushes that branch to GitHub, and opens a pull request when it is ready for review.

---

## Prerequisites

- Git installed locally (`git --version` to check).
- A GitHub account that has been added as a collaborator on this repository (ping Mikhail if you haven't been added).
- One of the following ways to authenticate to GitHub:
  - **SSH key** added to your GitHub account — see [Adding a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account). Recommended.
  - The [**GitHub CLI**](https://cli.github.com/) (`gh auth login`) — easiest if you don't want to manage SSH keys.
  - HTTPS + a personal access token (works, but more friction).

One-time git config (skip if you've done this before):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 1. Clone the repository

Pick whichever matches how you authenticate:

**SSH (recommended):**
```bash
git clone git@github.com:delostrinos/ours_research_program.git
cd ours_research_program
```

**GitHub CLI:**
```bash
gh repo clone delostrinos/ours_research_program
cd ours_research_program
```

**HTTPS:**
```bash
git clone https://github.com/delostrinos/ours_research_program.git
cd ours_research_program
```

---

## 2. Make sure your local `main` is up to date

Always start a new branch from the latest `main`:

```bash
git checkout main
git pull origin main
```

---

## 3. Create your own branch

Use a name that makes it clear who owns the branch and what it contains. The convention for this repo is `<your-name>/<short-description>`:

```bash
git checkout -b yourname/short-description
```

Examples:
- `mikhail/genetic-algo-baseline`
- `alex/loss-function-exploration`
- `priya/symbolic-pruning`

One branch per solution or experiment — don't pile unrelated work into a single branch, it makes review painful.

---

## 4. Work, commit, repeat

Make your changes, then stage and commit them. Prefer small, focused commits; commit messages should describe *why* the change is being made, not just what.

```bash
git status                       # see what changed
git add path/to/file.py          # stage specific files (avoid `git add .` unless you mean it)
git commit -m "Brief description of the change"
```

---

## 5. Push your branch to GitHub

The first time you push a new branch, set the upstream so future `git push` / `git pull` work without extra arguments:

```bash
git push -u origin yourname/short-description
```

After that, you can just run:

```bash
git push
```

---

## 6. Open a pull request

Once your branch is on GitHub:

1. Go to https://github.com/delostrinos/ours_research_program
2. You should see a banner offering to **Compare & pull request** — click it.
3. Make sure the **base** is `main` and the **compare** is your branch.
4. Write a short description of what your branch does and request a review.

Or from the command line:

```bash
gh pr create --base main --fill
```

Don't merge your own PR without a review.

---

## Keeping your branch in sync with `main`

While your branch is in flight, `main` may move forward. Pull in the latest `main` periodically so you don't end up with a giant conflict at review time:

```bash
git checkout main
git pull origin main
git checkout yourname/short-description
git merge main
# resolve any conflicts, commit, then:
git push
```

---

## Rules of the road

- **Never commit directly to `main`.** All changes land via pull request.
- **Pull before you push.** If `git push` is rejected because the remote has new commits on your branch, run `git pull --rebase` and try again.
- **Don't force-push to shared branches** (especially `main`). Force-pushing your own feature branch is okay if you know what you're doing.
- **When in doubt, ask before doing anything destructive** (`git reset --hard`, `git push --force`, deleting branches you didn't create). Ping Mikhail.
