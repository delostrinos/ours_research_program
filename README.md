# ours_research_program

Symbolic regression for mathematical inference. Made by Spongebob Squarepants.

This repository is the shared workspace for our research group. We use a feature-branch workflow on top of `main`: every contributor branches off `main` for the solution they are working on, pushes that branch to GitHub, and opens a pull request when it is ready for review.

---

## Prerequisites

- Git installed locally (`git --version` to check).
- Python 3.10 or newer (`python3 --version` to check). See [Setting up Python](#setting-up-python) below if you need to install it.
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

## Setting up Python

The experiments in this repo are pure Python (NumPy + Matplotlib so far). Each contributor works inside a per-clone virtual environment, so dependencies stay isolated from your system Python.

### Install Python

- **macOS:** install via [Homebrew](https://brew.sh/): `brew install python`. (The system Python that ships with macOS works too, but Homebrew's is easier to keep current.)
- **Linux:** use your package manager, e.g. `sudo apt install python3 python3-venv` on Debian/Ubuntu.
- **Windows:** install from [python.org](https://www.python.org/downloads/) and tick **"Add Python to PATH"** in the installer.

Confirm the install with `python3 --version` (macOS/Linux) or `python --version` (Windows). You want 3.10 or newer.

### Create a virtual environment

Run this once, from the repo root, after cloning:

```bash
python3 -m venv .venv
```

This creates a `.venv/` directory containing a private copy of Python and its own `pip`. The `.venv/` directory is already in `.gitignore` so it never gets committed.

### Activate the environment

You need to activate the venv in every new terminal session before running scripts or installing packages.

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

Once activated, your prompt will be prefixed with `(.venv)` and plain `python` / `pip` will resolve to the venv's copies. When you're done, run `deactivate` to leave the environment.

> If you'd rather not activate, you can always call the venv's interpreter explicitly: `.venv/bin/python script.py` (macOS/Linux) or `.venv\Scripts\python.exe script.py` (Windows).

### Install dependencies

With the venv active, install everything in one go from the tracked `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you add a new dependency in your branch, install it with `pip install <package>`, then append the pinned line (`<package>==<version>`, copy the version from `pip freeze`) to `requirements.txt` and commit it. Other contributors will pick the new dep up on their next `pip install -r requirements.txt`.

---

## Running an experiment

Experiment scripts live under `experiments/`. With the venv active, run any of them directly:

```bash
python experiments/recover_circle_area.py
```

Scripts that produce figures (like the disk-area experiment) save them under `figures/` and may also open an interactive matplotlib window. The `figures/` directory is gitignored — figures are reproducible from the script, so we don't track the PNGs in git.

To view a saved figure on macOS:

```bash
open figures/visual_of_the_result.png
```

On Linux use `xdg-open`; on Windows use `start`.

If a script opens an interactive window via `plt.show()`, the script blocks until you close the window. Close it to let the script exit and free the terminal.

---

## Rules of the road

- **Never commit directly to `main`.** All changes land via pull request.
- **Pull before you push.** If `git push` is rejected because the remote has new commits on your branch, run `git pull --rebase` and try again.
- **Don't force-push to shared branches** (especially `main`). Force-pushing your own feature branch is okay if you know what you're doing.
- **When in doubt, ask before doing anything destructive** (`git reset --hard`, `git push --force`, deleting branches you didn't create). Ping Mikhail.

testing123
