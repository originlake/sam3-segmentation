# Source control for the installed skill

This checkout tracks https://github.com/originlake/sam3-segmentation as `origin`.
Local `main` tracks `origin/main`, preserving the fork's existing history.
The skill lives under `skills/sam3-segmentation/`.

The installed skill path is a symlink to that folder, so edits made through the
installed path also appear in this checkout. A backup of the previous installed
folder is kept separately from the checkout.

From the repository root, review and save changes with:

```sh
git status --short
git diff
git add <changed-files>
git diff --cached
git commit -m "Describe the skill change"
git push origin main
```

Commit and push only after review and authorization. Configuring the remote does
not upload local edits. Keep photos and generated annotation runs outside the
checkout; caches, environments, credentials and model weights are ignored.
Dependency manifests and lockfiles remain tracked.

Before reinstalling the skill, check for local edits and preserve the symlink:
an installer may replace it with a standalone copy, disconnecting future edits
from this working tree. Local Git history alone is not an off-machine backup.
