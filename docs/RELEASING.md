# Releasing CoNeX

## Steps for a maintainer

1. Bump the version in `setup.py` (`version="X.Y.Z"`).
2. Add a new entry at the top of `HISTORY.rst` with the version and date.
3. Commit both changes to `main`.
4. Create and push an annotated tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.

## What the workflow does

`release.yml` runs on every `v*` tag push (and manually via
`workflow_dispatch` with `target: testpypi` or `none`):

- `verify`: checks the tag against `setup.py` and `HISTORY.rst`
  (`scripts/check_release.py`) and runs the test suite.
- `build`: builds the sdist and wheel, runs `twine check`, installs each
  in a fresh venv and imports `conex`, uploads `dist/` as an artifact.
- `github-release`: creates a GitHub Release with the dist files attached.
- `publish-pypi`: publishes to PyPI via trusted publishing (tag pushes only).
- `publish-testpypi`: publishes to TestPyPI (manual dispatch only).

## One-time PyPI trusted-publisher setup

1. In the GitHub repo: Settings → Deployments → Environments → create
   `pypi` (and `testpypi`) environments.
2. On PyPI: Project → cnrl-conex → Publishing → Add a new trusted publisher.
   - Owner: `cnrl`, Repository: `CoNeX`
   - Environment: `pypi` (repeat for `testpypi`).
3. No API token is stored in the repo; the workflow uses OIDC
   (`permissions: id-token: write`).
