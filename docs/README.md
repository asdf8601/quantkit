# quantkit documentation

Read the documentation here on GitHub, even when the
[documentation website](https://asdf8601.github.io/quantkit/) is unavailable.

- [Installation](reference/install.md)
- [Quick start](reference/quickstart.md)
- [Portfolio statistics and examples](reference/portfolios.md)
- [Complete API reference](reference/autoapi/index.md)
- [Documentation contents](reference/index.md)

## Updating documentation

Edit the guides in `source/*.rst` or the docstrings in `../src/quantkit/`.
The files in `reference/` are generated: do not edit them directly.
From the repository root:

```bash
uv sync --locked --no-default-groups --group docs
uv run --no-sync make -C docs repo
uv run --no-sync make -C docs check-repo
uv run --no-sync make -C docs html SPHINXOPTS="-W --keep-going"
```

Commit the changed Markdown files alongside their sources. CI regenerates a
clean copy and rejects missing, obsolete or changed files. HTML and caches are
build artifacts and are not committed. Open `build/html/index.html` locally to
preview the website.

## GitHub Pages

The `Documentation` workflow validates every pull request and publishes the
latest `master` documentation after a successful build. It can also be run
manually from the Actions tab, selecting `master`. Other branches do not deploy.

In repository **Settings → Pages**, the publishing source must be **GitHub
Actions**. The `github-pages` environment must allow deployments from `master`.
The site is served at <https://asdf8601.github.io/quantkit/> without a custom
domain. Release tags continue to control PyPI publishing separately.

To investigate a failed publication, inspect the `Documentation` workflow's
build and deploy jobs and the `github-pages` environment. The committed copy
above remains available independently of the deployment.
