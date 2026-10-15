# Project page maintenance

The public research README focuses on using TomoDETR. Website source and publishing files are retained here for maintainers; neither is required to train or run the model.

Preview from the repository root:

```bash
python -m http.server 8000 --directory site
```

Open http://localhost:8000. After committing changes on `main`, publish with:

```bash
bash deployment/publish-pages.sh
```

GitHub Pages is configured to serve the root of `gh-pages`. The script publishes only `site/` to that branch. The optional workflow examples are inactive; branch-based publishing is the active deployment path.
