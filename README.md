# Mark van der Broek Blog

Personal blog built with [Quarto](https://quarto.org/).

## Local Development

Install project tools and Python dependencies:

```bash
mise install
mise run sync
```

Preview the site with live reload:

```bash
uv run quarto preview
```

Render the static site into `docs/`:

```bash
mise run render
```

## Project Structure

- `index.qmd`, `about.qmd`, `projects.qmd`, `useful-links.qmd`: top-level website pages
- `posts/`: blog posts
- `images/`: shared image assets
- `_archive/notebooks/`: archived notebook source material from the previous site setup
- `docs/`: rendered output published through GitHub Pages

## Deployment

GitHub Pages is deployed through the workflow in [.github/workflows/quarto-publish.yml](/Users/mvanderbroek/Projects/blog/.github/workflows/quarto-publish.yml).
