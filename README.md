# ACE-Ego Project Page

This repository contains the static project homepage for:

**ACE-Ego: Unifying Egocentric Human and Robotic Data for VLA Pretraining**

The page is plain HTML, CSS, and JavaScript, so it can be deployed directly with
GitHub Pages.

## Content Layout

- `index.html` contains the paper title, authors, abstract, method overview,
  real-robot demos, benchmark tables, figures, and BibTeX.
- `assets/videos/` contains web-compressed MP4 demos for GitHub Pages.
- `assets/posters/` contains video poster frames.
- `assets/figures/` contains web PNG figures rendered from the paper PDFs,
  including the teaser, method, data pipeline, fine-tuning coverage, and result figures.
- `assets/ACE_Logo.png` contains the ACE Robotics logo used in the hero.

The raw `整理视频/` directory and `整理视频.zip` are intentionally ignored because
they are too large for normal GitHub hosting.

## Local Preview

Open `index.html` directly in a browser, or run a small static server:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## GitHub Pages

1. Push this directory to a GitHub repository.
2. Open repository settings.
3. Go to Pages.
4. Select the branch that contains `index.html`.
5. Use the repository root as the publishing directory.
