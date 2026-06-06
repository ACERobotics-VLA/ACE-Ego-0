# ACE-Ego Project Page

This repository contains a static academic project homepage inspired by modern
robotics and vision-language-action project pages. It is intentionally plain
HTML, CSS, and JavaScript so it can be deployed directly with GitHub Pages.

## Edit Checklist

- Change the title, authors, affiliations, and links in `index.html`.
- Replace `assets/hero-poster.svg` with your teaser poster or keep it as a fallback.
- Add videos as `assets/teaser.mp4`, `assets/demo-1.mp4`, and `assets/demo-2.mp4`.
- Replace `assets/framework.svg` with your actual method figure.
- Update the abstract, result table, resource links, and BibTeX entry.

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
