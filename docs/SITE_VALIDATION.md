# Project page interaction checks

Checked on 2026-10-15 with Chromium, WebGL rendering, and mobile touch emulation. The acceptance suite passes all 16 groups, covering 26 stage selections (4 in 3D, 4 in the original comparison, and 18 baseline stages).

## Coverage

- Navigation anchors, duplicate IDs, local resource links and downloads.
- Both rendered 3D views respond to rotation and zoom; screenshots are compared to verify actual scene changes.
- All stage selections, play/pause, loop wrap, camera reset, section depth, cutaway, reference centers and tomogram-section visibility.
- Independent 2D/3D controls; fractional scrubbing resumes without jumping backwards.
- All three baseline selectors, intermediate stages, legends and color scales.
- Disclosure sections, clipboard copying, denied-clipboard fallback and actual `.bib` file contents.
- Hidden-tab playback state and reduced-motion behavior.
- Vertical touch rotation and subsequent taps; layout widths 320, 390 and 768 pixels.
- A failed volume request and a lost WebGL context expose working 2D controls and disable unavailable 3D actions.

No JavaScript errors or unexpected failed local resources occurred in the normal browsing path. Fault-injection checks deliberately simulate unavailable 3D resources separately.

The four distinct external destinations were also checked: GitHub repository, getting-started document and license returned HTTP 200; OpenReview redirects automated requests to its access-check page. These external services are outside this site's control.

## Fixes from this audit

- Scope legacy stage selectors to buttons so they cannot alter the 3D container's state attributes.
- Snap manual 2D scrubbing to the displayed stage before resuming playback.
- Observe the expanded projection viewer independently, so its player is not gated by the much larger page section.
- Preserve playback button state while the browser tab is hidden.
- Give the 3D canvas ownership of touch rotation and release drag state on lost pointer capture.
- Disable 3D actions after a renderer failure instead of leaving unresponsive buttons enabled.

## Repeat the checks

Requires Playwright and its Chromium browser. Start the static preview, then:

```bash
SITE_URL=http://127.0.0.1:8000/ node tools/check_site.cjs
```

`SITE_URL` can also point to the deployed project page. `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` optionally select an existing local Playwright installation and browser executable. The suite uses software WebGL in headless Chromium for reproducibility. These checks do not claim physical-device or Safari/Firefox coverage.
