# Vercel hosting

Import `Mstolte02/cfb-model-v3` into Vercel, use repository root `.` and production
branch `main`. The checked-in `vercel.json` supplies framework, install command,
build command and output directory. No API keys are needed to publish the site.

The build runs the browser-model regression tests and `scripts/publish.py`, which
copies `viz/` into `dist/`, excludes the working WAR diagnostic, and fingerprints
JavaScript and CSS. Routes under `power-xray/`, logos, JSON data and team artwork
remain at their existing relative paths. Client-side hash links remain valid.
Do not add a catch-all SPA rewrite: the site has real nested HTML pages.

Vercel deploys application changes directly from Git; website publication does
not wait for GitHub Actions or rebuild the season simulations.

## Model and data updates

The six-hour market capture retains its current cadence and credentials in
GitHub Actions. After updating results and live ratings, it dispatches
`pages.yml` (now **Refresh playoff projections**). That R workflow simulates
preseason/current playoff outcomes and commits the two refreshed JSON files.
Vercel then publishes the updated data. Model refreshes still require a GitHub
runner; hosting migration does not migrate the scheduled computations.

Both scheduled commits and manual source pushes should produce Vercel Git
integration deployments. Verify the first bot-authored data commit in Vercel.
If account policies reject automation authors, configure a Vercel Deploy Hook
for that workflow before relying on unattended data publication.

## Cutover checks

- A production Vercel deployment passes the existing JavaScript tests.
- The main page, Mark's Playoff, Projected bracket, and Power Rating X-Ray open.
- Data files return JSON; nested images and scripts load.
- A model-data commit triggers a Vercel build.
- Share the Vercel production URL after successful deployment.

The existing GitHub Pages copy is left available during cutover. Its old queued
runs do not block Vercel publication. Disable Pages hosting after the Vercel
production deployment is verified if the old address is no longer needed.
