# kchicken-game.vercel.app — 《從菜雞到股神》 site files

Built from `bsg-learning-system-20261019` @ 4dd2cf9 (100-lesson course) (apps/become-stock-god-v2/dist) as the **open-all review build**: every lesson
and layer is open (`window.BSG_OPEN_ALL=true` injected in index.html) for testing. Payment is simulated. The cover is
the intact original art. For the paid launch, rebuild without that line.

Deploy: pushing this branch runs `.github/workflows/kchicken-deploy.yml`, which publishes these files to the Vercel
project `kchicken-game` (needs repository secret `VERCEL_TOKEN`). Or connect the Vercel project to this repository
and branch with Framework "Other", no build command, output directory ".".

GitHub Pages: Settings → Pages → Deploy from a branch → `kchicken-site` / `(root)` → https://xchixmeng-svg.github.io/alphapilot-market-data-v2/

Vercel project `k-chicken` (alias kchicken-game.vercel.app) is Git-connected to this repository. Its Ignored Build Step only
lets `kchicken-site` build, so the daily data commits on `main` no longer replace the site. Production branch in Vercel is
still `main`, so a push here lands as a preview deployment that must be promoted (or set Production Branch to `kchicken-site`).
