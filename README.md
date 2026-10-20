# kchicken-game.vercel.app — 《從菜雞到股神》 site files

Built from `bsg-learning-system-20261019` (apps/become-stock-god-v2/dist) as the **open-all review build**: every lesson
and layer is open (`window.BSG_OPEN_ALL=true` injected in index.html) for testing. Payment is simulated. The cover is
the intact original art. For the paid launch, rebuild without that line.

Deploy: pushing this branch runs `.github/workflows/kchicken-deploy.yml`, which publishes these files to the Vercel
project `kchicken-game` (needs repository secret `VERCEL_TOKEN`). Or connect the Vercel project to this repository
and branch with Framework "Other", no build command, output directory ".".

GitHub Pages: Settings → Pages → Deploy from a branch → `kchicken-site` / `(root)` → https://xchixmeng-svg.github.io/alphapilot-market-data-v2/
