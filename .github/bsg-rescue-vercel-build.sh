#!/usr/bin/env bash
set -euo pipefail
URL='https://sdmntpraustraliaeast.oaiusercontent.com/files/00000000-f234-81fa-b958-60090c3762e7/raw?se=2026-10-18T05%3A26%3A11Z&sp=r&sv=2026-02-06&sr=b&scid=d1c1e4d2-583e-5747-831d-58643669ad75&skoid=b1bca990-dcac-4c31-8ca1-ac563cf24c54&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2026-10-18T01%3A58%3A56Z&ske=2026-10-19T01%3A58%3A56Z&sks=b&skv=2026-02-06&sig=q/m9szupqteFU4n41XyPFkhnjWnjmpMiBwEx2RtBD4A%3D'
curl -fL "$URL" -o /tmp/bsg-overlay.tgz
echo 'a7eea896c5a7056d25d5dd1f1b93a853394246b5d5ae2350e2a74889e706b2fb  /tmp/bsg-overlay.tgz' | sha256sum -c -
tar -xzf /tmp/bsg-overlay.tgz
rm -f apps/become-stock-god-v2/assets/chick_god.b64 apps/become-stock-god-v2/assets/hero.b64 apps/become-stock-god-v2/parts/app.09.part
cd apps/become-stock-god-v2
npm install --silent
npm run build:web
cat > /tmp/expected.sha256 <<'EOF'
1f624e52eb13687575c652754e1f90f52895343383430fd193d3b5ad1d18e96f  dist/apple-touch-icon.png
68cb3ebeccfe523ea078ba2a27482942ea793aef72d790c81e5fadb08edf38a5  dist/icon-192.png
f080bcdede1084d04e9ca7e30ed7b7aa71d99aa2fcba7d6d7312ceb547d1cec  dist/icon-512.png
2bd95ace80ab2b7d95bb1152de8a842cebff0790137511a2dbfdc9dc70ddc6dc  dist/index.html
bfd5da8ecf14e3b8830d075eb20a500bf7d4348af410453a5cf44056ae9dae02  dist/manifest.webmanifest
EOF
sha256sum -c /tmp/expected.sha256
