#!/usr/bin/env bash
# Build the Roku app, and optionally sideload it onto a Roku in developer mode.
#   roku/build.sh                                       # makes roku/belle-verriere.zip
#   ROKU_IP=192.168.1.50 ROKU_PASS=secret roku/build.sh  # ...and installs it
# The app talks to this Mac at its current Wi-Fi address; set SERVER_URL to override.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
SERVER_URL=${SERVER_URL:-http://$($PY -c 'import sys; sys.path.insert(0, ".."); from run import lan_ip; print(lan_ip())'):8765}

BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
cp -R manifest source components "$BUILD/"
sed -i '' "s|__SERVER_URL__|$SERVER_URL|" "$BUILD/manifest"
mkdir "$BUILD/images"
$PY - "$BUILD/images" <<'EOF'
import sys
from PIL import Image, ImageOps
src = Image.open("../examples/reference.png").convert("RGB")
for name, size in (("icon_fhd.png", (540, 405)), ("icon_hd.png", (290, 218))):
    ImageOps.fit(src, size, centering=(0.5, 0.35)).save(f"{sys.argv[1]}/{name}")
EOF

rm -f belle-verriere.zip
(cd "$BUILD" && zip -qr - .) > belle-verriere.zip
echo "Built roku/belle-verriere.zip (server: $SERVER_URL)"

if [ -n "${ROKU_IP:-}" ]; then
  curl -sS --user "rokudev:${ROKU_PASS:?set ROKU_PASS to your Roku developer password}" --digest \
    -F "mysubmit=Install" -F "archive=@belle-verriere.zip" "http://$ROKU_IP/plugin_install" \
    | grep -o 'Install Success\|Identical to previous version\|Install Failure[^<]*' || true
fi
