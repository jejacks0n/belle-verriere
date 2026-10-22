#!/usr/bin/env bash
# Open the display page as a full-screen kiosk window on the second screen (the window panel).
# Uses a separate Chrome profile so it never disturbs your normal browser windows.
set -euo pipefail
URL=${1:-http://localhost:8765/display}

# Ask macOS where the non-main screen is (Chrome wants top-left coordinates in points).
POS=$(osascript -l JavaScript -e '
ObjC.import("AppKit");
var screens = $.NSScreen.screens.js;
var main = screens[0].frame;
var out = "";
for (var i = 1; i < screens.length; i++) {
  var f = screens[i].frame;
  var top = main.size.height - (f.origin.y + f.size.height);
  out = Math.round(f.origin.x) + "," + Math.round(top);
}
out;' 2>/dev/null || true)

if [ -z "$POS" ]; then
  echo "Only one screen found. Opening the display here; drag it to the panel and press Cmd+Ctrl+F."
  POS="0,0"
fi

CHROME="/Applications/Google Chrome.app"
if [ -d "$CHROME" ]; then
  open -na "$CHROME" --args \
    --user-data-dir="$HOME/.stained-window-display" \
    --window-position="$POS" --kiosk --app="$URL" \
    --no-first-run --disable-session-crashed-bubble --noerrdialogs --autoplay-policy=no-user-gesture-required
else
  echo "Google Chrome not found; opening in your default browser instead. Make it full screen on the panel."
  open "$URL"
fi
