#!/usr/bin/env bash
# Adds HandsToVoice to the desktop's application menu (per user, no sudo).
# Undo with: rm ~/.local/share/applications/handstovoice.desktop
set -e
APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
DEST="$HOME/.local/share/applications/handstovoice.desktop"
mkdir -p "$(dirname "$DEST")"
cat > "$DEST" <<EOF
[Desktop Entry]
Type=Application
Name=HandsToVoice
Comment=Kinyarwanda Sign Language recognition and voice conversion
Exec="$APP_DIR/run.sh"
Path=$APP_DIR
Icon=$APP_DIR/assets/logo.png
Terminal=false
Categories=Education;Accessibility;
EOF
chmod +x "$APP_DIR/run.sh"
echo "Installed: search for \"HandsToVoice\" in your applications menu."
