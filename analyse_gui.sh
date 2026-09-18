#!/usr/bin/env bash
# Launch the analysis GUI, creating/updating the local venv first if needed.
# Usage: ./analyse_gui.sh [file.wav ...]
#        ./analyse_gui.sh --install-desktop     (add "Audio Analysis" with icon to the app menu)
#        ./analyse_gui.sh --uninstall-desktop
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
APP_ID="audio-analysis"
DESKTOP_FILE="${DATA_HOME}/applications/${APP_ID}.desktop"
ICON_SVG="${DATA_HOME}/icons/hicolor/scalable/apps/${APP_ID}.svg"
ICON_PNG="${DATA_HOME}/icons/hicolor/256x256/apps/${APP_ID}.png"

refresh_desktop_caches() {
    gtk-update-icon-cache -f -t "${DATA_HOME}/icons/hicolor" >/dev/null 2>&1 || true
    update-desktop-database "${DATA_HOME}/applications" >/dev/null 2>&1 || true
}

if [ "${1:-}" = "--install-desktop" ]; then
    install -Dm644 "${ROOT_DIR}/gui/icon.svg" "${ICON_SVG}"
    install -Dm644 "${ROOT_DIR}/gui/icon.png" "${ICON_PNG}"
    mkdir -p "$(dirname "${DESKTOP_FILE}")"
    # StartupWMClass must match the Tk className set in gui/app.py
    cat > "${DESKTOP_FILE}" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Audio Analysis
Comment=Compare impulse-response analyses side by side
Exec="${ROOT_DIR}/analyse_gui.sh" %F
Icon=${APP_ID}
Terminal=false
Categories=AudioVideo;Audio;
MimeType=audio/x-wav;audio/wav;
StartupWMClass=Audio-analysis
DESKTOP
    refresh_desktop_caches
    echo "Installed ${DESKTOP_FILE}"
    exit 0
fi

if [ "${1:-}" = "--uninstall-desktop" ]; then
    rm -f "${DESKTOP_FILE}" "${ICON_SVG}" "${ICON_PNG}"
    refresh_desktop_caches
    echo "Removed ${DESKTOP_FILE}"
    exit 0
fi

cd "${ROOT_DIR}"

STAMP=".venv/.installed"
if [ ! -x .venv/bin/python ] || [ ! -f "${STAMP}" ] || [ requirements.txt -nt "${STAMP}" ]; then
    bash setup.sh
    touch "${STAMP}"
fi

if ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
    echo "ERROR: tkinter is not available. Install it with: sudo apt install python3-tk" >&2
    exit 1
fi

# Resolve file arguments before they're interpreted relative to ROOT_DIR
files=()
for f in "$@"; do files+=("$(cd "${OLDPWD}" && readlink -f "$f")"); done

exec .venv/bin/python -m gui "${files[@]}"
