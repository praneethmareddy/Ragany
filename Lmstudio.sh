#!/bin/bash

# --- CONFIGURATION ---
# IMPORTANT: Check https://lmstudio.ai/download for the latest version link if this one is outdated.
# Current as of late 2024/early 2025
DOWNLOAD_URL="https://installers.lmstudio.ai/linux/x64/0.3.26-2/LM-Studio-0.3.26-2-x64.AppImage"
FILENAME="LM-Studio.AppImage"
INSTALL_DIR="$HOME/.local/bin"
ICON_PATH="$HOME/.local/share/icons"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Starting LM Studio Installation ===${NC}"

# 1. Check & Install Dependencies (libfuse2)
echo -e "${BLUE}[1/5] Checking dependencies...${NC}"
if dpkg -l | grep -q libfuse2; then
    echo -e "${GREEN}✓ libfuse2 is already installed.${NC}"
else
    echo -e "${RED}libfuse2 is missing. Installing it now (requires sudo)...${NC}"
    sudo apt update && sudo apt install -y libfuse2
fi

# 2. Setup Directories
echo -e "${BLUE}[2/5] Setting up installation directories...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$ICON_PATH"

# 3. Download LM Studio
echo -e "${BLUE}[3/5] Downloading LM Studio...${NC}"
if [ -f "$INSTALL_DIR/$FILENAME" ]; then
    echo -e "Existing file found. Backing it up..."
    mv "$INSTALL_DIR/$FILENAME" "$INSTALL_DIR/${FILENAME}.bak"
fi

wget -O "$INSTALL_DIR/$FILENAME" "$DOWNLOAD_URL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Download failed! Please check your internet connection or update the DOWNLOAD_URL in the script.${NC}"
    exit 1
fi

chmod +x "$INSTALL_DIR/$FILENAME"
echo -e "${GREEN}✓ Download complete and executable permissions set.${NC}"

# 4. Create Desktop Entry (Shortcut)
echo -e "${BLUE}[4/5] Creating Desktop Shortcut...${NC}"
# Download a logo for the icon
wget -q -O "$ICON_PATH/lmstudio.png" "https://lmstudio.ai/static/media/lmstudio-logo.82b2024c.png"

cat > "$HOME/.local/share/applications/lm-studio.desktop" << EOF
[Desktop Entry]
Name=LM Studio
Comment=Run local LLMs
Exec=$INSTALL_DIR/$FILENAME
Icon=$ICON_PATH/lmstudio.png
Terminal=false
Type=Application
Categories=Development;Science;AI;
StartupWMClass=LM Studio
EOF

update-desktop-database "$HOME/.local/share/applications" 2>/dev/null
echo -e "${GREEN}✓ Desktop shortcut created.${NC}"

# 5. Bootstrap "lms" CLI Tool
echo -e "${BLUE}[5/5] Setting up 'lms' command line tool...${NC}"
# LM Studio needs to be run once to unpack files, but we can try to bootstrap if the cache exists.
# Usually, the bootstrap binary lives in ~/.cache/lm-studio/bin/lms
LMS_BIN_PATH="$HOME/.cache/lm-studio/bin/lms"

if [ -f "$LMS_BIN_PATH" ]; then
    echo "Found lms binary. Bootstrapping..."
    "$LMS_BIN_PATH" bootstrap
    echo -e "${GREEN}✓ 'lms' command enabled. Restart your terminal to use it.${NC}"
else
    echo -e "${BLUE}ℹ️  NOTE: To enable the 'lms' terminal command:${NC}"
    echo "   1. Open LM Studio once."
    echo "   2. Run this command: ~/.cache/lm-studio/bin/lms bootstrap"
fi

echo -e "${GREEN}=== Installation Complete! ===${NC}"
echo -e "You can now launch LM Studio from your applications menu or by running:"
echo -e "  $INSTALL_DIR/$FILENAME"
