#!/bin/bash

echo "====================================================="
echo "     🚀 LM Studio Headless Server Setup (v2 FIXED)    "
echo "====================================================="

# -------------------------------
# STEP 1 — Dependencies
# -------------------------------
echo "➡️ Installing required packages..."
sudo apt update -y
sudo apt install -y wget curl unzip

# -------------------------------
# STEP 2 — Remove old/broken binary
# -------------------------------
if [ -f "lmstudio-server" ]; then
    echo "➡️ Removing old lmstudio-server binary..."
    rm -f lmstudio-server
fi

# -------------------------------
# STEP 3 — Download correct LM Studio server binary
# -------------------------------
echo "➡️ Downloading LM Studio Server from correct URL..."
curl -L "https://releases.lmstudio.ai/linux/lmstudio-server" -o lmstudio-server

# Check if download succeeded
if [ ! -f "lmstudio-server" ]; then
    echo "❌ ERROR: LM Studio server binary did not download."
    exit 1
fi

chmod +x lmstudio-server

# Validate file size (> 50MB)
FILESIZE=$(stat -c%s "lmstudio-server")
if [ $FILESIZE -lt 50000000 ]; then
    echo "❌ ERROR: The lmstudio-server binary is too small (corrupted download)."
    echo "   Expected > 50MB, got ${FILESIZE} bytes"
    exit 1
fi

echo "✅ LM Studio binary downloaded successfully (size OK)."

# -------------------------------
# STEP 4 — Download models
# -------------------------------
echo "➡️ Downloading Qwen2-VL-14B model..."
./lmstudio-server download Qwen/Qwen2-VL-14B-Instruct || {
    echo "❌ ERROR downloading Qwen2-VL-14B"
    exit 1
}

echo "➡️ Downloading SigLIP SO400M embedding model..."
./lmstudio-server download google/siglip-so400m || {
    echo "❌ ERROR downloading SigLIP model"
    exit 1
}

# -------------------------------
# STEP 5 — Start Qwen2-VL-14B server
# -------------------------------
echo "➡️ Starting Qwen2-VL-14B server on port 1234..."
nohup ./lmstudio-server start Qwen/Qwen2-VL-14B-Instruct --port 1234 > llm.log 2>&1 &

sleep 5

echo "➡️ Checking Qwen2-VL server..."
curl -s http://localhost:1234/v1/models || echo "⚠️ LLM server not ready yet."

# -------------------------------
# STEP 6 — Start SigLIP SO400M embedding server
# -------------------------------
echo "➡️ Starting SigLIP embedding server on port 1235..."
nohup ./lmstudio-server start google/siglip-so400m --port 1235 > embed.log 2>&1 &

sleep 5

echo "➡️ Checking SigLIP embedding server..."
curl -s http://localhost:1235/v1/models || echo "⚠️ Embedding server not ready yet."

# -------------------------------
# STEP 7 — Show installed models
# -------------------------------
echo "➡️ Installed models:"
./lmstudio-server list || echo "⚠️ Could not list models."

# -------------------------------
# DONE!
# -------------------------------
echo "====================================================="
echo " 🎉 LM Studio Setup Complete!"
echo " -----------------------------------------------------"
echo " LLM Server:       http://localhost:1234/v1"
echo " Embedding Server: http://localhost:1235/v1"
echo " Qwen2-VL-14B logs: llm.log"
echo " SigLIP logs:       embed.log"
echo "====================================================="
