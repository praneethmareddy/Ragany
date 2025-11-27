#!/bin/bash

echo "==============================================="
echo " 🚀 LM STUDIO HEADLESS SERVER SETUP (UBUNTU) "
echo "==============================================="

# 1. Install dependencies
echo "➡️ Installing dependencies..."
sudo apt update
sudo apt install -y wget unzip curl

# 2. Download LM Studio server binary
echo "➡️ Downloading LM Studio Server..."
wget -O lmstudio-server https://desktop-release.lmstudio.ai/linux/lmstudio-server

if [ ! -f "lmstudio-server" ]; then
    echo "❌ ERROR: Could not download LM Studio server."
    exit 1
fi

chmod +x lmstudio-server
echo "✅ LM Studio server downloaded."

# 3. Debug: Print version
echo "➡️ Checking LM Studio server version..."
./lmstudio-server --version || echo "⚠️ Version command may not be available."

# 4. Download Qwen2-VL-14B
echo "➡️ Downloading Qwen2-VL-14B-Instruct..."
./lmstudio-server download Qwen/Qwen2-VL-14B-Instruct

# 5. Download SigLIP SO400M embedding model
echo "➡️ Downloading SigLIP SO400M..."
./lmstudio-server download google/siglip-so400m

# 6. Start models in background (LLM + Embedding)
echo "➡️ Starting Qwen2-VL-14B model (port 1234)..."
nohup ./lmstudio-server start Qwen/Qwen2-VL-14B-Instruct --port 1234 > llm.log 2>&1 &

sleep 5
echo "➡️ Checking if LLM server is running..."
curl -s http://localhost:1234/v1/models || echo "⚠️ LLM server not responding yet."

echo "➡️ Starting SigLIP embedding server (port 1235)..."
nohup ./lmstudio-server start google/siglip-so400m --port 1235 > embed.log 2>&1 &

sleep 5
echo "➡️ Checking if embedding server is running..."
curl -s http://localhost:1235/v1/models || echo "⚠️ Embedding server not responding yet."

# 7. List models
echo "➡️ Listing downloaded models..."
./lmstudio-server list

echo "==============================================="
echo " 🎉 LM Studio setup complete! "
echo " LLM:  http://localhost:1234/v1  (Qwen2-VL-14B)"
echo " EMB:  http://localhost:1235/v1  (SigLIP)"
echo "==============================================="
