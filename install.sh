#!/bin/bash
# =============================================================================
# ECO-Gradian IoT Edge Device — One-Shot Installer for Orange Pi / RPi
# =============================================================================
# Usage:
#   curl -sSL https://your-server/install.sh | bash
#   OR:
#   chmod +x install.sh && ./install.sh
# =============================================================================
set -e

INSTALL_DIR="/opt/eco_iot"
SERVICE_NAME="eco_iot"
VENV_DIR="$INSTALL_DIR/.venv"

echo "============================================="
echo "  🌿 ECO-Gradian IoT Edge Installer"
echo "============================================="
echo ""

# ── 1. System Dependencies ────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    libopencv-dev python3-opencv \
    v4l-utils \
    > /dev/null 2>&1
echo "  ✅ System packages installed"

# ── 2. Create install directory ────────────────────────────────────────────────
echo "[2/6] Setting up $INSTALL_DIR..."
sudo mkdir -p "$INSTALL_DIR"
sudo chown "$(whoami):$(whoami)" "$INSTALL_DIR"

# Copy project files
cp -r "$(dirname "$0")/"* "$INSTALL_DIR/" 2>/dev/null || true
echo "  ✅ Project files copied"

# ── 3. Python Virtual Environment ─────────────────────────────────────────────
echo "[3/6] Creating Python venv..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$INSTALL_DIR/requirements.txt" -q
echo "  ✅ Python venv ready"

# ── 4. Create required directories ────────────────────────────────────────────
echo "[4/6] Creating directories..."
mkdir -p "$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR/data"
mkdir -p "$INSTALL_DIR/models"
echo "  ✅ Directories created"

# ── 5. Generate default API key ────────────────────────────────────────────────
echo "[5/6] Generating default API key..."
cd "$INSTALL_DIR"
"$VENV_DIR/bin/python" main.py --generate-key "default-installer"
echo "  ✅ Default API key generated"

# ── 6. Install systemd service ─────────────────────────────────────────────────
echo "[6/6] Installing systemd service..."
sudo cp "$INSTALL_DIR/eco_iot.service" "/etc/systemd/system/${SERVICE_NAME}.service" 2>/dev/null || true
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "  ✅ systemd service installed"

echo ""
echo "============================================="
echo "  🎉 Installation Complete!"
echo "============================================="
echo ""
echo "  Start the service:"
echo "    sudo systemctl start eco_iot"
echo ""
echo "  Or run manually:"
echo "    cd $INSTALL_DIR"
echo "    source .venv/bin/activate"
echo "    python main.py --camera 0 --port 8080"
echo ""
echo "  Generate more API keys:"
echo "    python main.py --generate-key \"Camera #2\""
echo ""
echo "  Check status:"
echo "    curl http://localhost:8080/health"
echo ""
echo "============================================="
