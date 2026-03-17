#!/bin/bash
# Download MedCLIP pretrained weights manually when network is available

WEIGHT_URL="https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt"
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
WEIGHT_FILE="$CACHE_DIR/medclip_vit_weight.txt"

echo "Downloading MedCLIP pretrained weights..."
echo "URL: $WEIGHT_URL"
echo "Destination: $WEIGHT_FILE"
echo ""

# Create cache directory if it doesn't exist
mkdir -p "$CACHE_DIR"

# Download using curl (with retry)
if command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L -o "$WEIGHT_FILE.tmp" "$WEIGHT_URL" --retry 3 --connect-timeout 30
    if [ $? -eq 0 ]; then
        mv "$WEIGHT_FILE.tmp" "$WEIGHT_FILE"
        echo "✓ Download successful!"
        echo "File saved to: $WEIGHT_FILE"
        ls -lh "$WEIGHT_FILE"
    else
        echo "✗ Download failed with curl"
        rm -f "$WEIGHT_FILE.tmp"
        exit 1
    fi
elif command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget -O "$WEIGHT_FILE.tmp" "$WEIGHT_URL" --tries=3 --timeout=30
    if [ $? -eq 0 ]; then
        mv "$WEIGHT_FILE.tmp" "$WEIGHT_FILE"
        echo "✓ Download successful!"
        echo "File saved to: $WEIGHT_FILE"
        ls -lh "$WEIGHT_FILE"
    else
        echo "✗ Download failed with wget"
        rm -f "$WEIGHT_FILE.tmp"
        exit 1
    fi
else
    echo "✗ Neither curl nor wget found. Install one of them to download weights."
    exit 1
fi

echo ""
echo "Weights are now cached and ready for use."
echo "Restart your application to use the pretrained weights."
