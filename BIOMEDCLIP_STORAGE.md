# BiomedCLIP Model Storage

## Where are BiomedCLIP weights stored?

BiomedCLIP uses **HuggingFace's automatic caching system** - weights are NOT stored with pip.

### Storage Location
```
~/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/
```

### Size
- **747 MB** (compressed model files)

### How it works

1. **First Run**: When you initialize BiomedCLIP, it automatically:
   - Downloads from HuggingFace Hub
   - Caches to `~/.cache/huggingface/hub/`
   - This is a one-time download (~2-5 minutes depending on network)

2. **Subsequent Runs**: 
   - Loads from local cache instantly
   - No network required
   - Fast initialization (~5 seconds)

### Code Flow

```python
from open_clip import create_model_from_pretrained, get_tokenizer

# This triggers automatic download on first run
model, preprocess = create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
```

The `hf-hub:` prefix tells open_clip to:
1. Check local cache first
2. If not found, download from HuggingFace
3. Cache for future use

### Verify Cache

Check if BiomedCLIP is cached:
```bash
ls ~/.cache/huggingface/hub/ | grep BiomedCLIP
```

Expected output:
```
models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

Check cache size:
```bash
du -sh ~/.cache/huggingface/hub/models--microsoft--BiomedCLIP*
```

### Clear Cache (if needed)

To force re-download or free space:
```bash
rm -rf ~/.cache/huggingface/hub/models--microsoft--BiomedCLIP*
```

Next run will re-download automatically.

### Comparison: Old vs New

| Aspect | Old MedCLIP | New BiomedCLIP |
|--------|-------------|----------------|
| **Storage** | Manual `pretrained/` folder | Auto HuggingFace cache |
| **Location** | `Capstone_Project-CS[ID]/pretrained/` | `~/.cache/huggingface/hub/` |
| **Size** | 1GB+ (included in repo) | 747MB (cached separately) |
| **Download** | Manual via `main.py --download-models` | Automatic on first use |
| **Management** | Manual (git-ignored folder) | Automatic (HuggingFace handles it) |
| **Updates** | Manual re-download | `pip install --upgrade open_clip_torch` |

### Benefits

1. **No Manual Setup**: Just `pip install` and run
2. **No Git Bloat**: Weights not in project directory
3. **Shared Cache**: Multiple projects use same cached model
4. **Automatic Updates**: HuggingFace manages versioning
5. **Standard Location**: Follows Python/ML conventions

### Network Requirements

**First Run Only**:
- Requires internet to download from HuggingFace
- ~747MB download
- Proxy-friendly (uses standard HTTPS)

**Subsequent Runs**:
- No network required
- Runs completely offline

### Deployment

For production deployment:

**Option 1: Pre-cache on build**
```bash
# In Dockerfile or setup script
python -c "from open_clip import create_model_from_pretrained; create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"
```

**Option 2: Bundle cache**
```bash
# Copy cache to deployment
cp -r ~/.cache/huggingface/hub/models--microsoft--BiomedCLIP* /path/to/deployment/.cache/huggingface/hub/
```

**Option 3: Custom cache location**
```bash
# Set environment variable
export HF_HOME=/custom/path/to/cache
```

### Troubleshooting

**Issue**: Model not downloading
- Check internet connection
- Check HuggingFace Hub status: https://status.huggingface.co/
- Try with proxy settings if behind corporate firewall

**Issue**: Permission denied on cache
```bash
# Fix permissions
chmod -R 755 ~/.cache/huggingface/
```

**Issue**: Disk space
```bash
# Check available space
df -h ~/.cache/

# Clean old models (if multiple cached)
rm -rf ~/.cache/huggingface/hub/models--*
```

### Summary

✅ **No manual pretrained folder needed**  
✅ **Automatic caching via HuggingFace**  
✅ **Standard ML workflow**  
✅ **747MB storage in `~/.cache/huggingface/hub/`**  
✅ **Works with pip install**
