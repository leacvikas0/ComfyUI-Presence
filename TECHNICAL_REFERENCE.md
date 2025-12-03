# Flux 2 Technical Reference & Node Implementation Deep Dive

This document contains detailed technical information about Flux 2 architecture and the implementation details of the Presence node suite. This is for developers/debugging, not for the AI Director.

## Flux 2 Architecture (2024)

### Core Design
Flux 2 uses a **latent flow matching architecture** combining:
- **Mistral-3 24B Vision-Language Model (VLM):** Provides contextual understanding
- **Rectified Flow Transformer:** Handles compositional logic

### Model Specifications
- **Size:** 32 billion parameters (dev variant)
- **Resolution:** Up to 4 megapixels
- **Speed:** Sub-10 seconds with optimizations
- **VRAM:** 40% reduction via FP8 quantization
- **Multi-Reference:** Up to 10 reference images

### VAE (Variational Autoencoder)
- Fully retrained from scratch (not inherited from Flux 1)
- Apache 2.0 licensed
- Solves the "Learnability-Quality-Compression" trilemma

## Reference Latent Technical Details

### Why Lists, Not Batches?
Flux 2 expects `reference_latents` as a LIST of tensors, not a batched tensor:
- Flux 1: `[N, C, H, W]` (batched)
- Flux 2: `[[1,C,H,W], [1,C,H,W], ...]` (list)
- Reason: Allows variable resolutions, processed independently in attention

## The NHWC vs NCHW Problem

### The Issue
PyTorch standard: **NCHW** (Batch, Channels, Height, Width)
ComfyUI VAE wrapper: **NHWC** (Batch, Height, Width, Channels)

### Double-Permute Bug
Passing NCHW to ComfyUI VAE causes:
1. Your code: NHWC → NCHW (thinking it helps)
2. VAE wrapper: NCHW → NHWC (as it always does)
3. Result: H/W/C scrambled
4. Error: `expected input[1, 1024, 0, 1024] to have 3 channels`

### The Solution (v4.0)
```python
# 1. Load as NHWC
pixels_nhwc = torch.from_numpy(img_array)[None, ]

# 2. Resize using NCHW (PyTorch standard)
pixels_nchw = pixels_nhwc.permute(0, 3, 1, 2)
pixels_nchw = F.interpolate(pixels_nchw, size=(new_H, new_W))

# 3. Convert BACK to NHWC
pixels_nhwc = pixels_nchw.permute(0, 2, 3, 1).contiguous()

# 4. Verify
assert pixels_nhwc.shape[-1] == 3
assert 0 <= pixels_nhwc.min() <= 1

# 5. Encode (VAE handles [-1,1] scaling internally)
latent = vae.encode(pixels_nhwc)
```

## FluxAdaptiveInjector Implementation

### Full Process (8 Steps)

**1. Input Reception:**
```python
def inject_references(self, vae, conditioning, context_bundle=None, images=None):
```
Accepts either `context_bundle` or `images`, converts to unified list.

**2. Image Preprocessing:**
```python
pixels_nhwc = img[0].unsqueeze(0)  # [H,W,3] → [1,H,W,3]
```

**3. Resize to ~1MP:**
```python
current_pixels = H * W
if current_pixels > 1024*1024:
    scale = sqrt(1024*1024 / current_pixels)
    new_H = round((H * scale) / 64) * 64  # Must be multiple of 64
    new_W = round((W * scale) / 64) * 64
```

**4. NCHW → NHWC Dance:**
```python
pixels_nchw = pixels_nhwc.permute(0, 3, 1, 2)
pixels_nchw = F.interpolate(pixels_nchw, ...)
pixels_nhwc = pixels_nchw.permute(0, 2, 3, 1).contiguous()
```

**5. Format Verification:**
```python
assert pixels_nhwc.shape[-1] == 3
assert 0 <= pixels_nhwc.min() <= 1
```

**6. Device Management:**
```python
pixels_final = pixels_nhwc.to(vae.device).contiguous()
```
Move to device LAST to prevent CPU reversion.

**7. VAE Encoding:**
```python
latent = vae.encode(pixels_final)
if hasattr(latent, "sample"):
    latent = latent.sample()
```

**8. List Construction & Injection:**
```python
reference_latents = []
for img in images:
    latent = encode(img)
    reference_latents.append(latent)

# CRITICAL: Only inject if non-empty
if len(reference_latents) > 0:
    conditioning["reference_latents"] = reference_latents
```

## Common Pitfalls

### Pitfall 1: Empty List Injection
```python
# BAD: Crashes Flux
conditioning["reference_latents"] = []

# GOOD: Check first
if len(reference_latents) > 0:
    conditioning["reference_latents"] = reference_latents
```

### Pitfall 2: Wrong Format
```python
# BAD
latent = vae.encode(pixels_nchw)

# GOOD
latent = vae.encode(pixels_nhwc)
```

### Pitfall 3: Batching
```python
# BAD
reference_latents = torch.cat([lat1, lat2], dim=0)

# GOOD
reference_latents = [lat1, lat2]
```

### Pitfall 4: Device Mismatch
```python
# BAD
pixels = pixels.to("cuda")
pixels = pixels.permute(...)  # Might revert!

# GOOD
pixels = pixels.permute(...).contiguous()
pixels = pixels.to(vae.device)
```

## PresenceDirector State Management

### State Persistence (v2.0)
State is saved to `presence_state.json` in the project folder:
```json
{
  "queue": [...],
  "history": [...],
  "seen_files": [...]
}
```

### Why This Matters
- Prevents re-uploading old images to Gemini (saves API costs)
- Survives ComfyUI restarts
- Maintains project continuity

### Cost Optimization
- Images resized to ~1MP before upload (reduces Gemini token cost)
- Only new images uploaded (state tracking)
- Persistent memory prevents re-processing

## Performance Notes

### Flux 2 Optimizations
- FP8 quantization: 40% VRAM reduction
- Consumer GPU support (RTX 4090+)
- Sub-10 second generation

### Resolution Guidelines (Technical)
- All dimensions must be multiples of 64
- Max tested: 4MP
- Optimal for references: 1MP
- Aspect ratios: 1:1, 4:3, 3:4, 16:9

## Debugging Tips

### Check VAE Input
```python
print(f"Shape: {pixels.shape}")  # Should be [1, H, W, 3]
print(f"Range: [{pixels.min()}, {pixels.max()}]")  # Should be [0, 1]
print(f"Device: {pixels.device}")  # Should match VAE device
```

### Check Reference Latents
```python
print(f"Type: {type(reference_latents)}")  # Should be list
print(f"Count: {len(reference_latents)}")  # Should be > 0
print(f"Each shape: {[r.shape for r in reference_latents]}")
```

### Check Conditioning
```python
print(f"Has key: {'reference_latents' in conditioning[0][1]}")
print(f"Is list: {isinstance(conditioning[0][1]['reference_latents'], list)}")
```

---
End of Technical Reference
