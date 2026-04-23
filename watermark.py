"""
WHT-Based Dual Color Image Watermarking Algorithm
Implements: "A Novel Dual Color Image Watermarking Algorithm Using Walsh–Hadamard Transform
with Difference-Based Embedding Positions" (Symmetry 2026, 18, 65)
"""

import numpy as np
import cv2
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim_func


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BLOCK_SIZE = 4          # 4×4 blocks
BITS_PER_BLOCK = 4      # 4 watermark bits per block
T = 8                   # Quantization step (optimal per paper)
MU = 4.0                # Logistic map control parameter
X0 = 0.398              # Logistic map initial value

# 4×4 Hadamard matrix (Eq. 3 in paper)
H4 = np.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1,  1, -1, -1],
    [1, -1, -1,  1]
], dtype=np.float64)


# ─────────────────────────────────────────────
# Walsh-Hadamard Transform
# ─────────────────────────────────────────────

def wht(block):
    """Apply WHT: F = (1/N) * H4 * block  (Eq. 5)"""
    return (1.0 / BLOCK_SIZE) * H4 @ block.astype(np.float64)


def iwht(F):
    """Apply inverse WHT: f = H4 * F  (Eq. 6)"""
    return H4 @ F


# ─────────────────────────────────────────────
# Logistic Chaotic Encryption
# ─────────────────────────────────────────────

def logistic_sequence(length, mu=MU, x0=X0):
    """Generate chaotic sequence via logistic map (Eq. 12)"""
    seq = np.zeros(length)
    x = x0
    for i in range(length):
        x = mu * x * (1 - x)
        seq[i] = x
    return seq


def encrypt_bits(bits, mu=MU, x0=X0):
    """Scramble watermark bits using logistic chaotic map"""
    n = len(bits)
    seq = logistic_sequence(n, mu, x0)
    indices = np.argsort(seq)          # permutation from chaotic sequence
    encrypted = np.zeros(n, dtype=np.uint8)
    for new_pos, old_pos in enumerate(indices):
        encrypted[new_pos] = bits[old_pos]
    return encrypted, indices


def decrypt_bits(encrypted_bits, indices):
    """Inverse permutation to recover original bits"""
    n = len(encrypted_bits)
    bits = np.zeros(n, dtype=np.uint8)
    for new_pos, old_pos in enumerate(indices):
        bits[old_pos] = encrypted_bits[new_pos]
    return bits


# ─────────────────────────────────────────────
# Entropy-Based Block Selection
# ─────────────────────────────────────────────

def visual_entropy(block):
    """Visual entropy E1 = -Σ(pk * log(pk))  (Eq. 10)"""
    flat = block.flatten().astype(np.float64)
    # Build histogram over pixel value range
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    pk = hist / hist.sum()
    pk = pk[pk > 0]
    return -np.sum(pk * np.log(pk + 1e-12))


def edge_entropy(block):
    """Edge entropy E2 = Σ(pk * exp(1 - pk))  (Eq. 11)"""
    flat = block.flatten().astype(np.float64)
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    pk = hist / hist.sum()
    pk = pk[pk > 0]
    return np.sum(pk * np.exp(1 - pk))


def block_score(block):
    """BlockScore = E1 + E2"""
    return visual_entropy(block) + edge_entropy(block)


def get_blocks(channel, block_size=BLOCK_SIZE):
    """Divide channel into non-overlapping 4×4 blocks, return list of (row, col, block)"""
    h, w = channel.shape
    blocks = []
    for r in range(0, h - block_size + 1, block_size):
        for c in range(0, w - block_size + 1, block_size):
            blk = channel[r:r+block_size, c:c+block_size].copy()
            blocks.append((r, c, blk))
    return blocks


def select_blocks(channel, n_blocks_needed):
    """Select n_blocks_needed lowest-entropy blocks (Section 3.1.2)"""
    blocks = get_blocks(channel)
    scored = [(block_score(blk), r, c) for (r, c, blk) in blocks]
    scored.sort(key=lambda x: x[0])   # ascending entropy
    selected = [(r, c) for (_, r, c) in scored[:n_blocks_needed]]
    return selected


# ─────────────────────────────────────────────
# Difference-Based Coefficient Pair Selection
# ─────────────────────────────────────────────

def get_coefficient_pairs(F):
    """
    Divide a 4×4 WHT coefficient matrix into 8 horizontal 1×2 pairs.
    Returns list of (row, col1, col2) for each pair.
    Section 3.1.3 / Figure 1.
    """
    pairs = []
    for row in range(BLOCK_SIZE):
        for col in range(0, BLOCK_SIZE, 2):
            pairs.append((row, col, col + 1))
    return pairs   # 8 pairs total


def select_best_pairs(F, n=4):
    """Select the 4 pairs with smallest absolute coefficient difference"""
    pairs = get_coefficient_pairs(F)
    diffs = [(abs(F[r, c1] - F[r, c2]), r, c1, c2) for (r, c1, c2) in pairs]
    diffs.sort(key=lambda x: x[0])
    return [(r, c1, c2) for (_, r, c1, c2) in diffs[:n]]


# ─────────────────────────────────────────────
# Watermark Embedding (Algorithm 1)
# ─────────────────────────────────────────────

def embed_bits_in_block(F, bits_4, T=T):
    """
    Embed 4 watermark bits into a WHT coefficient block.
    Modifies the 4 selected coefficient pairs in-place.
    Returns modified F and the positions used.
    """
    positions = select_best_pairs(F, n=4)
    F_mod = F.copy()
    for k, (r, c1, c2) in enumerate(positions):
        a, b = F_mod[r, c1], F_mod[r, c2]
        avg = (a + b) / 2.0
        wbit = bits_4[k]
        if wbit == 1:
            # Always embed bit 1: ensure c1 > c2
            F_mod[r, c1] = avg + T / 2
            F_mod[r, c2] = avg - T / 2
        else:  # wbit == 0
            # Always embed bit 0: ensure c1 < c2
            F_mod[r, c1] = avg - T / 2
            F_mod[r, c2] = avg + T / 2
    return F_mod, positions


def embed_channel(channel, bits, block_positions):
    """Embed watermark bits into one channel at given block positions"""
    ch = channel.astype(np.float64)
    embedding_map = {}   # (r,c) -> list of pair positions used

    bit_idx = 0
    for (r, c) in block_positions:
        if bit_idx + BITS_PER_BLOCK > len(bits):
            break
        block = ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
        F = wht(block)
        bits_4 = bits[bit_idx:bit_idx + BITS_PER_BLOCK]
        F_mod, positions = embed_bits_in_block(F, bits_4, T)
        block_rec = iwht(F_mod)
        ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = block_rec
        embedding_map[(r, c)] = positions
        bit_idx += BITS_PER_BLOCK

    ch = np.clip(ch, 0, 255).astype(np.uint8)
    return ch, embedding_map


# ─────────────────────────────────────────────
# Watermark Extraction (Algorithm 2)
# ─────────────────────────────────────────────

def extract_bits_from_block(F, positions):
    """Extract watermark bits by comparing coefficient pairs (Algorithm 2)"""
    bits = []
    for (r, c1, c2) in positions:
        if F[r, c1] > F[r, c2]:
            bits.append(1)
        else:
            bits.append(0)
    return bits


def extract_channel(channel, block_positions, embedding_map):
    """Extract bits from one channel using stored embedding positions"""
    ch = channel.astype(np.float64)
    bits = []
    for (r, c) in block_positions:
        if (r, c) not in embedding_map:
            continue
        positions = embedding_map[(r, c)]
        block = ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
        F = wht(block)
        bits.extend(extract_bits_from_block(F, positions))
    return np.array(bits, dtype=np.uint8)


# ─────────────────────────────────────────────
# Image ↔ Bits Conversion
# ─────────────────────────────────────────────

def image_to_bits(img_array):
    """Convert image array to flat binary array (MSB first per byte)"""
    flat = img_array.flatten()
    bits = np.unpackbits(flat.astype(np.uint8))
    return bits


def bits_to_image(bits, shape):
    """Convert flat binary array back to image"""
    # Ensure correct length
    total = shape[0] * shape[1] * (shape[2] if len(shape) == 3 else 1) * 8
    bits = bits[:total]
    if len(bits) < total:
        bits = np.concatenate([bits, np.zeros(total - len(bits), dtype=np.uint8)])
    flat = np.packbits(bits.astype(np.uint8))
    return flat.reshape(shape)


# ─────────────────────────────────────────────
# Main Embed / Extract API
# ─────────────────────────────────────────────

def embed_watermark(cover_img, watermark_img):
    """
    Full embedding pipeline (Section 3.1).
    Returns watermarked image and metadata needed for extraction.
    """
    # Resize watermark to paper spec (32×32) if needed
    wm_resized = cv2.resize(watermark_img, (32, 32), interpolation=cv2.INTER_AREA)

    # Split channels
    cover_r, cover_g, cover_b = cv2.split(cover_img)
    wm_r, wm_g, wm_b = cv2.split(wm_resized)

    # Total bits needed per channel
    wm_bits_per_channel = 32 * 32 * 8
    n_blocks_needed = wm_bits_per_channel // BITS_PER_BLOCK  # blocks per channel

    metadata = {'wm_shape': wm_resized.shape, 'channels': {}}

    result_channels = []
    for ch_name, cover_ch, wm_ch in [('R', cover_r, wm_r),
                                      ('G', cover_g, wm_g),
                                      ('B', cover_b, wm_b)]:
        # Step 1: Select low-entropy blocks
        block_positions = select_blocks(cover_ch, n_blocks_needed)

        # Step 2: Convert watermark channel to bits
        wm_bits = image_to_bits(wm_ch)

        # Step 3: Logistic chaotic encryption
        encrypted_bits, perm_indices = encrypt_bits(wm_bits)

        # Step 4: Embed
        watermarked_ch, embedding_map = embed_channel(cover_ch, encrypted_bits, block_positions)

        result_channels.append(watermarked_ch)
        metadata['channels'][ch_name] = {
            'block_positions': block_positions,
            'embedding_map': embedding_map,
            'perm_indices': perm_indices,
        }

    watermarked = cv2.merge(result_channels)
    return watermarked, metadata


def extract_watermark(watermarked_img, metadata):
    """
    Full extraction pipeline (Section 3.2).
    Returns extracted watermark image.
    """
    wm_shape = metadata['wm_shape']
    wm_bits_per_channel = 32 * 32 * 8

    wm_r, wm_g, wm_b = cv2.split(watermarked_img)
    extracted_channels = []

    for ch_name, wm_ch in [('R', wm_r), ('G', wm_g), ('B', wm_b)]:
        ch_meta = metadata['channels'][ch_name]
        block_positions = ch_meta['block_positions']
        embedding_map = ch_meta['embedding_map']
        perm_indices = ch_meta['perm_indices']

        # Extract encrypted bits
        encrypted_bits = extract_channel(wm_ch, block_positions, embedding_map)

        # Ensure correct length
        encrypted_bits = encrypted_bits[:wm_bits_per_channel]
        if len(encrypted_bits) < wm_bits_per_channel:
            encrypted_bits = np.concatenate([
                encrypted_bits,
                np.zeros(wm_bits_per_channel - len(encrypted_bits), dtype=np.uint8)
            ])

        # Decrypt using inverse logistic permutation
        original_bits = decrypt_bits(encrypted_bits, perm_indices)

        # Convert bits back to image channel
        ch_img = bits_to_image(original_bits, (32, 32))
        extracted_channels.append(ch_img)

    extracted_wm = cv2.merge(extracted_channels)
    return extracted_wm


# ─────────────────────────────────────────────
# Metrics (Section 4)
# ─────────────────────────────────────────────

def compute_psnr(original, watermarked):
    """PSNR averaged across R,G,B channels (Eq. 13-14)"""
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    channel_psnr = []
    for i in range(3):
        mse = np.mean((original[:, :, i] - watermarked[:, :, i]) ** 2)
        if mse == 0:
            channel_psnr.append(100.0)
        else:
            max_val = np.max(original[:, :, i]) ** 2
            channel_psnr.append(10 * np.log10(max_val / mse))
    return np.mean(channel_psnr)


def compute_ssim(original, watermarked):
    """SSIM between original and watermarked image (Eq. 15)"""
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    wm_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    score = ssim_func(orig_gray, wm_gray, data_range=255)
    return float(score)


def compute_nc(original_wm, extracted_wm):
    """Normalized Correlation between watermarks (Eq. 16)"""
    w = original_wm.astype(np.float64)
    w_prime = extracted_wm.astype(np.float64)
    numerator = np.sum(w * w_prime)
    denominator = np.sqrt(np.sum(w ** 2)) * np.sqrt(np.sum(w_prime ** 2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_ber(original_wm, extracted_wm):
    """Bit Error Rate between original and extracted watermark (Eq. 17)"""
    orig_bits = image_to_bits(original_wm)
    ext_bits = image_to_bits(extracted_wm)
    min_len = min(len(orig_bits), len(ext_bits))
    errors = np.sum(orig_bits[:min_len] != ext_bits[:min_len])
    return float(errors / min_len)


# ─────────────────────────────────────────────
# Attack Simulation
# ─────────────────────────────────────────────

def apply_attack(img, attack_type, **kwargs):
    """Apply various attacks to a watermarked image"""
    result = img.copy()

    if attack_type == 'gaussian_noise':
        var = kwargs.get('variance', 0.01)
        noise = np.random.normal(0, var * 255, img.shape).astype(np.float64)
        result = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    elif attack_type == 'jpeg_compression':
        quality = kwargs.get('quality', 85)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, enc = cv2.imencode('.jpg', img, encode_params)
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    elif attack_type == 'cropping':
        percent = kwargs.get('percent', 0.1)
        h, w = img.shape[:2]
        crop_h = int(h * percent)
        crop_w = int(w * percent)
        result = img.copy()
        result[h - crop_h:, :] = 0
        result[:, w - crop_w:] = 0

    elif attack_type == 'rotation':
        angle = kwargs.get('angle', 1)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(img, M, (w, h))

    elif attack_type == 'scaling':
        scale = kwargs.get('scale', 0.95)
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h))
        result = cv2.resize(scaled, (w, h))

    return result


# ─────────────────────────────────────────────
# Utility: Image encode/decode helpers
# ─────────────────────────────────────────────

def numpy_to_base64(img_array):
    """Convert numpy BGR image to base64 PNG string"""
    success, buffer = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_numpy(b64_str):
    """Convert base64 image string to numpy BGR array"""
    img_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def file_bytes_to_numpy(file_bytes):
    """Convert uploaded file bytes to numpy BGR array"""
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def prepare_cover_image(img, target_size=256):
    """Resize cover image to 256×256 as per paper"""
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)


# =============================================================================
# ENHANCEMENT LAYER  —  MOD-1 through MOD-7
#
# The core algorithm above (WHT, IWHT, logistic encryption, entropy scoring,
# coefficient-pair embedding/extraction, metrics) is NOT modified.
# All modifications below are additive layers that wrap or extend the core.
# =============================================================================

try:
    from scipy.signal import wiener as _scipy_wiener
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ─────────────────────────────────────────────
# MOD-1  Adaptive Embedding Strength
# ─────────────────────────────────────────────

def adaptive_T(block, T_base=T, T_min=4.0, T_max=14.0):
    """
    MOD-1: Per-block adaptive quantization step T.

    Smooth blocks (low variance) → T closer to T_min (less distortion → higher PSNR/SSIM).
    Textured blocks (high variance) → T closer to T_max (stronger signal → better NC/BER).

    Human eyes are MORE sensitive to distortion in smooth regions, so we embed
    lighter there. In textured regions we embed stronger for attack robustness.
    T_min=4 is safe: Wiener filter changes WHT coefficients by < 2 units, and
    T/2=2 leaves a margin. T_max=14 gives robust embedding in textured blocks.
    """
    variance = np.var(block.astype(np.float64))
    # Normalise variance against a typical "textured" block (~300 units²)
    norm_var = min(variance / 300.0, 1.0)   # 0 = smooth, 1 = textured
    return float(T_min + norm_var * (T_max - T_min))


# ─────────────────────────────────────────────
# MOD-2  WHT + SVD Hybrid (QIM embedding)
# ─────────────────────────────────────────────

def embed_bits_svd_qim(F, bits_4, T_local):
    """
    MOD-2: Embed 4 bits into a WHT coefficient matrix using a hybrid SVD + QIM
    approach. Only the two largest singular values S[0] and S[1] are modified
    via QIM (they dominate block energy and survive clipping / JPEG well).
    The remaining 2 bits use simple pair-comparison on reconstructed coefficients
    identical to the original algorithm — ensuring backward-compatible fallback.

    QIM step size: eff_T = max(T_local, S[k] * 0.15)  ensures the watermark
    signal is >= 15% of the singular value, surviving uint8 rounding.

    Returns: (F_mod, eff_T_list)  where eff_T_list[k] is the step used for S[k].
    """
    U, S, Vt = np.linalg.svd(F, full_matrices=True)
    S_mod = S.copy()
    eff_T_list = []
    # Embed bits 0-1 in S[0] and S[1] using QIM
    for k in range(min(2, len(S))):
        eff_T = max(T_local, abs(float(S[k])) * 0.15)
        eff_T_list.append(eff_T)
        base = np.floor(S[k] / eff_T) * eff_T
        S_mod[k] = base + (0.75 if bits_4[k] == 1 else 0.25) * eff_T
    # Embed bits 2-3 in S[2] and S[3] via sign comparison (pair comparison style)
    for k in range(2, min(4, len(S))):
        eff_T_list.append(T_local)          # placeholder for extraction
        avg_s = (S_mod[k] + S_mod[min(k+1, len(S)-1)]) / 2.0 if k + 1 < len(S) else S_mod[k]
        if bits_4[k] == 1:
            S_mod[k] = avg_s + T_local / 2
        else:
            S_mod[k] = avg_s - T_local / 2
    S_mat = np.zeros_like(F)
    for i in range(len(S_mod)):
        S_mat[i, i] = S_mod[i]
    F_mod = U @ S_mat @ Vt
    return F_mod, eff_T_list


def extract_bits_svd_qim(F, eff_T_list):
    """
    MOD-2: Extract 4 bits. Uses QIM for the first 2 (S[0], S[1]) and sign
    comparison for bits 2-3 (S[2] vs S[3]) to match the embedding scheme.
    """
    _, S, _ = np.linalg.svd(F, full_matrices=True)
    bits = []
    # QIM read for bits 0-1
    for k in range(min(2, len(eff_T_list), len(S))):
        eff_T = eff_T_list[k]
        frac = (S[k] % eff_T) / eff_T
        bits.append(1 if frac >= 0.5 else 0)
    # Sign comparison for bits 2-3
    for k in range(2, min(len(eff_T_list), len(S))):
        if k + 1 < len(S):
            bits.append(1 if S[k] > S[k+1] else 0)
        else:
            bits.append(0)
    return bits[:4]


# ─────────────────────────────────────────────
# MOD-3  Hamming (7,4) Error Correction Coding
# ─────────────────────────────────────────────

# Systematic (7,4) Hamming generator matrix  [I4 | P]
_HAMMING_G = np.array([
    [1, 0, 0, 0,  1, 1, 0],
    [0, 1, 0, 0,  1, 0, 1],
    [0, 0, 1, 0,  0, 1, 1],
    [0, 0, 0, 1,  1, 1, 1],
], dtype=np.uint8)

# Parity-check matrix  [P^T | I3]
_HAMMING_H = np.array([
    [1, 1, 0, 1,  1, 0, 0],
    [1, 0, 1, 1,  0, 1, 0],
    [0, 1, 1, 1,  0, 0, 1],
], dtype=np.uint8)

# Syndrome → error position (0-indexed); syndrome as tuple of 3 bits
_HAMMING_SYNDROME_TABLE = {
    (1, 0, 0): 4,
    (0, 1, 0): 5,
    (0, 0, 1): 6,
    (1, 1, 0): 0,
    (1, 0, 1): 1,
    (0, 1, 1): 2,
    (1, 1, 1): 3,
}


def hamming_encode(bits):
    """
    MOD-3: Encode a flat bit array using Hamming (7,4).
    Input length must be divisible by 4; output length = input * 7/4.
    Pads to nearest multiple of 4 if needed.
    """
    bits = np.array(bits, dtype=np.uint8)
    # Pad to multiple of 4
    rem = len(bits) % 4
    if rem:
        bits = np.concatenate([bits, np.zeros(4 - rem, dtype=np.uint8)])
    n_words = len(bits) // 4
    encoded = np.zeros(n_words * 7, dtype=np.uint8)
    for i in range(n_words):
        d = bits[i*4:(i+1)*4]
        c = (d @ _HAMMING_G) % 2          # 7-bit codeword
        encoded[i*7:(i+1)*7] = c
    return encoded


def hamming_decode(coded_bits):
    """
    MOD-3: Decode and error-correct Hamming (7,4) codewords.
    Corrects up to 1-bit error per 7-bit codeword.
    Returns the recovered data bits (length = n_codewords * 4).
    """
    coded_bits = np.array(coded_bits, dtype=np.uint8)
    # Pad to multiple of 7
    rem = len(coded_bits) % 7
    if rem:
        coded_bits = np.concatenate([coded_bits, np.zeros(7 - rem, dtype=np.uint8)])
    n_words = len(coded_bits) // 7
    decoded = np.zeros(n_words * 4, dtype=np.uint8)
    for i in range(n_words):
        r = coded_bits[i*7:(i+1)*7].copy()
        syndrome = tuple(int(x) for x in (r @ _HAMMING_H.T) % 2)
        if syndrome != (0, 0, 0):
            err_pos = _HAMMING_SYNDROME_TABLE.get(syndrome)
            if err_pos is not None:
                r[err_pos] ^= 1           # flip erroneous bit
        decoded[i*4:(i+1)*4] = r[:4]     # first 4 bits are data
    return decoded


# ─────────────────────────────────────────────
# MOD-5  Extended Attack Types (Salt & Pepper,
#         Gaussian Blur, Histogram Equalisation)
# ─────────────────────────────────────────────

def _apply_attack_extended(img, attack_type, **kwargs):
    """
    MOD-5: Additional attack types beyond the original apply_attack().
    Falls back to the original function for known attack types.
    """
    result = img.copy()

    if attack_type == 'salt_pepper':
        prob = kwargs.get('prob', 0.02)
        flat = result.reshape(-1, result.shape[2] if img.ndim == 3 else 1)
        rng = np.random.default_rng(42)
        mask = rng.random(flat.shape[0])
        flat[mask < prob / 2] = 0
        flat[(mask >= prob / 2) & (mask < prob)] = 255
        result = flat.reshape(img.shape)

    elif attack_type == 'gaussian_blur':
        ksize = kwargs.get('ksize', 3)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        result = cv2.GaussianBlur(img, (ksize, ksize), 0)

    elif attack_type == 'histogram_equalization':
        channels = cv2.split(img)
        eq_chs = [cv2.equalizeHist(ch) for ch in channels]
        result = cv2.merge(eq_chs)

    elif attack_type == 'median_filter':
        ksize = kwargs.get('ksize', 3)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        result = cv2.medianBlur(img, ksize)

    else:
        # Delegate to ORIGINAL (pre-patch) attack function for known types
        result = _original_apply_attack(img, attack_type, **kwargs)

    return result


# Monkey-patch: make apply_attack transparently support all attack types
_original_apply_attack = apply_attack


def apply_attack(img, attack_type, **kwargs):          # noqa: F811
    """
    Extended apply_attack supporting original types plus:
      - 'salt_pepper'            (prob=0.02)
      - 'gaussian_blur'          (ksize=3)
      - 'histogram_equalization'
      - 'median_filter'          (ksize=3)
    """
    return _apply_attack_extended(img, attack_type, **kwargs)


# ─────────────────────────────────────────────
# MOD-6  HVS-Guided Block Selection
# ─────────────────────────────────────────────

def hvs_block_score(block, lambda_edge=0.5):
    """
    MOD-6: Perceptual block score = original entropy score
           + lambda * Sobel edge magnitude.
    Blocks with high edge density are penalised so embedding is steered
    away from visually sensitive regions (edges), raising PSNR and SSIM.
    """
    base = block_score(block)
    b = block.astype(np.float64)
    # Approximate Sobel via finite differences on the 4×4 block
    gy = np.diff(b, axis=0)          # shape (3,4)
    gx = np.diff(b, axis=1)          # shape (4,3)
    edge_mag = np.mean(np.abs(gy)) + np.mean(np.abs(gx))
    return base + lambda_edge * edge_mag


def hvs_select_blocks(channel, n_blocks_needed, lambda_edge=0.5):
    """
    MOD-6: Select n_blocks_needed blocks using the HVS-guided score.
    Replaces select_blocks() in the enhanced pipeline.
    """
    blocks = get_blocks(channel)
    scored = [(hvs_block_score(blk, lambda_edge), r, c)
              for (r, c, blk) in blocks]
    scored.sort(key=lambda x: x[0])
    return [(r, c) for (_, r, c) in scored[:n_blocks_needed]]


# ─────────────────────────────────────────────
# MOD-7  Post-Embedding Wiener Filter Smoothing
# ─────────────────────────────────────────────

def post_process_channel(ch):
    """
    MOD-7: Apply a mild Wiener filter (3×3, low noise) to one channel of
    the watermarked image to smooth quantisation artefacts introduced by
    coefficient modification, raising PSNR and SSIM without blurring edges.
    Falls back to a 3×3 median filter if scipy is unavailable.
    """
    ch_f = ch.astype(np.float64)
    if _SCIPY_AVAILABLE:
        # Add a safe minimum noise floor to avoid divide-by-zero in uniform regions
        smoothed = _scipy_wiener(ch_f, mysize=3, noise=max(0.5, np.var(ch_f) * 1e-4))
        smoothed = np.nan_to_num(smoothed, nan=0.0, posinf=255.0, neginf=0.0)
    else:
        smoothed = cv2.medianBlur(ch.astype(np.uint8), 3).astype(np.float64)
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def pre_process_channel_for_extraction(ch):
    """
    MOD-5 (pre-extraction): Mild Wiener denoise applied to a channel before
    bit extraction when the image may have been attacked with noise.
    Reduces noise floor without blurring edges, improving bit-read accuracy.
    """
    return post_process_channel(ch)


# ─────────────────────────────────────────────
# Enhanced Channel Embed / Extract
# ─────────────────────────────────────────────

def _embed_channel_enhanced(channel, bits, block_positions, config):
    """
    Drop-in replacement for embed_channel() that applies MOD-1 and MOD-2.
    Returns (watermarked_channel, enhanced_embedding_map).
    enhanced_embedding_map[( r,c)] = {'T_local': float, 'positions': list | None}
      positions is None when SVD-QIM is used (MOD-2).
    """
    use_svd = config.get('use_svd', True)
    use_adaptive_T = config.get('use_adaptive_T', True)

    ch = channel.astype(np.float64)
    embedding_map_enh = {}

    bit_idx = 0
    for (r, c) in block_positions:
        if bit_idx + BITS_PER_BLOCK > len(bits):
            break
        block = ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
        F = wht(block)
        bits_4 = bits[bit_idx:bit_idx + BITS_PER_BLOCK]

        # MOD-1: compute adaptive T for this block
        T_local = adaptive_T(block) if use_adaptive_T else float(T)

        if use_svd:
            # MOD-2: SVD + QIM embedding — returns (F_mod, eff_T_list)
            F_mod, eff_T_list = embed_bits_svd_qim(F, bits_4, T_local)
            positions = None
        else:
            # Original pair-comparison embedding (with adaptive T)
            F_mod, positions = embed_bits_in_block(F, bits_4, T_local)
            eff_T_list = None

        block_rec = iwht(F_mod)
        ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = block_rec
        embedding_map_enh[(r, c)] = {
            'T_local': T_local,
            'positions': positions,
            'eff_T_list': eff_T_list,   # stored for exact QIM extraction
        }
        bit_idx += BITS_PER_BLOCK

    ch = np.clip(ch, 0, 255).astype(np.uint8)
    return ch, embedding_map_enh


def _extract_channel_enhanced(channel, block_positions, embedding_map_enh,
                               config, pre_denoise=False):
    """
    Drop-in replacement for extract_channel() that applies MOD-2 and MOD-5.
    """
    use_svd = config.get('use_svd', True)

    ch = channel.astype(np.float64)

    # MOD-5: optional pre-extraction denoising
    if pre_denoise:
        ch = pre_process_channel_for_extraction(
            np.clip(ch, 0, 255).astype(np.uint8)
        ).astype(np.float64)

    bits = []
    for (r, c) in block_positions:
        if (r, c) not in embedding_map_enh:
            continue
        meta = embedding_map_enh[(r, c)]
        T_local = meta['T_local']
        positions = meta['positions']
        block = ch[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
        F = wht(block)

        if use_svd:
            # MOD-2: SVD + QIM extraction — uses stored eff_T_list
            b = extract_bits_svd_qim(F, meta['eff_T_list'])
        else:
            # Original pair-comparison extraction
            b = extract_bits_from_block(F, positions)
        bits.extend(b)

    return np.array(bits, dtype=np.uint8)


# ─────────────────────────────────────────────
# MOD-4  2× Redundant Embedding + Majority Vote
# ─────────────────────────────────────────────

def _embed_channel_redundant(channel, bits, block_positions, config):
    """
    MOD-4: Embed the same bit sequence twice in two non-overlapping halves
    of block_positions. Returns (watermarked_channel, meta_list) where
    meta_list = [embedding_map_copy1, embedding_map_copy2].
    """
    half = len(block_positions) // 2
    pos1 = block_positions[:half]
    pos2 = block_positions[half:half*2]

    # Trim bits to what fits in half the blocks
    max_bits = (half * BITS_PER_BLOCK)
    bits_trimmed = bits[:max_bits]

    ch, map1 = _embed_channel_enhanced(channel, bits_trimmed, pos1, config)
    ch, map2 = _embed_channel_enhanced(ch, bits_trimmed, pos2, config)
    return ch, [map1, map2], pos1, pos2


def _extract_channel_redundant(channel, pos1, pos2, map1, map2, config,
                                pre_denoise=False):
    """
    MOD-4: Extract two copies and return majority-voted bits.
    """
    bits1 = _extract_channel_enhanced(channel, pos1, map1, config, pre_denoise)
    bits2 = _extract_channel_enhanced(channel, pos2, map2, config, pre_denoise)
    min_len = min(len(bits1), len(bits2))
    voted = np.where(bits1[:min_len] + bits2[:min_len] >= 1, 1, 0).astype(np.uint8)
    return voted


# ─────────────────────────────────────────────
# Default Enhancement Configuration
# ─────────────────────────────────────────────

ENHANCED_CONFIG = {
    # ── Active by default ─────────────────────────────────────────────────────
    'use_adaptive_T':      True,   # MOD-1: smooth blocks → lower T (better PSNR/SSIM)
                                   #        textured blocks → higher T (better NC/BER)
    'use_hvs_selection':   True,   # MOD-6: avoid embedding in visually-sensitive edge regions
    'use_post_processing': True,   # MOD-7: mild Wiener filter smooths quantisation artefacts
    'lambda_edge':         0.5,    # MOD-6: edge-penalty weight for block selection

    # ── Optional (off by default) ─────────────────────────────────────────────
    # ECC inflates embedded bits by 1.75× → 75% more blocks modified → worse PSNR/SSIM.
    # Enable only when BER after heavy attacks is the priority.
    'use_ecc':             False,  # MOD-3: Hamming (7,4) error correction

    # Redundancy needs 2× blocks; combine with use_ecc=False only.
    'use_redundancy':      False,  # MOD-4: 2× redundant embedding + majority vote

    # SVD-QIM is most robust with lossless PNG; set True if JPEG is not needed.
    'use_svd':             False,  # MOD-2: WHT+SVD hybrid QIM embedding

    # pre_denoise applies Wiener BEFORE extraction — useful for attacked images.
    # Keeping it True always causes double-filtering and corrupts pair comparisons.
    'pre_denoise':         False,  # MOD-5: set True only for attacked-image extraction
}


# ─────────────────────────────────────────────
# Enhanced Main Embed / Extract API
# ─────────────────────────────────────────────

def embed_watermark_enhanced(cover_img, watermark_img, config=None):
    """
    Enhanced embedding pipeline.
    Wraps the core algorithm with MOD-1 … MOD-7 enhancements.

    config keys (all optional, defaults from ENHANCED_CONFIG):
        use_adaptive_T, use_svd, use_ecc, use_redundancy,
        use_hvs_selection, use_post_processing, pre_denoise, lambda_edge

    Returns:
        watermarked  — numpy BGR uint8 image
        metadata     — dict required by extract_watermark_enhanced()
    """
    if config is None:
        config = ENHANCED_CONFIG.copy()

    use_ecc       = config.get('use_ecc', True)
    use_redundancy = config.get('use_redundancy', False)
    use_hvs       = config.get('use_hvs_selection', True)
    use_post      = config.get('use_post_processing', True)
    lambda_edge   = config.get('lambda_edge', 0.5)

    # Resize watermark to 32×32 (paper spec)
    wm_resized = cv2.resize(watermark_img, (32, 32), interpolation=cv2.INTER_AREA)

    cover_r, cover_g, cover_b = cv2.split(cover_img)
    wm_r, wm_g, wm_b = cv2.split(wm_resized)

    # Bit budget per channel
    raw_bits_per_ch = 32 * 32 * 8   # 8192 bits

    # MOD-3: ECC inflates bit count (8192 data → 14336 coded)
    # 14336 / 4 = 3584 blocks needed  ≤  4096 available — fits in 256×256
    if use_ecc:
        coded_bits_per_ch = (raw_bits_per_ch // 4) * 7   # = 14336
    else:
        coded_bits_per_ch = raw_bits_per_ch               # = 8192

    n_blocks_needed = coded_bits_per_ch // BITS_PER_BLOCK  # 3584 or 2048

    # MOD-4: redundancy needs 2× the blocks
    if use_redundancy:
        n_blocks_needed = n_blocks_needed * 2   # 4096 — exact fit, ECC must be off

    metadata = {
        'wm_shape': wm_resized.shape,
        'config': config,
        'channels': {},
    }

    result_channels = []
    for ch_name, cover_ch, wm_ch in [
        ('R', cover_r, wm_r),
        ('G', cover_g, wm_g),
        ('B', cover_b, wm_b),
    ]:
        # Step 1: Block selection — MOD-6 (HVS) or original entropy
        if use_hvs:
            block_positions = hvs_select_blocks(cover_ch, n_blocks_needed, lambda_edge)
        else:
            block_positions = select_blocks(cover_ch, n_blocks_needed)

        # Step 2: Watermark channel → bits
        wm_bits = image_to_bits(wm_ch)   # 8192 bits

        # Step 3: MOD-3 — Hamming encode
        if use_ecc:
            wm_bits = hamming_encode(wm_bits)   # 14336 bits

        # Step 4: Logistic chaotic encryption (original, unchanged)
        encrypted_bits, perm_indices = encrypt_bits(wm_bits)

        # Step 5: Embed — MOD-4 redundant or MOD-1/MOD-2 enhanced
        if use_redundancy:
            watermarked_ch, maps, pos1, pos2 = _embed_channel_redundant(
                cover_ch, encrypted_bits, block_positions, config)
            ch_meta = {
                'block_positions': block_positions,
                'pos1': pos1, 'pos2': pos2,
                'map1': maps[0], 'map2': maps[1],
                'perm_indices': perm_indices,
                'redundant': True,
                'ecc': use_ecc,
                'coded_bits_per_ch': coded_bits_per_ch,
            }
        else:
            watermarked_ch, embedding_map_enh = _embed_channel_enhanced(
                cover_ch, encrypted_bits, block_positions, config)
            ch_meta = {
                'block_positions': block_positions,
                'embedding_map_enh': embedding_map_enh,
                'perm_indices': perm_indices,
                'redundant': False,
                'ecc': use_ecc,
                'coded_bits_per_ch': coded_bits_per_ch,
            }

        # Step 6: MOD-7 — post-embedding Wiener smoothing
        if use_post:
            watermarked_ch = post_process_channel(watermarked_ch)

        result_channels.append(watermarked_ch)
        metadata['channels'][ch_name] = ch_meta

    watermarked = cv2.merge(result_channels)
    return watermarked, metadata


def extract_watermark_enhanced(watermarked_img, metadata):
    """
    Enhanced extraction pipeline.
    Uses the config and metadata produced by embed_watermark_enhanced().
    Returns the extracted watermark as a 32×32 BGR numpy uint8 image.
    """
    wm_shape = metadata['wm_shape']
    config   = metadata.get('config', ENHANCED_CONFIG)
    pre_denoise = config.get('pre_denoise', True)

    wm_r_ch, wm_g_ch, wm_b_ch = cv2.split(watermarked_img)
    extracted_channels = []

    for ch_name, wm_ch in [('R', wm_r_ch), ('G', wm_g_ch), ('B', wm_b_ch)]:
        ch_meta = metadata['channels'][ch_name]
        perm_indices      = ch_meta['perm_indices']
        redundant         = ch_meta.get('redundant', False)
        use_ecc           = ch_meta.get('ecc', True)
        coded_bits_per_ch = ch_meta.get('coded_bits_per_ch', 14336)
        raw_bits_per_ch   = 32 * 32 * 8   # 8192

        # Step 1: Extract encrypted bits (MOD-4 or standard)
        if redundant:
            encrypted_bits = _extract_channel_redundant(
                wm_ch,
                ch_meta['pos1'], ch_meta['pos2'],
                ch_meta['map1'], ch_meta['map2'],
                config, pre_denoise=pre_denoise)
        else:
            encrypted_bits = _extract_channel_enhanced(
                wm_ch,
                ch_meta['block_positions'],
                ch_meta['embedding_map_enh'],
                config, pre_denoise=pre_denoise)

        # Pad / trim to expected coded length
        encrypted_bits = encrypted_bits[:coded_bits_per_ch]
        if len(encrypted_bits) < coded_bits_per_ch:
            encrypted_bits = np.concatenate([
                encrypted_bits,
                np.zeros(coded_bits_per_ch - len(encrypted_bits), dtype=np.uint8)
            ])

        # Step 2: Logistic decrypt (original, unchanged)
        decoded_bits = decrypt_bits(encrypted_bits, perm_indices)

        # Step 3: MOD-3 — Hamming decode + error correction
        if use_ecc:
            decoded_bits = hamming_decode(decoded_bits)   # back to 8192 bits

        # Trim / pad to exact raw bit length
        decoded_bits = decoded_bits[:raw_bits_per_ch]
        if len(decoded_bits) < raw_bits_per_ch:
            decoded_bits = np.concatenate([
                decoded_bits,
                np.zeros(raw_bits_per_ch - len(decoded_bits), dtype=np.uint8)
            ])

        # Step 4: Bits → image channel
        ch_img = bits_to_image(decoded_bits, (32, 32))
        extracted_channels.append(ch_img)

    extracted_wm = cv2.merge(extracted_channels)
    return extracted_wm
