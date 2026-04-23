"""
Flask backend for WHT-based Color Image Watermarking Application
Routes: /embed, /extract, /attack, /metrics
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import cv2
import base64
import pickle
import os

from watermark import (
    embed_watermark, extract_watermark,
    embed_watermark_enhanced, extract_watermark_enhanced,
    compute_psnr, compute_ssim, compute_nc, compute_ber,
    apply_attack, numpy_to_base64, file_bytes_to_numpy,
    prepare_cover_image, ENHANCED_CONFIG
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload

# ─────────────────────────────────────────────
# In-memory session store (single-user demo)
# ─────────────────────────────────────────────
session_store = {
    'cover_img': None,             # 256×256 cover used by algorithm
    'cover_original': None,        # full-res uploaded cover (for sharp display)
    'watermark_img': None,         # 32×32 watermark used by algorithm
    'watermark_original_size': None, # (h, w) of uploaded watermark
    'watermarked_img': None,       # 256×256 watermarked result
    'metadata': None,              # embedding metadata
    'attacked_img': None,          # attacked image (256×256)
    'extracted_wm': None,          # most recently extracted watermark (32×32)
    'enhanced': True,
}


def decode_image_from_request(field_name):
    """Read an uploaded image file from request.files"""
    if field_name not in request.files:
        return None, f"No file uploaded for field '{field_name}'"
    f = request.files[field_name]
    if f.filename == '':
        return None, "Empty filename"
    img_bytes = f.read()
    try:
        img = file_bytes_to_numpy(img_bytes)
        return img, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/embed', methods=['POST'])
def embed():
    """
    POST /embed
    Form data: cover_image (file), watermark_image (file)
    Returns JSON: { watermarked_b64, cover_b64, psnr, ssim }
    """
    cover_img, err = decode_image_from_request('cover_image')
    if err:
        return jsonify({'error': f'Cover image error: {err}'}), 400

    watermark_img, err = decode_image_from_request('watermark_image')
    if err:
        return jsonify({'error': f'Watermark image error: {err}'}), 400

    try:
        # Keep original cover for sharp display later
        orig_h, orig_w = cover_img.shape[:2]
        cover_original = cover_img.copy()

        # Resize to 256×256 for the algorithm (paper spec)
        cover_256 = prepare_cover_image(cover_img, 256)
        watermark_img_small = cv2.resize(watermark_img, (32, 32))

        # Determine mode: enhanced (default) or base
        data = request.form.to_dict()
        use_enhanced = data.get('mode', 'enhanced') != 'base'

        # Run embedding algorithm on 256×256 cover
        if use_enhanced:
            watermarked_256, metadata = embed_watermark_enhanced(
                cover_256, watermark_img, config=ENHANCED_CONFIG.copy())
        else:
            watermarked_256, metadata = embed_watermark(cover_256, watermark_img)

        # Compute imperceptibility metrics on the 256×256 pair (paper consistent)
        psnr = compute_psnr(cover_256, watermarked_256)
        ssim = compute_ssim(cover_256, watermarked_256)

        # ── Sharp display: overlay distortion delta onto full-res original ──
        # Instead of resizing the whole watermarked image (which blurs it),
        # compute only the tiny per-pixel distortion at 256×256, upscale that
        # delta, and add it to the original full-resolution cover.
        if (orig_h, orig_w) != (256, 256):
            delta = watermarked_256.astype(np.int16) - cover_256.astype(np.int16)
            delta_up = cv2.resize(
                delta.astype(np.float32), (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )
            watermarked_display = np.clip(
                cover_original.astype(np.int16) + delta_up.astype(np.int16),
                0, 255
            ).astype(np.uint8)
        else:
            watermarked_display = watermarked_256
            cover_original = cover_256

        # Save to session (algorithm always works at 256×256)
        session_store['cover_img'] = cover_256
        session_store['cover_original'] = cover_original
        session_store['watermark_img'] = watermark_img_small
        session_store['watermark_original_size'] = watermark_img.shape[:2]
        session_store['watermarked_img'] = watermarked_256
        session_store['metadata'] = metadata
        session_store['attacked_img'] = None
        session_store['extracted_wm'] = None
        session_store['enhanced'] = use_enhanced

        return jsonify({
            'watermarked_b64': numpy_to_base64(watermarked_display),
            'cover_b64': numpy_to_base64(cover_original),
            'watermark_b64': numpy_to_base64(watermark_img_small),
            'psnr': round(psnr, 4),
            'ssim': round(ssim, 6),
            'mode': 'enhanced' if use_enhanced else 'base',
            'message': 'Watermark embedded successfully'
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/extract', methods=['POST'])
def extract():
    """
    POST /extract
    JSON body: { use_attacked: bool }
    Returns JSON: { extracted_wm_b64, nc, ber }
    """
    data = request.get_json(silent=True) or {}
    use_attacked = data.get('use_attacked', False)

    if session_store['metadata'] is None:
        return jsonify({'error': 'No watermarked image in session. Run /embed first.'}), 400

    target_img = session_store['attacked_img'] if (use_attacked and session_store['attacked_img'] is not None) \
                 else session_store['watermarked_img']

    if target_img is None:
        return jsonify({'error': 'No image available for extraction'}), 400

    try:
        if session_store.get('enhanced', True):
            extracted_wm = extract_watermark_enhanced(target_img, session_store['metadata'])
        else:
            extracted_wm = extract_watermark(target_img, session_store['metadata'])
        session_store['extracted_wm'] = extracted_wm

        # Metrics computed at 32×32 (algorithm resolution)
        original_wm = session_store['watermark_img']
        nc = compute_nc(original_wm, extracted_wm)
        ber = compute_ber(original_wm, extracted_wm)

        # Upscale to original watermark size using INTER_NEAREST (no blur)
        orig_h, orig_w = session_store.get('watermark_original_size') or (32, 32)
        extracted_wm_display = cv2.resize(
            extracted_wm, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        return jsonify({
            'extracted_wm_b64': numpy_to_base64(extracted_wm_display),
            'nc': round(nc, 6),
            'ber': round(ber, 6),
            'message': 'Watermark extracted successfully'
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/attack', methods=['POST'])
def attack():
    """
    POST /attack
    JSON body: { attack_type: str, params: {...} }
    Returns JSON: { attacked_b64, nc, ber }
    """
    if session_store['watermarked_img'] is None:
        return jsonify({'error': 'No watermarked image. Run /embed first.'}), 400

    data = request.get_json(silent=True) or {}
    attack_type = data.get('attack_type', 'gaussian_noise')
    params = data.get('params', {})

    try:
        attacked = apply_attack(session_store['watermarked_img'], attack_type, **params)
        session_store['attacked_img'] = attacked

        # Extract from attacked image — enable pre_denoise to reduce attack noise
        if session_store.get('enhanced', True):
            import copy
            attack_meta_copy = copy.deepcopy(session_store['metadata'])
            attack_meta_copy['config']['pre_denoise'] = True
            extracted_wm = extract_watermark_enhanced(attacked, attack_meta_copy)
        else:
            extracted_wm = extract_watermark(attacked, session_store['metadata'])
        session_store['extracted_wm'] = extracted_wm

        original_wm = session_store['watermark_img']
        nc = compute_nc(original_wm, extracted_wm)
        ber = compute_ber(original_wm, extracted_wm)

        # Upscale extracted watermark using INTER_NEAREST (no blur)
        orig_h, orig_w = session_store.get('watermark_original_size') or (32, 32)
        extracted_wm_display = cv2.resize(
            extracted_wm, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        return jsonify({
            'attacked_b64': numpy_to_base64(attacked),
            'extracted_wm_b64': numpy_to_base64(extracted_wm_display),
            'nc': round(nc, 6),
            'ber': round(ber, 6),
            'message': f'Attack "{attack_type}" applied successfully'
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """
    GET /metrics
    Returns all computed metrics for current session
    """
    if session_store['cover_img'] is None or session_store['watermarked_img'] is None:
        return jsonify({'error': 'No session data. Run /embed first.'}), 400

    result = {}

    # PSNR & SSIM (imperceptibility)
    result['psnr'] = round(compute_psnr(session_store['cover_img'],
                                         session_store['watermarked_img']), 4)
    result['ssim'] = round(compute_ssim(session_store['cover_img'],
                                         session_store['watermarked_img']), 6)

    # NC & BER (robustness) - use extracted if available
    if session_store['extracted_wm'] is not None and session_store['watermark_img'] is not None:
        result['nc'] = round(compute_nc(session_store['watermark_img'],
                                         session_store['extracted_wm']), 6)
        result['ber'] = round(compute_ber(session_store['watermark_img'],
                                           session_store['extracted_wm']), 6)
    else:
        result['nc'] = None
        result['ber'] = None

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
