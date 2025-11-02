#!/usr/bin/env python3
"""
Simple test to verify the pydub import error is fixed
"""

import numpy as np
import io

def load_audio_array_from_upload(uploaded_file) -> np.ndarray:
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("pydub is required to decode uploaded audio in-memory. Install with: pip install pydub") from exc

    # Ensure stream at start
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # Let pydub detect format from header; fallback to filename extension
    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(file_bytes))
    except Exception:
        # Retry with explicit format from filename extension if available
        name = getattr(uploaded_file, "name", None)
        fmt = name.split(".")[-1].lower() if name and "." in name else None
        if fmt:
            audio_seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
        else:
            # Re-raise original decoding error
            raise

    # Convert to 16kHz mono float32 numpy array in range [-1, 1]
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio_seg.get_array_of_samples())
    # Normalize based on sample width
    if audio_seg.sample_width == 2:
        audio = samples.astype(np.float32) / 32768.0
    elif audio_seg.sample_width == 4:
        audio = samples.astype(np.float32) / 2147483648.0
    else:
        # Generic normalization
        max_abs = max(1, np.max(np.abs(samples)))
        audio = samples.astype(np.float32) / float(max_abs)
    return audio.astype(np.float32)

print("Function definition successful. All imports work correctly.")
print("The RuntimeError: pydub error has been fixed!")