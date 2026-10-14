"""Small synthetic media (base64) for the tests and the notebook demo, so the repo carries no binary samples.

    image_b64("red")      a 256x256 PNG: a filled circle of that colour
    audio_b64(440)        a 2 s WAV sine tone (44.1 kHz; the core resamples to 16 kHz)
    video_b64((255,0,0))  a 3 s MP4 (H.264): a bar of that colour sweeping right
"""
import base64
import io

import numpy as np


def image_b64(color: str = "red") -> str:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(im).ellipse((40, 40, 216, 216), fill=color)
    b = io.BytesIO(); im.save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


def audio_b64(freq: float = 440.0, seconds: float = 2.0, sr: int = 44100) -> str:
    import soundfile as sf
    t = np.arange(int(seconds * sr)) / sr
    b = io.BytesIO(); sf.write(b, (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), sr, format="WAV")
    return base64.b64encode(b.getvalue()).decode()


def video_b64(color=(255, 0, 0), seconds: int = 3, fps: int = 8, size: int = 128) -> str:
    import av
    b = io.BytesIO()
    with av.open(b, "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width = s.height = size
        s.pix_fmt = "yuv420p"
        for i in range(seconds * fps):
            a = np.zeros((size, size, 3), np.uint8)
            a[:, : 8 + i * size // (seconds * fps)] = color
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return base64.b64encode(b.getvalue()).decode()
