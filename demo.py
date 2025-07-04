import moderngl
import numpy as np
import imageio
import librosa
import pygame
from pygame import display
import subprocess
import os

# ==== Settings ====
AUDIO_FILE = 'input/godhand.mp3'  # Must be mono or stereo WAV/MP3
FPS = 30
DURATION = 50  # seconds of video
TEMP_VIDEO_FILE = 'temp/temp_video.mp4'
FINAL_VIDEO_FILE = 'output/final_output.mp4'
WIDTH, HEIGHT = 1920, 1080
BAR_COUNT = 128

# ==== Load Audio ====
y, sr = librosa.load(AUDIO_FILE, sr=None)
hop_length = int(sr / FPS)
stft = np.abs(librosa.stft(y, n_fft=BAR_COUNT * 2, hop_length=hop_length)) # short time Fourier transform
stft = stft[:BAR_COUNT, :DURATION * FPS]  # [freq_bins, frames]
stft = stft / np.max(stft)  # normalize

# ==== Setup Pygame + moderngl ====
pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
ctx = moderngl.create_context()
prog = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_pos;
        uniform float scale;
        void main() {
            gl_Position = vec4(in_pos.x, in_pos.y * scale, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.1, 0.8, 0.6, 1.0);
        }
    ''',
)

# Geometry: a vertical bar (scaled in shader)
bar_vertices = np.array([
    [-1.0, -1.0],
    [-0.9, -1.0],
    [-0.9,  1.0],
    [-1.0,  1.0],
], dtype='f4')

vbo = ctx.buffer(bar_vertices.tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')

# Framebuffer for offscreen rendering
fbo = ctx.simple_framebuffer((WIDTH, HEIGHT))
fbo.use()

# Video writer
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)

# ==== Render Loop ====
for frame in range(DURATION * FPS):
    fbo.clear(0.05, 0.0, 0.1, 1.0)

    magnitudes = stft[:, frame]
    for i, mag in enumerate(magnitudes):
        x_offset = (2.0 / BAR_COUNT) * i - 1.0
        scale = mag * 2.5

        bar_vertices[:, 0] = np.array([0, 0.02, 0.02, 0]) + x_offset
        vbo.write(bar_vertices.astype('f4').tobytes())
        prog['scale'].value = scale
        vao.render(moderngl.TRIANGLE_FAN)

    # Read framebuffer and save to video
    pixels = fbo.read(components=3, alignment=1)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    writer.append_data(np.flip(image, axis=0))  # flip Y-axis

writer.close()
pygame.quit()

# ==== Combine with audio ====
subprocess.run([
    'ffmpeg',
    '-y',
    '-i', TEMP_VIDEO_FILE,
    '-i', AUDIO_FILE,
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-shortest',
    FINAL_VIDEO_FILE
])

# Remove the temp file
os.remove(TEMP_VIDEO_FILE)

print(f"✅ Final video with audio saved as {FINAL_VIDEO_FILE}")
