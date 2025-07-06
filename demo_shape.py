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
ALPHA_UP = 0.7   # EMA Fast response when values increase
ALPHA_DOWN = 0.15 # EMA Slow decay when values decrease

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
            // Scale upward from bottom: bottom stays at -1.0, top scales from -1.0
            float scaledY = -1.0 + (in_pos.y + 1.0) * scale;
            gl_Position = vec4(in_pos.x, scaledY, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
    ''',
)

# Geometry: a vertical bar that grows from bottom (scaled in shader)
bar_vertices = np.array([
    [-1.0, -1.0],  # bottom-left (stays at bottom)
    [-0.9, -1.0],  # bottom-right (stays at bottom)
    [-0.9,  0.0],  # top-right (will be scaled upward)
    [-1.0,  0.0],  # top-left (will be scaled upward)
], dtype='f4')

vbo = ctx.buffer(bar_vertices.tobytes()) # vertex buffer object
vao = ctx.simple_vertex_array(prog, vbo, 'in_pos') # vertex array object

# Framebuffer for offscreen rendering
fbo = ctx.simple_framebuffer((WIDTH, HEIGHT)) # framebuffer object
fbo.use()

# Video writer
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)

# ==== Render Loop ====
prev_background_color = np.zeros(4, dtype='f4')  # Initialize previous background color

for frame in range(DURATION * FPS):
    magnitudes = stft[:, frame]

    low = np.sum(magnitudes[:BAR_COUNT // 10])
    mid = np.sum(magnitudes[BAR_COUNT // 10:BAR_COUNT * 3 // 2])
    high = np.sum(magnitudes[BAR_COUNT * 3 // 4:])

    loudness = np.sum(magnitudes)
    total = low + mid + high

    # Normalize the frequency bands to create color ratios
    if total > 0:
        low_ratio = low / total
        mid_ratio = mid / total
        high_ratio = high / total
    else:
        low_ratio = mid_ratio = high_ratio = 0
    
    # Scale by overall loudness for brightness
    new_background_color = np.array([low_ratio * loudness, mid_ratio * loudness, high_ratio * loudness, 1.0], dtype='f4')
    
    # Asymmetric EMA: fast up, slow down
    background_color = np.zeros_like(new_background_color)
    for i in range(3):  # RGB channels only (skip alpha)
        if new_background_color[i] > prev_background_color[i]:
            # Use fast alpha for increases
            background_color[i] = ALPHA_UP * new_background_color[i] + (1 - ALPHA_UP) * prev_background_color[i]
        else:
            # Use slow alpha for decreases
            background_color[i] = ALPHA_DOWN * new_background_color[i] + (1 - ALPHA_DOWN) * prev_background_color[i]
    background_color[3] = 1.0  # Keep alpha at 1.0

    fbo.clear(*background_color)

    for i, mag in enumerate(magnitudes):
        x_offset = (2.0 / BAR_COUNT) * i - 1.0 # OpenGL coordinates range from -1 to 1
        scale = mag * 2 # Scale bar height based on magnitude

        bar_vertices[:, 0] = np.array([0, 0.02, 0.02, 0]) + x_offset
        vbo.write(bar_vertices.astype('f4').tobytes())
        prog['scale'].value = scale
        vao.render(moderngl.TRIANGLE_FAN)

    # Read framebuffer and save to video
    pixels = fbo.read(components=3, alignment=1)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    writer.append_data(np.flip(image, axis=0))  # flip Y-axis
    
    # Update previous background color for next frame
    prev_background_color = background_color.copy()

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

print(f"Final video with audio saved as {FINAL_VIDEO_FILE}")
