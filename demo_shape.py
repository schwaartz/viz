import moderngl
import numpy as np
import imageio
import librosa
import pygame
from pygame import display
import subprocess
import os
from visuals.create_shape import create_shape
from utils.sigmoid import sigmoid

# ==== Settings ====
AUDIO_FILE = 'input/spooky_beat.mp3'  # Must be mono or stereo WAV/MP3
FPS = 30
DURATION = 22  # seconds of video
TEMP_VIDEO_FILE = 'temp/temp_video.mp4'
FINAL_VIDEO_FILE = 'output/spooky_beat.mp4'
WIDTH, HEIGHT = 1920, 1080
BAR_COUNT = 128
ALPHA_UP = 0.7   # EMA Fast response when values increase
ALPHA_DOWN = 0.15 # EMA Slow decay when values decrease
RPM = 80 # Revolutions per minute for the spinning animation

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

# Shape shader program
circle_prog = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_pos;
        uniform float radius;
        void main() {
            gl_Position = vec4(in_pos.x * radius, in_pos.y * radius, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // Black circle
        }
    ''',
)

fbo = ctx.simple_framebuffer((WIDTH, HEIGHT)) # framebuffer object
fbo.use()

# Video writer
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)

# ==== Render Loop ====
prev_background_color = np.zeros(4, dtype='f4')  # Initialize previous background color
prev_radius = 0.0  # Initialize previous size for circle
prev_freq_center_of_gravity = 0.0  # Initialize previous center of gravity for frequency bands
curr_rotation = 0.0  # Initialize current rotation angle

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
    prev_background_color = background_color.copy()

    fbo.clear(*background_color)

    # Render the circle that scales with loudness
    circle_radius = 0.1 + loudness * 0.3  # Base size 0.1, grows with loudness
    if circle_radius > prev_radius:
        # Use fast alpha for increases
        circle_radius = ALPHA_UP * circle_radius + (1 - ALPHA_UP) * prev_radius
    else:
        # Use slow alpha for decreases
        circle_radius = ALPHA_DOWN * circle_radius + (1 - ALPHA_DOWN) * prev_radius
    prev_radius = circle_radius

    circle_prog['radius'].value = circle_radius
    rotations_per_frame = loudness * RPM / (60 * FPS)
    curr_rotation = curr_rotation + rotations_per_frame * 2 * np.pi
    freq_center_of_gravity = low_ratio * 0.0 + mid_ratio * 0.5 + high_ratio * 1.0
    if freq_center_of_gravity > prev_freq_center_of_gravity:
        # Use fast alpha for increases
        freq_center_of_gravity = ALPHA_UP * freq_center_of_gravity + (1 - ALPHA_UP) * prev_freq_center_of_gravity
    else:
        # Use slow alpha for decreases
        freq_center_of_gravity = ALPHA_DOWN * freq_center_of_gravity + (1 - ALPHA_DOWN) * prev_freq_center_of_gravity
    prev_freq_center_of_gravity = freq_center_of_gravity

    circle_vao = create_shape(2.5*freq_center_of_gravity, curr_rotation, pert_num=5, ctx=ctx, prog=circle_prog, correction_factor=HEIGHT / WIDTH)
    circle_vao.render(moderngl.TRIANGLE_FAN)

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
    '-map', '0:v:0',  # Map video from first input
    '-map', '1:a:0',  # Map audio from second input
    '-shortest',
    FINAL_VIDEO_FILE
])

# Remove the temp file
os.remove(TEMP_VIDEO_FILE)

print(f"Final video with audio saved as {FINAL_VIDEO_FILE}")