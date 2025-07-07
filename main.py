import moderngl
import numpy as np
import imageio
import pygame
import pygame
import subprocess
import os
from visuals.create_shape import create_shape
from audio.audio_processing import short_time_fourrier_transform, get_audio_info, AudioInfo
from utils.ema import apply_background_color_asymmetric_ema, apply_asymmetric_ema
from constants import (
    TEMP_VIDEO_FILE,
    FINAL_VIDEO_FILE,
    AUDIO_FILE,
    FPS,
    DURATION,
    WIDTH,
    HEIGHT,
    NUM_FREQ,
    ALPHA_UP,
    ALPHA_DOWN,
    RPM,
    CIRCLE_BASE_SIZE,
    CIRCLE_SCALE_FACTOR,
)

# ==== Visuals ====
pygame.init()
pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
ctx = moderngl.create_context()
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)
shape_prog = ctx.program(
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


# ==== Audio ====
stft = short_time_fourrier_transform() # Load and process audio data
audio_info = get_audio_info(stft, NUM_FREQ)  # Calculate audio information for the first frame


# ==== Render Loop ====
prev_bg_color = np.zeros(4, dtype='f4')
prev_radius = 0.0
prev_avg_freq = 0.0
curr_rotation = 0.0 

for frame in range(DURATION * FPS):
    curr_info = audio_info[frame]

    # Set background color
    new_bg_color = np.array([*curr_info.color, 1.0], dtype='f4')
    bg_color = apply_background_color_asymmetric_ema(prev_bg_color, new_bg_color, ALPHA_UP, ALPHA_DOWN)
    prev_bg_color = bg_color.copy()
    fbo.clear(*bg_color)

    # Determine the size based on loudness
    new_radius = CIRCLE_BASE_SIZE + curr_info.loudness * CIRCLE_SCALE_FACTOR
    radius = apply_asymmetric_ema(prev_radius, new_radius, ALPHA_UP, ALPHA_DOWN)
    prev_radius = radius
    shape_prog['radius'].value = radius

    # Determine the rotation 
    rotations_per_frame = curr_info.loudness * RPM / (60 * FPS)
    curr_rotation = curr_rotation + rotations_per_frame * 2 * np.pi

    # Determine the average frequency
    new_avg_freq = curr_info.avg_freq
    avg_freq = apply_asymmetric_ema(prev_avg_freq, new_avg_freq, ALPHA_UP, ALPHA_DOWN)
    prev_avg_freq = avg_freq

    # Create and render the shape
    shape_vao = create_shape(2.5*avg_freq, curr_rotation, perturbations=5, ctx=ctx, prog=shape_prog, correction_factor=HEIGHT / WIDTH)
    shape_vao.render(moderngl.TRIANGLE_FAN)

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
os.remove(TEMP_VIDEO_FILE)
print(f"Final video with audio saved as {FINAL_VIDEO_FILE}")