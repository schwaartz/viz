import moderngl
import numpy as np
import imageio
import subprocess
import os
import time
from visuals.create_shape import create_shape
from audio.audio_processing import short_time_fourrier_transform, get_audio_info, AudioInfo
from utils.ema import apply_background_color_asymmetric_ema, apply_asymmetric_ema
from constants import (
    PERTURBATION_MAX_AMOUNT,
    TEMP_VIDEO_FILE,
    FINAL_VIDEO_FILE,
    AUDIO_FILE,
    FPS,
    DURATION,
    WIDTH,
    HEIGHT,
    NUM_FREQ,
    ALPHA_UP_COLOR,
    ALPHA_DOWN_COLOR,
    ALPHA_UP_RADIUS,
    ALPHA_DOWN_RADIUS,
    ALPHA_UP_AVG_FREQ,
    ALPHA_DOWN_AVG_FREQ,
    RPM,
    CIRCLE_BASE_SIZE,
    CIRCLE_SCALE_FACTOR,
    USE_FIXED_PERT_NUM,
    FIXED_PERT_NUM
)


# ==== Visuals ====
ctx = moderngl.create_context(standalone=True)
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)
shape_prog = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_pos;
        void main() {
            gl_Position = vec4(in_pos.x, in_pos.y, 0.0, 1.0);
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
print("Starting audio processing...")
audio_start = time.time()
stft = short_time_fourrier_transform() # Load and process audio data
audio_info = get_audio_info(stft, NUM_FREQ)  # Calculate audio information for the first frame
audio_end = time.time()


# ==== Render Loop ====
print("Starting rendering...")
render_start = time.time()
prev_bg_color = np.zeros(4, dtype='f4')
prev_radius = 0.0
prev_avg_freq = 0.0
prev_pert_num_float = 0.0
curr_rotation = 0.0 

for frame in range(DURATION * FPS):
    curr_info: AudioInfo = audio_info[frame]

    # Set background color
    new_bg_color = np.array([*curr_info.color, 1.0], dtype='f4')
    bg_color = apply_background_color_asymmetric_ema(prev_bg_color, new_bg_color, ALPHA_UP_COLOR, ALPHA_DOWN_COLOR)
    prev_bg_color = bg_color.copy()
    fbo.clear(*bg_color)

    # Determine the size based on loudness
    new_radius = CIRCLE_BASE_SIZE + curr_info.loudness * CIRCLE_SCALE_FACTOR
    radius = apply_asymmetric_ema(prev_radius, new_radius, ALPHA_UP_RADIUS, ALPHA_DOWN_RADIUS)
    prev_radius = radius

    # Determine the rotation 
    rotations_per_frame = curr_info.loudness * RPM / (60 * FPS)
    curr_rotation = curr_rotation + rotations_per_frame * 2 * np.pi

    # Determine the average frequency
    new_avg_freq = curr_info.avg_freq
    avg_freq = apply_asymmetric_ema(prev_avg_freq, new_avg_freq, ALPHA_UP_AVG_FREQ, ALPHA_DOWN_AVG_FREQ)
    prev_avg_freq = avg_freq

    # Determine the number of perturbations
    if USE_FIXED_PERT_NUM:
        pert_num_float = FIXED_PERT_NUM
    else:
        new_pert_num_float = avg_freq * PERTURBATION_MAX_AMOUNT
        pert_num_float = apply_asymmetric_ema(prev_pert_num_float, new_pert_num_float, ALPHA_UP_AVG_FREQ, ALPHA_DOWN_AVG_FREQ)
        prev_pert_num_float = pert_num_float

    # Create and render the shape
    shape_vao = create_shape(radius, avg_freq, curr_rotation, int(pert_num_float), ctx, prog=shape_prog)
    shape_vao.render(moderngl.TRIANGLE_FAN)

    # Read framebuffer and save to video
    pixels = fbo.read(components=3, alignment=1)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    writer.append_data(np.flip(image, axis=0))  # flip Y-axis

writer.close()
render_end = time.time()


# ==== Combine with audio ====
print("Starting FFmpeg...")
ffmpeg_start = time.time()
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
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ffmpeg_end = time.time()

os.remove(TEMP_VIDEO_FILE)

total_time = ffmpeg_end - audio_start
print(f"\nTIMING SUMMARY")
print(f"To render a total of {DURATION} seconds of video at {FPS} FPS ({DURATION * FPS} frames):")
print(f" - Audio processing: {audio_end - audio_start:.2f}s ({((audio_end - audio_start) / total_time * 100):.1f}%)")
print(f" - Rendering:        {render_end - render_start:.2f}s ({((render_end - render_start) / total_time * 100):.1f}%)")
print(f" - FFmpeg:           {ffmpeg_end - ffmpeg_start:.2f}s ({((ffmpeg_end - ffmpeg_start) / total_time * 100):.1f}%)")
print(f" - Total:            {total_time:.2f}s")
print(f"\nFinal video with audio saved as {FINAL_VIDEO_FILE}")