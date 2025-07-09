import moderngl
import numpy as np
import imageio
import subprocess
import os
import time
from visuals.create_shape import create_shape
from audio.audio_processing import short_time_fourrier_transform, get_audio_info, AudioInfo
from utils.ema import apply_asymmetric_ema
from utils.load_shader import load_shader_program
from constants import *


# ==== Visuals ====
ctx = moderngl.create_context(standalone=True)
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)
shape_prog = load_shader_program(ctx, 'shaders/shape.vert', 'shaders/shape.frag')
bg_wave_prog = load_shader_program(ctx, 'shaders/wave.vert', 'shaders/wave.frag')

# Create fullscreen quad for wave rendering
bg_quad_vertices = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0], 
], dtype='f4')
bg_quad_vbo = ctx.buffer(bg_quad_vertices.tobytes())
bg_quad_vao = ctx.simple_vertex_array(bg_wave_prog, bg_quad_vbo, 'in_pos')

fbo = ctx.simple_framebuffer((WIDTH, HEIGHT)) # framebuffer object
fbo.use()


# ==== Audio ====
print("Starting audio processing...")
audio_start = time.time()
stft = short_time_fourrier_transform() # Load and process audio data
audio_info = get_audio_info(stft, NUM_FREQ)  # Calculate audio information for the first frame
audio_end = time.time()


# ==== Render Loop ====
print("Starting render loop...")
render_loop_start = time.time()
total_rendering_time = 0.0

frame_since_last_wave = 0
active_waves = []
curr_rotation = 0.0 
prev_radius = 0.0
prev_avg_freq = 0.0
prev_portr_num_float = 0.0
prev_brightness = 1.0
prev_color = np.array([0.0, 0.0, 0.0])

for frame in range(DURATION * FPS):
    curr_info: AudioInfo = audio_info[frame]
    current_color = np.array(curr_info.color)
    color_diff = np.linalg.norm(current_color - prev_color)
    frame_since_last_wave += 1
    
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

    # Determine the number of portrusions
    if USE_FIXED_PORTR_NUM:
        portr_num_float = FIXED_PORTR_NUM
    else:
        new_portr_num_float = avg_freq * PORTR_MAX_AMOUNT
        portr_num_float = apply_asymmetric_ema(prev_portr_num_float, new_portr_num_float, ALPHA_UP_AVG_FREQ, ALPHA_DOWN_AVG_FREQ)
        prev_portr_num_float = portr_num_float

    # Check if new wave should spawn
    if (frame == 0 or color_diff > COLOR_CHANGE_THRESHOLD or
        frame_since_last_wave > MAX_FRAMES_BETWEEN_WAVES or len(active_waves) == 0):
        active_waves.append({
            'color': curr_info.color,
            'radius': 0.0
        })
        prev_color = current_color.copy()
        frame_since_last_wave = 0  # Reset counter
    
    # Update all active waves
    for wave in active_waves.copy():
        dynamic_speed = BASE_WAVE_SPEED + curr_info.loudness * WAVE_SPEED_MULTIPLIER
        wave['radius'] += dynamic_speed
        if wave['radius'] > WAVE_REMOVAL_RADIUS:
            active_waves.remove(wave)
    
    # Keep only the most recent waves (performance optimization)
    if len(active_waves) > MAX_WAVES:
        active_waves = active_waves[-MAX_WAVES:]
    
    # Clear framebuffer
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    
    # Prepare wave data for shader
    wave_colors = []
    wave_radii = []
    
    for wave in active_waves:
        wave_colors.append(wave['color']) # vec3
        wave_radii.append(wave['radius']) # float
    
    # Pad arrays to buffer size
    while len(wave_colors) < MAX_WAVES:
        wave_colors.append([0.0, 0.0, 0.0])
    while len(wave_radii) < MAX_WAVES:
        wave_radii.append(0.0)
    
    # Set shader uniforms
    bg_wave_prog['wave_colors'].value = wave_colors
    bg_wave_prog['wave_radii'].value = wave_radii
    bg_wave_prog['num_waves'].value = len(active_waves)
    bg_wave_prog['wave_thickness'].value = WAVE_THICKNESS
    bg_wave_prog['brightness'].value = BRIGHTNESS

    # Create the shape
    shape_vao = create_shape(radius, avg_freq, curr_rotation, int(portr_num_float), ctx, prog=shape_prog)

    # Render the waves and the shape
    render_start = time.time()
    bg_quad_vao.render(moderngl.TRIANGLE_FAN)
    shape_vao.render(moderngl.TRIANGLE_FAN)
    render_end = time.time()
    total_rendering_time += render_end - render_start

    # Read framebuffer and save to video
    pixels = fbo.read(components=3, alignment=1)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    writer.append_data(np.flip(image, axis=0))  # flip Y-axis

writer.close()
render_loop_end = time.time()


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
print(f" - Render Loop:        {render_loop_end - render_loop_start:.2f}s ({((render_loop_end - render_loop_start) / total_time * 100):.1f}%)")
print(f"   - Actual time spend rendering frames: {total_rendering_time:.2f}s ({(total_rendering_time / (render_loop_end - render_loop_start) * 100):.1f}%)")
print(f" - FFmpeg:           {ffmpeg_end - ffmpeg_start:.2f}s ({((ffmpeg_end - ffmpeg_start) / total_time * 100):.1f}%)")
print(f" - Total:            {total_time:.2f}s")
print(f"\nFinal video with audio saved as {FINAL_VIDEO_FILE}")