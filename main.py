import moderngl
import numpy as np
import imageio
import subprocess
import os
import time
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from constants import *
from visuals.create_circle import create_circle
from audio.audio_processing import short_time_fourrier_transform, get_audio_info, AudioInfo
from utils.ema import apply_asymmetric_ema
from utils.load_shader import load_shader_program
from utils.timing_summary import print_timing_summary


# ==== Pretty Printing ====
console = Console()
console.log("Starting program")


# ==== Visuals ====
ctx = moderngl.create_context(standalone=True)
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)

shape_prog = load_shader_program(ctx, 'shaders/shape.vert', 'shaders/shape.frag')
shape_prog['protr_base_thickness'].value = PROTR_BASE_THINNESS 
shape_prog['protr_thickness_factor'].value = PROTR_THICKENING_FACTOR  
shape_prog['height_width_ratio'].value = HEIGHT / WIDTH
shape_prog['protr_amount'].value = PROTR_AMOUNT
shape_prog['protr_scale'].value = PROTR_SCALE
shape_prog['protr_variability'].value = PROTR_VARIABILITY

bg_wave_prog = load_shader_program(ctx, 'shaders/wave.vert', 'shaders/wave.frag')
bg_quad_vertices = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0], 
], dtype='f4')
bg_quad_vbo = ctx.buffer(bg_quad_vertices.tobytes())
bg_quad_vao = ctx.simple_vertex_array(bg_wave_prog, bg_quad_vbo, 'in_pos')
shape_vao = create_circle(CIRCLE_BASE_SIZE, ctx, shape_prog)

fbo = ctx.simple_framebuffer((WIDTH, HEIGHT))
fbo.use()


# ==== Audio ====
# STFT calculation
console.log(f"Processing audio file [bold]{AUDIO_FILE}[/bold]")
audio_start = time.time()
stft = short_time_fourrier_transform()

# Info extraction from STFT
console.log("Extracting audio information")
audio_info = get_audio_info(stft, NUM_FREQ)
audio_duration = time.time() - audio_start


# ==== Render Loop ====
console.log("Starting render loop\n")
render_loop_start = time.time()
total_rendering_time = 0.0
total_writing_time = 0.0

frame_since_last_wave = 0
active_waves = []
curr_rotation = 0.0 
prev_radius_scaler = 0.0
prev_avg_freq = 0.0
prev_brightness = 1.0
prev_color = np.array([0.0, 0.0, 0.0])

with Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    "[cyan]{task.completed}/{task.total} frames",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
    console=Console()
) as progress:
    render_task = progress.add_task("Rendering and storing frames", total=len(audio_info))
    
    for frame in range(len(audio_info)):
        curr_info: AudioInfo = audio_info[frame]
        current_color = np.array(curr_info.color)
        color_diff = np.linalg.norm(current_color - prev_color)
        frame_since_last_wave += 1
        
        # Determine the size based on loudness
        new_radius_scaler = curr_info.loudness * CIRCLE_SCALE_FACTOR
        radius_scaler = apply_asymmetric_ema(prev_radius_scaler, new_radius_scaler, ALPHA_UP_RADIUS, ALPHA_DOWN_RADIUS)
        prev_radius_scaler = radius_scaler

        # Determine the rotation 
        rotations_per_frame = curr_info.loudness * RPM_MULTIPLIER / (60 * FPS)
        curr_rotation = curr_rotation + rotations_per_frame * 2 * np.pi

        # Determine the average frequency
        new_avg_freq = curr_info.avg_freq
        avg_freq = apply_asymmetric_ema(prev_avg_freq, new_avg_freq, ALPHA_UP_AVG_FREQ, ALPHA_DOWN_AVG_FREQ)
        prev_avg_freq = avg_freq

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
            if wave['radius'] > WAVE_REMOVAL_RADIUS and len(active_waves) > 1:
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
        
        # Set shader uniforms for the waves
        bg_wave_prog['wave_colors'].value = wave_colors
        bg_wave_prog['wave_radii'].value = wave_radii
        bg_wave_prog['num_waves'].value = len(active_waves)
        bg_wave_prog['wave_thickness'].value = WAVE_THICKNESS
        bg_wave_prog['brightness'].value = BRIGHTNESS

        # Set the shader uniforms for the shape
        shape_prog['rotation'].value = curr_rotation
        shape_prog['radius_scale'].value = radius_scaler
        shape_prog['avg_freq'].value = avg_freq

        # Render the waves and the shape
        render_start = time.time()
        bg_quad_vao.render(moderngl.TRIANGLE_FAN)
        shape_vao.render(moderngl.TRIANGLE_FAN)
        render_duration = time.time() - render_start
        total_rendering_time += render_duration

        # Read framebuffer and save to video
        pixels = fbo.read(components=3, alignment=1)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        write_start = time.time()
        writer.append_data(np.flip(image, axis=0))  # flip Y-axis
        write_duration = time.time() - write_start
        total_writing_time += write_duration
        
        # Update progress
        progress.update(render_task, advance=1)

writer.close()
render_loop_duration = time.time() - render_loop_start


# ==== Combine with audio ====
ffmpeg_start = time.time()
console.print("\n")
console.log("Combining video with audio using FFmpeg")
process = subprocess.Popen([
    'ffmpeg',
    '-y',
    '-i', TEMP_VIDEO_FILE,
    '-i', AUDIO_FILE,
    '-c:v', 'copy',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-shortest',
    FINAL_VIDEO_FILE
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
process.communicate() # Wait for process to finish
ffmpeg_duration = time.time() - ffmpeg_start

os.remove(TEMP_VIDEO_FILE)


# ==== Timing Summary ====
print_timing_summary(
    console,
    audio_duration,
    render_loop_duration,
    ffmpeg_duration,
    total_rendering_time,
    total_writing_time,
    len(audio_info)
)