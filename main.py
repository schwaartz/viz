import moderngl
import numpy as np
import imageio
import subprocess
import os
import time
import pygame
from pygame.locals import *
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from visuals.create_circle import create_circle
from audio.audio_processing import short_time_fourrier_transform, get_audio_info, AudioInfo
from utils.ema import apply_asymmetric_ema
from utils.load_shader import load_shader_program
from utils.timing_summary import print_timing_summary
from config import VisualConfig, load_config


# ==== Config ====
console = Console()
console.log("Starting program")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
config: VisualConfig = load_config(console=console)


# ==== Initialize Pygame with OpenGL ====
pygame.init()
pygame.display.set_mode((config.width, config.height), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Audio Visualizer - Live Preview")


# ==== Visuals ====
# Create ModernGL context from pygame's OpenGL context
ctx = moderngl.create_context()
writer = imageio.get_writer(config.temp_file, fps=config.fps)

shape_prog = load_shader_program(ctx, 'shaders/shape.vert', 'shaders/shape.frag')
shape_prog['protr_base_thickness'].value = config.protrusion_base_thickness 
shape_prog['protr_thickness_factor'].value = config.protrusion_thickening_factor  
shape_prog['height_width_ratio'].value = config.height / config.width
shape_prog['protr_amount'].value = config.num_protrusions
shape_prog['protr_scale'].value = config.protrusion_scale
shape_prog['protr_variability'].value = config.protrusion_variability

bg_wave_prog = load_shader_program(ctx, 'shaders/wave.vert', 'shaders/wave.frag')
bg_quad_vertices = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0], 
], dtype='f4')
bg_quad_vbo = ctx.buffer(bg_quad_vertices.tobytes())
bg_quad_vao = ctx.simple_vertex_array(bg_wave_prog, bg_quad_vbo, 'in_pos')
shape_vao = create_circle(ctx, shape_prog, config)

# Create framebuffer for video output
fbo = ctx.simple_framebuffer((config.width, config.height))


# ==== Audio ====
console.log(f"Processing audio file [bold]{config.audio_file}[/bold]")
audio_start = time.time()
stft = short_time_fourrier_transform(config)

console.log("Extracting audio information")
audio_info = get_audio_info(stft, config)
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        curr_info: AudioInfo = audio_info[frame]
        current_color = np.array(curr_info.color)
        color_diff = np.linalg.norm(current_color - prev_color)
        frame_since_last_wave += 1
        
        # Calculate values
        new_radius_scaler = curr_info.loudness * config.circle_loudness_scale_factor
        radius_scaler = apply_asymmetric_ema(prev_radius_scaler, new_radius_scaler, config.alpha_up_radius, config.alpha_down_radius)
        prev_radius_scaler = radius_scaler

        rotations_per_frame = curr_info.loudness * config.rotation_speed / (60 * config.fps)
        curr_rotation = curr_rotation + rotations_per_frame * 2 * np.pi

        new_avg_freq = curr_info.avg_freq
        avg_freq = apply_asymmetric_ema(prev_avg_freq, new_avg_freq, config.alpha_up_avg_freq, config.alpha_down_avg_freq)
        prev_avg_freq = avg_freq

        # Wave spawning logic
        if (frame == 0 or color_diff > config.color_change_threshold or
            frame_since_last_wave > config.max_frames_between_waves or len(active_waves) == 0):
            active_waves.append({
                'color': curr_info.color,
                'radius': 0.0
            })
            prev_color = current_color.copy()
            frame_since_last_wave = 0

        # Update waves
        for wave in active_waves.copy():
            dynamic_speed = config.base_wave_speed + curr_info.loudness * config.wave_speed_loudness_scale_factor
            wave['radius'] += dynamic_speed
            if wave['radius'] > config.wave_removal_radius and len(active_waves) > 1:
                active_waves.remove(wave)

        if len(active_waves) > config.max_waves:
            active_waves = active_waves[-config.max_waves:]
        
        # Prepare wave data
        wave_colors = []
        wave_radii = []
        
        for wave in active_waves:
            wave_colors.append(wave['color'])
            wave_radii.append(wave['radius'])
        
        while len(wave_colors) < config.max_waves:
            wave_colors.append([0.0, 0.0, 0.0])
        while len(wave_radii) < config.max_waves:
            wave_radii.append(0.0)
        
        # Set uniforms
        bg_wave_prog['wave_colors'].value = wave_colors
        bg_wave_prog['wave_radii'].value = wave_radii
        bg_wave_prog['num_waves'].value = len(active_waves)
        bg_wave_prog['wave_thickness'].value = config.wave_thickness
        bg_wave_prog['brightness'].value = config.brightness

        shape_prog['rotation'].value = curr_rotation
        shape_prog['radius_scale'].value = radius_scaler
        shape_prog['avg_freq'].value = avg_freq

        # Render to BOTH screen and framebuffer
        render_start = time.time()

        # Only render every 10th frame to reduce load (for preview)
        if frame % 10 == 0:
            ctx.screen.use()
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            bg_quad_vao.render(moderngl.TRIANGLE_FAN)
            shape_vao.render(moderngl.TRIANGLE_FAN)
        
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        bg_quad_vao.render(moderngl.TRIANGLE_FAN)
        shape_vao.render(moderngl.TRIANGLE_FAN)
        render_duration = time.time() - render_start
        total_rendering_time += render_duration

        pygame.display.flip()

        pixels = fbo.read(components=3, alignment=1)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape((config.height, config.width, 3))
        write_start = time.time()
        writer.append_data(np.flip(image, axis=0))
        write_duration = time.time() - write_start
        total_writing_time += write_duration

        progress.update(render_task, advance=1)

pygame.quit()
writer.close()
render_loop_duration = time.time() - render_loop_start


# ==== Combine with audio ====
ffmpeg_start = time.time()
console.print("\n")
console.log("Combining video with audio using FFmpeg")
process = subprocess.Popen([
    'ffmpeg',
    '-y',
    '-i', config.temp_file,
    '-i', config.audio_file,
    '-c:v', 'copy',
    '-map', '0:v:0',
    '-map', '1:a:0',
    '-shortest',
    config.output_file
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
process.communicate()
ffmpeg_duration = time.time() - ffmpeg_start

os.remove(config.temp_file)


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