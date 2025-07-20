import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import moderngl
import numpy as np
import time
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from pygame.locals import *
from audio.audio_processing import  AudioInfo
from functions.ema import apply_asymmetric_ema
from config import VisualConfig


def render_loop(ctx: moderngl.Context, writer, audio_info: list,
                config: VisualConfig, bg_wave_prog: moderngl.Program,
                shape_prog: moderngl.Program, bg_quad_vao: moderngl.VertexArray,
                shape_vao: moderngl.VertexArray, fbo: moderngl.Framebuffer,
                console: Console) -> tuple:
    """
    Main render loop that processes audio information and renders frames accordingly.
    It also shows a live preview and a progress bar in the console while saving the frames
    to a video file.
    :param ctx: ModernGL context.
    :param writer: ImageIO writer object to save frames.
    :param audio_info: List of AudioInfo objects containing audio data.
    :param config: VisualConfig object with settings.
    :param bg_wave_prog: Background wave shader program.
    :param shape_prog: Shape shader program.
    :param bg_quad_vao: Vertex array object for the background quad.
    :param shape_vao: Vertex array object for the shape.
    :param fbo: Framebuffer for rendering.
    :param console: Console for logging.
    :return: Tuple containing render loop duration, total rendering time, and total writing time.
    """
    console.log("Starting render loop\n")
    render_loop_start = time.time()
    prev_color = np.array([0.0, 0.0, 0.0])
    frame_since_last_wave = 0
    active_waves = []
    rotation = 0.0
    timings = {
        "render_loop": 0.0,
        "total_rendering": 0.0,
        "total_writing": 0.0
    }
    ema_vars = {
        "prev_radius_scale": 0.0,
        "prev_avg_freq": 0.0,
    }

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", "•", "[cyan]{task.completed}/{task.total} frames", "•",
        TimeElapsedColumn(), "•",
        TimeRemainingColumn(),
        console=Console()
    ) as progress:
        render_task = progress.add_task("Rendering and storing frames", total=len(audio_info))
        
        for frame in range(len(audio_info)):
            _check_pygame_quit(writer)
            curr_info: AudioInfo = audio_info[frame]
            
            radius_scale, avg_freq = _apply_emas(curr_info, ema_vars, config)
            rotation = _update_rotation(rotation, curr_info.loudness, config)
            wave_info = _process_waves(config, frame, curr_info, active_waves, prev_color, frame_since_last_wave)
            active_waves, prev_color, frame_since_last_wave = wave_info
            
            _set_wave_uniforms(bg_wave_prog, active_waves, config)
            _set_shape_uniforms(shape_prog, radius_scale, avg_freq, rotation)

            _render_frame(ctx, fbo, bg_quad_vao, shape_vao, frame, timings, writer, config)

            progress.update(render_task, advance=1)

    pygame.quit()
    writer.close()
    timings['render_loop'] = time.time() - render_loop_start
    return (timings['render_loop'], timings['total_rendering'], timings['total_writing'])


# Private helper functions from here to the end

def _check_pygame_quit(writer) -> None:
    """
    Check if the Pygame window has been closed.
    If it has, close the Pygame window and exit the program.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            writer.close()
            exit()

def _apply_emas(curr_info: AudioInfo, ema_vars, config: VisualConfig) -> tuple:
    """
    Apply exponential moving averages to the current audio information.
    :param curr_info: Current AudioInfo object containing audio data.
    :param ema_vars: Dictionary containing previous EMA values.
    :param config: VisualConfig object with settings.
    :return: Tuple containing updated radius scale and average frequency.
    """
    new_radius_scale = curr_info.loudness * config.circle_loudness_scale_factor
    radius_scale = apply_asymmetric_ema(ema_vars['prev_radius_scale'], new_radius_scale, config.alpha_up_radius, config.alpha_down_radius)
    ema_vars['prev_radius_scale'] = radius_scale

    new_avg_freq = curr_info.avg_freq
    avg_freq = apply_asymmetric_ema(ema_vars['prev_avg_freq'], new_avg_freq, config.alpha_up_avg_freq, config.alpha_down_avg_freq)
    ema_vars['prev_avg_freq'] = avg_freq

    return radius_scale, avg_freq

def _update_rotation(curr_rotation: float, loudness: float, config: VisualConfig) -> float:
    """
    Update the current rotation based on the loudness and configuration settings.
    :param curr_rotation: Current rotation value.
    :param loudness: Loudness value from the current audio information.
    :param config: VisualConfig object with settings.
    :return: Updated rotation value.
    """
    rotations_per_frame = loudness * config.rotation_speed / (60 * config.fps)
    return curr_rotation + rotations_per_frame * 2 * np.pi

def _process_waves(config: VisualConfig, frame: int, curr_info: AudioInfo,
                  active_waves: list, prev_color: np.ndarray,
                  frame_since_last_wave: int) -> tuple:

    """ Process the wave spawning logic based on the current audio information.
    :param config: VisualConfig object
    :param frame: Current frame number.
    :param curr_info: Current AudioInfo object.
    :param active_waves: List of currently active waves.
    :param prev_color: Previous color used for wave spawning.
    :param frame_since_last_wave: Number of frames since the last wave was spawned.
    :return: Updated active_waves, prev_color, and frame_since_last_wave.
    """
    frame_since_last_wave += 1
    current_color = np.array(curr_info.color)
    color_diff = np.linalg.norm(current_color - prev_color)

    # Check if a new wave should be spawned
    if (frame == 0 or color_diff > config.color_change_threshold or
        frame_since_last_wave > config.max_frames_between_waves or len(active_waves) == 0):
        active_waves.append({'color': curr_info.color, 'radius': 0.0})
        prev_color = current_color.copy()
        frame_since_last_wave = 0

    # Update waves
    for wave in active_waves.copy():
        dynamic_speed = config.base_wave_speed + curr_info.loudness * config.wave_speed_loudness_scale_factor
        wave['radius'] += dynamic_speed
        if wave['radius'] > config.wave_removal_radius and len(active_waves) > 1:
            active_waves.remove(wave)

    # Limit the number of active waves
    if len(active_waves) > config.max_waves:
        active_waves = active_waves[-config.max_waves:]

    return active_waves, prev_color, frame_since_last_wave

def _set_wave_uniforms(wave_prog: moderngl.Program, active_waves: list, config: VisualConfig) -> None:
    """
    Set the uniforms for the wave shader program.
    :param wave_prog: Wave shader program.
    :param active_waves: List of currently active waves.
    :param config: VisualConfig object with settings.
    """
    wave_colors = []
    wave_radii = []

    for wave in active_waves:
        wave_colors.append(wave['color'])
        wave_radii.append(wave['radius'])

    while len(wave_colors) < config.max_waves:
        wave_colors.append([0.0, 0.0, 0.0])
    while len(wave_radii) < config.max_waves:
        wave_radii.append(0.0)

    wave_prog['wave_colors'].value = wave_colors
    wave_prog['wave_radii'].value = wave_radii
    wave_prog['num_waves'].value = len(active_waves)
    wave_prog['wave_thickness'].value = config.wave_thickness
    wave_prog['brightness'].value = config.brightness

def _set_shape_uniforms(shape_prog: moderngl.Program, radius_scale: float, avg_freq: float, rotation: float) -> None:
    """
    Set the uniforms for the shape shader program.
    :param shape_prog: Shape shader program.
    :param radius_scale: Radius scale value.
    :param avg_freq: Average frequency value.
    :param rotation: Current rotation value.
    """
    shape_prog['rotation'].value = rotation
    shape_prog['radius_scale'].value = radius_scale
    shape_prog['avg_freq'].value = avg_freq

def _render_frame(ctx: moderngl.Context, fbo: moderngl.Framebuffer, bg_quad_vao: moderngl.VertexArray, shape_vao: moderngl.VertexArray, frame: int, timings: dict, writer, config: VisualConfig) -> None:
    """ Render the current frame using the shader programs to a framebuffer and to
    the screen, then write the framebuffer to the video file.
    :param ctx: ModernGL context.
    :param fbo: Framebuffer for rendering.
    :param bg_quad_vao: Vertex array object for the background quad.
    :param shape_vao: Vertex array object for the shape.
    :param frame: Current frame number.
    :param timings: Dictionary to store timing information.
    :param writer: ImageIO writer object to save frames.
    :param config: VisualConfig object with settings.
    """
    # Only render every 10th frame (for preview)
    if frame % 10 == 0:
        ctx.screen.use()
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        render_start = time.time()
        bg_quad_vao.render(moderngl.TRIANGLE_FAN)
        shape_vao.render(moderngl.TRIANGLE_FAN)
        timings['total_rendering'] += time.time() - render_start
    
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    render_start = time.time()
    bg_quad_vao.render(moderngl.TRIANGLE_FAN)
    shape_vao.render(moderngl.TRIANGLE_FAN)
    timings['total_rendering'] += time.time() - render_start

    pygame.display.flip()

    pixels = fbo.read(components=3, alignment=1)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((config.height, config.width, 3))
    write_start = time.time()
    writer.append_data(np.flip(image, axis=0))
    timings['total_writing'] += time.time() - write_start