import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import moderngl
import numpy as np
import imageio
import subprocess
import time
from rich.console import Console
from vao.create_circle import create_circle_vao
from vao.create_quad import create_quad_vao
from audio.audio_processing import short_time_fourrier_transform, get_audio_info
from shaders.utils.load_shader import load_shader_program
from timing_summary import print_timing_summary
from config import VisualConfig, load_config
from argument_parser import parse_arguments
from render_loop import render_loop


def main():
    """
    Main function to run the audio visualizer. It initializes the Pygame window,
    sets up the ModernGL context, loads shaders, processes audio, and runs the
    render loop. Finally, it combines the rendered video with audio using
    FFmpeg.
    """
    args = parse_arguments()
    console = Console()
    console.log("Starting program")
    config: VisualConfig = load_config(config_file=args.config, console=console)

    _initialize_pygame(config)
    ctx = moderngl.create_context()
    writer = imageio.get_writer(config.temp_file, fps=config.fps)

    shape_prog = load_shader_program(ctx, 'shaders/shape.vert', 'shaders/shape.frag')
    _set_shape_prog_uniforms(shape_prog, config)
    wave_prog = load_shader_program(ctx, 'shaders/wave.vert', 'shaders/wave.frag')
    quad_vao = create_quad_vao(ctx, wave_prog)
    shape_vao = create_circle_vao(ctx, shape_prog, config)

    console.log(f"Processing audio file [bold]{args.input_audio}[/bold]")
    audio_info, audio_duration = _process_audio(args.input_audio, config)

    timings = render_loop(ctx, writer, audio_info, config, wave_prog, shape_prog, quad_vao, shape_vao, console)
    (render_loop_duration, total_rendering_time, total_writing_time) = timings

    console.log("\n", "Combining video with audio using FFmpeg")
    ffmpeg_duration = _combine_audio_with_video(config, args)

    print_timing_summary(
        console,
        audio_duration,
        render_loop_duration,
        ffmpeg_duration,
        total_rendering_time,
        total_writing_time,
        len(audio_info),
        config,
        args.output if args.output else 'output.mp4'
    )

def _initialize_pygame(config: VisualConfig) -> None:
    """
    Initialize Pygame with the specified configuration.
    :param config: VisualConfig object containing settings.
    """
    pygame.init()
    pygame.display.set_mode((config.width, config.height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Audio Visualizer - Live Preview")

def _set_shape_prog_uniforms(shape_prog: moderngl.Program, config: VisualConfig) -> None:
    """
    Set the uniforms for the shape shader program based on the config.
    :param shape_prog: The shader program for shapes.
    :param config: The VisualConfig object containing settings.
    """
    shape_prog['protr_base_thickness'].value = config.protrusion_base_thickness 
    shape_prog['protr_thickness_factor'].value = config.protrusion_thickening_factor  
    shape_prog['height_width_ratio'].value = config.height / config.width
    shape_prog['protr_amount'].value = config.num_protrusions
    shape_prog['protr_scale'].value = config.protrusion_scale
    shape_prog['protr_variability'].value = config.protrusion_variability

def _process_audio(audio_file: str, config: VisualConfig) -> tuple:
    """
    Process the audio file to extract the short-time Fourier transform (STFT)
    and audio information.
    :param audio_file: Path to the input audio file.
    :param config: VisualConfig object with settings.
    :return: Tuple containing AudioInfo and the time it took.
    """
    start_time = time.time()
    stft = short_time_fourrier_transform(audio_file, config)
    audio_info = get_audio_info(stft, config)
    processing_time = time.time() - start_time
    return audio_info, processing_time

def _combine_audio_with_video(config: VisualConfig, args) -> float:
    """
    Combine the rendered video frames with the audio using FFmpeg.
    :param config: VisualConfig object with settings.
    :param args: Command line arguments containing input audio and output file.
    :return: Duration of the FFmpeg processing.
    """
    ffmpeg_start = time.time()
    default_output_file = args.input_audio.replace('.mp3', '.mp4')
    process = subprocess.Popen([
        'ffmpeg',
        '-y',
        '-i', config.temp_file,
        '-i', args.input_audio,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        args.output if args.output else default_output_file, 
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()
    ffmpeg_duration = time.time() - ffmpeg_start
    os.remove(config.temp_file)
    return ffmpeg_duration

if __name__ == "__main__":
    main()