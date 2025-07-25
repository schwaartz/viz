from rich.console import Console
from rich.table import Table
from rich import box
from config import VisualConfig

def print_timing_summary(console: Console, audio_duration: float,
                         render_loop_duration: float, ffmpeg_duration: float,
                         total_rendering_time: float, total_writing_time: float,
                         total_frames: int, config: VisualConfig, output_file: str) -> None:
    """
    Prints a summary of the timing for each stage of the process.
    :param console: The console to print to.
    :param audio_duration: Duration of audio processing.
    :param render_loop_duration: Duration of the render loop.
    :param ffmpeg_duration: Duration of the FFmpeg processing.
    :param total_rendering_time: Total time spent rendering frames.
    :param total_writing_time: Total time spent writing frames to video.
    :param total_frames: Total number of frames rendered.
    :param config: VisualConfig object with settings.
    :param output_file: Path to the final output video file.
    """
    total_time = audio_duration + render_loop_duration + ffmpeg_duration
    table = Table(title="TIMING SUMMARY", box=box.ROUNDED)
    table.add_column("Stage", style="bold cyan")
    table.add_column("Time (s)", justify="right")
    table.add_column("Percent", justify="right")
    table.add_row("Audio processing", 
                f"{audio_duration:.2f}", 
                f"{((audio_duration) / total_time * 100):.1f}%")
    table.add_row("Render Loop", 
                f"{render_loop_duration:.2f}", 
                f"{((render_loop_duration) / total_time * 100):.1f}%")
    table.add_row("  - Rendering frames", 
                f"{total_rendering_time:.2f}", 
                f"{(total_rendering_time / (render_loop_duration) * 100):.1f}%")
    table.add_row("  - Writing frames", 
                f"{total_writing_time:.2f}", 
                f"{(total_writing_time / (render_loop_duration) * 100):.1f}%")
    table.add_row("FFmpeg", 
                f"{ffmpeg_duration:.2f}", 
                f"{((ffmpeg_duration) / total_time * 100):.1f}%")
    table.add_row("Total", 
                f"{total_time:.2f}", 
                "100%")
    console.log(f"Rendered {config.duration} seconds of video at {config.fps} FPS ({total_frames} frames)")
    console.log(f"Final video with audio saved as [bold][underlined]{output_file}[/underlined][/bold]")
    console.print("\n", table, "\n")