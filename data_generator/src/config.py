import json
from pathlib import Path
from dataclasses import dataclass, asdict
from rich.console import Console

@dataclass
class VisualConfig:
    """
    Configuration for the audio visualizer. If the names of the parameters are
    not clear enough, please take a look at the source code for more
    information. Also note that some of the parameters are using in speed
    calculations that depend on the FPS. Therefore this class has a method to
    rescale the constants based on the actual FPS. If this function is not
    called when the FPS is changed (from the default of 30), the speed of the
    visuals will change accordingly.
    """
    # File settings
    temp_file: str = 'temp/temp_video.mp4'
    duration: int = 200
    fps: int = 60
    width: int = 1920
    height: int = 1088
    
    # Shape settings
    circle_base_size: float = 0.06
    circle_loudness_scale_factor: float = 5.0
    rotation_speed: float = 1000
    protrusion_scale: float = 0.25
    num_protrusions: int = 6
    protrusion_variability: float = 2.0
    protrusion_base_thickness: float = 0.01
    protrusion_thickening_factor: float = 3.5
    shape_vertices: int = 1000
    
    # Wave settings
    base_wave_speed: float = 0.03
    wave_speed_loudness_scale_factor: float = 0.25
    wave_thickness: float = 0.40
    brightness: float = 1.2
    color_change_threshold: float = 0.025
    max_frames_between_waves: int = 15
    wave_removal_radius: float = 4.0
    max_waves: int = 32
    
    # EMA settings
    alpha_up_radius: float = 0.9
    alpha_down_radius: float = 0.2
    alpha_up_avg_freq: float = 0.8
    alpha_down_avg_freq: float = 0.15
    alpha_up_bg_speed: float = 0.3
    alpha_down_bg_speed: float = 0.05

    # Audio settings
    num_frequency_bands: int = 128
    freq_band_weight_func_exponent: float = 0.2 # lower value = higher weight for lower freq

    def rescale_constants_based_on_fps(self):
        """
        Rescale constants based on the actual FPS. This is because many
        calculations depend on the amount of frames. The base FPS is used to
        scale the constants, so that the speed of the visuals remains consistent
        regardless of the FPS.
        """
        base_fps = 30
        scaling_factor = base_fps / self.fps
        self.rotation_speed *= scaling_factor
        self.base_wave_speed *= scaling_factor
        self.wave_speed_loudness_scale_factor *= scaling_factor
        self.max_frames_between_waves = int(self.max_frames_between_waves * scaling_factor)
        self.circle_loudness_scale_factor *= scaling_factor

        self.alpha_up_radius *= scaling_factor
        self.alpha_down_radius *= scaling_factor
        self.alpha_up_avg_freq *= scaling_factor
        self.alpha_down_avg_freq *= scaling_factor
        self.alpha_up_bg_speed *= scaling_factor
        self.alpha_down_bg_speed *= scaling_factor


def load_config(config_file: str = 'config.json', console: Console = None) -> VisualConfig:
    """
    Load configuration from JSON file, create default if not exists.
    :param config_file: Path to the configuration file.
    :return: VisualConfig object with loaded settings.
    """
    config_path = Path(config_file)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = json.load(f)
        if console: console.log(f"Loaded configuration from {config_file}")
        config = VisualConfig(**data)
    else:
        config = VisualConfig() # default config
        save_config(config, config_file)
        if console: console.log(f"Created default config file: {config_file}")
    
    config.rescale_constants_based_on_fps()
    return config

def save_config(config: VisualConfig, config_file: str = 'config.json'):
    """
    Save configuration to JSON file
    :param config: VisualConfig object to save.
    :param config_file: Path where to save the configuration.
    """
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)