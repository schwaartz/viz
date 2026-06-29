# Audio Visualizer

Creates animated videos with shapes and colors that react to music.
The visuals are dependent on the volume and the frequencies of the music.
All parameters can be adjusted through the use of a dedicated config file.
The video is rendered using GLSL.

![Audio Visualizer Example](images/example.gif)
Example GIF for Faster by A.Oonthebeat

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required)
sudo apt install ffmpeg  # Debian
```

# Basic usage

**Note:** Run these commands from the `/src` directory

```bash
# Running the main program
python -m data_generator.main <mp3_input_path>

# With options
python -m data_generator.main <mp3_input_path> -o <mp4_output_path> -c config.json
```

## Options

- `-o, --output` Output video file
- `-c, --config` Custom config file

## Configuration

Edit `config.json` to customize visuals. Key settings:

- `circle_loudness_scale_factor` Shape size response
- `base_wave_speed` Wave animation speed
- `brightness` Overall brightness
- `fps`, `width`, `height` Quality settings
- ...

Since the name of the config file is a command line option, it is possible
to store multiple different config files.

## License

MIT License - see [LICENSE](LICENSE.txt)
