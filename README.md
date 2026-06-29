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
# Running the generator
python -m data_generator.generate <mp3_input_path>

# With options
python -m data_generator.generate <mp3_input_path> -o <mp4_output_path> -c config.json
```

## Options

- `-o, --output` Output video file
- `-c, --config` Custom config file

## Configuration

Edit `config.json` to customize visuals. The file can be found under
`src/data_generator`. Key settings:

- `circle_loudness_scale_factor` Shape size response
- `base_wave_speed` Wave animation speed
- `brightness` Overall brightness
- `fps`, `width`, `height` Quality settings
- ...

Since the name of the config file is a command line option, it is possible
to store multiple different config files.

---

# Video Prediction

There is also a separate module under the `src` directory named
`video_prediction`. This module contains the necessary Python code to predict
the output of the audio visualizer given some input audio.

This is experimental and only serves as an interesting proof of concept. The
learned model is obviously much slower than the true video generator and
produces videos at a lower resolution to compensate.

The model itself is a simple RNN with transposed 2D convolutional layers. It
takes in a number of audio spectra and its previous state and predicts the next
frame. It does this for a certain window of time. To generate longer videos, it
generates the result in chunks. Practically, the input $x$ has the shape $(B, 1,
F, T_c)$ where $B$ is the batch size, $F$ is the number of frequency buckets and
$T_c$ is the number of audio features per segment. The output $y$ has the shape
$(B, T_v, 3, H, W)$ where $T_v$ is the number of frames and $H$ and $W$
correspond to the height and width respectively. There are 3 color channels,
hence the number 3.

There is no config file for this module. The specific numbers that were used
here aren't all that important. They can be found in `src/video_prediction/constants.py`.
The exact architecture can be found in `src/video_prediction/model.py`.

## Using the predictor

To train the model, you must first generate some examples using the generator
tool. The audio files go in one directory, while the output videos go in a
different directory. The names of the files must match. (e.g. `song1.mp3` in the
audio directory corresponds to `song1.mp4` in the video directory)

```bash
# Generating the dataset
python -m video_prediction.preprocess_dataset --audio-dir <input_audio_directory> --video-dir <output_video_directory>

# Train the model
python -m video_prediction.train

# Make a prediction
python -m video_prediction.predict -i <input_audio_path> -o <output_video_path>
```
