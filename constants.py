# === General Constants ===
AUDIO_FILE = 'input/spooky_beat.mp3'
FINAL_VIDEO_FILE = 'output/spooky_beat.mp4'
TEMP_VIDEO_FILE = 'temp/temp_video.mp4'
WIDTH, HEIGHT = 1920, 1088
DURATION = 200  # in seconds
FPS = 60


# === Audio settings ===
NUM_FREQ = 128
LOWER_FREQ_WEIGHT_FUNC_EXPONENT = 0.2 # The lower, the higher the weight of lower frequencies


# === Visual settings ===
# Shape and animation
RPM_MULTIPLIER = 1000 # Revolutions per minute for the spinning animation (also depends on loudness)
CIRCLE_BASE_SIZE = 0.06  # Base size of the circle
CIRCLE_SCALE_FACTOR = 5.0  # Scale factor for circle size based on loud
PROTR_SCALE = 0.25  # Factor to normalize protrusion in the shape 
PROTR_AMOUNT = 6 # This also generally influences the amount protrusions
VERTECES = 1000  # Number of verteces in the shape
PROTR_VARIABILITY = 2.0 # Variability factor for protrusion lengths
PROTR_BASE_THINNESS = 0.01 # Base thickness of protrusions
PROTR_THICKENING_FACTOR = 3.5 # Factor to scale protrusion thickness based on the frequency of the sound

# Background wave settings
BASE_WAVE_SPEED = 0.03  # Base speed of waves
WAVE_SPEED_MULTIPLIER = 0.25  # Higher multiplier for more variation
WAVE_THICKNESS = 0.40  # Thicker waves for better blending
COLOR_CHANGE_THRESHOLD = 0.025  # Much lower threshold to spawn waves more frequently
MAX_FRAMES_BETWEEN_WAVES = 15  # More frequent spawning of waves, assumes FPS is 30 (see later scaling)
WAVE_REMOVAL_RADIUS = 4.0  # Larger removal threshold for waves
BRIGHTNESS = 1.2 # Use fixed brightness for background waves
MAX_WAVES = 32 # Maximum number of waves on screen, DO NOT CHANGE, HARDCODED ELSEWHERE

# Asymmetric EMA settings
ALPHA_UP_RADIUS = 0.9
ALPHA_DOWN_RADIUS = 0.2
ALPHA_UP_AVG_FREQ = 0.8
ALPHA_DOWN_AVG_FREQ = 0.15
ALPHA_UP_BG_SPEED = 0.3
ALPHA_DOWN_BG_SPEED = 0.05


# === Tweak Constants Based on FPS ===
""" Since many calculations depend on the amount of frames, we define a base FPS constant.
We then scale the constants based on the actual FPS used in the video, so that the speed
of the visuals remains consistent regardless of the FPS."""
BASE_FPS = 30 # Base FPS for calculations
scaling_factor = BASE_FPS / FPS

# Scale constants based on the actual FPS
RPM_MULTIPLIER = RPM_MULTIPLIER * scaling_factor
BASE_WAVE_SPEED = BASE_WAVE_SPEED * scaling_factor
WAVE_SPEED_MULTIPLIER = WAVE_SPEED_MULTIPLIER * scaling_factor
MAX_FRAMES_BETWEEN_WAVES = int(MAX_FRAMES_BETWEEN_WAVES * scaling_factor)

# Asymmetric EMA scaling
ALPHA_UP_RADIUS = ALPHA_UP_RADIUS * scaling_factor
ALPHA_DOWN_RADIUS = ALPHA_DOWN_RADIUS * scaling_factor
ALPHA_UP_AVG_FREQ = ALPHA_UP_AVG_FREQ * scaling_factor
ALPHA_DOWN_AVG_FREQ = ALPHA_DOWN_AVG_FREQ * scaling_factor
ALPHA_UP_BG_SPEED = ALPHA_UP_BG_SPEED * scaling_factor
ALPHA_DOWN_BG_SPEED = ALPHA_DOWN_BG_SPEED * scaling_factor
