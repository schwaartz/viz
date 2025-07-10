AUDIO_FILE = 'input/faster.mp3'
FINAL_VIDEO_FILE = 'output/faster.mp4'
TEMP_VIDEO_FILE = 'temp/temp_video.mp4'
WIDTH, HEIGHT = 1920, 1088 # 1088 otherwise FFMPEG complains
DURATION = 35  # in seconds
FPS = 30

# === Audio settings ===
NUM_FREQ = 128
LOWER_FREQ_WEIGHT_FUNC_EXPONENT = 0.2 # The lower, the higher the weight of lower frequencies

# === Visual settings ===
# Shape and animation
RPM = 600 # Revolutions per minute for the spinning animation
CIRCLE_BASE_SIZE = 0.06  # Base size of the circle
CIRCLE_SCALE_FACTOR = 3.3  # Scale factor for circle size based on loud
PROTR_SCALE = 0.25  # Factor to normalize protrusion in the shape 
PROTR_AMOUNT = 6 # This also generally influences the amount protrusions
VERTECES = 1000  # Number of verteces in the shape
PROTR_VARIABILITY = 2.2 # Variability factor for protrusion lengths
PROTR_BASE_THINNESS = 0.01 # Base thickness of protrusions
PROTR_THICKENING_FACTOR = 3.5 # Factor to scale protrusion thickness based on the frequency of the sound

# Background wave settings
BASE_WAVE_SPEED = 0.02  # Base speed of waves
WAVE_SPEED_MULTIPLIER = 0.25  # Higher multiplier for more variation
WAVE_THICKNESS = 0.6  # Thicker waves for better blending
COLOR_CHANGE_THRESHOLD = 0.05  # Much lower threshold to spawn waves more frequently
MAX_FRAMES_BETWEEN_WAVES = 15  # More frequent spawning of waves
WAVE_REMOVAL_RADIUS = 4.0  # Larger removal threshold for waves
BRIGHTNESS = 1.2 # Use fixed brightness for background waves
MAX_WAVES = 32 # Maximum number of waves on screen, DO NOT CHANGE, HARDCODED ELSEWHERE

# Asymmetric EMA settings
ALPHA_UP_RADIUS = 0.8
ALPHA_DOWN_RADIUS = 0.2
ALPHA_UP_AVG_FREQ = 0.6
ALPHA_DOWN_AVG_FREQ = 0.10
ALPHA_UP_BG_SPEED = 0.4
ALPHA_DOWN_BG_SPEED = 0.05
ALPHA_UP_PROTR = 0.4
ALPHA_DOWN_PROTR = 0.05