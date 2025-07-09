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
RPM = 500 # Revolutions per minute for the spinning animation
CIRCLE_BASE_SIZE = 0.08  # Base size of the circle
CIRCLE_SCALE_FACTOR = 2  # Scale factor for circle size based on loud
PORTR_SCALE = 0.60  # Factor to normalize protrusion in the shape 
MAX_PROTR_AMOUNT = 10 # This also generally influences the amount portrusions
VERTECES = 1000  # Number of verteces in the shape
PORTR_VARIABILITY = 2.0 # Variability factor for protrusion lengths
MAX_ACTIVE_PROTRUSIONS = 10  # Maximum number of protrusions to be active at once, DO NOT CHANGE, HARDCODED ELSEWHERE
PROTR_BASE_THICKNESS = 1.0 # Base thickness of protrusions
PROTR_THICKNESS_FACTOR = 3.0 # Factor to scale protrusion thickness based on the frequency of the sound
MIN_PROTRUSIONS = 6 # The amount of protrusions, too little will look janky

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
ALPHA_UP_COLOR = 0.4 # Not in use at the moment   
ALPHA_DOWN_COLOR = 0.05 # Not in use at the moment
ALPHA_UP_RADIUS = 0.8
ALPHA_DOWN_RADIUS = 0.2
ALPHA_UP_AVG_FREQ = 0.6
ALPHA_DOWN_AVG_FREQ = 0.10
ALPHA_UP_BG_SPEED = 0.4
ALPHA_DOWN_BG_SPEED = 0.05
ALPHA_UP_PROTR = 0.4
ALPHA_DOWN_PROTR = 0.05

# Static portrusion number
USE_FIXED_PORTR_NUM = True
FIXED_PORTR_NUM = 6