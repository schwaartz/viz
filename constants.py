AUDIO_FILE = 'input/halogen.mp3'
FINAL_VIDEO_FILE = 'output/halogen.mp4'
TEMP_VIDEO_FILE = 'temp/temp_video.mp4'
WIDTH, HEIGHT = 1920, 1088 # 1088 otherwise FFMPEG complains
DURATION = 100  # in seconds
FPS = 30

# === Audio settings ===
NUM_FREQ = 128
LOWER_FREQ_WEIGHT_FUNC_EXPONENT = 0.2 # The lower, the higher the weight of lower frequencies

# === Visual settings ===
# Shape and animation
RPM = 500 # Revolutions per minute for the spinning animation
CIRCLE_BASE_SIZE = 0.05  # Base size of the circle
CIRCLE_SCALE_FACTOR = 0.35  # Scale factor for circle size based on loud
PORTRUSION_SCALE = 0.60  # Factor to normalize protrusion in the shape 
PERTURBATION_MAX_AMOUNT = 10 # This also generally influences the amount perturbations, useless if USE_FIXED_PERT_NUM = True
SEGMENTS = 400  # Number of segments in the shape
PORTRUSION_VARIABILITY = 2.0 # Variability factor for protrusion lengths

# Background wave settings
BASE_WAVE_SPEED = 0.02  # Base speed of waves
WAVE_SPEED_MULTIPLIER = 0.25  # Higher multiplier for more variation
WAVE_THICKNESS = 0.6  # Thicker waves for better blending
COLOR_CHANGE_THRESHOLD = 0.05  # Much lower threshold to spawn waves more frequently
MAX_FRAMES_BETWEEN_WAVES = 15  # More frequent spawning of waves
WAVE_REMOVAL_RADIUS = 4.0  # Larger removal threshold for waves
BRIGHTNESS = 1.2 # Use fixed brightness for background waves
MAX_WAVES = 32 # DO NOT CHANGE, HARDCODED ELSEWHERE

# Asymmetric EMA settings
ALPHA_UP_COLOR = 0.4 # Not in use at the moment   
ALPHA_DOWN_COLOR = 0.05 # Not in use at the moment
ALPHA_UP_RADIUS = 0.7
ALPHA_DOWN_RADIUS = 0.15
ALPHA_UP_AVG_FREQ = 0.6
ALPHA_DOWN_AVG_FREQ = 0.10
ALPHA_UP_NUM_PERT = 0.2 # Does nothing if USE_FIXED_PERT_NUM = True
ALPHA_DOWN_NUM_PERT = 0.2 # Does nothing if USE_FIXED_PERT_NUM = True
ALPHA_UP_BG_SPEED = 0.4
ALPHA_DOWN_BG_SPEED = 0.05

# Static perturbation number
USE_FIXED_PERT_NUM = True
FIXED_PERT_NUM = 6