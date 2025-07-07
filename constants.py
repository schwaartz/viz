AUDIO_FILE = 'input/spooky_beat.mp3'
FPS = 30
DURATION = 50  # in seconds
TEMP_VIDEO_FILE = 'temp/temp_video.mp4'
FINAL_VIDEO_FILE = 'output/spooky_beat.mp4'
WIDTH, HEIGHT = 1920, 1080
NUM_FREQ = 128


# === Visual settings ===

# Shape and animation
RPM = 350 # Revolutions per minute for the spinning animation
CIRCLE_BASE_SIZE = 0.1  # Base size of the circle
CIRCLE_SCALE_FACTOR = 0.15  # Scale factor for circle size based on loud
PORTRUSION_FACTOR = 0.30  # Factor to normalize protrusion in the shape 
PERTURBATION_MAX_AMOUNT = 10 # This also generally influences the amount perturbations

# Asymmetric EMA settings
ALPHA_UP_COLOR = 0.7   
ALPHA_DOWN_COLOR = 0.08
ALPHA_UP_RADIUS = 0.7
ALPHA_DOWN_RADIUS = 0.15
ALPHA_UP_AVG_FREQ = 0.7
ALPHA_DOWN_AVG_FREQ = 0.15
ALPHA_UP_NUM_PERT = 0.2 # Does nothing if USE_FIXED_PERT_NUM = True
ALPHA_DOWN_NUM_PERT = 0.2 # Does nothing if USE_FIXED_PERT_NUM = True

# Static perturbation number
USE_FIXED_PERT_NUM = True
FIXED_PERT_NUM = 10