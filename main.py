import moderngl
import numpy as np
import imageio
import subprocess
import os
import time
from visuals.create_shape import create_shape
from audio.audio_processing import short_time_fourrier_transform, get_audio_info, AudioInfo
from utils.ema import apply_background_color_asymmetric_ema, apply_asymmetric_ema
from constants import (
    PERTURBATION_MAX_AMOUNT,
    TEMP_VIDEO_FILE,
    FINAL_VIDEO_FILE,
    AUDIO_FILE,
    FPS,
    DURATION,
    WIDTH,
    HEIGHT,
    NUM_FREQ,
    ALPHA_UP_COLOR,
    ALPHA_DOWN_COLOR,
    ALPHA_UP_RADIUS,
    ALPHA_DOWN_RADIUS,
    ALPHA_UP_AVG_FREQ,
    ALPHA_DOWN_AVG_FREQ,
    ALPHA_UP_BG_SPEED,
    ALPHA_DOWN_BG_SPEED,
    RPM,
    CIRCLE_BASE_SIZE,
    CIRCLE_SCALE_FACTOR,
    USE_FIXED_PERT_NUM,
    FIXED_PERT_NUM,
    BACKGROUND_SPEED
)


# ==== Visuals ====
ctx = moderngl.create_context(standalone=True)
writer = imageio.get_writer(TEMP_VIDEO_FILE, fps=FPS)
shape_prog = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_pos;
        void main() {
            gl_Position = vec4(in_pos.x, in_pos.y, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);  // Black circle
        }
    ''',
)

# Replace the fragment shader in bg_wave_prog with this brighter version
bg_wave_prog = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_pos;
        out vec2 frag_pos;
        void main() {
            frag_pos = in_pos;
            gl_Position = vec4(in_pos.x, in_pos.y, 0.0, 1.0);
        }
    ''',
    fragment_shader='''
        #version 330
        in vec2 frag_pos;
        uniform vec3 wave_colors[32];  // Array of wave colors
        uniform float wave_radii[32];  // Array of wave radii
        uniform int num_waves;
        uniform float wave_thickness;
        uniform float brightness;
        out vec4 fragColor;
        
        void main() {
            float dist = length(frag_pos);
            vec3 final_color = vec3(0.0, 0.0, 0.0);  // Start with black
            float total_weight = 0.0;
            
            // Blend waves with extended center coverage
            for (int i = 0; i < num_waves && i < 32; i++) {
                float wave_radius = wave_radii[i];
                
                // Calculate distance from wave center
                float distance_from_wave = abs(dist - wave_radius);
                
                // Extended blending area for better center coverage
                float blend_thickness = wave_thickness * 1.5;  // Increased coverage
                
                if (distance_from_wave < blend_thickness) {
                    // Steeper falloff for more distinct colors
                    float weight = 1.0 - pow(distance_from_wave / blend_thickness, 1.5);  // Gentler falloff
                    
                    // Add weighted color contribution
                    final_color += wave_colors[i] * weight;
                    total_weight += weight;
                }
            }
            
            // Fallback color if no waves cover this pixel
            if (total_weight <= 0.0 && num_waves > 0) {
                final_color = wave_colors[0] * 0.3;  // Use first wave color as fallback
            } else if (total_weight > 0.0) {
                final_color = final_color / total_weight * brightness;
            }
            
            fragColor = vec4(final_color, 1.0);
        }
    ''',
)

# Create fullscreen quad for wave rendering
bg_quad_vertices = np.array([
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0], 
], dtype='f4')
bg_quad_vbo = ctx.buffer(bg_quad_vertices.tobytes())
bg_quad_vao = ctx.simple_vertex_array(bg_wave_prog, bg_quad_vbo, 'in_pos')

fbo = ctx.simple_framebuffer((WIDTH, HEIGHT)) # framebuffer object
fbo.use()


# ==== Audio ====
print("Starting audio processing...")
audio_start = time.time()
stft = short_time_fourrier_transform() # Load and process audio data
audio_info = get_audio_info(stft, NUM_FREQ)  # Calculate audio information for the first frame
audio_end = time.time()


# ==== Render Loop ====
print("Starting rendering...")
render_start = time.time()
prev_bg_speed = 0.0
prev_radius = 0.0
prev_avg_freq = 0.0
prev_pert_num_float = 0.0
curr_rotation = 0.0 
prev_brightness = 1.0  # Add this line

# Wave system for color gradients
active_waves = []
max_waves = 32
base_wave_speed = 0.02  # Lower base speed
wave_speed_multiplier = 10.0  # Higher multiplier for more variation
wave_thickness = 0.6  # Thicker waves for better coverage
prev_color = np.array([0.0, 0.0, 0.0])
color_change_threshold = 0.05  # Much lower threshold to spawn waves more frequently
frame_since_last_wave = 0  # Track frames since last wave

for frame in range(DURATION * FPS):
    curr_info: AudioInfo = audio_info[frame]
    current_color = np.array(curr_info.color)
    
    # Only add new wave when color changes significantly OR when no waves for too long
    color_diff = np.linalg.norm(current_color - prev_color)
    frame_since_last_wave += 1
    
    if frame == 0 or color_diff > color_change_threshold or frame_since_last_wave > 15:  # More frequent spawning
        active_waves.append({
            'color': curr_info.color,
            'radius': 0.0
        })
        prev_color = current_color.copy()
        frame_since_last_wave = 0  # Reset counter
    
    # Update all active waves with more dramatic speed variation
    for wave in active_waves[:]:  # Copy list to safely modify
        # Much more dramatic speed variation based on loudness
        dynamic_speed = base_wave_speed * (1.0 + curr_info.loudness * wave_speed_multiplier)
        wave['radius'] += dynamic_speed
        
        # Remove waves that have moved off-screen
        if wave['radius'] > 4.0:  # Larger removal threshold
            active_waves.remove(wave)
    
    # More aggressive emergency spawning - check if center is covered
    center_covered = False
    for wave in active_waves:
        if wave['radius'] >= wave_thickness:  # Wave is large enough to cover center
            center_covered = True
            break
    
    if not center_covered or len(active_waves) == 0:
        active_waves.append({
            'color': curr_info.color,
            'radius': 0.0
        })
    
    # Keep only the most recent waves (performance optimization)
    if len(active_waves) > max_waves:
        active_waves = active_waves[-max_waves:]
    
    # Clear framebuffer
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    
    # Prepare wave data for shader
    wave_colors = []
    wave_radii = []
    
    for wave in active_waves:
        wave_colors.append(wave['color'])  # Add as vec3 (list of 3 floats)
        wave_radii.append(wave['radius'])
    
    # Pad arrays to shader size
    while len(wave_colors) < max_waves:
        wave_colors.append([0.0, 0.0, 0.0])  # Add as vec3, not individual floats
    while len(wave_radii) < max_waves:
        wave_radii.append(0.0)
    
    # Set shader uniforms
    bg_wave_prog['wave_colors'].value = wave_colors[:max_waves]  # Pass vec3 array
    bg_wave_prog['wave_radii'].value = wave_radii[:max_waves]
    bg_wave_prog['num_waves'].value = len(active_waves)
    bg_wave_prog['wave_thickness'].value = wave_thickness
    
    # Use fixed brightness (no flickering)
    bg_wave_prog['brightness'].value = 1.2  # Constant brightness
    
    # Render wave background
    bg_quad_vao.render(moderngl.TRIANGLE_FAN)

    # Determine background speed
    new_bg_speed = curr_info.loudness * BACKGROUND_SPEED
    bg_speed = apply_asymmetric_ema(prev_bg_speed, new_bg_speed, ALPHA_UP_BG_SPEED, ALPHA_DOWN_BG_SPEED)
    prev_bg_speed = bg_speed

    # Determine the size based on loudness
    new_radius = CIRCLE_BASE_SIZE + curr_info.loudness * CIRCLE_SCALE_FACTOR
    radius = apply_asymmetric_ema(prev_radius, new_radius, ALPHA_UP_RADIUS, ALPHA_DOWN_RADIUS)
    prev_radius = radius

    # Determine the rotation 
    rotations_per_frame = curr_info.loudness * RPM / (60 * FPS)
    curr_rotation = curr_rotation + rotations_per_frame * 2 * np.pi

    # Determine the average frequency
    new_avg_freq = curr_info.avg_freq
    avg_freq = apply_asymmetric_ema(prev_avg_freq, new_avg_freq, ALPHA_UP_AVG_FREQ, ALPHA_DOWN_AVG_FREQ)
    prev_avg_freq = avg_freq

    # Determine the number of perturbations
    if USE_FIXED_PERT_NUM:
        pert_num_float = FIXED_PERT_NUM
    else:
        new_pert_num_float = avg_freq * PERTURBATION_MAX_AMOUNT
        pert_num_float = apply_asymmetric_ema(prev_pert_num_float, new_pert_num_float, ALPHA_UP_AVG_FREQ, ALPHA_DOWN_AVG_FREQ)
        prev_pert_num_float = pert_num_float

    # Create and render the shape
    shape_vao = create_shape(radius, avg_freq, curr_rotation, int(pert_num_float), ctx, prog=shape_prog)
    shape_vao.render(moderngl.TRIANGLE_FAN)

    # Read framebuffer and save to video
    pixels = fbo.read(components=3, alignment=1)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    writer.append_data(np.flip(image, axis=0))  # flip Y-axis

writer.close()
render_end = time.time()


# ==== Combine with audio ====
print("Starting FFmpeg...")
ffmpeg_start = time.time()
subprocess.run([
    'ffmpeg',
    '-y',
    '-i', TEMP_VIDEO_FILE,
    '-i', AUDIO_FILE,
    '-c:v', 'copy',
    '-map', '0:v:0',  # Map video from first input
    '-map', '1:a:0',  # Map audio from second input
    '-shortest',
    FINAL_VIDEO_FILE
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
ffmpeg_end = time.time()

os.remove(TEMP_VIDEO_FILE)

total_time = ffmpeg_end - audio_start
print(f"\nTIMING SUMMARY")
print(f"To render a total of {DURATION} seconds of video at {FPS} FPS ({DURATION * FPS} frames):")
print(f" - Audio processing: {audio_end - audio_start:.2f}s ({((audio_end - audio_start) / total_time * 100):.1f}%)")
print(f" - Rendering:        {render_end - render_start:.2f}s ({((render_end - render_start) / total_time * 100):.1f}%)")
print(f" - FFmpeg:           {ffmpeg_end - ffmpeg_start:.2f}s ({((ffmpeg_end - ffmpeg_start) / total_time * 100):.1f}%)")
print(f" - Total:            {total_time:.2f}s")
print(f"\nFinal video with audio saved as {FINAL_VIDEO_FILE}")