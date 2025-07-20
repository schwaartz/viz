#version 330
in vec2 frag_pos;
uniform vec3 wave_colors[32];
uniform float wave_radii[32];
uniform int num_waves;
uniform float wave_thickness;
uniform float brightness;
out vec4 fragColor;

void main() {
    float dist = length(frag_pos);
    vec3 final_color = vec3(0.0, 0.0, 0.0);
    float total_weight = 0.0;
    
    for (int i = 0; i < num_waves && i < 32; i++) {
        float wave_radius = wave_radii[i];
        float distance_from_wave = abs(dist - wave_radius);
        float blend_thickness = wave_thickness * 2.0;  // Increased coverage
        if (distance_from_wave < blend_thickness) {
            // More gradual falloff for better blending
            float weight = 1.0 - smoothstep(0.0, blend_thickness, distance_from_wave);
            final_color += wave_colors[i] * weight;
            total_weight += weight;
        }
    }
    
    // Fallback strategies for center coverage
    if (total_weight <= 0.001 && num_waves > 0) {
        float smallest_radius = 999.0;
        int newest_wave_idx = 0;
        for (int i = 0; i < num_waves && i < 32; i++) {
            float wave_radius = wave_radii[i];
            if (wave_radius < smallest_radius) {
                smallest_radius = wave_radius;
                newest_wave_idx = i;
            }
        }
        final_color = wave_colors[newest_wave_idx] * 0.9;
        total_weight = 1.0;
    }
    
    if (total_weight > 0.0) {
        final_color = (final_color / total_weight) * brightness;
    }
    
    fragColor = vec4(final_color, 1.0);
}