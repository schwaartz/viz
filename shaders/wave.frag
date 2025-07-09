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
        float distance_from_wave = abs(dist - wave_radius);
        float blend_thickness = wave_thickness * 1.5; // Increase thickness for better blending
        if (distance_from_wave < blend_thickness) {
            float weight = 1.0 - pow(distance_from_wave / blend_thickness, 1.5);  // Gentler falloff
            final_color += wave_colors[i] * weight;
            total_weight += weight;
        }
    }
    
    // Fallback color if no waves cover this pixel
    if (total_weight <= 0.0 && num_waves > 0) {
        final_color = wave_colors[0] * 0.3;  // Use first wave color as fallback
    } else if (total_weight > 0.0) {
        final_color = brightness * final_color / total_weight;
    }
    
    fragColor = vec4(final_color, 1.0);
}