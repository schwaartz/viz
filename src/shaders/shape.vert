#version 330
in vec2 in_pos;
uniform float avg_freq; // Average frequency of the sound
uniform float protr_amount; // Amount of protrusions
uniform float protr_scale; // Scaler for protrusions
uniform float protr_base_thickness; // Base thickness of protrusions
uniform float protr_thickness_factor; // Factor to scale protrusion thickness
uniform float height_width_ratio; // Height to width ratio
uniform float rotation; // Angle of rotation
uniform float radius_scale; // Scale factor for the radius 
uniform float protr_variability; // Variability factor for protrusion lengths

void main() {
    float x = in_pos.x * (1.0 + radius_scale);
    float y = in_pos.y * (1.0 + radius_scale);
    if (in_pos.x == 0.0 && in_pos.y == 0.0) {
        gl_Position = vec4(x * height_width_ratio, y, 0.0, 1.0); // Central point is excluded
        return;
    }
    float theta = atan(y, x);
    float radius = length(vec2(x, y)); 
    float total_protr = 0.0;
    if (protr_amount > 0.0) {
        float protr_size = protr_scale * pow(avg_freq, protr_variability);
        float power = protr_base_thickness + protr_thickness_factor * avg_freq;
        float protr = protr_size * pow((sin(protr_amount * theta + rotation) + 1.0) / 2.0, power);
        total_protr += protr;
    }
    x = (radius + total_protr) * cos(theta);
    y = (radius + total_protr) * sin(theta);
    gl_Position = vec4(x * height_width_ratio, y, 0.0, 1.0);
}