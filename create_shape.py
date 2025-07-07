import numpy as np
import moderngl

SEGMENTS = 128

def create_shape(freq_center_of_gravity: float, angle: float, perturbations: int, ctx: moderngl.Context, prog: moderngl.Program, correction_factor: float) -> moderngl.VertexArray:
    """
    Creates a vertex array object (VAO) representing a circular shape with optional perturbations and rotation.
    Parameters:
        freq_center_of_gravity (float): Determines the spikiness or frequency-based deformation of the shape.
        angle (float): The rotation angle (in radians) to apply to the shape.
        perturbations (int): The number of perturbations to apply to the outer edge of the shape.
        ctx (moderngl.Context): The ModernGL context used to create buffers and VAOs.
        prog (moderngl.Program): The shader program to use for rendering the shape.
        correction_factor (float): A factor to correct the aspect ratio of the shape, typically HEIGHT / WIDTH.
    Returns:
        moderngl.VertexArray: A vertex array object representing the generated shape.
    Notes:
        - The shape is constructed as a triangle fan, starting from the center.
        - The circle geometry is corrected for aspect ratio using HEIGHT and WIDTH.
        - The function assumes the existence of global variables SEGMENTS, HEIGHT, WIDTH, and circle_prog.
    """
    # Create circle geometry (triangle fan)
    vertices = []
    vertices.append([0.0, 0.0])  # Center point
    for i in range(SEGMENTS + 1):
        theta = 2.0 * np.pi * i / SEGMENTS
        portrusion = np.sin(perturbations * theta + angle) * freq_center_of_gravity + 2.0
        portrusion = portrusion / 3 # Normalize to avoid excessive protrusion
        x = portrusion * np.cos(theta) * correction_factor # Correction to get a circle, not an ellipse
        y = portrusion * np.sin(theta)
        vertices.append([x, y])

    vertices = np.array(vertices, dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')
    return vao
