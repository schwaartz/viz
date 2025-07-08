import numpy as np
import moderngl
from constants import PORTRUSION_SCALE, HEIGHT, WIDTH, SEGMENTS, PORTRUSION_VARIABILITY

def create_shape(radius: float, avg_freq: float, angle: float, pert_num: int, ctx: moderngl.Context, prog: moderngl.Program) -> moderngl.VertexArray:
    """
    Creates a vertex array object (VAO) representing a circular shape with optional perturbations and rotation.
    Parameters:
        avg_freq (float): Determines the spikiness or frequency-based deformation of the shape.
        angle (float): The rotation angle (in radians) to apply to the shape.
        pert_num (int): The number of perturbations to apply to the outer edge of the shape.
        ctx (moderngl.Context): The ModernGL context used to create buffers and VAOs.
        prog (moderngl.Program): The shader program to use for rendering the shape.
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
        portusion_size = avg_freq ** PORTRUSION_VARIABILITY  # Scale protrusion size based on average frequency
        portrusion = portusion_size * ((np.sin(pert_num * theta + angle) + 1.0) / 2) ** 2 # Taking the square just looks good
        portrusion = portrusion * PORTRUSION_SCALE # Normalize to avoid excessive protrusion
        portrusion = radius + portrusion # The circle is at least the radius size everywhere

        correction_factor = HEIGHT / WIDTH
        x = portrusion * np.cos(theta) * correction_factor # Correction to get a circle, not an ellipse
        y = portrusion * np.sin(theta)
        vertices.append([x, y])

    vertices = np.array(vertices, dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')
    return vao
