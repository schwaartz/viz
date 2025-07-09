import numpy as np
import moderngl
from constants import PORTR_SCALE, HEIGHT, WIDTH, SEGMENTS, PORTR_VARIABILITY

def create_shape(radius: float, avg_freq: float, angle: float, portr_num: int, ctx: moderngl.Context, prog: moderngl.Program) -> moderngl.VertexArray:
    """
    Creates a vertex array object (VAO) representing a circular shape with optional portrusions and rotation.
    :param radius: Base radius of the circle.
    :param avg_freq: Average frequency used to scale protrusions.
    :param angle: Angle in radians to rotate the shape.
    :param portr_num: Number of protrusions to create.
    :param ctx: ModernGL context.
    :param prog: ModernGL shader program.
    :return: A ModernGL VertexArray object containing the circle geometry.
    """
    # Create circle geometry (triangle fan)
    vertices = []
    vertices.append([0.0, 0.0])  # Center point
    for i in range(SEGMENTS + 1):
        theta = 2.0 * np.pi * i / SEGMENTS
        portusion_size = avg_freq ** PORTR_VARIABILITY  # Scale protrusion size based on average frequency
        portrusion = portusion_size * ((np.sin(portr_num * theta + angle) + 1.0) / 2) ** 2 # Taking the square just looks good
        portrusion = portrusion * PORTR_SCALE # Normalize to avoid excessive protrusion
        portrusion = radius + portrusion # The circle is at least the radius size everywhere

        correction_factor = HEIGHT / WIDTH
        x = portrusion * np.cos(theta) * correction_factor # Correction to get a circle, not an ellipse
        y = portrusion * np.sin(theta)
        vertices.append([x, y])

    vertices = np.array(vertices, dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')
    return vao
