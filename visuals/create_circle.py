import numpy as np
import moderngl
from params import PROTR_SCALE, HEIGHT, WIDTH, VERTECES, PROTR_VARIABILITY

def create_circle(radius: float, ctx: moderngl.Context, prog: moderngl.Program) -> moderngl.VertexArray:
    """
    Creates a vertex array object (VAO) representing a circle.
    :param radius: Radius of the circle.
    :param ctx: ModernGL context.
    :param prog: ModernGL shader program.
    :return: A ModernGL VertexArray object containing the circle geometry.
    """
    vertices = []
    vertices.append([0.0, 0.0])  # Center point
    for i in range(VERTECES + 1):
        theta = 2.0 * np.pi * i / VERTECES
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        vertices.append([x, y])

    vertices = np.array(vertices, dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')
    return vao