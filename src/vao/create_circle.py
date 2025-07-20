import numpy as np
import moderngl
from config import VisualConfig

def create_circle(ctx: moderngl.Context, prog: moderngl.Program, config: VisualConfig) -> moderngl.VertexArray:
    """
    Creates a vertex array object (VAO) representing a circle.
    :param ctx: ModernGL context.
    :param prog: ModernGL shader program.
    :param config: VisualConfig object with settings.
    :return: A ModernGL VertexArray object containing the circle geometry.
    """
    vertices = []
    vertices.append([0.0, 0.0])  # Center point
    for i in range(config.shape_vertices + 1):
        theta = 2.0 * np.pi * i / config.shape_vertices
        r = config.circle_base_size
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        vertices.append([x, y])

    vertices = np.array(vertices, dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_pos')
    return vao