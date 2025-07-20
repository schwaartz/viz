import numpy as np
import moderngl

def create_quad(ctx: moderngl.Context, shader_program: moderngl.Program) -> moderngl.VertexArray:
    """
    Create a quad vertex array object for rendering a full-screen quad.
    :param ctx: ModernGL context.
    :param shader_program: Shader program to use for rendering the quad.
    :return: A ModernGL VertexArray object for the quad.
    """
    bg_quad_vertices = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0], 
    ], dtype='f4')
    bg_quad_vbo = ctx.buffer(bg_quad_vertices.tobytes())
    return ctx.simple_vertex_array(shader_program, bg_quad_vbo, 'in_pos')