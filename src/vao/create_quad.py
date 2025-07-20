import numpy as np
import moderngl

def create_quad_vao(ctx: moderngl.Context, shader_program: moderngl.Program) -> moderngl.VertexArray:
    """
    Create a quad vertex array object for rendering a full-screen quad.
    :param ctx: ModernGL context.
    :param shader_program: Shader program to use for rendering the quad.
    :return: A ModernGL VertexArray object for the quad.
    """
    quad_vertices = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0], 
    ], dtype='f4')
    quad_vbo = ctx.buffer(quad_vertices.tobytes())
    vao = ctx.simple_vertex_array(shader_program, quad_vbo, 'in_pos')
    return vao