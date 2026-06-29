from pathlib import Path
import moderngl


_SHADER_ROOT = Path(__file__).resolve().parents[2]

def load_shader(filepath: str) -> str:
    """
    Load shader source code from a file.
    :param filepath: Path to the shader file.
    :return: Shader source code as a string.
    """
    path = Path(filepath)
    if not path.is_absolute():
        # Resolve bundled shader files relative to the package, not the cwd.
        path = _SHADER_ROOT / path

    with open(path, 'r') as file:
        return file.read()

def load_shader_program(ctx: moderngl.Context, vertex_path: str, fragment_path: str) -> moderngl.Program:
    """
    Load and compile a shader program from files.
    :param ctx: ModernGL context.
    :param vertex_path: Path to the vertex shader file.
    :param fragment_path: Path to the fragment shader file.
    :return: A ModernGL Program object.
    """
    vertex_shader = load_shader(vertex_path)
    fragment_shader = load_shader(fragment_path)
    return ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )