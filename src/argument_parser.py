import argparse

def parse_arguments():
    """
    Parses command line arguments for the audio visualizer.
    :return: Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description='Audio Visualizer - Generate stunning visuals from audio')
    parser.add_argument('input_audio', 
                       help='Input audio file (mp3, wav, flac, etc.)')
    parser.add_argument('-o', '--output', 
                       help='Output video file (default: input filename with .mp4 extension)')
    parser.add_argument('-c', '--config', 
                       default='config.json',
                       help='Configuration file (default: config.json)')
    
    return parser.parse_args()