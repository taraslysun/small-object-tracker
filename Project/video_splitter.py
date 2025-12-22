from moviepy.editor import VideoFileClip

def split_video(input_path, segments):
    """
    input_path: Path to your video file (e.g., 'video.mp4')
    segments: List of tuples [(start_sec, end_sec), ...]
    """
    video = VideoFileClip(input_path)
    
    for i, (start, end) in enumerate(segments):
        print(f"Processing segment {i+1}: {start}s to {end}s...")
        new_clip = video.subclip(start, end)
        output_filename = f"part_{i+1}_{start}_to_{end}.mp4"
        new_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")
        
    video.close()

my_segments = [(1,11)]
split_video("./data/city_longer.mov", my_segments)