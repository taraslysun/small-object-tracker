import os
from generate_image_sequence import video_to_images
from image_blurrer import batch_gaussian_blur
from image_greyscaler import batch_greyscale


def prepare_video(path_to_video, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    IMAGES_PATH = output_folder + "/images"
    GREYSCALE_PATH = output_folder + "/greyscale_frames"
    BLURRED_PATH = output_folder + "/blurred_frames"
    COMBINED_PATH = output_folder + "/combined_frames"

    video_to_images(path_to_video, IMAGES_PATH)
    batch_greyscale(IMAGES_PATH, GREYSCALE_PATH)
    batch_gaussian_blur(IMAGES_PATH, BLURRED_PATH, blur_radius=2)
    batch_gaussian_blur(GREYSCALE_PATH, COMBINED_PATH, blur_radius=2)

    print(f"Video at {path_to_video} is prepared and saved to {output_folder}.")

list_of_data = [
    ("../data/videos/bees_long.mp4", "../data/images/bees_long"),
    # ("./data/videos/bees_medium.mp4", "./data/images/bees_medium"),
    # ("../data/videos/bees_short.mp4", "../data/images/bees_short"),
    # ("../data/videos/bees1_longer.mp4", "../data/images/bees1_longer"),
    # ("../data/videos/bees2_longer.mp4", "../data/images/bees2_longer"),
    # ("../data/videos/videoplayback.mp4", "../data/images/videoplayback")
]

for pair in list_of_data:
    prepare_video(*pair)