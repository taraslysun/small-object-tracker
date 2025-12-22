import os
from PIL import Image, ImageFilter

def batch_gaussian_blur(input_folder, output_folder, blur_radius=2):
    """
    input_folder: Folder containing your images
    output_folder: Where to save blurred images
    blur_radius: Intensity of the blur (default is 2)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        save_path = os.path.join(output_folder, filename)

        with Image.open(img_path) as img:
            blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_img.save(save_path)
                
        print(f"Blurred: {filename}")

    print(f"\nSuccess! All images blurred with radius {blur_radius}.")
