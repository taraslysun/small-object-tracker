import os
from PIL import Image

def batch_greyscale(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        save_path = os.path.join(output_folder, filename)

        with Image.open(img_path) as img:
            grayscale_img = img.convert('L')
            grayscale_img.save(save_path)
                
        print(f"Processed: {filename}")

    print("\nBatch processing complete.")
