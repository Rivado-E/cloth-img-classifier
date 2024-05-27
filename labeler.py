import os
import subprocess
import json


def display_image_with_feh(image_path):
    feh_process = subprocess.Popen(['feh', image_path])
    return feh_process


def label_images_in_folder(folder_path):
    images = [img for img in os.listdir(
        folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

    labels = {}
    i = 0

    for img in images:
        image_path = os.path.join(folder_path, img)

        feh_process = display_image_with_feh(image_path)

        label = input(f"\nLabel for {img} (0 or 1): ")
        print(label)
        if label == "":
            label = "0"

        feh_process.terminate()

        labels[image_path] = label
        i += 1
        if i == 10:
            return labels

    return labels


def save_labels_to_json(labels, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(labels, json_file, indent=4)


folder_path = '/home/rivaldo/work/folded/data/0/left/rgb'

json_output_path = f'{folder_path}/labels.json'

labels = label_images_in_folder(folder_path)

save_labels_to_json(labels, json_output_path)

print(f"Labels saved to {json_output_path}")
