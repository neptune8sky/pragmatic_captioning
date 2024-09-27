import matplotlib.pyplot as plt
import os
from PIL import Image
import textwrap
from utils.io import from_yaml


def create_collage(folder_name, images_data):
    """
    Create a collage of images with pragmatic captions.

    Args:
        folder_name (str): Name of the folder containing images.
        images_data (dict): Dictionary containing image paths and captions.

    Returns:
        matplotlib.figure.Figure: The created figure object.
    """
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    fig.suptitle(folder_name, fontsize=14)

    for idx, (image_path, data) in enumerate(images_data.items()):
        img = Image.open(image_path)
        axs[idx].imshow(img)
        axs[idx].axis('off')

        caption = data.get('PragCaption', 'No caption available')
        wrapped_caption = textwrap.wrap(caption, width=30)
        axs[idx].text(0.5, -0.1, '\n'.join(wrapped_caption),
                      ha='center', va='center', transform=axs[idx].transAxes,
                      fontsize=8, wrap=True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    return fig


def main():
    """
    Main function to process YAML data and create collages.
    """
    yaml_file = 'out/pragmatic_dataset.yaml'
    output_folder = 'out/pragmatic_plots'

    os.makedirs(output_folder, exist_ok=True)

    data = from_yaml(yaml_file)

    for folder_name, images_data in data.items():
        fig = create_collage(folder_name, images_data)

        output_path = os.path.join(output_folder, f"{folder_name}.png")
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"Saved collage for {folder_name} to {output_path}")


if __name__ == "__main__":
    main()
