import cv2 as cv
import numpy as np
from pathlib import Path

def load_images(folder):
    img_folder = Path(folder)
    sorting = sorted(img_folder.glob("*.jpg"))
    images = [cv.imread(str(file), cv.IMREAD_GRAYSCALE) for file in sorting]
    print(f"Total images loaded: {len(images)}")
    return images

def save_ply(filename, verts, colors=None):
    verts = verts.T
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, v in enumerate(verts):
            if colors is not None:
                c = colors[i]
                f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
            else:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
