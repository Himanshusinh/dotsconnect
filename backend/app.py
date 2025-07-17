from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.spatial import distance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_ordered_path(coords):
    visited = set()
    ordered = []

    current = coords[0]
    visited.add(tuple(current))
    ordered.append(current)

    for _ in range(len(coords) - 1):
        dists = distance.cdist([current], coords)[0]
        dists = np.ma.masked_array(dists, mask=[tuple(c) in visited for c in coords])
        nearest_idx = np.argmin(dists)
        current = coords[nearest_idx]
        visited.add(tuple(current))
        ordered.append(current)

    return np.array(ordered)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), dots: int = Form(...)):
    temp = NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    img = cv2.imread(temp.name, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    skeleton = skeletonize(binary // 255).astype(np.uint8)
    y_coords, x_coords = np.where(skeleton > 0)
    coords = np.column_stack((x_coords, y_coords))

    ordered_path = calculate_ordered_path(coords)
    distances = np.cumsum(np.sqrt(np.sum(np.diff(ordered_path, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)

    target_distances = np.linspace(0, distances[-1], dots)
    sampled_points = []
    for td in target_distances:
        idx = np.searchsorted(distances, td)
        sampled_points.append(ordered_path[min(idx, len(ordered_path) - 1)])

    sampled_points = np.array(sampled_points)
    x_sampled, y_sampled = sampled_points[:, 0], sampled_points[:, 1]

    # Dot size doubled (was 2, now 4)
    dot_size = 16

    output_path = "output.svg"
    plt.figure(figsize=(10, 10), facecolor="black")  # Set background black
    plt.scatter(x_sampled, y_sampled, s=dot_size, color="white")  # White dots
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, format="svg", facecolor="black")  # Save with black background
    plt.close()

    return FileResponse(output_path, media_type="image/svg+xml", filename="dotted_output.svg")