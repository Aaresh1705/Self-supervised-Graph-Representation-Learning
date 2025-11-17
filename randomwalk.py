#!/usr/bin/env python3

# import torch
# import torch_geometric
# import nx
import numpy as np
from PIL import Image, ImageCms

fps = 30
iterations_per_frame = 400
seconds = 60

frames = seconds * fps
iterations = frames * iterations_per_frame

pos = np.asarray((127, 127, 127, 127, 127), dtype=np.uint8)
pic = np.full((255, 255, 3), (0, 0, 0), dtype=np.uint8)  # L=128, a=128, b=128 ≈ neutral gray
allpics = np.zeros((frames, 255, 255, 3), dtype=np.uint8)

maxes = np.asarray([255, 255, 255, 255, 255])

print("simming...")
for i in range(iterations):
    pos = (pos + np.random.choice([-1, 0, 1], (5,))) % maxes
    x, y, r, g, b = pos
    pic[x, y] = [r, g, b]
    if i % iterations_per_frame == 0:
        frame_i = i // iterations_per_frame
        allpics[frame_i] = pic
        print(f"frame {frame_i}/{frames}")
        
print("pilling...")
frames = [Image.fromarray(img) for img in allpics]

print("giffing...")
frames[0].save(
    "animation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=(1.0/fps)*1000, # millis per frame
    loop=0)
