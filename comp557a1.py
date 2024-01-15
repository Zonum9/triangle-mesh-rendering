import math
import igl
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="bunnyLowRes.obj")
parser.add_argument("--width", type=int, default=1440, help="Width of off screen framebuffer")
parser.add_argument("--height", type=int, default=720, help="Height of off screen framebuffer")
parser.add_argument("--px", type=int, default=1, help="Size of pixel in on screen framebuffer")
parser.add_argument("--test", type=int, help="run a numbered unit test")
args = parser.parse_args()
ti.init(arch=ti.cpu) # can also use ti.gpu
px = args.px # Size of pixel in on screen framebuffer
width, height = args.width//px, args.height//px # Size of off screen framebuffer
pix = np.zeros((width, height, 3), dtype=np.float32)
depth = np.zeros((width, height, 1), dtype=np.float32)
pixti = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width*px, height*px))
V, _, N, T, _, TN = igl.read_obj(args.file) #read mesh with normals