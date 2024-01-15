import math
import igl
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="cube.obj")
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





@ti.kernel
# copy pixels from small framebuffer to large framebuffer
def copy_pixels():
    for i, j in pixels:
        if px<2 or (tm.mod(i,px)!=0 and tm.mod(j,px)!=0):
            pixels[i,j] = pixti[i//px,j//px]
gui = ti.GUI("Rasterizer", res=(width*px, height*px))
t = 0 # time step for time varying transformaitons
translate = np.array([ width/2,height/2,0 ]) # translate to center of window
scale = 200/px*np.eye(3) # scale to fit in the window
while gui.running:
    pix.fill(0) # clear pixel buffer
    for i in range(width):
        if i%2==0:
            j=min(i,719)
            # for j in range(height):
            pix[i][j][0]=99;
            pix[i][j][1]=99;
            pix[i][j][2]=99;
    
    depth.fill(-math.inf) # clear depth buffer
    #time varying transformation
    c,s = math.cos(1.2*t),math.sin(1.2*t)
    Ry = np.array([[ c, 0, s],[ 0, 1, 0],[-s, 0, c]])
    c,s = math.cos(t),math.sin(t)
    Rx = np.array([[ 1, 0, 0],[ 0, c, s],[ 0,-s, c]])
    c,s = math.cos(1.8*t),math.sin(1.8*t)
    Rz = np.array([[ c, s, 0],[-s, c, 0],[ 0, 0, 1]])
    Vt = (scale @ Ry @ Rx @ Rz @ V.T).T
    Vt = Vt + translate
    Nt = (Ry @ Rx @ Rz @ N.T).T
    # draw!
    pixti.from_numpy(pix)
    copy_pixels()
    gui.set_image(pixels)
    gui.show()
    t += 0.001