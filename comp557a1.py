import math
import igl
import numpy as np
import numpy.random as random
import taichi as ti
import taichi.math as tm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="bunny.obj")
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




# exit()

def project(Point3d):
    x,y,z=Point3d
    x+=.5
    y+=.5
    v=y/z
    u=x/z
    return [u,v]

def toWorldCoordinates(objectCoordinates2D,scale=1):
    oX,oY=objectCoordinates2D
    originX= width/2
    originY=height/2
    worldX=originX+oX*scale
    worldY=originY+oY*scale
    return np.clip([worldX,worldY],[0,0],[width-1,height-1])

def drawTri(triangle):
    for vIndex in triangle:
        x,y=V2dWorldCoords[vIndex]
        pix[x,y,0]=0
        pix[x,y,1]=255
        pix[x,y,2]=0

    
    p1=V2dWorldCoords[triangle[0]]
    p2=V2dWorldCoords[triangle[1]]
    p3=V2dWorldCoords[triangle[2]]

    boundingBoxMinX= min(p1[0],p2[0],p3[0])
    boundingBoxMinY= min(p1[1],p2[1],p3[1])
    boundingBoxMaxX= max(p1[0],p2[0],p3[0])
    boundingBoxMaxY= max(p1[1],p2[1],p3[1])

    if args.test == 1:
        r=random.rand()
        g=random.rand()
        b=random.rand()
        for i in range(boundingBoxMinX,boundingBoxMaxX):
            for j in range(boundingBoxMinY,boundingBoxMaxY):
                pix[i,j,0]=r
                pix[i,j,1]=g
                pix[i,j,2]=b
    else:
        for i in range(boundingBoxMinX,boundingBoxMaxX):
            for j in range(boundingBoxMinY,boundingBoxMaxY):
                alpha,beta,gamma=cartesianToBary([i,j],[p2,p3,p1])
                if alpha>0 and beta>0 and gamma>0:
                    pix[i,j,0]=alpha
                    pix[i,j,1]=beta
                    pix[i,j,2]=gamma
    # drawLine(p1,p2)
    # drawLine(p2,p3)
    # drawLine(p3,p1)

def cartesianToBary(point,triangleVertices):
    a, b, c = triangleVertices    
    x1,y1=a
    x2,y2=b
    x3,y3=c
    x, y = point    
    # Calculate the barycentric coordinates
    det = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)    
    if det==0:
        return 1,1,1
    alpha=((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/det
    beta=((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/det
    gamma=1-alpha-beta    
    return alpha, beta, gamma


V2d= np.array([project(p) for p in V])
V2dWorldCoords=np.array([toWorldCoordinates(p,scale=10) for p in V2d], dtype=int)


# exit()

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
    # for x,y in V2dWorldCoords:
    #     pix[x,y,0]=255
    #     pix[x,y,1]=255
    #     pix[x,y,2]=255
    for triangle in T:
        drawTri(triangle)



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