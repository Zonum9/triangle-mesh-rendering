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
Vertices, _, Normals, Tris, _, TN = igl.read_obj(args.file) #read mesh with normals

def drawToPix(x,y,r=1,g=1,b=1):
    pix[x,y,0]=r
    pix[x,y,1]=g
    pix[x,y,2]=b

def projectAndScale(Point3D,scale=1):    
    x,y,z=Point3D
    z=1 #note this makes it orthographic, but can change if necessary
    u=x/z
    v=y/z    
    originX= width/2
    originY=height/2
    worldX=originX+u*scale
    worldY=originY+v*scale
    #clip to fit in the buffer
    return np.clip([worldX,worldY],[0,0],[width-1,height-1]).astype(int)


def drawTri(triangle):
    vertex3d=np.zeros((3,3))
    for i,vIndex in enumerate(triangle):
        vertex3d[i]=Vertices[vIndex]

    scale =500
    p1=projectAndScale(vertex3d[0],scale)    
    p2=projectAndScale(vertex3d[1],scale)    
    p3=projectAndScale(vertex3d[2],scale)

    #these values have already been clipped to be within the buffer 
    boundingBoxMinX= min(p1[0],p2[0],p3[0])
    boundingBoxMinY= min(p1[1],p2[1],p3[1])
    boundingBoxMaxX= max(p1[0],p2[0],p3[0])
    boundingBoxMaxY= max(p1[1],p2[1],p3[1])

    

    if args.test == 1:
        r=random.rand()
        g=random.rand()
        b=random.rand()
        for x in range(boundingBoxMinX,boundingBoxMaxX):
            for y in range(boundingBoxMinY,boundingBoxMaxY):
                drawToPix(x,y,r,g,b)
    elif args.test==2:
        for x in range(boundingBoxMinX,boundingBoxMaxX):
            for y in range(boundingBoxMinY,boundingBoxMaxY):
                alpha,beta,gamma=cartesianToBary2D([x,y],[p2,p3,p1])
                if alpha>0 and beta>0 and gamma>0:
                    z = alpha*vertex3d[0][2]+beta*vertex3d[1][2]+gamma*vertex3d[2][2]
                    if(depth[x,y] < z):
                        depth[x,y]=z
                        drawToPix(x,y,alpha,beta,gamma)
    else:
        #todo interpolate normal vectors
        drawLine(p1,p2)
        drawLine(p2,p3)
        drawLine(p3,p1)


#todo this was taken form wikipedia, try to understand it later
def drawLine(p1,p2):
    x0,y0=p1
    x1,y1=p2
    if abs(y1-y0)<abs(x1-x0):
        if x0>x1:
            plotLineLow(x1,y1,x0,y0)
        else:
            plotLineLow(x0,y0,x1,y1)
    else:
        if y0>y1:
            plotLineHigh(x1,y1,x0,y0)
        else:
            plotLineHigh(x0,y0,x1,y1)

def plotLineLow(x0,y0,x1,y1):
    dx=x1-x0
    dy=y1-y0
    yi=1
    if dy<0:
        yi=-1
        dy=-dy
    D=(2*dy)-dx
    y=y0
    for x in range(x0,x1):
        drawToPix(x,y)
        if D>0:
            y=y+yi
            D=D+(2*(dy-dx))
        else:
            D=D+2*dy

def plotLineHigh(x0,y0,x1,y1):
    dx=x1-x0
    dy=y1-y0
    xi=1
    if dx<0:
        xi=-1
        dx=-dx
    D=(2*dx)-dy
    x=x0
    for y in range(y0,y1):
        drawToPix(x,y)
        if D>0:
            x=x+xi
            D=D+(2*(dx-dy))
        else:
            D=D+2*dx
def cartesianToBary2D(point2D,triangleVertices):
    a, b, c = triangleVertices    
    xa,ya=a
    xb,yb=b
    xc,yc=c
    x, y = point2D        
    # Calculate the barycentric coordinates
    triangleArea = (yb-yc)*(xa-xc)+(xc-xb)*(ya-yc)    
    if triangleArea==0:
        return 0,0,0
    alpha=((yb-yc)*(x-xc)+(xc-xb)*(y-yc))/triangleArea
    beta=((y-yc)*(xa-xc)+(xc-x)*(ya-yc))/triangleArea
    gamma=1-alpha-beta    
    return alpha, beta, gamma

def triangleNormal(triangleVertices3D):
    a,b,c =triangleVertices3D
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    n = np.cross(np.subtract(b,a),np.subtract(c,a))
    return n


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
    depth.fill(-math.inf) # clear depth buffer
    #time varying transformation
    cos,sin = math.cos(1.2*t),math.sin(1.2*t)
    Rotation_y = np.array([[ cos, 0, sin],[ 0, 1, 0],[-sin, 0, cos]])
    cos,sin = math.cos(t),math.sin(t)
    Rotation_x = np.array([[ 1, 0, 0],[ 0, cos, sin],[ 0,-sin, cos]])
    cos,sin = math.cos(1.8*t),math.sin(1.8*t)
    Rotation_z = np.array([[ cos, sin, 0],[-sin, cos, 0],[ 0, 0, 1]])
    Vt = (scale @ Rotation_y @ Rotation_x @ Rotation_z @ Vertices.T).T
    Vt = Vt + translate
    Nt = (Rotation_y @ Rotation_x @ Rotation_z @ Normals.T).T

    for triangle in Tris:
        drawTri(triangle)
    # draw!
    pixti.from_numpy(pix)
    copy_pixels()
    gui.set_image(pixels)
    gui.show()
    t += 0.001