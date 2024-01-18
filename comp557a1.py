
# todo add the thigns at the top
import math
import igl
import numpy as np
import numpy.random as random
import taichi as ti
import taichi.math as tm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="cube.obj")
parser.add_argument("--width", type=int, default=1440, help="Width of off screen framebuffer")
parser.add_argument("--height", type=int, default=720, help="Height of off screen framebuffer")
parser.add_argument("--px", type=int, default=10, help="Size of pixel in on screen framebuffer")
parser.add_argument("--test", type=int, help="run a numbered unit test")
parser.add_argument("--timeStep", type=float, default=0.01, help="time skip between frames (makes animations faster or slower)")
parser.add_argument("--rotateFalse", action='store_true', help="setting this flag disable the mesh rotation")
parser.add_argument("--scale", type=int,default=200, help="scale at which the mesh will be rendered")
parser.add_argument("--nonSmoothNormals", action='store_true', help="setting this will make the model look low poly")

args = parser.parse_args()
ti.init(arch=ti.cpu) # can also use ti.gpu
px = args.px # Size of pixel in on screen framebuffer
width, height = args.width//px, args.height//px # Size of off screen framebuffer
pix = np.zeros((width, height, 3), dtype=np.float32)
RotateFalse = args.rotateFalse
TimeStep=args.timeStep
Depth = np.zeros((width, height, 1), dtype=np.float32)
pixti = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width*px, height*px))
Vertices, _, Normals, Tris, _, TN = igl.read_obj(args.file) #read mesh with normals
argsScale=args.scale

def drawToPix(x,y,r=1,g=1,b=1):
    pix[x,y,0]=r
    pix[x,y,1]=g
    pix[x,y,2]=b

def project(Point3D):   
    x,y,_=Point3D
    return x,y

#method used to render without scale matrix transformation
def projectAndScale(Point3D,scale): 
    x,y,z=Point3D
    originX= width/2
    originY=height/2
    worldX=originX+x*scale
    worldY=originY+y*scale
    return worldX,worldY

#lots of repeated code, but wanted to avoid having if conditions inside of the for loops
def drawTri(triangle,normal,rotateDisabled,scale):
    scale/=px
    vertices3D=np.zeros((3,3))
    normalPoints=np.zeros((3,3))
    if rotateDisabled:
        for i,vIndex in enumerate(triangle):
            vertices3D[i]=Vertices[vIndex]
        for i,nIndex in enumerate(normal):
            normalPoints[i]=Normals[nIndex]
        p1=projectAndScale(vertices3D[0],scale)    
        p2=projectAndScale(vertices3D[1],scale)    
        p3=projectAndScale(vertices3D[2],scale)
    else:
        for i,vIndex in enumerate(triangle):
            vertices3D[i]=VerticesTransformed[vIndex]
        for i,nIndex in enumerate(normal):
            normalPoints[i]=NormalsTransformed[nIndex]
        p1=project(vertices3D[0])
        p2=project(vertices3D[1])
        p3=project(vertices3D[2]) 
    
    boundingBoxMinX= np.clip(np.floor(min(p1[0],p2[0],p3[0])),0,width).astype(int)
    boundingBoxMinY= np.clip(np.floor(min(p1[1],p2[1],p3[1])),0,height).astype(int)
    boundingBoxMaxX= np.clip(np.ceil(max(p1[0],p2[0],p3[0])),0,width).astype(int)
    boundingBoxMaxY= np.clip(np.ceil(max(p1[1],p2[1],p3[1])),0,height).astype(int)
    

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
                alpha,beta,gamma=cartesianToBary([x,y],[p1,p2,p3])
                if alpha>=0 and beta>=0 and gamma>=0:
                    z = alpha*vertices3D[0][2]+beta*vertices3D[1][2]+gamma*vertices3D[2][2]
                    if(Depth[x,y] < z):
                        Depth[x,y]=z
                        drawToPix(x,y,alpha,beta,gamma)
    else:
        for x in range(boundingBoxMinX,boundingBoxMaxX):
            for y in range(boundingBoxMinY,boundingBoxMaxY):
                alpha,beta,gamma=cartesianToBary([x,y],[p1,p2,p3])
                if alpha>=0 and beta>=0 and gamma>=0:
                    z = alpha*vertices3D[0][2]+beta*vertices3D[1][2]+gamma*vertices3D[2][2]                    
                    if(Depth[x,y] < z):
                        Depth[x,y]=z
                        normalZ = alpha*normalPoints[0][2]+beta*normalPoints[1][2]+gamma*normalPoints[2][2]
                        if normalZ >= 0:
                            drawToPix(x,y,normalZ,normalZ,normalZ)
                        else:
                            drawToPix(x,y,0,0,0)

#transforms 2d point into bary cords for a give triangle                                           
def cartesianToBary(point2D,triangleVertices):
    a, b, c = triangleVertices    
    xa,ya=a
    xb,yb=b
    xc,yc=c
    x, y = point2D        
    triangleArea = (yb-yc)*(xa-xc)+(xc-xb)*(ya-yc)    
    if triangleArea==0:
        return 0,0,0
    alpha=((yb-yc)*(x-xc)+(xc-xb)*(y-yc))/triangleArea
    beta=((y-yc)*(xa-xc)+(xc-x)*(ya-yc))/triangleArea
    gamma=1-alpha-beta    
    return alpha, beta, gamma

#retunrs non normalized normals
def triangleNormal(triangle):
    triangleVertices3D= np.zeros((3,3))
    for i,vIndex in enumerate(triangle):
        triangleVertices3D[i]=Vertices[vIndex]
    a,b,c =triangleVertices3D
    n = np.cross(b-a,c-a)
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
scale = argsScale/px*np.eye(3) # scale to fit in the window

if args.nonSmoothNormals:
    Normals= np.empty(shape=(0,3))
    TN = np.zeros(Tris.shape).astype(int)
    for faceIndex,triangle in enumerate(Tris):        
        n=triangleNormal(triangle)    
        Normals=np.append(Normals,[n/np.linalg.norm(n)],axis=0)
        for i in range(3):
            TN[faceIndex,i]=len(Normals)-1

elif Normals.shape[0]==0:
    Normals= np.zeros((len(Vertices),3))
    TN = np.zeros(Tris.shape).astype(int)
    for faceIndex,triangle in enumerate(Tris):        
        n=triangleNormal(triangle)    
        for i,vertexIndex in enumerate(triangle):
            Normals[vertexIndex]+=n
            TN[faceIndex,i]=vertexIndex
    for i,n in enumerate(Normals):
        Normals[i]=n/np.linalg.norm(n)

while gui.running:
    pix.fill(0) # clear pixel buffer
    Depth.fill(-math.inf) # clear depth buffer    
    #time varying transformation
    cos,sin = math.cos(1.2*t),math.sin(1.2*t)
    Rotation_y = np.array([[ cos, 0, sin],[ 0, 1, 0],[-sin, 0, cos]])
    cos,sin = math.cos(t),math.sin(t)
    Rotation_x = np.array([[ 1, 0, 0],[ 0, cos, sin],[ 0,-sin, cos]])
    cos,sin = math.cos(1.8*t),math.sin(1.8*t)
    Rotation_z = np.array([[ cos, sin, 0],[-sin, cos, 0],[ 0, 0, 1]])
    VerticesTransformed = (scale @ Rotation_y @ Rotation_x @ Rotation_z @ Vertices.T).T
    VerticesTransformed = VerticesTransformed + translate
    NormalsTransformed = (Rotation_y @ Rotation_x @ Rotation_z @ Normals.T).T

    for triangle,normal in zip(Tris,TN):
        drawTri(triangle,normal,RotateFalse,argsScale)
    # draw!
    pixti.from_numpy(pix)
    copy_pixels()
    gui.set_image(pixels)
    gui.show()
    t += TimeStep