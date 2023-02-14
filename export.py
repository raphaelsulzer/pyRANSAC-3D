
import os
import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt

class Exporter:

    def __init__(self):
        pass


    def export_deleted_points(self,path,points,count,subfolder="deleted_points",color=None):

        c = color if color is not None else [0.5,0.5,0.5]


        """this writes one group of points"""

        path = os.path.join(path,subfolder)
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path, str(count) + '.obj')
        f = open(filename, 'w')
        for i,v in enumerate(points):
            f.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], c[0], c[1], c[2]))
        f.close()


    def export_plane(self,path,plane,points,count,subpaths=["planes","point_groups"],color=None):


        c = color if color is not None else [0.5,0.5,0.5]


        os.makedirs(os.path.join(path,subpaths[1]), exist_ok=True)
        filename = os.path.join(path, subpaths[1], str(count) + '.obj')
        f = open(filename, 'w')
        for j, v in enumerate(points):
            f.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], c[0], c[1], c[2]))
        f.close()


        ## project verts to plane
        ## https://www.baeldung.com/cs/3d-point-2d-plane
        k = (-plane[-1] - plane[0] * points[:, 0] - plane[1] * points[:, 1] - plane[2] * points[:, 2]) / \
            (plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
        pp = np.asarray([points[:, 0] + k * plane[0], points[:, 1] + k * plane[1], points[:, 2] + k * plane[2]]).transpose()

        # plt.figure()
        # plt.scatter(pp[:,0],pp[:,1])
        # plt.axis('equal')
        # plt.show()

        ch = ConvexHull(pp[:, :2])
        verts = ch.points[ch.vertices]
        verts = np.hstack((verts, pp[ch.vertices, 2, np.newaxis]))

        os.makedirs(os.path.join(path,subpaths[0]), exist_ok=True)
        filename = os.path.join(path, subpaths[0], str(count) + '.obj')
        f = open(filename, 'w')
        fstring = 'f'
        for j, v in enumerate(verts):
            f.write('v {} {} {} {} {} {}\n'.format(v[0], v[1], v[2], c[0], c[1], c[2]))
            fstring += ' {}'.format(j + 1)
        f.write(fstring)

        f.close()


    def export_planes(self,path,planes,points,color=None):

        """this writes all planes"""


        os.makedirs(os.path.join(path,"planes"), exist_ok=True)
        os.makedirs(os.path.join(path,"point_groups"), exist_ok=True)

        for i, plane in enumerate(planes):
            c = np.random.random(size=3)

            filename = os.path.join(path, "point_groups", str(i) + '.obj')
            f = open(filename, 'w')
            fstring='f'
            for j,v in enumerate(points[i]):
                f.write('v {} {} {} {} {} {}\n'.format(v[0],v[1],v[2],c[0],c[1],c[2]))
            f.close()

            p = points[i]

            # project verts to plane
            # https://www.baeldung.com/cs/3d-point-2d-plane
            k = (-plane[-1] -plane[0]*p[:, 0] -plane[1]*p[:, 1] -plane[2]*p[:, 2]) / \
                (plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
            pp = np.asarray([p[:, 0] + k * plane[0], p[:, 1] + k * plane[1], p[:, 2] + k * plane[2]]).transpose()


            ch = ConvexHull(pp[:,:2])
            verts = ch.points[ch.vertices]
            verts = np.hstack((verts, pp[ch.vertices,2,np.newaxis]))




            filename = os.path.join(path, "planes", str(i) + '.obj')
            f = open(filename, 'w')
            fstring='f'
            for j,v in enumerate(verts):
                f.write('v {} {} {} {} {} {}\n'.format(v[0],v[1],v[2],c[0],c[1],c[2]))
                fstring+=' {}'.format(j+1)
            f.write(fstring)

            f.close()
