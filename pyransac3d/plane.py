import copy
import random

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from skimage import measure
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.spatial.transform import Rotation as R

class Plane:
    """
    Implementation of planar RANSAC.

    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.

    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.

    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")

    ---
    """

    def __init__(self,count=0):
        self.pt_inliers_id = []
        self.occ_inliers_id = []
        self.equation = []
        self.occ_score = 0
        self.counter = count


    # def optimize_plane(self,points):
    #
    #     # from here:https://stackoverflow.com/questions/35118419/wrong-result-for-best-fit-plane-to-set-of-points-with-scipy-linalg-lstsq
    #
    #     A = np.c_[points[:,0], points[:,1], np.ones(points.shape[0])]
    #     C, _, _, _ = lstsq(A, points[:,2],overwrite_a=False,overwrite_b=False,check_finite=True,lapack_driver='gelsy')
    #
    #     # return np.array((C[0], C[1], -1., C[2]))
    #     return [C[0], C[1], -1., C[2]]

    def optimize_plane(self, points, plane):
        # from here:https://stackoverflow.com/questions/35118419/wrong-result-for-best-fit-plane-to-set-of-points-with-scipy-linalg-lstsq
        def model(params, xyz):
            a, b, c, d = params
            x, y, z = xyz
            length_squared = a ** 2 + b ** 2 + c ** 2
            return ((a * x + b * y + c * z + d) ** 2 / length_squared).sum()

        def unit_length(params):
            a, b, c, d = params
            return a ** 2 + b ** 2 + c ** 2 - 1

        # x,y,z = points[:,0],points[:,1],points[:,2]
        # constrain the vector perpendicular to the plane be of unit length
        cons = ({'type': 'eq', 'fun': unit_length})
        sol = minimize(model, plane, args=[points[:,0],points[:,1],points[:,2]], constraints=cons)
        return tuple(sol.x)

    def project_points_to_plane(self,points,plane):

        ### project inlier points to plane
        ## https://www.baeldung.com/cs/3d-point-2d-plane
        k = (-plane[-1] - plane[0] * points[:, 0] - plane[1] * points[:, 1] - plane[2] * points[:, 2]) / \
            (plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
        pp = np.asarray([points[:, 0] + k * plane[0], points[:, 1] + k * plane[1], points[:, 2] + k * plane[2]])
        ## make e1 and e2 (see bottom of page linked above)
        ## take a starting vector (e0) and take a component of this vector which is nonzero (see here: https://stackoverflow.com/a/33758795)
        z = np.argmax(np.abs(plane[:3]))
        y = (z+1)%3
        x = (y+1)%3
        e0 = np.array(plane[:3])
        e0 = e0/np.linalg.norm(e0)
        e1 = np.zeros(3)
        ## reverse the non-zero component and put if on a different axis
        e1[x], e1[y], e1[z] = e0[x], -e0[z], e0[y]
        ## take the cross product of e0 and e1 to make e2
        e2 = np.cross(e0,e1)
        e12 = np.array([e1,e2])
        return (e12@pp).transpose()



    def plot_inliers(self,points,plane):

        pp = self.project_points_to_plane(points,plane)

        plt.figure()
        plt.scatter(pp[:,0],pp[:,1])
        plt.axis('equal')
        plt.show()


    def segment(self, im):

        ## see example usage here: https://scipy-lectures.org/packages/scikit-image/auto_examples/plot_labels.html
        labels, num_labels = measure.label(im, background=0, return_num=True)

        self.plt = plt.figure(figsize=(9, 3.5))
        plt.subplot(121)
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(labels, cmap='nipy_spectral')
        plt.axis('off')

        plt.tight_layout()
        # plt.show(block=True)

        return labels, num_labels

    def cluster_inliers(self, plane, points, res=50):

        ### project points to plane
        pp = self.project_points_to_plane(points,plane)

        ### discretise the projected points
        xymin = np.min(pp)
        xymax = np.max(pp)
        xynorm = (pp-xymin)*(res-1)/(xymax-xymin)
        xyindex = xynorm.astype(int)

        ### make the grid and activate every cell which has a point in it
        grid = np.zeros(shape=(res, res))
        grid[xyindex[:, 0], xyindex[:, 1]] = 1

        ### find connected components in the grid
        labels, num_labels = self.segment(grid)

        ### nothing to do if I have only one group,
        ### therefore export all points that were input, ie a list of id's with len(input points)
        if num_labels == 1:
            return [np.arange(points.shape[0])]

        ### if there are multiple groups, export a list of list of id's
        groups = []
        ## iterate over each group
        for label in range(1,num_labels+1):
            row, col = np.where(labels==label)
            group_points = []
            for i,_ in enumerate(row):
                group_points.append(np.where((xynorm[:, 0] >= row[i]) & (xynorm[:, 0] < (row[i] + 1)) & \
                         (xynorm[:, 1] >= col[i]) & (xynorm[:, 1] < (col[i] + 1))))
            group_points = np.concatenate(group_points, axis=1).flatten()
            ## check if the group adheres to the minPoint criterion
            if group_points.shape[0] > self.minPoints:
                groups.append(group_points)

        if len(groups) > 0:
            return groups
        else:
            ### if all new groups are too small, export none, and the fit function will return the same as if no plane was detected (=empty lists for best_eq etc)
            return None




    def point_plane_dist(self,pts,plane_eq):

        # Distance from a point to a plane
        # https://mathworld.wolfram.com/Point-PlaneDistance.html
        return (plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]) \
                  / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)


    def get_pt_inliers(self,pts,plane_eq,thresh):
        dist_pt = self.point_plane_dist(pts, plane_eq)
        return np.where(np.abs(dist_pt) <= thresh)[0]


    def fit(self, pts, thresh=0.05, minPoints=100, maxIteration=1000,
            optimization=False, segmentation=False):
        """
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers

        ---
        """
        n_points = pts.shape[0]
        best_eq = []
        best_pt_inliers = []

        for it in range(maxIteration):

            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # get indexes where distance is smaller than the threshold
            pt_id_inliers = self.get_pt_inliers(pts, plane_eq, thresh)


            if (len(pt_id_inliers) > len(best_pt_inliers)) and (len(pt_id_inliers) >= minPoints):
                best_eq = plane_eq
                best_pt_inliers = pt_id_inliers



        if segmentation:
            if len(best_eq) > 0:
                split_groups = self.cluster_inliers(best_eq,pts[best_pt_inliers])
                self.plt.set_facecolor('red')
                if split_groups is not None:
                    # print("Found {} component(s) plane".format(len(split_groups)))
                    split_best_pt_inliers = []
                    for sp in split_groups:
                        # new_plane = self.optimize_plane(pts[best_pt_inliers[sp]])
                        # new_inliers = self.get_pt_inliers(pts,new_plane,thresh)
                        # self.plot_inliers(pts[new_inliers],new_plane)
                        # best_eq = new_plane
                        # split_best_pt_inliers.append(new_inliers)
                        split_best_pt_inliers.append(best_pt_inliers[sp])
                    self.equation = best_eq
                    self.pt_inliers_id = split_best_pt_inliers

                    self.plt.set_facecolor('green')

                self.plt.suptitle("[{}] Point Plane".format(self.counter),color='white',size=18, weight='bold')
                self.plt.show()
        else:
            self.equation = [best_eq]
            self.pt_inliers_id = [best_pt_inliers]

        return self.equation, self.pt_inliers_id


    def get_occ_inliers(self,occ_tgt,pts_tgt,plane_eq):

        ## check if points are left or right of plane (same as distance check, but scaling (denominator) is not necessary)
        occ_pt = np.sign(plane_eq[0] * pts_tgt[:, 0] + plane_eq[1] * pts_tgt[:, 1] + plane_eq[2] * pts_tgt[:, 2] + plane_eq[3])

        ## now just check for correct inside/outside classification here, one orientation check is enough
        # check inside on the right or outside on the left
        occ = (occ_pt / 2) + 0.5  # put the side to [0,1] values
        check = ((occ_tgt + occ) > 1).sum()
        inside = check / np.nansum(occ)
        occ = 1 - occ
        occ_tgt = 1 - occ_tgt
        check = ((occ_tgt + occ) > 1).sum()
        outside = check / np.nansum(occ)
        in_out = np.vstack((inside, outside))

        best_side = np.nanargmax(in_out)
        occ_score = in_out[best_side, 0]

        occ_id_inliers = np.where(occ == best_side)[0]

        return occ_score, occ, occ_id_inliers, in_out, best_side


    def fit_with_occ(self, pts, normals=None, pts_tgt=None, occ_tgt=None,
                     thresh=0.05, minPoints=100, minOccScore=0.5, maxIteration=1000,
                     optimization=False, segmentation=False):
        """
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers

        ---
        """

        self.minPoints = minPoints

        n_points = pts.shape[0]
        best_eq = []
        best_pt_inliers = []
        best_occ_inliers = []
        best_occ_score = 0

        # np.random.seed(42)
        for it in range(maxIteration):

            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            if np.isnan(pt_samples).any():
                maxIteration+=1
                continue



            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[2]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]


            # get indexes where distance is smaller than the threshold
            pt_id_inliers = self.get_pt_inliers(pts,plane_eq,thresh)


            # get occ inliers
            occ_score, occ, occ_id_inliers, in_out, best_side = self.get_occ_inliers(occ_tgt,pts_tgt,plane_eq) # i do not need the /sqrt for this, if I want to do v high level optimization


            # if (len(pt_id_inliers) >= minPoints) and (len(pt_id_inliers) > len(best_pt_inliers)) and (occ_score > best_occ_score) and (occ_score >= minOccScore):
            if (len(pt_id_inliers) >= minPoints) and (occ_score > best_occ_score) and (occ_score >= minOccScore):

                ## if the occupancy classification is also good on the other side of the plane
                ## then take away these points too, ie take away all occ points
                if in_out[np.invert(best_side),0] > minOccScore:
                    print("all occupancy points are out")
                    occ_id_inliers = np.arange(occ.shape[0])

                best_eq = plane_eq
                best_pt_inliers = pt_id_inliers

                best_occ_score = occ_score
                best_occ_inliers = occ_id_inliers

        if segmentation:
            if len(best_eq) > 0:
                split_groups = self.cluster_inliers(best_eq,pts[best_pt_inliers])
                self.plt.set_facecolor('red')
                if split_groups is not None:
                    # print("Found {} component(s) plane".format(len(split_groups)))
                    split_best_pt_inliers = []
                    best_eqs = []
                    for sp in split_groups:
                        # new_plane = self.optimize_plane(pts[best_pt_inliers[sp]],best_eq)
                        # new_inliers = self.get_pt_inliers(pts,new_plane,thresh)
                        # self.plot_inliers(pts[new_inliers],new_plane)
                        # best_eq = new_plane
                        # split_best_pt_inliers.append(new_inliers)
                        split_best_pt_inliers.append(best_pt_inliers[sp])
                    self.equation = best_eq
                    self.pt_inliers_id = split_best_pt_inliers
                    self.occ_score = best_occ_score
                    self.occ_inliers_id = best_occ_inliers

                    self.plt.set_facecolor('green')


                self.plt.suptitle("[{}] Occupancy Plane".format(self.counter),color='white',size=18, weight='bold')
                self.plt.show()
        else:
            self.equation = best_eq
            self.pt_inliers_id = [best_pt_inliers]
            self.occ_score = best_occ_score
            self.occ_inliers_id = best_occ_inliers





        return self.equation, self.pt_inliers_id, self.occ_score, self.occ_inliers_id