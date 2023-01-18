import random

import numpy as np


class Plane:
    """
    Implementation of planar RANSAC.

    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.

    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.

    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")

    ---
    """

    def __init__(self):
        self.pt_inliers = []
        self.occ_inliers = []
        self.equation = []
        self.occ_score = 0

    def fit(self, pts, thresh=0.05, minPoints=100, maxIteration=1000):
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
        best_inliers = []

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

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            # pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if (len(pt_id_inliers) > len(best_inliers)) and (len(pt_id_inliers) >= minPoints):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            self.pt_inliers = best_inliers
            self.equation = best_eq

        return self.equation, self.pt_inliers



    def fit_with_occ(self, pts, normals=None, pts_tgt=None, occ_tgt=None, thresh=0.05, minPoints=100, minOccScore=0.5, maxIteration=1000):
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

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            # pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)


            occ_pt = (
                plane_eq[0] * pts_tgt[:, 0] + plane_eq[1] * pts_tgt[:, 1] + plane_eq[2] * pts_tgt[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            ## now just check for correct inside/outside classification here, one orientation check is enough
            # check inside on the right or outside on the left
            occ = (np.sign(occ_pt)/2)+0.5# put the side to [0,1] values
            check = ((occ_tgt+occ)>1).sum()
            inside = check/np.nansum(occ)
            occ = 1-occ
            occ_tgt = 1-occ_tgt
            check = ((occ_tgt+occ)>1).sum()
            outside = check/np.nansum(occ)
            in_out = np.vstack((inside,outside))

            best_side = np.nanargmax(in_out)
            occ_score = in_out[best_side,0]

            occ_id_inliers = np.where(occ==best_side)[0]

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            # if (len(pt_id_inliers) > len(best_pt_inliers)) and (occ_score > best_occ_score):

            # normals


            # if (len(pt_id_inliers) >= minPoints) and (len(pt_id_inliers) > len(best_pt_inliers)) and (occ_score > best_occ_score) and (occ_score >= minOccScore):
            if (len(pt_id_inliers) >= minPoints) and (occ_score > best_occ_score) and (occ_score >= minOccScore):
                best_eq = plane_eq
                best_pt_inliers = pt_id_inliers
                best_occ_inliers = occ_id_inliers
                best_occ_score = occ_score
            self.occ_score = best_occ_score
            self.occ_inliers = best_occ_inliers
            self.pt_inliers = best_pt_inliers
            self.equation = best_eq

        return self.equation, self.pt_inliers, self.occ_score, self.occ_inliers