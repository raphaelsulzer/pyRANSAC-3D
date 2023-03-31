import copy
import trimesh

PYTHONPATH="/home/rsulzer/python"

import sys, os
from treelib import Node, Tree
sys.path.append(os.path.join(PYTHONPATH,"dsr-benchmark"))
from datasets import modelnet10, berger
from fractions import Fraction
from sage.all import polytopes, QQ, RR, Polyhedron
import numpy as np
import pyransac3d as ransac
from export import PlaneExporter
sys.path.append(os.path.join(PYTHONPATH,"ksr-benchmark"))
from main import Benchmark
import string, subprocess
import open3d as o3d
from skimage import measure

from tqdm import tqdm

sys.path.append(os.path.join(PYTHONPATH,"..","cpp","check_mesh_contains","build","release"))
import libMeshContains as MC


class PBR:


    def __init__(self):

        self.exporter = PlaneExporter()

    def _inequalities(self,plane):
        """
        Inequalities from plane parameters.

        Parameters
        ----------
        plane: (4,) float
            Plane parameters

        Returns
        -------
        positive: (4,) float
            Inequality of the positive half-plane
        negative: (4,) float
            Inequality of the negative half-plane
        """
        positive = [QQ(plane[-1]), QQ(plane[0]), QQ(plane[1]), QQ(plane[2])]
        negative = [QQ(-element) for element in positive]
        return positive, negative


    def write_face(self,m,facet,color=[1,0,0],count=-1):

        c = np.random.random(size=3)

        path = os.path.join(os.path.dirname(m["ransac"]),"facets")
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(count)+'.obj')

        ss = facet.render_solid().obj_repr(facet.render_solid().default_render_params())

        f = open(filename,'w')
        # for out in ss[2:4]:
        #     for line in out:
        #         f.write(line+"\n")
        for v in ss[2]: # colored vertex
            f.write(v)
            for c in color:
                f.write(" "+str(c))
            f.write("\n")
        for fa in ss[3]: # facet
            f.write(fa)

        f.close()
        a=5

    def write_cell(self, m, polyhedron, points=None, count=-1):

        c = np.random.random(size=3)

        path = os.path.join(os.path.dirname(m['ransac']),"cells")
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(count)+'.obj')

        ss = polyhedron.render_solid().obj_repr(polyhedron.render_solid().default_render_params())

        f = open(filename,'w')
        for out in ss[2:4]:
            for line in out:
                f.write(line+"\n")

        if points is not None:
            for p in points:
                f.write("v {} {} {} {} {} {}\n".format(p[0],p[1],p[2],c[0],c[1],c[2]))

        f.close()

    def get_bounding_box(self,m):

        self.bounding_verts = []
        # points = np.load(m["pointcloud"])["points"]
        points = np.load(m["occ"])["points"]

        ppmin = points.min(axis=0)
        ppmax = points.max(axis=0)

        ppmin = [-40,-40,-40]
        ppmax = [40,40,40]

        pmin=[]
        for p in ppmin:
            pmin.append(Fraction(str(p)))
        pmax=[]
        for p in ppmax:
            pmax.append(Fraction(str(p)))

        self.bounding_verts.append(pmin)
        self.bounding_verts.append([pmin[0],pmax[1],pmin[2]])
        self.bounding_verts.append([pmin[0],pmin[1],pmax[2]])
        self.bounding_verts.append([pmin[0],pmax[1],pmax[2]])
        self.bounding_verts.append(pmax)
        self.bounding_verts.append([pmax[0],pmin[1],pmax[2]])
        self.bounding_verts.append([pmax[0],pmax[1],pmin[2]])
        self.bounding_verts.append([pmax[0],pmin[1],pmin[2]])

        self.bounding_poly = Polyhedron(vertices=self.bounding_verts)

    def construct(self,m):
        ex = Exporter()
        ## load surface and volume points of shape
        scan = np.load(m["scan"])
        points = scan["points"]
        # normals = scan["normals"]
        occ = np.load(m["occ"])
        points_tgt = occ["points"]
        occ_tgt = np.unpackbits(occ["occupancies"]).astype(float)


        # iteratively slice convexes with planes
        cell_count = 0
        plane_count = 0
        unprocessed_occ_pts = copy.deepcopy(points_tgt)
        unprocessed_pts = copy.deepcopy(points)

        self.get_bounding_box(m)
        self.write_cell(m,self.bounding_poly)

        tree = Tree()
        dd = {"cell":self.bounding_poly,"pt_ids":np.arange(points.shape[0]),"occ_pt_ids":np.arange(points_tgt.shape[0]),"needs split":True}
        tree.create_node(tag=cell_count, identifier=cell_count,data=dd) # root node

        group_verts = []
        planes = []
        group_num_points = []


        children = tree.expand_tree(0,filter= lambda x: x.data["needs split"],mode=Tree.DEPTH)
        maxIter = 200; iter=0
        for child in children: # while all nodes are not fully split, continue the loop

            iter += 1
            if iter > maxIter:
                print("maxIter reacher")
                break

            tree[0].data["needs split"] = False

            color = np.random.random(size=3)

            # if not tree[child].is_leaf():
            #     continue

            current_cell = tree[child].data["cell"]
            current_pt_ids = tree[child].data["pt_ids"]
            current_occ_pt_ids = tree[child].data["occ_pt_ids"]

            minPoints = 48
            if points[current_pt_ids].shape[0] < 4:
                tree[child].data["needs split"] = False
                print("Finished cell {} with {} points left".format(child,points[current_pt_ids].shape[0]))
                continue

            if np.isnan(points_tgt).all() or np.isnan(points).all():
                print("all occupancy or surface points are treated")
                break

            rp = ransac.Plane()
            # best_eq, best_inliers = plane1.fit(points, thresh=0.875, minPoints=50, maxIteration=100)
            th = 0.1
            plane_eq = []
            while len(plane_eq) == 0 and minPoints >= 2:
                surf_plane = 0
                plane_eq, inliers, occ_score, occ_inliers, perfect_split = rp.fit_with_occ(points[current_pt_ids], pts_tgt=points_tgt[current_occ_pt_ids],
                                                                                          occ_tgt=occ_tgt[current_occ_pt_ids],
                                                                                          thresh=th, minPoints=minPoints,
                                                                                          minOccScore=0.98,
                                                                                          maxIteration=1000,
                                                                                          optimization=False,
                                                                                          segmentation=True, segmentation_resolution=20)
                print("cell {} is difficult to process".format(child))
                minPoints = minPoints / 2

            if minPoints < 2:
                minPoints = 48
                while len(plane_eq) == 0 and minPoints >= 2:
                    # if len(plane_eq)==0:
                    occ_score = 0
                    surf_plane = 1
                    occ_inliers = []
                    plane_eq, inliers = rp.fit(points[current_pt_ids], thresh=th, minPoints=minPoints, maxIteration=1000,
                                                       optimization=False, segmentation=True,segmentation_resolution=40)
                    minPoints = minPoints / 2

            # if len(plane_eq) == 0:
            #     minPoints = minPoints/2
            #     print("cell {} is difficult to process".format(child))

            if minPoints < 2:
                # tree[child].data["needs split"] = False
                # print("Finished cell {} with {} points left".format(child,points[current_pt_ids].shape[0]))
                continue


            ex.export_deleted_points(os.path.dirname(m["ransac"]), points[current_pt_ids,:],
                                     count=child, subfolder="available_points",color=color)

            ex.export_deleted_points(os.path.dirname(m["ransac"]), points_tgt[current_occ_pt_ids,:],
                                     count=child, subfolder="cell_points",color=color)
            ## check the side
            occs = plane_eq[0] * points_tgt[current_occ_pt_ids, 0] + plane_eq[1] * points_tgt[current_occ_pt_ids, 1] + plane_eq[2] * points_tgt[current_occ_pt_ids, 2] + plane_eq[3]
            poccs = np.where(occs > 0)[0]
            noccs = np.where(occs < 0)[0]

            ptss = plane_eq[0] * points[current_pt_ids, 0] + plane_eq[1] * points[current_pt_ids, 1] + plane_eq[2] * points[current_pt_ids, 2] + plane_eq[3]
            pptss= np.where(ptss > th)[0]
            nptss= np.where(ptss < -th)[0]

            print("{}: {} plane with score {:.2f} split cell {} into cells {}/{} with n_points {}/{}; {}/{} surface points processed; {}/{} occ points processed".format(
                plane_count, "surface" if surf_plane else "occ", occ_score, child, cell_count + 1, cell_count + 2, poccs.shape[0], noccs.shape[0],
                len(group_verts), points.shape[0],
                0,100000))


            hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in
                                                self._inequalities(plane_eq)]

            hspace_positive = current_cell.intersection(hspace_positive)
            hspace_negative = current_cell.intersection(hspace_negative)

            # if hspace_positive.dim() != 3 or hspace_negative.dim() != 3:
            #     continue

            if(not hspace_positive.is_empty()): # should maybe adopt the dim != 3 method of the original code instead
                cell_count+=1
                pneeds_split = list(poccs) != list(occ_inliers)
                pneeds_split = np.invert(perfect_split) if perfect_split else pneeds_split
                dd = {"cell": hspace_positive, "pt_ids": current_pt_ids[pptss], "occ_pt_ids": current_occ_pt_ids[poccs], "needs split": pneeds_split}
                tree.create_node(tag=cell_count,identifier=cell_count,data=dd,parent=tree[child].identifier)
                self.write_cell(m,hspace_positive,count=cell_count)

            if(not hspace_negative.is_empty()):
                cell_count+=1
                nneeds_split = list(noccs) != list(occ_inliers)
                nneeds_split = np.invert(perfect_split) if perfect_split else nneeds_split
                dd = {"cell": hspace_negative, "pt_ids": current_pt_ids[nptss], "occ_pt_ids":current_occ_pt_ids[noccs], "needs split": nneeds_split}
                tree.create_node(tag=cell_count,identifier=cell_count,data=dd,parent=tree[child].identifier)
                self.write_cell(m,hspace_negative,count=cell_count)

            if(not hspace_negative.is_empty() and not hspace_positive.is_empty()):
                facet = hspace_positive.intersection(hspace_negative)
                self.write_face(m, facet, color=color, count=plane_count)

            abc = list(string.ascii_lowercase)
            abc = [""]+abc
            for i,ins in enumerate(inliers):
                ex.export_plane(os.path.dirname(m["ransac"]), plane_eq, points[current_pt_ids,:][ins,:],
                                        count=str(plane_count)+abc[i], color=color)

                soc = [1,0,0] if surf_plane else [0,1,0]
                ex.export_plane(os.path.dirname(m["ransac"]), plane_eq, points[current_pt_ids, :][ins, :],
                                count=str(plane_count) + abc[i], color=soc, subpaths=["planes_so","point_groups_so"])

                unprocessed_pts[current_pt_ids[ins], :] = None
                planes.append(plane_eq)
                group_verts += list(current_pt_ids[ins])
                group_num_points.append(len(ins))

            unprocessed_occ_pts[current_occ_pt_ids[occ_inliers],:] = None
            ex.export_deleted_points(os.path.dirname(m["ransac"]), unprocessed_occ_pts[~np.isnan(unprocessed_occ_pts).all(axis=1)], count="888",color=[1.0,0.0,0.0])
            ex.export_deleted_points(os.path.dirname(m["ransac"]), unprocessed_pts[~np.isnan(unprocessed_pts).all(axis=1)], count="999",color=[1.0,0.0,0.0])

            ex.export_deleted_points(os.path.dirname(m["ransac"]), points_tgt[current_occ_pt_ids,:][poccs, :],
                                     count=str(plane_count)+"p{}".format("" if pneeds_split else "-"), color=color)
            ex.export_deleted_points(os.path.dirname(m["ransac"]), points_tgt[current_occ_pt_ids,:][noccs, :],
                                     count=str(plane_count)+"n{}".format("" if nneeds_split else "-"), color=color)



            plane_count+=1


        tree.show()


        np.savez(m["ransac"],
                 points=scan["points"],normals=scan["normals"],
                 group_parameters=np.array(planes),
                 group_num_points=np.array(group_num_points).astype(int),
                 group_points=np.array(group_verts)
                 )

    def print_loss(self,points,occ,plane):
        occ = 1-occ

        a, b, c, d = plane[0],plane[1],plane[2],plane[3]
        x, y, z, o = points[:,0],points[:,1],points[:,2],occ
        dist = a * x + b * y + c * z + d
        sigmoid_dist = 1 / (1 + np.exp(-dist * 10))
        asd = 1-sigmoid_dist
        loss = ((sigmoid_dist * o)+asd).sum()
        print(loss)

    def test_optim(self,m):

        ex = Exporter()
        ## load surface and volume points of shape
        scan = np.load(m["scan"])
        points = scan["points"]
        # normals = scan["normals"]
        occ = np.load(m["occ"])
        points_tgt = occ["points"]
        occ_tgt = np.unpackbits(occ["occupancies"]).astype(float)

        ## export bb
        self.get_bounding_box(m)
        self.write_cell(m,self.bounding_poly)


        ## export init plane
        init_plane = [0,0.1,-1.1,-37.5]
        hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in self._inequalities(init_plane)]
        hspace_positive = self.bounding_poly.intersection(hspace_positive)
        hspace_negative = self.bounding_poly.intersection(hspace_negative)
        facet = hspace_positive.intersection(hspace_negative)
        self.write_face(m, facet, count=-2)

        rp = ransac.Plane()
        plane_eq = rp.optimize_plane_occ(points_tgt,occ_tgt,init_plane)



        ## export refined plane
        hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in self._inequalities(plane_eq)]
        hspace_positive = self.bounding_poly.intersection(hspace_positive)
        hspace_negative = self.bounding_poly.intersection(hspace_negative)
        facet = hspace_positive.intersection(hspace_negative)
        self.write_face(m, facet, count=-1)

        print("before")
        self.print_loss(points_tgt,occ_tgt,init_plane)
        print("after")
        self.print_loss(points_tgt,occ_tgt,plane_eq)

        a=5



    def marching_cubes(self,m,res=196):

        self.get_bounding_box(m)

        x = np.linspace(self.bounding_poly.bounding_box()[0][0],self.bounding_poly.bounding_box()[1][0],res)
        y = np.linspace(self.bounding_poly.bounding_box()[0][1],self.bounding_poly.bounding_box()[1][1],res)
        z = np.linspace(self.bounding_poly.bounding_box()[0][2],self.bounding_poly.bounding_box()[1][2],res)
        xv, yv, zv = np.meshgrid(x,y,z)

        positions = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).transpose()


        mc = MC.Checker()
        mc.loadMesh(m["mesh"])
        contains = mc.check(positions)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        colors = []
        for co in contains:
            col = [1,0,0] if co else [0,0,1]
            colors.append(col)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        os.makedirs(os.path.join(m["path"], "grid_points"),exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(m["path"], "grid_points", "{}.ply".format(m["model"])), pcd)

        occupancy_grid = contains.reshape((res, res, res)).astype(int)
        occupancy_grid = occupancy_grid.swapaxes(1,0)

        # Extract the surface using marching cubes
        verts, faces, normals, values = measure.marching_cubes(occupancy_grid, level=0.5, gradient_direction='ascent')

        # Create an Open3D mesh from the surface vertices and faces
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.triangles = o3d.utility.Vector3iVector(faces)


        # # Scale the mesh by the maximum distance
        # mesh.scale(1/40, mesh.get_center())
        #
        center = mesh.get_axis_aligned_bounding_box().get_center()
        mesh.translate(-center)

        # Scale the mesh using the calculated scaling factor
        mesh.scale(max(self.bounding_poly.bounding_box()[1])/(res/2), [0, 0, 0])


        # Export the mesh to a file in OBJ format
        os.makedirs(os.path.join(m["path"], "mc_mesh"), exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(m["path"], "mc_mesh", "{}.ply".format(m["model"])), mesh)

        pcd = mesh.sample_points_uniformly(number_of_points=100000)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(os.path.join(os.path.dirname(m["pointcloud_ply"]),"{}_{}.ply".format(m["model"],str(res))),pcd)

        q=5

    def extract_planes(self,m):

        scan = np.load(m["scan"])
        points = scan["points"]
        # normals = scan["normals"]
        occ = np.load(m["occ"])
        points_tgt = occ["points"]
        occ_tgt = np.unpackbits(occ["occupancies"]).astype(float)

        group_verts = []
        group_points  = []
        planes = []
        group_num_points = []

        occ_scores = []

        it = 0; maxIter = 100
        plane_count = 0; maxPlanes = 50
        maxbar = tqdm(maxIter,leave=False)

        files=[]

        while len(planes) < maxPlanes:
            np.random.seed(None)
            c = np.random.random(size=3)
            it+=1
            maxbar.update(1)
            if it > maxIter:
                print("stopped at global maxiter")
                break

            if np.isnan(points_tgt).all() or np.isnan(points).all():
                print("all occupancy or surface points are treated")
                break

            plane1 = ransac.Plane(count=plane_count)
            # best_eq, best_inliers = plane1.fit(points, thresh=0.875, minPoints=50, maxIteration=100)
            best_eq, best_inliers, best_occ_score, best_occ_pts = plane1.fit_with_occ(points, pts_tgt=points_tgt, occ_tgt=occ_tgt,
                                                                                      thresh=0.05, minPoints=20, minOccScore=0.96, maxIteration=5000,
                                                                                      optimization=True, segmentation=True)
            if len(best_eq) == 0:
                best_eq, best_inliers = plane1.fit(points, thresh=0.1, minPoints=20, maxIteration=5000,
                                                   optimization=True, segmentation=True)

            if len(best_eq) == 0:
                continue

            occ_scores.append({plane_count:best_occ_score})
            for i,bi in enumerate(best_inliers):
                group_verts += list(bi)
                gp = points[bi]
                group_points.append(gp)
                group_num_points.append(len(gp))
                planes.append(best_eq)

                if best_occ_score > 0.98: # occ points are removed
                    fname = os.path.join(os.path.dirname(m["ransac"]),"planes",str(plane_count)+"r.obj")
                    files.append(fname)
                    # export plane and take out the processed surface pts
                    self.exporter.export_plane(os.path.dirname(m["ransac"]), best_eq, gp, count=str(plane_count)+"r", color=c)
                    points[bi, :] = None
                    # export and then take out the processed occupancy points, only once, for split planes, that is why there is the np.isnan
                    if(not np.isnan(points_tgt[best_occ_pts,:]).any()):
                        self.exporter.export_deleted_points(os.path.dirname(m["ransac"]), points_tgt[best_occ_pts, :], count=str(plane_count)+"r",color=c)
                        points_tgt[best_occ_pts, :] = None
                        occ_tgt[best_occ_pts] = None
                        fname = os.path.join(os.path.dirname(m["ransac"]), "deleted_points", str(plane_count) + "r.obj")
                        files.append(fname)
                else: # occ point are not removed
                    fname = os.path.join(os.path.dirname(m["ransac"]),"planes",str(plane_count)+".obj")
                    files.append(fname)
                    if best_occ_score > 0:
                        fname = os.path.join(os.path.dirname(m["ransac"]),"deleted_points",str(plane_count)+".obj")
                        files.append(fname)
                    self.exporter.export_plane(os.path.dirname(m["ransac"]), best_eq, gp, count=plane_count, color=c)
                    points[bi, :] = None
                    self.exporter.export_deleted_points(os.path.dirname(m["ransac"]), points_tgt[best_occ_pts, :], count=plane_count,color=c)

                plane_count+=1

        maxbar.close()
        print(occ_scores)
        assert(len(group_points)==len(planes))
        # save to ply
        np.savez(m["ransac"],
                 points=scan["points"],normals=scan["normals"],
                 group_parameters=np.array(planes),
                 group_num_points=np.array(group_num_points).astype(int),
                 group_points=np.array(group_verts)
                 )

        subprocess.Popen(["meshlab"]+files)






if __name__ == '__main__':

    pbr = PBR()

    abspy_k = 1
    ksr_k = 1


    path = "/home/rsulzer/data/reconbench"
    ds = berger.Berger(path=path)
    models = ds.getModels(scan_conf="1",ksr_k=ksr_k,abspy_k=abspy_k,hint="anchor")

    path = "/home/rsulzer/data/reconbench"
    bm = Benchmark(path)
    for m in models:

        # pbr.test_optim(m)

        pbr.marching_cubes(m)

        bm.clear(m)
        pbr.construct(m)

