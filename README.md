# Open3D Guide

My personal guide to the great Python library [Open3D](https://www.open3d.org/).

I followed these resources:

- [**Open3D Basic Tutorial**](https://www.open3d.org/docs/latest/tutorial/Basic/index.html)
- The [Open3D Official Documentation](https://www.open3d.org/docs/release/index.html)
- [Open3D Python Tutorial, by Nicolai Nielsen](https://www.youtube.com/watch?v=zF3MreN1w6c&list=PLkmvobsnE0GEZugH1Di2Cr_f32qYkv7aN)
- [pointcloud_tutorial, by Jeff Delmerico](https://github.com/mxagar/pointcloud_tutorial)
- [3D Data Processing with Open3D](https://towardsdatascience.com/3d-data-processing-with-open3d-c3062aadc72e)

Also, look at this [Point Cloud Library (PCL)](https://pointclouds.org/) compilation of mine, where the below listed topics are shown using PCL:

[mxagar/tool_guides/pcl](https://github.com/mxagar/tool_guides/tree/master/pcl)

- Point cloud creation and management.
- Point (and cloud) feature exploration: normals, PFHs, moments, etc.
- Filtering and segmentation: voxel grid filtering, projection, outlier removal, RANSAC, shape segmentation, etc.
- Registration, i.e., Matching: ICP.
- Surface processing: resampling, convex hulls, projections, triangulation, etc.
- Visualization: normals, coordinate systems, etc.
- Data structures: KD-tree, voxelmaps and octrees, etc.

Table of contents:

- [Open3D Guide](#open3d-guide)
  - [Setup and File Structure](#setup-and-file-structure)
  - [1. Introduction and File IO](#1-introduction-and-file-io)
  - [2. Point Clouds](#2-point-clouds)
  - [3. Meshes](#3-meshes)
  - [4. Transformations](#4-transformations)
  - [5. Rest of Modules](#5-rest-of-modules)
  - [Authorship](#authorship)

## Setup and File Structure

Install in a Python environment:

```bash
pip install open3d
```

The repository consists of three main folders:

- [`notebooks/`](./notebooks): Personal notebooks based on the [**Open3D Basic Tutorial**](https://www.open3d.org/docs/latest/tutorial/Basic/index.html); the sections below contain code summaries from those notebooks.
- [`examples/`](./examples): Official example files from [https://github.com/isl-org/Open3D/tree/main/examples/python](https://github.com/isl-org/Open3D/tree/main/examples/python).
- [`models/`](./models): Several models both from Open3D repositories as well as from [mxagar/tool_guides/pcl](https://github.com/mxagar/tool_guides/tree/master/pcl), i.e., PCD files from PCL.

## 1. Introduction and File IO

Notebook: [`01_Intro_File_IO.ipynb`](./notebooks/01_Intro_File_IO.ipynb).

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/file_io.html](https://www.open3d.org/docs/latest/tutorial/Basic/file_io.html).

Summary of contents:

- Load pointclouds, meshes, images
- Visualize in new window and in notebook
- Save files with desired formats
- Download models from the internet with `o3d.data`: [https://www.open3d.org/docs/release/python_api/open3d.data.html](https://www.open3d.org/docs/release/python_api/open3d.data.html)

```python
import sys
import os

# Add the directory containing 'examples' to the Python path
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)  # Parent directory
sys.path.append(parent_directory)

import open3d as o3d
from examples import open3d_example as o3dex
import numpy as np

# Here, the same file is opened locally
pcd = o3d.io.read_point_cloud("../models/fragment.ply")
print(pcd) # 196133 points
print(np.asarray(pcd.points))

# A new visualization window is opened
# Keys:
#  [/]          : Increase/decrease field of view.
#  R            : Reset view point.
#  Ctrl/Cmd + C : Copy current view status into the clipboard.
#  Ctrl/Cmd + V : Paste view status from clipboard.
#  Q, Esc       : Exit window.
#  H            : Print help message.
#  P, PrtScn    : Take a screen capture.
#  D            : Take a depth capture.
#  O            : Take a capture of current rendering settings.
# IMPORTANT: Press Q to exit the viewer; the notebook cell waits for that!
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# We can inspect the docstring of each function with help()
help(o3d.visualization.draw_geometries)

# In-Notebook web visualizer (but with a worse quality)
o3d.web_visualizer.draw(pcd,                                  
                        lookat=[2.6172, 2.0475, 1.532],
                        up=[-0.0694, -0.9768, 0.2024])

# IO Pointcloud
pcd = o3d.io.read_point_cloud("../models/fragment.pcd")
print(pcd)
# Save file
# The format is passed in the extension, or, optionally in the argument format='xyz'
# Supported formats: 
# xyz: [x, y, z]
# xyzn: [x, y, z, nx, ny, nz]
# xyzrgb: [x, y, z, r, g, b]
# pts: [x, y, z, i, r, g, b]
# ply
# pcd
o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

# IO Mesh
mesh = o3d.io.read_triangle_mesh("../models/monkey.ply")
print(mesh)
# Save file
# The format is passed in the extension
# Supported formats: 
# ply, stl, obj, off, gltf/glb
o3d.io.write_triangle_mesh("copy_monkey.ply", mesh)

# IO Image
img = o3d.io.read_image("../models/lenna.png")
print(img)
# Save file
# Supported formats: JPG, PNG
o3d.io.write_image("copy_of_lena.jpg", img)

# We can download data using `o3d.data`; a list of all possible models is provided here:
# https://www.open3d.org/docs/release/python_api/open3d.data.html
armadillo = o3d.data.ArmadilloMesh()
armadillo_mesh = o3d.io.read_triangle_mesh(armadillo.path)
bunny = o3d.data.BunnyMesh()
bunny_mesh = o3d.io.read_triangle_mesh(bunny.path)

# Visualize the mesh
print(bunny_mesh)
o3d.visualization.draw_geometries([bunny_mesh], window_name='3D Mesh Visualization')
```

## 2. Point Clouds

Notebook: [`01_Intro_File_IO.ipynb`](./notebooks/01_Intro_File_IO.ipynb).

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html](https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html).

Summary of contents:

- Visualize a point cloud: `o3d.visualization.draw_geometries()`
- Voxel downsampling: `pc.voxel_down_sample()`
- Vertex normal estimation: `pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(...)`
- Access estimated vertex normals as Numpy arrays: `np.asarray(pc.normals)[:10, :]`
- Crop point cloud: `cropped_pc = polygon_volume.crop_point_cloud(pc)`
- Paint point cloud: `pc.paint_uniform_color([1, 0.5, 0])`
- Point cloud distance and selection: `dists = pc1.compute_point_cloud_distance(pc2)`, `pc.select_by_index(ind)`
- Bounding volumes (AABB, OBB): `pc.get_axis_aligned_bounding_box()`, `pc.get_oriented_bounding_box()`
- Convex hull and sampling: `pc = mesh.sample_points_poisson_disk()`, `hull_mes, _ = pc.compute_convex_hull()`
- DBSCAN clustering: `pc.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)`
- Plane segmentation: `plane_model, inliers = pc.segment_plane()`
- (Visually) Hidden point removal: `pc.hidden_point_removal(camera, radius)`

```python
import sys
import os

# Add the directory containing 'examples' to the Python path
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)  # Parent directory
sys.path.append(parent_directory)

import open3d as o3d
from examples import open3d_example as o3dex
import numpy as np

## -- Visualize point cloud

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("../models/fragment.ply")
print(pcd)
print(np.asarray(pcd.points))
# A new visualization window is opened
# Points rendered as surfels
# Keys:
#  [/]          : Increase/decrease field of view.
#  R            : Reset view point.
#  Ctrl/Cmd + C : Copy current view status into the clipboard.
#  Ctrl/Cmd + V : Paste view status from clipboard.
#  Q, Esc       : Exit window.
#  H            : Print help message.
#  P, PrtScn    : Take a screen capture.
#  D            : Take a depth capture.
#  O            : Take a capture of current rendering settings.
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

## -- Voxel downsampling

# Voxel downsampling
# 1. Points are bucketed into voxels.
# 2. Each occupied voxel generates exactly one point by averaging all points inside.
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

## -- Vertex normal estimation

print("Recompute the normal of the downsampled point cloud")
# Compute normals: estimate_normals()
# The function finds adjacent points and calculates the principal axis of the adjacent points using covariance analysis.
# The function takes an instance of KDTreeSearchParamHybrid class as an argument. 
# The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor.
# It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.
# NOTE: normal direction is chosen to comply with original ones, else arbitrary
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
# Visualize points and normals: toggle on/off normals with N
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)

## -- Access estimated vertex normals as Numpy arrays

print("Print a normal vector of the 0th point")
print(downpcd.normals[0])

# Use help() extensively to check all available variables/proterties/functions!
help(downpcd)

# Normal vectors can be transformed as a numpy array using np.asarray
print("Print the normal vectors of the first 10 points")
print(np.asarray(downpcd.normals)[:10, :])

## -- Crop point cloud

# Download the cropping demo
# The demo consists of the living room PLY `fragment.ply` and a JSON which contains a bounding polygon
demo_crop = o3d.data.DemoCropPointCloud()

# Once we have the polygon which encloses our desired region, cropping is easy
print("Load a polygon volume and use it to crop the original point cloud")
# Read a json file that specifies polygon selection area
vol = o3d.visualization.read_selection_polygon_volume(
    "../models/cropped.json"
)
# Filter out points. Only the chair remains.
chair = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

## -- Paint point cloud

print("Paint chair")
# Paint all the points to a uniform color.
# The color is in RGB space, [0, 1] range.
chair.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

## -- Point cloud distance and selection

# Load data
pcd = o3d.io.read_point_cloud("../models/fragment.ply")
vol = o3d.visualization.read_selection_polygon_volume(
    "../models/cropped.json")
chair = vol.crop_point_cloud(pcd)

# Compute the distance from a source point cloud to a target point cloud.
# I.e., it computes for each point in the source point cloud the distance to the closest point in the target point cloud
# pcd: 196133 points
# chair: 31337 points
# dists: 196133 items
# np.where yields a tuple witha unique array -> [0]
# With select_by_index all indices from pcd are taken which have a distance larger than 0.01
# Since chair is contained in pcd, this is equivalent to removing chair from pcd
dists = pcd.compute_point_cloud_distance(chair)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
pcd_without_chair = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_without_chair],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

## -- Bounding volumes (AABB, OBB)

# Get the AABB and the OBB of a point cloud
# Then visualize them
aabb = chair.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = chair.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([chair, aabb, obb],
                                  zoom=0.7,
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

## -- Convex hull and sampling

# Download data
bunny = o3d.data.BunnyMesh()
bunny_mesh = o3d.io.read_triangle_mesh(bunny.path) # ../models/BunnyMesh.ply

# Before computing the convex hull, the point cloud is sampled.
# sample_points_poisson_disk(): each point has approximately the same distance
# to the neighbouring points (blue noise).
# Method is based on Yuksel, "Sample Elimination for Generating Poisson Disk Sample Sets", EUROGRAPHICS, 2015
# number_of_points: Number of points that should be sampled.
pcl = bunny_mesh.sample_points_poisson_disk(number_of_points=2000)

# Compute the convex hull of the sampled point cloud (based in Qhull)
# A triangle mesh is returned
hull, _ = pcl.compute_convex_hull()
# The conv hull traingle mesh a line set is created for visualization purposes
# and lines painted in red
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))

# Visualize downsampled pointcloud as well as convex hull represented with lines
o3d.visualization.draw_geometries([pcl, hull_ls])

## -- DBSCAN clustering

import matplotlib.pyplot as plt

# Load model
pcd = o3d.io.read_point_cloud("../models/fragment.ply")

# DBSCAN two parameters:
# - eps defines the distance to neighbors in a cluster 
# - and min_points defines the minimum number of points required to form a cluster.
# The function returns labels, where the label -1 indicates noise.
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True)
    )

# Plot points with colors
max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
# Vector3dVector: Convert float64 numpy array of shape (n, 3) to Open3D format
# https://www.open3d.org/docs/release/python_api/open3d.utility.html#open3d-utility
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

## -- Plane segmentation

# Segmententation of geometric primitives (only plane?) from point clouds using RANSAC
# - distance_threshold defines the maximum distance a point can have to an estimated plane to be considered an inlier,
# - ransac_n defines the number of points that are randomly sampled to estimate a plane,
# - and num_iterations defines how often a random plane is sampled and verified.
# The function then returns the plane as (a,b,c,d) such that for each point (x,y,z) on the plane we have ax+by+cz+d=0.
# The function further returns a list of indices of the inlier points.
pcd = o3d.io.read_point_cloud("../models/fragment.pcd")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
# Plane model
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Plot
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

## -- (Visually) Hidden point removal

# Download data
armadillo = o3d.data.ArmadilloMesh()
armadillo_mesh = o3d.io.read_triangle_mesh(armadillo.path) # ../models/ArmadilloMesh.ply


# First, we load a mesh and sample points on it
print("Convert mesh to a point cloud and estimate dimensions")
pcd = armadillo_mesh.sample_points_poisson_disk(5000)
diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
)
o3d.visualization.draw_geometries([pcd])

# Imagine you want to render a point cloud from a given view point, 
# but points from the background leak into the foreground because they are not occluded by other points.
# For this purpose we can apply a hidden point removal algorithm
print("Define parameters used for hidden_point_removal")
camera = [0, 0, diameter]
radius = diameter * 100

print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

print("Visualize result")
pcd = pcd.select_by_index(pt_map)
o3d.visualization.draw_geometries([pcd])

```

## 3. Meshes

Notebook: [`03_Meshes.ipynb`](./notebooks/03_Meshes.ipynb).

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/mesh.html](https://www.open3d.org/docs/latest/tutorial/Basic/mesh.html).

Summary of contents:

- Load and check properties: `read_triangle_mesh()`, `mesh.vertices`, `mesh.triangles`
- Visualize a mesh: `o3d.visualization.draw_geometries([mesh])`
- Surface normal estimation: `mesh.compute_vertex_normals()`, `mesh.triangle_normals`
- Crop a mesh using Numpy slicing
- Paint a mesh: `mesh1.paint_uniform_color([1, 0.5, 0])`
- Check properties: `is_edge_manifold`, `is_vertex_manifold`, `is_self_intersecting`, `is_watertight`, `is_orientable`.
- Mesh filtering:
  - Average filter: `mesh.filter_smooth_simple(...)`
  - Laplacian: `mesh.filter_smooth_laplacian(...)`
  - Taubin: `mesh.filter_smooth_taubin(...)`
- Sampling mesh surfaces with points:
  - Uniform: `mesh.sample_points_uniformly(number_of_points=500)`
  - Poison: `mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5)`
- Mesh subdivision: `mesh.subdivide_midpoint(...)`, `mesh.subdivide_loop(...)`.
- Mesh simplification:
  - Vertex clustering: `mesh.simplify_vertex_clustering(...)`
  - Mesh decimation: `mesh.simplify_quadric_decimation(...)`
- Connected components: `mesh.cluster_connected_triangles()`

```python
import sys
import os
import copy

# Add the directory containing 'examples' to the Python path
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)  # Parent directory
sys.path.append(parent_directory)

import open3d as o3d
from examples import open3d_example as o3dex
import numpy as np

## -- Load and Check Properties

# Download data
dataset = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(dataset.path) # ../models/KnotMesh.ply

print(mesh)
# Open3D provides direct memory access to these fields via numpy
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

## --  Visualize a mesh

print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")
o3d.visualization.draw_geometries([mesh])
print("A mesh with no normals and no colors does not look good.")

## -- Surface normal estimation

# Rendering is much better with normals
print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])

## -- Crop mesh using Numpy slicing

print("We make a partial mesh of only the first half triangles.")
# Make a copy
mesh1 = copy.deepcopy(mesh)
# Vector3iVector: Convert int32 numpy array of shape (n, 3) to Open3D format
# https://www.open3d.org/docs/release/python_api/open3d.utility.html#open3d-utility
# Take 1/2 of triangle normals and triangles
mesh1.triangles = o3d.utility.Vector3iVector(
    np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
print(mesh1.triangles)
o3d.visualization.draw_geometries([mesh1])

## -- Paint mesh

print("Painting the mesh")
mesh1.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([mesh1])

## -- Check properties

def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(o3dex.edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(o3dex.edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(o3dex.edges_to_lineset(mesh, edges, (1, 0, 1)))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

check_properties('Knot', o3dex.get_knot_mesh())
#check_properties('Moebius', o3d.geometry.TriangleMesh.create_moebius(twists=1))
check_properties("non-manifold edge", o3dex.get_non_manifold_edge_mesh())
check_properties("non-manifold vertex", o3dex.get_non_manifold_vertex_mesh())
check_properties("open box", o3dex.get_open_box_mesh())
check_properties("intersecting_boxes", o3dex.get_intersecting_boxes_mesh())

## -- Mesh filtering

# - Average Filtering

# Add noise to vertices in Numpy
print('create noisy mesh')
mesh_in = o3dex.get_knot_mesh()
vertices = np.asarray(mesh_in.vertices)
noise = 5
vertices += np.random.uniform(0, noise, size=vertices.shape)
# Convert Numpy to O3D format
mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
mesh_in.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_in])

# Average filter
# The simplest filter is the average filter.
# A given vertex v_i is given by the average of the adjacent vertices N.
print('filter with average with 1 iteration')
mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

print('filter with average with 5 iterations')
mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

# - Laplacian

# Normalized weights that relate to the distance of the neighboring vertices
# The problem with the average and Laplacian filter is that they lead to a shrinkage of the triangle mesh
print('filter with Laplacian with 10 iterations')
mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=10)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

print('filter with Laplacian with 50 iterations')
mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

# - Taubin filter

# The problem with the average and Laplacian filter is that they lead to a shrinkage of the triangle mesh
# The application of two Laplacian filters with different strength parameters can prevent the mesh shrinkage
print('filter with Taubin with 10 iterations')
mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=10)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

print('filter with Taubin with 100 iterations')
mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=100)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

## -- Sampling mesh surfaces with points

mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
# Uniform sampling: fast, but can lead to clusters of points
pcd = mesh.sample_points_uniformly(number_of_points=500)
o3d.visualization.draw_geometries([pcd])

mesh = o3dex.get_bunny_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
# Uniform sampling: fast, but can lead to clusters of points
pcd = mesh.sample_points_uniformly(number_of_points=50000)
o3d.visualization.draw_geometries([pcd])

# Uniform sampling can yield clusters of points on the surface, 
# while a method called Poisson disk sampling can evenly distribute the points on the surface
# by eliminating redundant (high density) samples.
# We have 2 options to provide the initial point cloud to remove from
# 1) Default via the parameter init_factor: 
# The method first samples uniformly a point cloud from the mesh 
# with init_factor x number_of_points and uses this for the elimination.
mesh = o3d.geometry.TriangleMesh.create_sphere()
pcd = mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5)
o3d.visualization.draw_geometries([pcd])
# 2) One can provide a point cloud and pass it to the sample_points_poisson_disk method.
# Then, this point cloud is used for elimination.
pcd = mesh.sample_points_uniformly(number_of_points=2500)
pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
o3d.visualization.draw_geometries([pcd])

mesh = o3dex.get_bunny_mesh()
pcd = mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=5)
o3d.visualization.draw_geometries([pcd])

pcd = mesh.sample_points_uniformly(number_of_points=50000)
pcd = mesh.sample_points_poisson_disk(number_of_points=10000, pcl=pcd)
o3d.visualization.draw_geometries([pcd])

## -- Mesh subdivision

# In mesh subdivision we divide each triangle into a number of smaller triangles
# In the simplest case, we compute the midpoint of each side per triangle
# and divide the triangle into four smaller triangles: subdivide_midpoint.
mesh = o3d.geometry.TriangleMesh.create_box()
mesh.compute_vertex_normals()
print(
    f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
mesh = mesh.subdivide_midpoint(number_of_iterations=1)
print(
    f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

# Another subdivision method: [Loop1987]
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
print(
    f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
mesh = mesh.subdivide_loop(number_of_iterations=2)
print(
    f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

mesh = o3dex.get_knot_mesh()
mesh.compute_vertex_normals()
print(
    f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
mesh = mesh.subdivide_loop(number_of_iterations=1)
print(
    f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

## -- Mesh simplification

# - Vertex clustering

mesh_in = o3dex.get_bunny_mesh()
mesh_in.compute_vertex_normals()
print(
    f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh_in])

# The vertex clustering method pools all vertices that fall
# into a voxel of a given size to a single vertex
# Parameters 
# - contraction: how the vertices are pooled; o3d.geometry.SimplificationContraction.Average 
# computes a simple average.
# - voxel_size
voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
print(f'voxel_size = {voxel_size:e}')
mesh_smp = mesh_in.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Average)
print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh_smp])

# Now, the voxel size is 2x
voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
print(f'voxel_size = {voxel_size:e}')
mesh_smp = mesh_in.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Average)
print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh_smp])

# - Mesh decimation

# We select a single triangle that minimizes an error metric and removes it.
# This is repeated until a required number of triangles is achieved.
# Stopping criterium: target_number_of_triangles 
mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=6500)
print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh_smp])

mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=1700)
print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh_smp])

## -- Connected components

# Spurious triangles added randomly scattered
print("Generate data")
mesh = o3dex.get_bunny_mesh().subdivide_midpoint(number_of_iterations=2)
vert = np.asarray(mesh.vertices)
min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
for _ in range(30):
    cube = o3d.geometry.TriangleMesh.create_box()
    cube.scale(0.005, center=cube.get_center())
    cube.translate(
        (
            np.random.uniform(min_vert[0], max_vert[0]),
            np.random.uniform(min_vert[1], max_vert[1]),
            np.random.uniform(min_vert[2], max_vert[2]),
        ),
        relative=False,
    )
    mesh += cube
mesh.compute_vertex_normals()
print("Show input mesh")
o3d.visualization.draw_geometries([mesh])

# Cluster connected components:
# We can compute the connected components of triangles, i.e., the clusters of triangles which are connected.
# This is useful in image/3D model reconstruction
print("Cluster connected triangles")
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)

print("Show mesh with small clusters removed")
mesh_0 = copy.deepcopy(mesh)
triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
mesh_0.remove_triangles_by_mask(triangles_to_remove)
o3d.visualization.draw_geometries([mesh_0])

print("Show largest cluster")
mesh_1 = copy.deepcopy(mesh)
largest_cluster_idx = cluster_n_triangles.argmax()
triangles_to_remove = triangle_clusters != largest_cluster_idx
mesh_1.remove_triangles_by_mask(triangles_to_remove)
o3d.visualization.draw_geometries([mesh_1])
```

## 4. Transformations



## 5. Rest of Modules



## Authorship

I compiled this guide following and modifying the cited resources, so most of it is not a creative original work of mine.

Mikel Sagardia, 2024.  
No guarantees.  
