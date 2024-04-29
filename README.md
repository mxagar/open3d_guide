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
  - [Setup](#setup)
  - [1. Introduction and File IO](#1-introduction-and-file-io)
  - [2. Point Clouds](#2-point-clouds)
  - [Authorship](#authorship)

## Setup

Install in a Python environment:

```bash
pip install open3d
```

I have copied the used models to [`models/`](./models/). Some of them come from [mxagar/tool_guides/pcl](https://github.com/mxagar/tool_guides/tree/master/pcl), i.e., they are PCD files from PCL &mdash; Open3D supports bith ASCII and binary PCDs, as well as PLYs, among others.

The guide is organized in notebooks, contained and named chronologically in the folder [`notebooks/`](./notebooks/).

## 1. Introduction and File IO

Notebook: [`01_Intro_File_IO.ipynb`](./notebooks/01_Intro_File_IO.ipynb).

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/file_io.html](https://www.open3d.org/docs/latest/tutorial/Basic/file_io.html).

Summary of contents:

- Load pointclouds, meshes, images
- Visualize in new window and in notebook
- Save files with desired formats
- Download models from the internet with `o3d.data`: [https://www.open3d.org/docs/release/python_api/open3d.data.html](https://www.open3d.org/docs/release/python_api/open3d.data.html)

```python
import open3d as o3d
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
import open3d as o3d
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

## Authorship

I compiled this guide following and modifying the cited resources, so most of it is not a creative original work of mine.

Mikel Sagardia, 2024.  
No guarantees.  
