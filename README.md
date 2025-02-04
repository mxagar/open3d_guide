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

## Table of Contents

- [Open3D Guide](#open3d-guide)
  - [Table of Contents](#table-of-contents)
  - [Setup and File Structure](#setup-and-file-structure)
    - [How to Use the Repository Contents](#how-to-use-the-repository-contents)
    - [Known Issues](#known-issues)
  - [1. Introduction and File IO](#1-introduction-and-file-io)
  - [2. Point Clouds](#2-point-clouds)
  - [3. Meshes](#3-meshes)
  - [4. Transformations](#4-transformations)
  - [5. Rest of Modules](#5-rest-of-modules)
    - [RGBD Images and Odometry](#rgbd-images-and-odometry)
    - [Visualization](#visualization)
    - [KDTree](#kdtree)
    - [ICP Registration](#icp-registration)
    - [Working with Numpy](#working-with-numpy)
    - [Tensor](#tensor)
    - [Voxelization](#voxelization)
  - [5. Use Cases](#5-use-cases)
    - [Capturing 3D Models with Your Phone](#capturing-3d-models-with-your-phone)
    - [3D-2D-3D Projection of a Scene](#3d-2d-3d-projection-of-a-scene)
  - [Authorship](#authorship)

## Setup and File Structure

If you have already a dedicated Python environment, just install Open3D via pip:

```bash
# I created this guide using version 0.18 (Windows 11) and 0.16.1 (Apple M1)
pip install open3d
```

If you don't have a dedicated Python environment yest, a quick recipe to getting started by using [conda](https://conda.io/projects/conda/en/latest/index.html) is the following:

```bash
# Set proxy, if required

# Create environment, e.g., with conda, to control Python version
conda create -n 3d python=3.10 pip
conda activate 3d

# Install pip-tools
python -m pip install -U pip-tools

# Generate pinned requirements.txt
pip-compile requirements.in

# Sync/Install (missing) pinned requirements
pip-sync requirements.txt

# Alternatively: you can install pinned requirements with pip, as always
python -m pip install -r requirements.txt

# If required, add new dependencies to requirements.in and sync
# i.e., update environment
pip-compile requirements.in
pip-sync requirements.txt

# Optional: if you's like to export you final conda environment config
conda env export > environment.yml
# Optional: If required, to delete the conda environment
conda remove --name 3d --all
```

### How to Use the Repository Contents

The repository consists of three main folders:

- [`notebooks/`](./notebooks): Personal notebooks based mainly on the [**Open3D Basic Tutorial**](https://www.open3d.org/docs/latest/tutorial/Basic/index.html); the sections below contain code summaries from those notebooks.
- [`examples/`](./examples): Official example files from [https://github.com/isl-org/Open3D/tree/main/examples/python](https://github.com/isl-org/Open3D/tree/main/examples/python).
- [`models/`](./models): Several models both from Open3D repositories as well as from [mxagar/tool_guides/pcl](https://github.com/mxagar/tool_guides/tree/master/pcl), i.e., PCD files from PCL.

**Sections 1-4** contain the most important and basic topics necessary to start using Open3D: File I/O, Point clouds, Meshes and Transformations. Each of the topics has

- a dedicated notebook in [`notebooks/`](./notebooks)
- and a code summary taken from the associated notebook.

**Section 5** contains the rest of the topics, which have also a dedicated notebook, but

- they don't have a dedicated section
- and their code is mostly only in the notebook.

**Section 6** contains specific use cases (or complex examples/solution recipes) I will be adding with time.

### Known Issues

:warning: Mac/Apple M1 wheels (latest version to date 0.16.1) cause an OpenGL error when we launch the visualization; if the code is in a script it is not that big of an issue from the UX perspective, but if the code is on a notebook, the kernel crashes and it needs to be restarted.

- Github issue: [isl-org/Open3D/issues/1673](https://github.com/isl-org/Open3D/issues/1673).

:warning: OpenGL GPU support is not provided for AMD chips (Open3D 0.18); instead of using `open3d.visualization.draw`, we should use `open3d.visualization.draw_geometries`, which is a basic rendering scheme.

- Github issue: [isl-org/Open3D/issues/4852](https://github.com/isl-org/Open3D/issues/4852)

:warning: Headless rendering is not possible for Windows (Open3D 0.18).

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

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/transformation.html](https://www.open3d.org/docs/latest/tutorial/Basic/transformation.html).

Summary of contents:

- Translate: `mesh.translate()`
- Rotate: `mesh.rotate()`
  - `get_rotation_matrix_from_xyz`
  - `get_rotation_matrix_from_axis_angle`
  - `get_rotation_matrix_from_quaternion`
- Scale: `mesh.scale()`
- General (homogeneous) transformation: `mesh.transform()`

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

## -- Translate

# Factory function which creates a mesh coordinate frame
# Check other factory functions with help(o3d.geometry.TriangleMesh)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# Translate mesh and deepcopy
mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
print(f'Center of mesh: {mesh.get_center()}')
# The method get_center returns the mean of the TriangleMesh vertices.
# That means that for a coordinate frame created at the origin [0,0,0],
# get_center will return [0.05167549 0.05167549 0.05167549]
print(f'Center of mesh tx: {mesh_tx.get_center()}')
print(f'Center of mesh ty: {mesh_ty.get_center()}')
o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])

# The method takes a second argument relative that is by default set to True.
# If set to False, the center of the geometry is translated directly to the position specified
# in the first argument.
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_mv = copy.deepcopy(mesh).translate((2, 2, 2), relative=False)
print(f'Center of mesh: {mesh.get_center()}')
print(f'Center of translated mesh: {mesh_mv.get_center()}')
o3d.visualization.draw_geometries([mesh, mesh_mv])

## -- Rotate

# We pass a rotation matrix R to rotate
# There are many conversion functions to get R
# - Convert from Euler angles with get_rotation_matrix_from_xyz (where xyz can also be of the form yzx, zxy, xzy, zyx, and yxz)
# - Convert from Axis-angle representation with get_rotation_matrix_from_axis_angle
# - Convert from Quaternions with get_rotation_matrix_from_quaternion
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_r = copy.deepcopy(mesh)
R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
mesh_r.rotate(R, center=(0, 0, 0))
o3d.visualization.draw_geometries([mesh, mesh_r])

# The function rotate has a second argument center that is by default set to True.
# This indicates that the object is first centered prior to applying the rotation
# and then moved back to its previous center. 
# If this argument is set to False, then the rotation will be applied directly, 
# such that the whole geometry is rotated around the coordinate center.
# This implies that the mesh center can be changed after the rotation.
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_r = copy.deepcopy(mesh).translate((2, 0, 0))
mesh_r.rotate(mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4)),
              center=(0, 0, 0))
o3d.visualization.draw_geometries([mesh, mesh_r])

## -- Scale

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_s = copy.deepcopy(mesh).translate((2, 0, 0))
mesh_s.scale(0.5, center=mesh_s.get_center())
o3d.visualization.draw_geometries([mesh, mesh_s])

# The scale method also has a second argument center that
# is set to True by default. If it is set to False,
# then the object is not centered prior to scaling such that
# the center of the object can move due to the scaling operation
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_s = copy.deepcopy(mesh).translate((2, 1, 0))
mesh_s.scale(0.5, center=(0, 0, 0))
o3d.visualization.draw_geometries([mesh, mesh_s])

## -- Transform

# Open3D also supports a general transformation 
# defined by a 4×4 homogeneous transformation matrix using the method transform.
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
T = np.eye(4)
T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
T[0, 3] = 1
T[1, 3] = 1.3
print(T)
mesh_t = copy.deepcopy(mesh).transform(T)
o3d.visualization.draw_geometries([mesh, mesh_t])

```

## 5. Rest of Modules

### RGBD Images and Odometry

Sources: 

- [https://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html](https://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html).
- [https://www.open3d.org/docs/latest/tutorial/Basic/rgbd_odometry.html](https://www.open3d.org/docs/latest/tutorial/Basic/rgbd_odometry.html).

Notebook: [`05_RGBD_Images.ipynb`](./notebooks/05_RGBD_Images.ipynb)

Summary of contents:

- Redwood dataset: RGB, Depth and Co.
- RGBD Odometry
  - Camera parameters: 
    - `o3d.camera.PinholeCameraIntrinsic`
    - `o3d.io.read_pinhole_camera_intrinsic`
  - Read RGBD images:
    - `o3d.geometry.RGBDImage.create_from_color_and_depth`
    - `o3d.geometry.PointCloud.create_from_rgbd_image`
  - Compute odometry from two RGBD image pairs: `o3d.pipelines.odometry.compute_rgbd_odometry`
    - `o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()`
    - `o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()`
  - Visualize RGBD image pairs

### Visualization

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/visualization.html](https://www.open3d.org/docs/latest/tutorial/Basic/visualization.html).

Notebook: [`06_Visualization.ipynb`](./notebooks/06_Visualization.ipynb).

Summary of contents:

- Function `draw_geometries`
- Store viewpoint: `Ctrl+C`
- Geometry primitives:
  - `o3d.geometry.TriangleMesh.create_box`
  - `o3d.geometry.TriangleMesh.create_sphere`
  - `o3d.geometry.TriangleMesh.create_cylinder`
  - `o3d.geometry.TriangleMesh.create_coordinate_frame`
- Drawing line sets: `o3d.geometry.LineSet`

### KDTree

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html](https://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html).

Notebook: [`07_KDTree.ipynb`](./notebooks/07_KDTree.ipynb).

Summary of contents:

- Build KDTree from point cloud and find & visualize nearest points of a point
  - `pcd_tree = o3d.geometry.KDTreeFlann(pcd)`: create a KDTree
  - `pcd_tree.search_knn_vector_3d`: given a point, find the N nearest ones
  - `pcd_tree.search_radius_vector_3d`: given a point, find the ones within a radius R

### ICP Registration

Sources: 

- ICP: [https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html](https://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html).
- Global registrations: [https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html](https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html).
- Colored point cloud registrations: [https://www.open3d.org/docs/latest/tutorial/Advanced/colored_pointcloud_registration.html](https://www.open3d.org/docs/latest/tutorial/Advanced/colored_pointcloud_registration.html).

Notebook: [`08_ICP_Registration.ipynb`](./notebooks/08_ICP_Registration.ipynb).

Summary of contents:

- Prepare Input Data: Source and Target
- Point-to-point ICP
  - `o3d.pipelines.registration.registration_icp`
  - `o3d.pipelines.registration.TransformationEstimationPointToPoint()`
- Point-to-plane ICP
  - `o3d.pipelines.registration.TransformationEstimationPointToPlane()`

> This tutorial demonstrates the ICP (Iterative Closest Point) registration algorithm. It has been a mainstay of geometric registration in both research and industry for many years. The input are two point clouds and an initial transformation that roughly aligns the source point cloud to the target point cloud. The output is a refined transformation that tightly aligns the two point clouds. A helper function draw_registration_result visualizes the alignment during the registration process. In this tutorial, we show two ICP variants, the point-to-point ICP and the point-to-plane ICP [Rusinkiewicz2001].
>
> Both [ICP registration](https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html) and [Colored point cloud registration](https://www.open3d.org/docs/latest/tutorial/Advanced/colored_pointcloud_registration.html) are known as **local registration methods** because they rely on a rough alignment as initialization. Prior to a local registration we need some kind of [**global registration**](https://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html). This family of algorithms do not require an alignment for initialization. They usually produce less tight alignment results and are used as initialization of the local methods.

**This notebook deals with the local registration approach ICP**: we give a source and target point cloud already aligned and we obtain a more tight alignment.

**IMPORTANT: The point-to-plane ICP algorithm uses point normals; we need to estimate them if they are not available**.

### Working with Numpy

Sources: 

- Tutorial: [https://www.open3d.org/docs/latest/tutorial/Basic/working_with_numpy.html](https://www.open3d.org/docs/latest/tutorial/Basic/working_with_numpy.html).
- Conversion interfaces: [https://www.open3d.org/docs/latest/python_api/open3d.utility.html#open3d-utility](https://www.open3d.org/docs/latest/python_api/open3d.utility.html#open3d-utility).

Notebook: [`09_Numpy.ipynb`](./notebooks/09_Numpy.ipynb).

All data structures in Open3D are natively compatible with a NumPy buffer.

Common interfaces to use O3D and Numpy interchangeably are:

- `o3d.utility.Vector3dVector`; more opetions in [open3d.utility](https://www.open3d.org/docs/latest/python_api/open3d.utility.html#open3d-utility).
- `np.asarray(pcd.points)`.

The tutorial in the section notebook generates a variant of sync function using NumPy and visualizes the function using Open3D.

### Tensor

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/tensor.html](https://www.open3d.org/docs/latest/tutorial/Basic/tensor.html).

Notebook: [`10_Tensor.ipynb`](./notebooks/10_Tensor.ipynb).

> Tensor is a “view” of a data Blob with shape, stride, and a data pointer. It is a multidimensional and homogeneous matrix containing elements of single data type. It is used in Open3D to perform numerical operations. It supports GPU operations as well.

Summary of contents:

- Tensor creation
- Properties of a tensor
- Copy & device transfer
- Data types
- Type casting
- Numpy I/O with direct memory map
- PyTorch I/O with DLPack memory map
- Binary element-wise operations
- Unary element-wise operations
- Reduction
- Slicing, indexing, getitem, and setitem
- Advanced indexing
- Logical operations
- Comparision Operations
- Nonzero operations

### Voxelization

Source: 

- Tutorial: [https://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html](https://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html).
- [`open3d.geometry.VoxelGrid`](https://www.open3d.org/docs/latest/python_api/open3d.geometry.VoxelGrid.html#open3d.geometry.VoxelGrid).
- [`open3d.geometry.Voxel`](https://www.open3d.org/docs/latest/python_api/open3d.geometry.Voxel.html#open3d.geometry.Voxel)

Notebook: [`11_Voxelization.ipynb`](./notebooks/11_Voxelization.ipynb).

Summary of contents:

- Voxelize from triangle mesh: `o3d.geometry.VoxelGrid.create_from_triangle_mesh`
- Voxels and their data: `voxel_grid.get_voxels()`
- Voxel cubes for visualization
- Create a Voxelmap from the VoxelGrid: A cartesian occupancy map
- Voxelize from a point cloud: `o3d.geometry.VoxelGrid.create_from_point_cloud`
- Inclusion test: `voxel_grid.check_if_included`
- Voxel carving


## 5. Use Cases

### Capturing 3D Models with Your Phone

There are many phone apps to capture physical worlds objects which use many technologies:

- Just photos (photogrammetry)
- LIDAR data (e.g., from the new iPhones)
- etc.

One possible and free application is [RealityScan](https://www.unrealengine.com/en-US/realityscan), from [Unreal Engine](https://www.unrealengine.com).

Example capture (photogrammetry) with an iPhone SE (2020, iOS 18.2.1): [`models/ikea_cup_reality_scan_iphone/`](./models/ikea_cup_reality_scan_iphone/). The capture could be easily improved, I think it is not characteristic of the quality achievable with RealityScan; however, it is a good benchmark/example for some applications, because it has reconstruction mistakes, such as holes.

### 3D-2D-3D Projection of a Scene

The notebook [`notebooks/12_3D2D3D_Projections.ipynb`](./notebooks/12_3D2D3D_Projections.ipynb) contains the following topics:

- **Capture RGB and Depth Map Snapshots of a 3D Object**
- **Reconstruct Pointcloud from RGB + Depth Map + Camera Parameters**
- **Viewpoint Optimization**

![Reconstructed Pointcloud](./assets/2d3d_reconstruction.png)

The last topic, **Viewpoint Optimization**, contains an approximative heuristic to get a set of viewpoints that optimally cover the complete model.

Indeed it is not enough with being able to run a 3D-2D-3D projection, but we need to know how to navigate the scene and get the optimum set of viewpoints to capture the scene!

The presented method is not thoroughly debugged!

Idea:

- Get an initial set of viewpoints by projecting outwards the voxel centers of a coarse voxel grid.
- Create a finer voxel grid and go through the initial set of viewpoints in a loop:
  1. Each loop iteration, compute the priority map: how many fine voxel centers are seen if the camera is set in a viewpoint of the initial set
  2. Pick the viewpoint with highest fine voxel count
  3. Mark all fine voxels as seen
  4. Store viewpoint
  5. Next iteration (step 1)
- Loop finishes when all fine voxel centers have been seen.

## Authorship

I compiled this guide following and modifying the cited resources, so most of it is not a creative original work of mine.

Mikel Sagardia, 2024.  
No guarantees.  
