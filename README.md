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
```

## 2. Point Clouds

Notebook: [`01_Intro_File_IO.ipynb`](./notebooks/01_Intro_File_IO.ipynb).

Source: [https://www.open3d.org/docs/latest/tutorial/Basic/file_io.html](https://www.open3d.org/docs/latest/tutorial/Basic/file_io.html).

Summary of contents:

- A
- B
- B



## Authorship

I compiled this guide following and modifying the cited resources, so most of it is not a creative original work of mine.

Mikel Sagardia, 2024.  
No guarantees.  
