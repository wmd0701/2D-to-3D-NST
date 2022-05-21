# reference: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/io/ply_io.py
# PyTorch3D already provides a method save_ply to save mesh in ply format. 
# The reason why I implement a new save_ply method is that the method provided by PyTorch3D does not support per-vertex color.
# This is incredibly ridiculous, since the ply format outperforms the obj format on supporting vertex color.
# What is even more stupid is that in pytorch3d.io there is an intern function _save_ply which supports vertex color, 
# but when calling function _save_ply inside save_ply, the vertex color is set to None. 

import itertools
import struct
import sys
import warnings
from collections import namedtuple
from dataclasses import asdict, dataclass
from io import BytesIO, TextIOBase
from typing import List, Optional, Tuple

import numpy as np
import torch
from iopath.common.file_io import PathManager
from pytorch3d.io.utils import _check_faces_indices, _make_tensor, _open_file, PathOrStr
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds

def _write_ply_header(
    f,
    *,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor],
    verts_normals: Optional[torch.Tensor],
    verts_colors: Optional[torch.Tensor],
    ascii: bool,
    colors_as_uint8: bool,
) -> None:
    """
    Internal implementation for writing header when saving to a .ply file.
    Args:
        f: File object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        verts_colors: FloatTensor of shape (V, 3) giving vertex colors.
        ascii: (bool) whether to use the ascii ply format.
        colors_as_uint8: Whether to save colors as numbers in the range
                    [0, 255] instead of float32.
    """
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert faces is None or not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)
    assert verts_normals is None or (
        verts_normals.dim() == 2 and verts_normals.size(1) == 3
    )
    assert verts_colors is None or (
        verts_colors.dim() == 2 and verts_colors.size(1) == 3
    )

    if ascii:
        f.write(b"ply\nformat ascii 1.0\n")
    elif sys.byteorder == "big":
        f.write(b"ply\nformat binary_big_endian 1.0\n")
    else:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
    f.write(f"element vertex {verts.shape[0]}\n".encode("ascii"))
    f.write(b"property float x\n")
    f.write(b"property float y\n")
    f.write(b"property float z\n")
    if verts_normals is not None:
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")
    if verts_colors is not None:
        color_ply_type = b"uchar" if colors_as_uint8 else b"float"
        for color in (b"red", b"green", b"blue"):
            f.write(b"property " + color_ply_type + b" " + color + b"\n")
    if len(verts) and faces is not None:
        f.write(f"element face {faces.shape[0]}\n".encode("ascii"))
        f.write(b"property list uchar int vertex_index\n")
    f.write(b"end_header\n")

def _save_ply(
    f,
    *,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor],
    verts_normals: Optional[torch.Tensor],
    verts_colors: Optional[torch.Tensor],
    ascii: bool,
    decimal_places: Optional[int] = None,
    colors_as_uint8: bool,
) -> None:
    """
    Internal implementation for saving 3D data to a .ply file.
    Args:
        f: File object to which the 3D data should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        verts_colors: FloatTensor of shape (V, 3) giving vertex colors.
        ascii: (bool) whether to use the ascii ply format.
        decimal_places: Number of decimal places for saving if ascii=True.
        colors_as_uint8: Whether to save colors as numbers in the range
                    [0, 255] instead of float32.
    """
    _write_ply_header(
        f,
        verts=verts,
        faces=faces,
        verts_normals=verts_normals,
        verts_colors=verts_colors,
        ascii=ascii,
        colors_as_uint8=colors_as_uint8,
    )

    if not (len(verts)):
        warnings.warn("Empty 'verts' provided")
        return

    color_np_type = np.ubyte if colors_as_uint8 else np.float32
    verts_dtype = [("verts", np.float32, 3)]
    if verts_normals is not None:
        verts_dtype.append(("normals", np.float32, 3))
    if verts_colors is not None:
        verts_dtype.append(("colors", color_np_type, 3))

    vert_data = np.zeros(verts.shape[0], dtype=verts_dtype)
    vert_data["verts"] = verts.detach().cpu().numpy()
    if verts_normals is not None:
        vert_data["normals"] = verts_normals.detach().cpu().numpy()
    if verts_colors is not None:
        color_data = verts_colors.detach().cpu().numpy()
        if colors_as_uint8:
            vert_data["colors"] = np.rint(color_data * 255)
        else:
            vert_data["colors"] = color_data

    if ascii:
        if decimal_places is None:
            float_str = b"%f"
        else:
            float_str = b"%" + b".%df" % decimal_places
        float_group_str = (float_str + b" ") * 3
        formats = [float_group_str]
        if verts_normals is not None:
            formats.append(float_group_str)
        if verts_colors is not None:
            formats.append(b"%d %d %d " if colors_as_uint8 else float_group_str)
        formats[-1] = formats[-1][:-1] + b"\n"
        for line_data in vert_data:
            for data, format in zip(line_data, formats):
                f.write(format % tuple(data))
    else:
        if isinstance(f, BytesIO):
            # tofile only works with real files, but is faster than this.
            f.write(vert_data.tobytes())
        else:
            vert_data.tofile(f)

    if faces is not None:
        faces_array = faces.detach().cpu().numpy()

        _check_faces_indices(faces, max_index=verts.shape[0])

        if len(faces_array):
            if ascii:
                np.savetxt(f, faces_array, "3 %d %d %d")
            else:
                faces_recs = np.zeros(
                    len(faces_array),
                    dtype=[("count", np.uint8), ("vertex_indices", np.uint32, 3)],
                )
                faces_recs["count"] = 3
                faces_recs["vertex_indices"] = faces_array
                faces_uints = faces_recs.view(np.uint8)

                if isinstance(f, BytesIO):
                    f.write(faces_uints.tobytes())
                else:
                    faces_uints.tofile(f)

def save_ply2(
    f,
    verts: torch.Tensor,
    faces: Optional[torch.LongTensor] = None,
    verts_normals: Optional[torch.Tensor] = None,
    ascii: bool = False,
    decimal_places: Optional[int] = None,
    path_manager: Optional[PathManager] = None,
    verts_colors: Optional[torch.Tensor] = None,
    colors_as_uint8: Optional[bool] = True
) -> None:
    """
    Save a mesh to a .ply file.
    Args:
        f: File (or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals.
        ascii: (bool) whether to use the ascii ply format.
        decimal_places: Number of decimal places for saving if ascii=True.
        path_manager: PathManager for interpreting f if it is a str.
    """

    if len(verts) and not (verts.dim() == 2 and verts.size(1) == 3):
        message = "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if (
        faces is not None
        and len(faces)
        and not (faces.dim() == 2 and faces.size(1) == 3)
    ):
        message = "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if (
        verts_normals is not None
        and len(verts_normals)
        and not (
            verts_normals.dim() == 2
            and verts_normals.size(1) == 3
            and verts_normals.size(0) == verts.size(0)
        )
    ):
        message = "Argument 'verts_normals' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if path_manager is None:
        path_manager = PathManager()

    # verts_colors is actually of shape (1, N, 3)
    if verts_colors is not None:
        verts_colors = verts_colors[0]

    with _open_file(f, path_manager, "wb") as f:
        _save_ply(
            f,
            verts=verts,
            faces=faces,
            verts_normals=verts_normals,
            verts_colors=verts_colors,
            ascii=ascii,
            decimal_places=decimal_places,
            colors_as_uint8=colors_as_uint8,
        )