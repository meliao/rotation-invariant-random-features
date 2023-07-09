from typing import List, Tuple
import logging
import os
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import numba

from src.DataSet import ShapeNetEncoding

@numba.jit(nopython=True)
def _find_triangle_area(coords: np.ndarray) -> np.float32:
    """
    coords has shape (3, 3)
    
    return value is a scalar
    """
    ab = coords[0] - coords[1]
    bc = coords[1] - coords[2]
    
    inner_prod = np.dot(ab, bc)
    
    ab_len = np.sqrt(np.dot(ab, ab))
    bc_len = np.sqrt(np.dot(bc, bc))

    if ab_len == 0 or bc_len == 0:
        return 0.
    
    cos_theta = inner_prod / (ab_len * bc_len)
    
    # The next few lines are to avoid numerical precision errors that 
    # return something like |cos_theta| = 1.00000000000002 and throw an error in sqrt()
    if cos_theta < -1.:
        cos_theta = -1.
    elif cos_theta > 1.:
        cos_theta = 1.
    
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    out = ab_len * bc_len * sin_theta / 2

    return out

@numba.jit(nopython=True)
def _vertex_arr_to_areas(vertex_arr: np.ndarray) -> np.ndarray:
    out = np.zeros(vertex_arr.shape[0])
    for i in range(vertex_arr.shape[0]):
        x = _find_triangle_area(vertex_arr[i])
        out[i] = x
    return out


# @numba.jit(nopython=True)
def _center_and_normalize_point_cloud(arr: np.ndarray) -> np.ndarray:
    """
    This is meant to operate on point clouds. 
    arr has shape (n_deltas, 3)
    return value has shape (n_deltas, 3)
    """
    out = arr - np.mean(arr, axis=0)
    return out / np.max(np.abs(out))
    
    
def _off_point_cloud_to_vertex_arr(fp: str) -> Tuple[np.ndarray, np.ndarray]:
    lst_of_points = []
    set_of_vertex_tuples = set()
    with open(fp, 'r') as f:
        header = f.readline().strip() # header line
        if header == 'OFF':
            _ = f.readline() # second metadata line sometimes but not always appears

        for line in f:
            line_lst = line.split()
            if len(line_lst) == 3:
                float_arr = np.empty(3, dtype=np.float32)
                for i in range(3):
                    float_arr[i] = float(line_lst[i])
                lst_of_points.append(float_arr)
            else:
                vertex_lst = line.split()[1:]
                set_of_vertex_tuples.add("_".join(vertex_lst))
                
    out_arr_points = np.array(lst_of_points)
    out_arr_points = _center_and_normalize_point_cloud(out_arr_points)

    
    out_arr_vertices = _rearrange_points_and_vertices(out_arr_points, set_of_vertex_tuples)

    # This is responding to the IndexError in _rearrange_points_and_vertices
    if out_arr_vertices is None:
        return np.empty((0, 3)), np.empty((0, 3, 3))
    else:
        return out_arr_points, out_arr_vertices

def _rearrange_points_and_vertices(arr_points: np.ndarray, vertices_set: set) -> np.ndarray:
    """
    arr_points has shape (n_deltas, 3)
    vertices_set has length (n_faces)
    
    return value has shape (n_faces, 3, 3)
    """
        
    arr_vertices = np.full((len(vertices_set), 3), np.nan, dtype=np.int32)
    
    for i, s in enumerate(vertices_set):
        indices_lst = [int(x) for x in s.split('_')]
        arr_vertices[i] = np.array(indices_lst)


    n_polygons, _ = arr_vertices.shape
    
    out_arr_vertices = np.full((n_polygons, 3, 3), np.nan)

    try:
                
        for i, vertex_indices in enumerate(arr_vertices):
            out_arr_vertices[i] = arr_points[vertex_indices]

        return out_arr_vertices
        
    except IndexError:
        return None

class ShapeNetObject:
    def __init__(self, vertex_arr: np.ndarray, areas: np.ndarray) -> None:
        """
        vertex_arr has shape (n_polygons, 3, 3)
        areas has shape (n_polygons,)
        """
        self.vertex_arr = vertex_arr
        self.n_polygons = vertex_arr.shape[0]
        
        self.vertex_areas = areas.flatten()

        assert self.vertex_areas.sum() != 0.
        self.vertex_areas_normalized = self.vertex_areas / self.vertex_areas.sum()

        # self.vertex_areas_larger = self.vertex_areas / self.vertex_areas.min()

    def sample_polygons_2(self, n: int) -> np.ndarray:
        """
        return value has shape (n, 3)
        """
        polygons_to_sample = np.random.choice(np.arange(self.n_polygons), 
                                              n,
                                              replace=True, 
                                              p=self.vertex_areas_normalized)
        out = np.full((n, 3), np.nan)
        
        for i, p in enumerate(polygons_to_sample):
            r_1 = np.sqrt(np.random.uniform(0, 1))
            r_2 = np.random.uniform(0, 1)
            
            A = self.vertex_arr[p, 0]
            B = self.vertex_arr[p, 1]
            C = self.vertex_arr[p, 2]
            
            out[i] = ((1 - r_1) * A 
                      + (r_1 * (1 - r_2)) * B
                      + (r_2 * r_1) * C)
        
        return _center_and_normalize_point_cloud(out)


    def sample_polygons(self, n: int) -> np.ndarray:
        """
        This method returns n samples from the surface of the mesh. It is not a 
        uniform sampling.
        
        This method is used in the SPHNet data pre-processesing, and some of the code is 
        borrowed from this source: 
        https://github.com/adrienPoulenard/SPHnet/blob/d30e341aaddae4d45ea537ec4787f0e72b6463c1/SPHnet/utils/pointclouds_utils.py#L481
        
        return value has shape (n, 3)
        """
        # out = np.full((n, 3), np.nan)

        # n_samples_per_face = np.ceil(n * self.vertex_areas).astype(np.int32)
        # if n_samples_per_face.sum() < n:
        n_samples_per_face = np.ceil(n / self.n_polygons * np.ones_like(self.vertex_areas)).astype(np.int32)
        # print(self.vertex_areas.shape)
        # print(n_samples_per_face[:5], np.sum(n_samples_per_face))
        floor_num = np.sum(n_samples_per_face) - n
        floor_indices = np.random.choice(np.arange(self.n_polygons), floor_num, replace=True)
        for idx in floor_indices:
            n_samples_per_face[idx] -= 1
        # while floor_num > 0:
        #     indices = np.where(n_samples_per_face > 0)[0]
        #     floor_indices = np.random.choice(indices, floor_num, replace=True)
        #     n_samples_per_face[floor_indices] -= 1
        #     floor_num = np.sum(n_samples_per_face) - n
        # print(n_samples_per_face[:5], np.sum(n_samples_per_face))
    

        n_samples = np.sum(n_samples_per_face)
        # assert n_samples == n, n_samples
        sample_face_idx = np.zeros((n_samples, ), dtype=np.int32)
        acc = 0
        for face_idx, _n_sample in enumerate(n_samples_per_face):
            sample_face_idx[acc: acc + _n_sample] = face_idx
            acc += _n_sample
        r = np.random.rand(n_samples, 2)
        A = self.vertex_arr[sample_face_idx, 0]
        B = self.vertex_arr[sample_face_idx, 1]
        C = self.vertex_arr[sample_face_idx, 2]
        # A = vertices[faces[sample_face_idx, 0], :]
        # B = vertices[faces[sample_face_idx, 1], :]
        # C = vertices[faces[sample_face_idx, 2], :]
        P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + np.sqrt(r[:,0:1]) * r[:,1:] * C

        assert P.shape[0] == n, P.shape[0]
        return P
        


    def to_disk(self, fp: str) -> np.ndarray:
        
        out_dd = {'areas': self.vertex_areas,
                 'vertex_arr': self.vertex_arr}
        
        io.savemat(fp, out_dd)
        
    @classmethod
    def from_disk(cls, fp: str) -> None:
        in_dd = io.loadmat(fp)
        return cls(in_dd['vertex_arr'], in_dd['areas'])

    @classmethod
    def from_vertex_arr(cls, vertex_arr: np.ndarray) -> None:

        n_polygons, _, _ = vertex_arr.shape

        areas = _vertex_arr_to_areas(vertex_arr)
        assert areas.shape[0] == n_polygons
        assert areas.size == n_polygons
        assert areas.sum() != 0.
        return cls(vertex_arr, areas)
            

def _load_modelnet_class(class_dir: str, n_deltas: int, load_n: int) -> List[np.ndarray]:
    out = []
    file_lst = os.listdir(class_dir)
    if load_n is not None:
        file_lst = file_lst[:load_n]
    for d in file_lst:
        fp = os.path.join(class_dir, d)
        if os.path.isfile(fp):
            s = ShapeNetObject.from_disk(fp)
            try:
                out.append(s.sample_polygons(n_deltas))
            except ValueError:
                logging.warning("Having trouble with fp %s", fp)
                logging.warning("s.areas_normalized.shape, %s", s.vertex_areas_normalized.shape)
        else:
            logging.warning("Can't find file: %s", fp)
    return out

def create_shapenet_dset(data_dir: str,
                            dset_type: str,
                            n: int=None,
                            max_n_deltas: int=500) -> Tuple[np.ndarray]:
    point_clouds = []
    labels = []

    if n is not None:
        n_load_each_class = n // 40
    else:
        n_load_each_class = None

    class_dir_lst = os.listdir(data_dir)
    class_dir_lst.sort()

    for i, class_str in enumerate(class_dir_lst):
        class_dir = os.path.join(data_dir, class_str, dset_type)
        x = _load_modelnet_class(class_dir, max_n_deltas, n_load_each_class)
        point_clouds.extend(x)
        labels.extend([i] * len(x))

    perm = np.random.permutation(len(point_clouds))
    point_clouds_out = []
    labels_out = []
    for i in range(len(point_clouds)):
        perm_idx = perm[i]
        point_clouds_out.append(point_clouds[perm_idx])
        labels_out.append(labels[perm_idx])
    
    return point_clouds_out, labels_out

def points_to_shapenet_encoding(points: List[np.ndarray], 
                                labels: List[int], 
                                max_L: int, 
                                n_radial_params: int, 
                                max_radius: float,
                                bump_width: float) -> ShapeNetEncoding:
    
    max_n_deltas = np.max([i.shape[0] for i in points])
    logging.info("Found max_n_deltas: %s", max_n_deltas)
    n_samples = len(points)
    coords_cart = np.zeros((n_samples, max_n_deltas, 3))
    charges_out = np.zeros((n_samples, max_n_deltas))
    
    labels_out = np.array(labels)
    
    for i in range(n_samples):
        x = points[i]
        n_deltas = x.shape[0]
        coords_cart[i, :n_deltas] = x
        
        charges_out[i, :n_deltas] = np.ones(n_deltas)
        
    radial_params = np.linspace(0., max_radius, n_radial_params)
    return ShapeNetEncoding(coords_cart, max_L, radial_params, labels=labels_out, bump_width=bump_width)

def points_to_shapenet_cos_encoding(points: List[np.ndarray], 
                                labels: List[int], 
                                max_L: int, 
                                n_radial_params: int, 
                                max_radius: float) -> ShapeNetEncoding:
    
    max_n_deltas = np.max([i.shape[0] for i in points])
    n_samples = len(points)
    coords_cart = np.zeros((n_samples, max_n_deltas, 3))
    charges_out = np.zeros((n_samples, max_n_deltas))
    
    labels_out = np.array(labels)
    
    for i in range(n_samples):
        x = points[i]
        n_deltas = x.shape[0]
        coords_cart[i, :n_deltas] = x
        
        charges_out[i, :n_deltas] = np.ones(n_deltas)
        
    # radial_params = np.arange(1, n_radial_params+1) * np.pi
    assert n_radial_params <= 10
    radial_params = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]) * np.pi
    radial_params = radial_params[:n_radial_params]
    return ShapeNetEncoding(coords_cart, max_L, radial_params, labels=labels_out, poly_rad_funcs_bool=True)