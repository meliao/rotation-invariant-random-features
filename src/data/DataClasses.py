import numpy as np
# import scipy.special

from src.kernels import (cartesian_to_spherical_batch, rotate_spherical_points_batch, 
                        spherical_harmonics_from_coords,
                        spherical_to_cartesian_batch)
class DataSample:
    """
    Base class for representing a data sample. Coordinates are specified in 
    Cartesian and Polar coordinates. Child classes must implement the 
    following functions:
        from_cartesian_coords() classmethod
    """
    def __init__(self, coords_cart: np.ndarray, coords: np.ndarray, n_points: int) -> None:
        assert coords_cart.shape[0] == n_points
        assert coords.shape[0] == n_points
        self.n_points = n_points
        self.coords_cart = coords_cart
        self.coords = coords


    def get_radii(self) -> np.ndarray:
        return self.coords[:, 0]


    def get_angles_longitude(self) -> np.ndarray:
        return self.coords[:, 1]

    @staticmethod
    def cartesian_to_polar(coords_cart: np.ndarray) -> np.ndarray:
        assert coords_cart.shape[1] == 2
        radii = np.sqrt(np.square(coords_cart[:, 0]) + np.square(coords_cart[:, 1]))
        angles = np.arctan2(coords_cart[:, 1], coords_cart[:, 0])
        return np.vstack([radii, angles]).T


    @staticmethod
    def polar_to_cartesian(coords_polar: np.ndarray) -> np.ndarray:
        assert coords_polar.shape[1] == 2
        x_coords = coords_polar[:, 0] * np.cos(coords_polar[:, 1])
        y_coords = coords_polar[:, 0] * np.sin(coords_polar[:, 1])
        return np.vstack([x_coords, y_coords]).T

    @staticmethod
    def cartesian_to_spherical(coords_cart: np.ndarray) -> np.ndarray:
        """Input is a list of points in 3D cartesian space. 
                [:, 0] x axis
                [:, 1] y axis
                [:, 2] z axis
        Output is the same points in 3D spherical coordinates
                [:, 0] radius
                [:, 1] longituindal angle
                [:, 2] latitudinal angle
        """
        return cartesian_to_spherical_batch(coords_cart)

    @staticmethod
    def spherical_to_cartesian(coords_sphe: np.ndarray) -> np.ndarray:
        """
        Input is the same points in 3D spherical coordinates
                [:, 0] radius
                [:, 1] longituindal angle
                [:, 2] latitudinal angle
        Output is a list of points in 3D cartesian space. 
                [:, 0] x axis
                [:, 1] y axis
                [:, 2] z axis
        """
        return spherical_to_cartesian_batch(coords_sphe)

    def centered(self) -> None:
        """Must be implemented by child classes and return a new object
        """


    def rotate(self, rot: np.ndarray) -> None:
        """Must be implemented by child classes and return a new object
        """


class DataSample2D(DataSample):
    def __init__(self, coords_cart: np.ndarray, coords_polar: np.ndarray, n_points: int) -> None:
        assert coords_cart.shape[1] == 2
        assert coords_polar.shape[1] == 2
        super().__init__(coords_cart, coords_polar, n_points)
        # self.angle_diffs = self.get_angle_diffs()

    @classmethod
    def from_polar_coords(cls, coords_polar: np.ndarray) -> None:
        coords_cart = cls.polar_to_cartesian(coords_polar)
        n_points = coords_polar.shape[0]
        return cls(coords_cart, coords_polar, n_points)

    @classmethod
    def from_cartesian_coords(cls, coords_cart: np.ndarray) -> None:
        coords_polar = cls.cartesian_to_polar(coords_cart)
        n_points = coords_cart.shape[0]
        return cls(coords_cart, coords_polar, n_points)

    def centered(self) -> None:
        new_cart = self.coords_cart.copy()
        for i in range(new_cart.shape[1]):
            new_cart[:, i] = new_cart[:, i] - new_cart[:, i].mean()
        return DataSample2D.from_cartesian_coords(new_cart)

    def rotate(self, rot: np.ndarray) -> None:
        return DataSample2D.from_cartesian_coords(np.matmul(rot, self.coords_cart.T).T)

class DataSample3D(DataSample):
    def __init__(self, coords_cart: np.ndarray, coords_polar: np.ndarray, n_points: int) -> None:
        assert coords_cart.shape[1] == 3
        assert coords_polar.shape[1] == 3
        super().__init__(coords_cart, coords_polar, n_points)
        self.spherical_harmonic_evals = None
        self.spherical_harmonic_max_L = None

    def get_spherical_harmonic_evals(self, max_L: int) -> np.ndarray:
        if self.spherical_harmonic_evals is not None and max_L == self.spherical_harmonic_max_L:
            return self.spherical_harmonic_evals
        else:
            evals = spherical_harmonics_from_coords(self.coords, max_L)
            self.spherical_harmonic_evals = evals
            self.spherical_harmonic_max_L = max_L
            return self.spherical_harmonic_evals

    @classmethod
    def from_spherical_coords(cls, coords_sphe: np.ndarray) -> None:
        coords_cart = cls.spherical_to_cartesian(coords_sphe)
        n_points = coords_sphe.shape[0]
        return cls(coords_cart, coords_sphe, n_points)

    @classmethod
    def from_cartesian_coords(cls, coords_cart: np.ndarray) -> None:
        coords_sphe = cls.cartesian_to_spherical(coords_cart)
        n_points = coords_cart.shape[0]
        return cls(coords_cart, coords_sphe, n_points)

    def get_angles_latitude(self) -> np.ndarray:
        return self.coords[:, 2]

    def centered(self) -> None:
        new_cart = self.coords_cart.copy()
        if new_cart.shape[0] == 0:
            return DataSample3D.from_cartesian_coords(new_cart)
        for i in range(new_cart.shape[1]):
            new_cart[:, i] = new_cart[:, i] - new_cart[:, i].mean()
        return DataSample3D.from_cartesian_coords(new_cart)

    def rotate(self, rot: np.ndarray) -> None:
        return DataSample3D.from_spherical_coords(rotate_spherical_points_batch(self.coords, rot))