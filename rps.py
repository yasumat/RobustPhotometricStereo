#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Photometric Stereo in Python
"""
__author__ = "Yasuyuki Matsushita <yasumat@ist.osaka-u.ac.jp>"
__version__ = "0.1.0"
__date__ = "11 May 2018"

import psutil
import rpsnumerics
import numpy as np
from sklearn.preprocessing import normalize


class RPS(object):
    """
    Robust Photometric Stereo class
    """
    # Choice of solution methods
    L2_SOLVER = 0   # Conventional least-squares
    L1_SOLVER = 1   # L1 residual minimization
    L1_SOLVER_MULTICORE = 2 # L1 residual minimization (multicore)
    SBL_SOLVER = 3  # Sparse Bayesian Learning
    SBL_SOLVER_MULTICORE = 4    # Sparse Bayesian Learning (multicore)
    RPCA_SOLVER = 5    # Robust PCA

    def __init__(self):
        self.M = None   # measurement matrix in numpy array
        self.L = None   # light matrix in numpy array
        self.N = None   # surface normal matrix in numpy array
        self.height = None  # image height
        self.width = None   # image width
        self.foreground_ind = None    # mask (indices of active pixel locations (rows of M))
        self.background_ind = None    # mask (indices of inactive pixel locations (rows of M))

    def load_lighttxt(self, filename=None):
        """
        Load light file specified by filename.
        The format of lights.txt should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.txt
        """
        self.L = psutil.load_lighttxt(filename)

    def load_lightnpy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.L = psutil.load_lightnpy(filename)

    def load_images(self, foldername=None, ext=None):
        """
        Load images in the folder specified by the "foldername" that have extension "ext"
        :param foldername: foldername
        :param ext: file extension
        """
        self.M, self.height, self.width = psutil.load_images(foldername, ext)

    def load_npyimages(self, foldername=None):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        self.M, self.height, self.width = psutil.load_npyimages(foldername)

    def load_mask(self, filename=None):
        """
        Load mask image and set the mask indices
        In the mask image, pixels with zero intensity will be ignored.
        :param filename: filename of the mask image
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        mask = psutil.load_image(filename=filename)
        mask = mask.reshape((-1, 1))
        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        psutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        psutil.save_normalmap_as_npy(filename=filename, normal=self.N, height=self.height, width=self.width)

    def solve(self, method=L2_SOLVER):
        if self.M is None:
            raise ValueError("Measurement M is None")
        if self.L is None:
            raise ValueError("Light L is None")
        if self.M.shape[1] != self.L.shape[1]:
            raise ValueError("Inconsistent dimensionality between M and L")

        if method == RPS.L2_SOLVER:
            self._solve_l2()
        elif method == RPS.L1_SOLVER:
            self._solve_l1()
        elif method == RPS.L1_SOLVER_MULTICORE:
            self._solve_l1_multicore()
        elif method == RPS.SBL_SOLVER:
            self._solve_sbl()
        elif method == RPS.SBL_SOLVER_MULTICORE:
            self._solve_sbl_multicore()
        elif method == RPS.RPCA_SOLVER:
            self._solve_rpca()
        else:
            raise ValueError("Undefined solver")

    def _solve_l2(self):
        """
        Lambertian Photometric stereo based on least-squares
        Woodham 1980
        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        self.N = np.linalg.lstsq(self.L.T, self.M.T, rcond=None)[0].T
        self.N = normalize(self.N, axis=1)  # normalize to account for diffuse reflectance
        if self.background_ind is not None:
            for i in range(self.N.shape[1]):
                self.N[self.background_ind, i] = 0

    def _solve_l1(self):
        """
        Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        A = self.L.T
        self.N = np.zeros((self.M.shape[0], 3))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind

        for index in indices:
            b = np.array([self.M[index, :]]).T
            n = rpsnumerics.L1_residual_min(A, b)
            self.N[index, :] = n.ravel()
        self.N = normalize(self.N, axis=1)

    def _solve_l1_multicore(self):
        """
        Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        from multiprocessing import Pool
        import multiprocessing

        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        p = Pool(processes=multiprocessing.cpu_count()-1)
        normal = p.map(self._solve_l1_multicore_impl, indices)
        if self.foreground_ind is None:
            self.N = np.asarray(normal)
            self.N = normalize(self.N, axis=1)
        else:
            N = np.asarray(normal)
            N = normalize(N, axis=1)
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]

    def _solve_l1_multicore_impl(self, index):
        """
        Implementation of Lambertian Photometric stereo based on sparse regression (L1 residual minimization)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :param index: an index of a measurement (row of M)
        :return: a row vector of surface normal at pixel index specified by "index"
        """
        A = self.L.T
        b = np.array([self.M[index, :]]).T
        n = rpsnumerics.L1_residual_min(A, b)   # row vector of a surface normal at pixel "index"
        return n.ravel()

    def _solve_sbl(self):
        """
        Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        A = self.L.T
        self.N = np.zeros((self.M.shape[0], 3))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind

        for index in indices:
            b = np.array([self.M[index, :]]).T
            n = rpsnumerics.sparse_bayesian_learning(A, b)
            self.N[index, :] = n.ravel()
        self.N = normalize(self.N, axis=1)

    def _solve_sbl_multicore(self):
        """
        Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        from multiprocessing import Pool
        import multiprocessing

        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        p = Pool(processes=multiprocessing.cpu_count()-1)
        normal = p.map(self._solve_sbl_multicore_impl, indices)
        if self.foreground_ind is None:
            self.N = np.asarray(normal)
            self.N = normalize(self.N, axis=1)
        else:
            N = np.asarray(normal)
            N = normalize(N, axis=1)
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(self.N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]

    def _solve_sbl_multicore_impl(self, index):
        """
        Implementation of Lambertian Photometric stereo based on sparse regression (Sparse Bayesian learning)
        Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita, Kiyoharu Aizawa:
        Robust photometric stereo using sparse regression. CVPR 2012: 318-325

        :param index: an index of a measurement (row of M)
        :return: a row vector of surface normal at pixel index specified by "index"
        """
        A = self.L.T
        b = np.array([self.M[index, :]]).T
        n = rpsnumerics.sparse_bayesian_learning(A, b)   # row vector of a surface normal at pixel "index"
        return n.ravel()

    def _solve_rpca(self):
        """
        Photometric stereo based on robust PCA.
        Lun Wu, Arvind Ganesh, Boxin Shi, Yasuyuki Matsushita, Yongtian Wang, Yi Ma:
        Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery. ACCV (3) 2010: 703-717

        :return: None

        Compute surface normal : numpy array of surface normal (p \times 3)
        """
        if self.foreground_ind is None:
            _M = self.M.T
        else:
            _M = self.M[self.foreground_ind, :].T

        A, E, ite = rpsnumerics.rpca_inexact_alm(_M)    # RPCA Photometric stereo

        if self.foreground_ind is None:
            self.N = np.linalg.lstsq(self.L.T, A, rcond=None)[0].T
            self.N = normalize(self.N, axis=1)    # normalize to account for diffuse reflectance
        else:
            N = np.linalg.lstsq(self.L.T, A, rcond=None)[0].T
            N = normalize(N, axis=1)    # normalize to account for diffuse reflectance
            self.N = np.zeros((self.M.shape[0], 3))
            for i in range(self.N.shape[1]):
                self.N[self.foreground_ind, i] = N[:, i]
