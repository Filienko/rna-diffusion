""" Script for the Correlation Score metric.

Reference:
-----
Ramon Viñas, Helena Andrés-Terré, Pietro Liò, Kevin Bryson,
Adversarial generation of gene expression data,
Bioinformatics, Volume 38, Issue 3, February 2022, Pages 730–737,
https://doi.org/10.1093/bioinformatics/btab035

Original code:
-----
https://github.com/rvinas/adversarial-gene-expression.git

"""

# Imports
import numpy as np


def upper_diag_list(m_: np.array):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    ----
    Parameters:
        m_ (np.array): array of float. Shape=(N, N)
    Returns:
        list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # upper-diagonal matrix
    tril = np.zeros_like(m_) + np.nan
    tril = np.tril(tril)
    m += tril
    m = np.ravel(m)
    return m[~np.isnan(m)]


def pearson_correlation(x: np.array, y: np.array):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    ----
    Parameters:
        x (np.array): Gene matrix 1. Shape=(nb_samples, nb_genes_1)
        y (np.array): Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    Returns:
        Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        S = (a - a_off) / a_std
        S[np.isnan(S)] = (a - a_off)[np.isnan(S)]
        return S

    assert x.shape[0] == y.shape[0]
    x_ = standardize(x)
    y_ = standardize(y)
    return np.dot(x_.T, y_) / x.shape[0]


def gamma_coeff_score(x_test: np.array, x_gen: np.array):
    """
    Compute correlation score for two given expression matrices
    ----
    Parameters:
        x (np.array): matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
        y (np.array): matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    Returns:
        Gamma(D^X, D^Z)
    """
    dists_x = 1 - correlations_list(x_test, x_test)
    dists_y = 1 - correlations_list(x_gen, x_gen)
    gamma_dx_dy = pearson_correlation(dists_x, dists_y)

    return gamma_dx_dy


def correlations_list(x: np.array, y: np.array):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    ----
    Parameters:
        x (np.array): Gene matrix 1. Shape=(nb_samples, nb_genes_1)
        y (np.array): Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    Returns:
        corr_fn (np.array): correlation function taking x and y as inputs
    """
    corr = pearson_correlation(x, y)

    return upper_diag_list(corr)
