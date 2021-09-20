import numpy as np
import scipy.spatial

def pairwise_squared_distances(x, v):
    return scipy.spatial.distance.cdist(x, v)**2

def calculate_covariances(x, u, v, m):
    c, n = u.shape
    d = v.shape[1]
    um = u**m

    covariances = np.zeros((c, d, d))

    for i in range(c):
        xv = x - v[:,i]
        uxv = np.sum(np.matmul(um[:, i], np.matmul(xv, xv.T)))
        covariances[i] = uxv/np.sum(um[:, i])

    return covariances

def partition_coefficient(x, u, v, m):
    c, n = u.shape
    return np.square(u).sum()/n

def partition_entropy_coefficient(x, u, v, m):
    c, n = u.shape
    return -(u*np.log(u)).sum()/n

def normalized_partition_coefficient(x, u, v, m):
    c, n = u.shape
    return 1 - c/(c - 1)*(1 - partition_coefficient(x, u, v, m))

def fuzzy_hypervolume(x, u, v, m):
    covariances = calculate_covariances(x, u, v, m)
    return sum(np.sqrt(np.linalg.det(cov)) for cov in covariances)

def fukuyama_sugeno_index(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u**m

    v_mean = v.T.mean(axis=0)

    d2 = pairwise_squared_distances(x, v.T)

    distance_v_mean_squared = np.linalg.norm(v.T - v_mean, axis=1, keepdims=True)**2

    return np.sum(np.matmul(um.T,d2)) - np.sum(np.matmul(um,distance_v_mean_squared))

def xie_beni_index(x, u, v, m):
    n = x.shape[0]
    c = v.shape[0]

    um = u**m

    d2 = pairwise_squared_distances(x, v.T)
    v2 = pairwise_squared_distances(v.T, v.T)

    v2[v2 == 0.0] = np.inf

    return np.sum(np.matmul(um.T,d2))/(n*np.min(v2))

def beringer_hullermeier_index(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    d2 = pairwise_squared_distances(x, v.T)
    v2 = pairwise_squared_distances(v, v.T)

    v2[v2 == 0.0] = np.inf

    V = np.sum(u*d2.T, axis=1)/np.sum(u, axis=1)

    return np.sum(np.matmul((u**m).T,d2))/n*0.5*np.sum(np.outer(V, V)/v2)

def bouguessa_wang_sun_index(x, u, v, m):
    n, d = x.shape
    c = v.shape[0]

    x_mean = x.mean(axis=0)

    covariances = calculate_covariances(x, u, v, m)

    sep = np.einsum("ik,ij->", u**m, np.square(v - x_mean))
    comp = sum(np.trace(covariance) for covariance in covariances)

    return sep/comp
