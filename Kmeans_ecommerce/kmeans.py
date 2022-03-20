import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
from time import time

from PIL import Image
from skimage.io import imread

from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering


def select_centroids(X, k):

    c1_index = np.random.choice(len(X), replace=False)
    c1 = X[c1_index]
    centroids = [c1]
    Other_X = np.delete(X, np.where(X == c1), axis=0)

    for c_id in range(k-1):
        far_index = np.argmax(np.linalg.norm(Other_X-centroids[c_id], axis=1))
        new_centroid = Other_X[far_index]
        centroids.append(new_centroid)
        Other_X = np.delete(Other_X, np.where(Other_X == new_centroid), axis=0)

    return np.array(centroids)


# kmeans function
def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):

    if centroids == None:
        n_iter = 0
        old_cen = np.zeros((k, len(X[0])))
        uniq_x = np.array(list(set([tuple(t) for t in X])))
        centrids_idx = np.random.choice(len(uniq_x), k, replace=False)
        centrids = np.array(uniq_x[centrids_idx])

        while ((np.array(centrids) - np.array(old_cen)).any() > tolerance) and (n_iter < max_iter):
            old_cen = centrids
            dis_all = []
            for i in range(k):
                dis_k = np.sqrt(np.sum((X - [centrids[i] for _ in range(len(X))])**2, axis=1))
                dis_all.append(dis_k)
            label = np.argmin(np.array(dis_all), axis=0)
            new_centr_data = []
            for j in range(k):
                new_centr_data.append(X[np.where(label==j)])
            centrids = [np.sum(cluster, axis=0)/len(cluster) for cluster in new_centr_data]

            n_iter += 1

    if centroids == 'kmeans++':

        n_iter = 0
        centrids = select_centroids(X, k)
        old_cen = np.zeros([k,len(X[0])])

        while ((np.array(centrids) - np.array(old_cen)).any() > tolerance) and (n_iter < max_iter):
            old_cen = centrids
            dis_all = []
            for i in range(k):
                dis_k = np.sqrt(np.sum((X - [centrids[i] for _ in range(len(X))])**2, axis=1))
                dis_all.append(dis_k)
            label = np.argmin(np.array(dis_all), axis=0)
            new_centr_data = []
            for j in range(k):
                new_centr_data.append(X[np.where(label==j)])
            centrids = [np.sum(cluster, axis=0)/max(1,len(cluster)) for cluster in new_centr_data]

            n_iter += 1

    return np.array(centrids), label

def kmeans_steps(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):

    centorid_list = []
    label_list = []
    cluster_list = []

    if centroids == None:
        n_iter = 0
        old_cen = np.zeros((k, len(X[0])))
        uniq_x = np.array(list(set([tuple(t) for t in X])))
        centrids_idx = np.random.choice(len(uniq_x), k, replace=False)
        centrids = np.array(uniq_x[centrids_idx])

        while ((np.array(centrids) - np.array(old_cen)).any() > tolerance) and (n_iter < max_iter):
            old_cen = centrids
            dis_all = []
            for i in range(k):
                dis_k = np.sqrt(np.sum((X - [centrids[i] for _ in range(len(X))])**2, axis=1))
                dis_all.append(dis_k)
            label = np.argmin(np.array(dis_all), axis=0)
            new_centr_data = []
            for j in range(k):
                new_centr_data.append(X[np.where(label==j)])
            centrids = [np.sum(cluster, axis=0)/len(cluster) for cluster in new_centr_data]

            n_iter += 1
            centorid_list.append(np.array(centrids))
            label_list.append(label)
            cluster_list.append(new_centr_data)

    if centroids == 'kmeans++':
        n_iter = 0
        centrids = select_centroids(X, k)
        old_cen = np.zeros([k,len(X[0])])
        label = np.zeros(len(X))

        while ((np.array(centrids) - np.array(old_cen)).any() > tolerance) and (n_iter < max_iter):
            old_cen = centrids
            dis_all = []
            for i in range(k):
                dis_k = np.sqrt(np.sum((X - [centrids[i] for _ in range(len(X))])**2, axis=1))
                dis_all.append(dis_k)
            label = np.argmin(np.array(dis_all), axis=0)
            new_centr_data = []
            for j in range(k):
                new_centr_data.append(X[np.where(label==j)])
            centrids = [np.sum(cluster, axis=0)/max(1,len(cluster)) for cluster in new_centr_data]

            n_iter += 1
            centorid_list.append(np.array(centrids))
            label_list.append(label)
            cluster_list.append(new_centr_data)


    return np.array(centrids), label, centorid_list, label_list, cluster_list



def kmedians_steps(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):

    centorid_list = []
    label_list = []
    cluster_list = []

    if centroids == None:
        n_iter = 0
        old_cen = np.zeros((k, len(X[0])))
        uniq_x = np.array(list(set([tuple(t) for t in X])))
        centrids_idx = np.random.choice(len(uniq_x), k, replace=False)
        centrids = np.array(uniq_x[centrids_idx])

        while ((np.array(centrids) - np.array(old_cen)).any() > tolerance) and (n_iter < max_iter):
            old_cen = centrids
            dis_all = []
            for i in range(k):
                dis_k = np.sqrt(np.sum((X - [centrids[i] for _ in range(len(X))])**2, axis=1))
                dis_all.append(dis_k)
            label = np.argmin(np.array(dis_all), axis=0)
            new_centr_data = []
            for j in range(k):
                new_centr_data.append(X[np.where(label==j)])
            centrids = [np.median(cluster, axis=0) for cluster in new_centr_data]

            n_iter += 1
            centorid_list.append(np.array(centrids))
            label_list.append(label)
            cluster_list.append(new_centr_data)

    if centroids == 'kmeans++':
        n_iter = 0
        centrids = select_centroids(X, k)
        old_cen = np.zeros([k,len(X[0])])
        label = np.zeros(len(X))

        while ((np.array(centrids) - np.array(old_cen)).any() > tolerance) and (n_iter < max_iter):
            old_cen = centrids
            dis_all = []
            for i in range(k):
                dis_k = np.sqrt(np.sum((X - [centrids[i] for _ in range(len(X))])**2, axis=1))
                dis_all.append(dis_k)
            label = np.argmin(np.array(dis_all), axis=0)
            new_centr_data = []
            for j in range(k):
                new_centr_data.append(X[np.where(label==j)])
            centrids = [np.median(cluster, axis=0) for cluster in new_centr_data]

            n_iter += 1
            centorid_list.append(np.array(centrids))
            label_list.append(label)
            cluster_list.append(new_centr_data)

    return np.array(centrids), label, centorid_list, label_list, cluster_list


def in_cross_dist(cluster_list):
    in_group = []
    for iter_d in cluster_list:
        dist_incluster = np.mean([np.mean(pdist(i)) for i in iter_d])
        in_group.append(dist_incluster)

    cross_group = []
    for iter_d in cluster_list:
        dist_crocluster = np.mean([np.mean(cdist(iter_d[0],i)) for i in iter_d])
        cross_group.append(dist_crocluster)
    return in_group, cross_group


def load_glove():
    print('Loading Glove...')
    d = {}
    with open('/Users/caoyanan/Downloads/glove/glove.6B.50d.txt') as f:
        for line in f.readlines():
            linelist = line.split()
            d[linelist[0]] = np.array(linelist[1:], dtype = np.float32)
    print('Finished loading Glove.')
    return d

def doc2vec(words, gloves):
    vectors = []
    for w in words:
        try:
            vectors.append(gloves[w])
        except:
            continue
    centroid = np.mean(vectors, axis=0)
    return centroid

def item_name_vector(item_name, d):
    item_name_vector = []
    for i in range(200):
        item = item_name[i]
        word = item.lower().split(' ')
        vector = doc2vec(word, d)
        if np.isnan(vector).any():
            item_name.drop(index=i)
            continue
        else:
            item_name_vector.append(vector)
    return item_name_vector
