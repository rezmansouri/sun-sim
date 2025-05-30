#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:55:33 2018

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import scipy.io as sio 
import numpy as np
import glob
from .vtk import read_vtk
import math, multiprocessing, os

abspath = os.path.abspath(os.path.dirname(__file__))


def S3_normalize(data, norm_method='SD', mean=None, std=None, mi=None, ma=None):
    """

    Parameters
    ----------
    data : Nx1 numpy array,
        DESCRIPTION.
    norm_method : TYPE, optional
        DESCRIPTION. The default is 'SD'.
    mean : TYPE, optional
        DESCRIPTION. The default is None.
    std : TYPE, optional
        DESCRIPTION. The default is None.
    mi : TYPE, optional
        DESCRIPTION. The default is None.
    ma : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    if norm_method=='SD':
        data = data - np.median(data)
        data = data/np.std(data)
        
        index = np.where(data < -3)[0]
        data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
        index = np.where(data > 3)[0]
        data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))
        
        data = data/np.std(data)
        index = np.where(data < -3)[0]
        data[index] = -3 - (1 - np.exp(3 - np.abs(data[index])))
        index = np.where(data > 3)[0]
        data[index] = 3 + (1 - np.exp(3 - np.abs(data[index])))
        
    elif norm_method=='MinMax':
        raise NotImplementedError('NotImplementedError')
    elif norm_method=='Gaussian':
        data = (data - data.mean())/data.std()
    elif norm_method=='PriorGaussian':
        assert mean is not None and std is not None, "PriorGaussian needs prior mean and std"
        data = (data - mean)/std
    elif norm_method=='PriorMinMax':
        assert mi is not None and ma is not None, "PriorMinMax needs prior min and max"
        data = (data - mi)/(ma - mi) * 2. - 1.
    else:
        raise NotImplementedError('NotImplementedError')
        
    return data


def Get_neighs_order(rotated=0):
    neigh_orders_163842 = get_neighs_order(163842, rotated)
    neigh_orders_40962 = get_neighs_order(40962, rotated)
    neigh_orders_10242 = get_neighs_order(10242, rotated)
    neigh_orders_2562 = get_neighs_order(2562, rotated)
    neigh_orders_642 = get_neighs_order(642, rotated)
    neigh_orders_162 = get_neighs_order(162, rotated)
    neigh_orders_42 = get_neighs_order(42, rotated)
    neigh_orders_12 = get_neighs_order(12, rotated)
    
    return neigh_orders_163842, neigh_orders_40962, neigh_orders_10242,\
        neigh_orders_2562, neigh_orders_642, neigh_orders_162, neigh_orders_42, neigh_orders_12
  
def get_neighs_order(n_vertex, rotated=0):
    adj_mat_order = sio.loadmat(abspath +'/neigh_indices/adj_mat_order_'+ \
                                str(n_vertex) +'_rotated_' + str(rotated) + '.mat')
    adj_mat_order = adj_mat_order['adj_mat_order']
    neigh_orders = np.zeros((len(adj_mat_order), 7))
    neigh_orders[:,0:6] = adj_mat_order-1
    neigh_orders[:,6] = np.arange(len(adj_mat_order))
    neigh_orders = np.ravel(neigh_orders).astype(np.int64)
    
    return neigh_orders


def Get_swin_matrices_2order():
    def get_matrices(num_vertices):
        count_matrix = np.load(abspath + "/transformer_matrices/count_2order_%s_vertices.npy" % num_vertices)
        reverse_matrix = np.load(abspath + "/transformer_matrices/reverse_2order_%s_vertices.npy" % num_vertices)
        top_matrix = np.load(abspath + "/transformer_matrices/top_patches_2order_%s_vertices.npy" % num_vertices)
        down_matrix = np.load(abspath + "/transformer_matrices/down_patches_2order_%s_vertices_2.npy" % num_vertices)
        return {"count": count_matrix.astype(np.int32), "reverse": reverse_matrix.astype(np.int32), "top": top_matrix.astype(np.int32), "down":down_matrix.astype(np.int32)}
    matrices_162 = get_matrices("162")
    matrices_642 = get_matrices("642")
    matrices_2562 = get_matrices("2562")
    matrices_10242 = get_matrices("10242")
    matrices_40962 = get_matrices("40962")
    return matrices_40962, matrices_10242, matrices_2562, matrices_642, matrices_162

	
# def Get_swin_matrices_4order():
#     def get_matrices(vertices):
#         count_matrix = np.load("./transformer_matrices/count_4order_%s_vertices.npy" % vertices)
#         reverse_matrix = np.load("./transformer_matrices/reverse_4order_%s_vertices.npy" % vertices)
#         top_matrix = np.load("./transformer_matrices/top_patches_4order_%s_vertices.npy" % vertices)
#         down_matrix = np.load("./transformer_matrices/down_patches_4order_%s_vertices_2.npy" % vertices)
#         return {"count": count_matrix, "reverse": reverse_matrix, "top": top_matrix, "down":down_matrix}
#     matrices_642 = get_matrices("642")
#     matrices_2562 = get_matrices("2562")
#     matrices_10242 = get_matrices("10242")
#     matrices_40962 = get_matrices("40962")
#     return matrices_40962, matrices_10242, matrices_2562, matrices_642 


def Get_upconv_index(rotated=0):
    
    upconv_top_index_163842, upconv_down_index_163842 = get_upconv_index(abspath+'/neigh_indices/adj_mat_order_163842_rotated_' + str(rotated) + '.mat')
    upconv_top_index_40962, upconv_down_index_40962 = get_upconv_index(abspath+'/neigh_indices/adj_mat_order_40962_rotated_' + str(rotated) + '.mat')
    upconv_top_index_10242, upconv_down_index_10242 = get_upconv_index(abspath+'/neigh_indices/adj_mat_order_10242_rotated_' + str(rotated) + '.mat')
    upconv_top_index_2562, upconv_down_index_2562 = get_upconv_index(abspath+'/neigh_indices/adj_mat_order_2562_rotated_' + str(rotated) + '.mat')
    upconv_top_index_642, upconv_down_index_642 = get_upconv_index(abspath+'/neigh_indices/adj_mat_order_642_rotated_' + str(rotated) + '.mat')
    upconv_top_index_162, upconv_down_index_162 = get_upconv_index(abspath+'/neigh_indices/adj_mat_order_162_rotated_' + str(rotated) + '.mat')
    
    #TODO: return tuples of each level
    return upconv_top_index_163842, upconv_down_index_163842, upconv_top_index_40962, upconv_down_index_40962, upconv_top_index_10242, upconv_down_index_10242,  upconv_top_index_2562, upconv_down_index_2562,  upconv_top_index_642, upconv_down_index_642, upconv_top_index_162, upconv_down_index_162


def get_upconv_index(order_path):  
    adj_mat_order = sio.loadmat(order_path)
    adj_mat_order = adj_mat_order['adj_mat_order']
    adj_mat_order = adj_mat_order -1
    nodes = len(adj_mat_order)
    next_nodes = int((len(adj_mat_order)+6)/4)
    upconv_top_index = np.zeros(next_nodes).astype(np.int64) - 1
    for i in range(next_nodes):
        upconv_top_index[i] = i * 7 + 6
    upconv_down_index = np.zeros((nodes-next_nodes) * 2).astype(np.int64) - 1
    for i in range(next_nodes, nodes):
        raw_neigh_order = adj_mat_order[i]
        parent_nodes = raw_neigh_order[raw_neigh_order < next_nodes]
        assert(len(parent_nodes) == 2)
        for j in range(2):
            parent_neigh = adj_mat_order[parent_nodes[j]]
            index = np.where(parent_neigh == i)[0][0]
            upconv_down_index[(i-next_nodes)*2 + j] = parent_nodes[j] * 7 + index
    
    return upconv_top_index, upconv_down_index


def get_upsample_order(n_vertex):
    n_last = int((n_vertex+6)/4)
    neigh_orders = get_neighs_order(abspath+'/neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_0.mat')
    neigh_orders = neigh_orders.reshape(n_vertex, 7)
    neigh_orders = neigh_orders[n_last:,:]
    row, col = (neigh_orders < n_last).nonzero()
    assert len(row) == (n_vertex - n_last)*2, "len(row) == (n_vertex - n_last)*2, error!"
    
    u, indices, counts = np.unique(row, return_index=True, return_counts=True)
    assert len(u) == n_vertex - n_last, "len(u) == n_vertex - n_last, error"
    assert u.min() == 0 and u.max() == n_vertex-n_last-1, "u.min() == 0 and u.max() == n_vertex-n_last-1, error"
    assert (indices == np.asarray(list(range(n_vertex - n_last))) * 2).sum() == n_vertex - n_last, "(indices == np.asarray(list(range(n_vertex - n_last))) * 2).sum() == n_vertex - n_last, error"
    assert (counts == 2).sum() == n_vertex - n_last, "(counts == 2).sum() == n_vertex - n_last, error"
    
    upsample_neighs_order = neigh_orders[row, col]
    
    return upsample_neighs_order  



def get_par_fs_lookup_table():
    """ lookup_table for parcellatiion label,
        copy from freesurfer/FreeSurferColorLUT.txt
    """
    
    lookup_table_vec = np.array([25,  5,  25,
                                 25 ,100 , 40,
                                 125 ,100, 160,
                                 100 , 25,   0,
                                 120  ,70,  50,
                                 220  ,20, 100,
                                 220,  20,  10,
                                 180, 220, 140,
                                 220,  60, 220,
                                 180,  40, 120,
                                 140,  20, 140,
                                 20  ,30 ,140,
                                 35 , 75,  50,
                                 225, 140, 140,
                                 200,  35,  75,
                                 160, 100,  50,
                                 20 ,220 , 60,
                                 60 ,220 , 60,
                                 220, 180, 140,
                                 20 ,100 , 50,
                                 220,  60,  20,
                                 120, 100,  60,
                                 220,  20,  20,
                                 220, 180, 220,
                                 60 , 20 ,220,
                                 160, 140, 180,
                                 80 , 20, 140,
                                 75 , 50, 125,
                                 20 ,220, 160,
                                 20 ,180, 140,
                                 140, 220, 220,
                                 80 ,160 , 20,
                                 100,   0, 100,
                                 70 , 20, 170,
                                 150, 150, 200,
                                 255, 192 , 32 ])
    lookup_table_vec = lookup_table_vec.reshape([36,3])
    
    lookup_table_scalar = np.array([1639705,
                                    2647065,
                                    10511485,
                                    6500,
                                    3294840,
                                    6558940,
                                    660700,
                                    9231540,
                                    14433500,
                                    7874740,
                                    9180300,
                                    9182740,
                                    3296035,
                                    9211105,
                                    4924360,
                                    3302560,
                                    3988500,
                                    3988540,
                                    9221340,
                                    3302420,
                                    1326300,
                                    3957880,
                                    1316060,
                                    14464220,
                                    14423100,
                                    11832480,
                                    9180240,
                                    8204875,
                                    10542100,
                                    9221140,
                                    14474380,
                                    1351760,
                                    6553700,
                                    11146310,
                                    13145750,
                                    2146559])
    
    lookup_table_name =['unknown',
                        'bankssts' ,
                        'caudalanteriorcingulate',
                        'caudalmiddlefrontal',
                        'corpuscallosum',
                        'cuneus',
                        'entorhinal',
                        'fusiform' ,
                        'inferiorparietal' ,
                        'inferiortemporal' ,
                        'isthmuscingulate' ,
                        'lateraloccipital' ,
                        'lateralorbitofrontal' ,
                        'lingual' ,
                        'medialorbitofrontal' ,
                        'middletemporal' ,
                        'parahippocampal'  ,
                        'paracentral' ,
                        'parsopercularis' ,
                        'parsorbitalis' ,
                        'parstriangularis' ,
                        'pericalcarine' ,
                        'postcentral' ,
                        'posteriorcingulate' ,
                        'precentral' ,
                        'precuneus',
                        'rostralanteriorcingulate' ,
                        'rostralmiddlefrontal' ,
                        'superiorfrontal' ,
                        'superiorparietal' ,
                        'superiortemporal' ,
                        'supramarginal' ,
                        'frontalpole',
                        'temporalpole' ,
                        'transversetemporal', 
                        'insula']
    
    return lookup_table_vec, lookup_table_scalar, lookup_table_name




def convert_par_fs_vec_to_par_int(parc_vec):
    """
    Convert parcellation label from rgb vector to single number from 0 to 35

    Parameters
    ----------
    par_vec : numpy array
        size: N x 3
        type: int.

    Returns
    -------
    None.

    """
    assert parc_vec.shape[1] == 3, "size not correct"
    a = get_par_36_to_fs_vec()
    parc_int = np.where((parc_vec[:, np.newaxis, :] == a).all(2))[1]
    return parc_int

def convert_par_fs_int_to_par_vec(parc_int):
    """
    Convert parcellation label from int [0, 35] to rgb vector 

    Parameters
    ----------
    par_vec : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    a = get_par_36_to_fs_vec()
    return a[parc_int]


def get_par_35_to_fs_vec():
    """ Preprocessing for parcellatiion label """
        
    label_36_to_35 = []
    with open(abspath+'/neigh_indices/ROI_36_TO_NAMIC35.txt', "r") as f:
        for x in f:
            label_36_to_35.append(int(x.split()[-1]))
    label_36_to_35 = np.asarray(label_36_to_35)
    
    label_36_to_vec = []
    with open(abspath+'/neigh_indices/FScolortable.txt') as f:
        data=f.readlines()  #逐行读取txt并存成list。每行是list的一个元素，数据类型为str
        for i in range(len(data)):
            for j in range(len(list(data[0].split()))):   #len(list(data[0].split()))为数据列数
                label_36_to_vec.append(int(data[i].split()[j]))
    label_36_to_vec = np.asarray(label_36_to_vec)
    label_36_to_vec = np.reshape(label_36_to_vec,(36, 5))
    label_36_to_vec =  label_36_to_vec[:,1:4]
    
    return label_36_to_vec[label_36_to_35-1]



def get_orthonormal_vectors(n_ver, rotated=0):
    """
    get the orthonormal vectors
    
    n_vec: int, number of vertices, 42,162,642,2562,10242,...
    rotated: 0: original, 1: rotate 90 degrees along y axis, 2: then rotate 90 degrees along z axis
    return orthonormal matrix, shape: n_vec * 3 * 2
    """
    assert type(n_ver) is int, "n_ver, the number of vertices should be int type"
    assert n_ver in [42,162,642,2562,10242,40962,163842], "n_ver, the number of vertices should be the one of [42,162,642,2562,10242,40962,163842]"
    assert rotated in [0, 1, 2], "rotated should be in [0, 1, 2]"
   
    template = read_vtk(abspath+'/neigh_indices/sphere_'+str(n_ver)+'_rotated_'+str(rotated)+'.vtk')
    vertices = template['vertices'].astype(np.float64)
    
    x_0 = np.argwhere(vertices[:,0]==0)
    y_0 = np.argwhere(vertices[:,1]==0)
    inter_ind = np.intersect1d(x_0, y_0)
    
    En_1 = np.cross(np.array([0,0,1]), vertices)
    En_1[inter_ind] = np.array([1,0,0])
    En_2 = np.cross(vertices, En_1)
    
    En_1 = En_1/np.repeat(np.sqrt(np.sum(En_1**2, axis=1))[:,np.newaxis], 3, axis=1)  # normalize to unit orthonormal vector
    En_2 = En_2/np.repeat(np.sqrt(np.sum(En_2**2, axis=1))[:,np.newaxis], 3, axis=1)  # normalize to unit orthonormal vector
    En = np.transpose(np.concatenate((En_1[np.newaxis,:], En_2[np.newaxis,:]), 0), (1,2,0))
    
    return En

def get_patch_indices(n_vertex):
    """
    return all the patch indices and weights
    """
    indices_files = sorted(glob.glob(abspath+'/neigh_indices/*_indices.npy'))
    weights_files = sorted(glob.glob(abspath+'/neigh_indices/*_weights.npy'))
    
    assert len(indices_files) == len(weights_files), "indices files should have the same number as weights number"
    assert len(indices_files) == 163842, "Indices should have dimension 163842 "
    
    indices = [x.split('/')[-1].split('_')[0] for x in indices_files]
    weights = [x.split('/')[-1].split('_')[0] for x in weights_files]
    assert indices == weights, "indices are not consistent with weights!"
    
    indices = [int(x) for x in indices]
    weights = [int(x) for x in weights]
    assert indices == weights, "indices are not consistent with weights!"
    
    indices = np.zeros((n_vertex, 4225, 3)).astype(np.int32)
    weights = np.zeros((n_vertex, 4225, 3))
    
    for i in range(n_vertex):
        indices_file = abspath+'/neigh_indices/'+ str(i) + '_indices.npy'
        weights_file = abspath+'/neigh_indices/'+ str(i) + '_weights.npy'
        indices[i,:,:] = np.load(indices_file)
        weights[i,:,:] = np.load(weights_file)
    
    return indices, weights
        

def get_vertex_dis(n_vertex):
    vertex_dis_dic = {42: 54.6,
                      162: 27.5,
                      642: 13.8,
                      2562: 6.9,
                      10242: 3.4,
                      40962: 1.7,
                      163842: 0.8}
    return vertex_dis_dic[n_vertex]


def check_intersect_vertices_worker(vertices, faces, top_k):
    intersect = []
    for i in range(len(faces)):
        # print(i)
        face = faces[i,:]
        face_vert = vertices[face,:]
        orig_vertex_1 = face_vert[0]
        orig_vertex_2 = face_vert[1]
        orig_vertex_3 = face_vert[2]
        
        dis_0 = np.linalg.norm(vertices - orig_vertex_1, axis=1)
        ind_0 = np.argpartition(dis_0, top_k)[0:top_k]
        dis_1 = np.linalg.norm(vertices - orig_vertex_2, axis=1)
        ind_1 = np.argpartition(dis_1, top_k)[0:top_k]
        dis_2 = np.linalg.norm(vertices - orig_vertex_3, axis=1)
        ind_2 = np.argpartition(dis_2, top_k)[0:top_k]
        ind = np.intersect1d(ind_0, ind_1)
        ind = np.intersect1d(ind, ind_2)
        
        assert len(ind) > len(vertices)/6.0, "extremly ugly face" + str(i) + "-th face!"
       
        normal = np.cross(orig_vertex_1-orig_vertex_3, orig_vertex_2-orig_vertex_3)    # normals of the face
        if (normal == np.array([0,0,0])).all():
            intersect.append([i, 0])
            continue
        
        # use formula p(x) = <p1,n>/<x,n> * x to calculate the intersection with the triangle face
        ratio = np.sum(orig_vertex_1 * normal)/np.sum(vertices[ind,:] * normal, axis=1)
        P = np.repeat(ratio[:,np.newaxis], 3, axis=1) * vertices[ind,:]  # intersection points
        
        area_BCP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_2-P), axis=1)/2.0
        area_ACP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_1-P), axis=1)/2.0
        area_ABP = np.linalg.norm(np.cross(orig_vertex_2-P, orig_vertex_1-P), axis=1)/2.0
        area_ABC = np.linalg.norm(normal)/2.0
        
        tmp = area_BCP + area_ACP + area_ABP - area_ABC
        
        candi = []
        candi.append((tmp <= 1e-5).nonzero()[0])
        
        candi = ind[candi]
        for t in face:
            assert t in candi, "t not in candi, error"
        pending_del = []
        for t in face:
            pending_del.append(np.argwhere(candi==t)[0])
        candi = np.delete(candi, pending_del, 0)
        
        for k in range(len(candi)):
             intersect.append([i, candi[k]])
       
    return intersect
    
def check_intersect_vertices(vertices, faces):
    """
    vertices: N * 3, numpy array, float 64
    faces: (N*2-4) * 3, numpy array, int 64
    """
    
    assert vertices.shape[1] == 3, "vertices size not right"
    assert faces.shape[1] == 3, "faces size not right"
    # assert 2*len(vertices)-4 == len(faces), "vertices are not consistent with faces."
    
    vertices = vertices.astype(np.float64)
    vertices = vertices / np.linalg.norm(vertices, axis=1)[:,np.newaxis]  # normalize to 1
    top_k = int(len(vertices)/3.0)
    
    """ multiple processes method: 163842: 9.6s, 40962: 2.8s, 10242: 1.0s, 2562: 0.28s """
    pool = multiprocessing.Pool()
    cpus = multiprocessing.cpu_count()
    faces_num_per_cpu = math.ceil(faces.shape[0]/cpus)
    results = []
    
    for i in range(cpus):
        results.append(pool.apply_async(check_intersect_vertices_worker, 
                                        args=(vertices, faces[i*faces_num_per_cpu:(i+1)*faces_num_per_cpu,:], top_k)))

    pool.close()
    pool.join()

    intersect = []
    for i in range(cpus):
        intersect = intersect + results[i].get()
    
    intersect = np.asarray(intersect)
    print("Num of intersect tris:", len(intersect))
    return intersect.size == 0
