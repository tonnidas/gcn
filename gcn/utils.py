import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# input = a string of dataset name
# output = 
#   adj        = a adjacency matrix of size 2708, 2708 for cora
#   features   = a feature matrix of size 2708, 1433 for cora
#   y_train    = a binary matrix of size 2708, 7 for training cluster masks/labels (only first 140 rows have a label "1" among 7 columns for cora)
#   y_val      = a binary matrix of size 2708, 7 for training cluster masks/labels (only mid 500 rows have a label "1" among 7 columns for cora)
#   y_test     = a binary matrix of size 2708, 7 for training cluster masks/labels (only last 1000 rows have a label "1" among 7 columns for cora)
#   train_mask = an array of 2708 True/False indicating which nodes are training, 0 to 139th 'True' for cora
#   val_mask   = an array of 2708 True/False indicating which nodes are training, 140 to 639th 'True' for cora
#   test_mask  = an array of 2708 True/False indicating which nodes are training, 1708 to 2708th 'True' for cora
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    # debugPrint(test_idx_range)  # For cora, an array of numbers ranging 1708 to 2707

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # debugPrint(labels.shape, labels[-1], features.todense().shape, adj.todense().shape)
    # debugPrint(labels)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # debugPrint("Test =", len(idx_test), idx_test[0], idx_test[-1])
    # debugPrint("Train =", len(idx_train), idx_train[0], idx_train[-1])
    # debugPrint("Val =", len(idx_val), idx_val[0], idx_val[-1])

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # debugPrint("Train =", train_mask, train_mask.shape)
    # debugPrint("Val =", val_mask, val_mask.shape)
    # debugPrint("Test =", test_mask, test_mask.shape)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # debugPrint("y_train =", y_train, y_train[0], y_train.shape)
    # debugPrint("y_val =", y_val, y_val[140], y_val.shape)
    # debugPrint(features.todense()[0], features.todense().shape)
    # debugPrint("load_data", train_mask)

    # debugPrint("adj_sparse =", adj.todense(), adj.todense().shape, "features_sparse = ", features.todense(), features.todense().shape)
    # debugPrint("y_train =", y_train, y_train.shape, "y_val =", y_val, y_val.shape, "y_test =", y_test, y_test.shape)
    # debugPrint("train_mask =", train_mask, train_mask.shape, sum(train_mask), "val_mask =", val_mask, val_mask.shape, sum(val_mask), "test_mask =", test_mask, test_mask.shape, sum(test_mask))

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        # debugPrint(coords, values, shape)
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# input = a feature matrix of size 2708, 1433 for cora
# output = a normalied sparse feature matrix
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # debugPrint(rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    # debugPrint(r_inv)
    r_inv[np.isinf(r_inv)] = 0.
    # debugPrint(r_inv)
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # debugPrint(r_mat_inv.todense())
    # debugPrint(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)   # making sparse
    # debugPrint(adj)

    rowsum = np.array(adj.sum(1))
    # debugPrint(rowsum)
    
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # debugPrint("d_mat_inv_sqrt = ", d_mat_inv_sqrt.todense())
    r = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    # debugPrint("normalized = ", r.todense())
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # a = np.ones((2, 4))
    # b = np.zeros((2, 4))
    # adj = np.vstack((a, b))
    # debugPrint(adj)
    # debugPrint(adj + sp.eye(adj.shape[0]))
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # debugPrint(sparse_to_tuple(adj_normalized))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    # debugPrint("chebyshev_polynomials start")
    # debugPrint(adj.todense().shape)
    
    # a = np.ones((2, 4))
    # b = np.zeros((2, 4))
    # adj = np.vstack((a, b))

    adj_normalized = normalize_adj(adj)
    # debugPrint(adj_normalized.todense())
    
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    # debugPrint(laplacian.todense())

    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    # debugPrint(largest_eigval)
    
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    # debugPrint(scaled_laplacian.todense())

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    # debugPrint("second_last = ", t_k[-2].todense(), "last =", t_k[-1].todense())
    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
        # debugPrint("Inside loop second_last =", t_k[-2].todense(), "last =", t_k[-1].todense())

    # debugPrint("sparse_to_tuple = ", sparse_to_tuple(sp.csr_matrix(np.ones((2,4)))) )
    # debugPrint(sparse_to_tuple(t_k))
    # for i in sparse_to_tuple(t_k):
    #     debugPrint(i.toarray().todense())
    # debugPrint("chebyshev_polynomials end")

    # We get 4 matrix from here for cora
    return sparse_to_tuple(t_k)

def k_hop_distance(adj, k, dataset):
    """Calculate k hop adjacency up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating k-hop adj up to order {}...".format(k))

    t_k = list()
    t_k.append(sp.csr_matrix(adj))

    sum = mul = adj

    for i in range(1, k+1):
        mul = np.dot(mul, adj)
        sum = sum + mul

        tmp = sum.toarray()
        tmp = np.where(tmp > 0, 1, 0)
        tmp = tmp - np.diag(tmp.diagonal())

        w =  (k + 1 - i) / (k + 1)
        tmp = w * tmp

        t_k.append(sp.csr_matrix(tmp))
    
    print("End calculating k-hop adj up to order {}...".format(k))
    
    return sparse_to_tuple(t_k)

def debugPrint(*args):
    """Print variable names and values with line number."""
    import inspect, re

    caller = inspect.getframeinfo(inspect.stack()[1][0])
    title = f"##### {caller.filename}:{caller.lineno} #####"
    print(f"\n{title:#^100}")

    import traceback
    for line in traceback.format_stack()[:-2]:
        print(line.strip())
    print(f"{'':.^100}")

    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    vnames = map(str.strip, r.split(','))

    for _, (var,val) in enumerate(zip(vnames, args)):
        if var.startswith("'") or var.startswith("\""): 
            print(val)
        else: 
            print(f"{var} = {val}")
    
    print(f"{'':=^100}\n")

# adj = np.zeros((4,4))
# adj[0][1] = adj[0][3] = adj[1][0] = adj[1][2] = adj[2][1] = adj[3][0] = 1
# x = k_hop_distance(sp.csr_matrix(adj), 2, 'Cora')
# debugPrint(x)
