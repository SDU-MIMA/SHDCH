import numpy as np


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calc_map(qB, rB, query_L, retrieval_L):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    query_L = np.array(query_L)
    retrieval_L = np.array(retrieval_L)
    query_L = query_L[:, 8:len(query_L[0])]  # 第二层的label
    retrieval_L = retrieval_L[:, 8:len(retrieval_L[0])]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map


def calc_top_n(qB, rB, query_L, retrieval_L, n):

    num_query = query_L.shape[0]
    query_L = np.array(query_L)
    retrieval_L = np.array(retrieval_L)
    query_L = query_L[:, 8:len(query_L[0])]  # 第二层的label
    retrieval_L = retrieval_L[:, 8:len(retrieval_L[0])]
    res = 0
    topn = []
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if (tsum == 0):
            continue;
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        topn.append([iter,ind[0:n] + 3000])
        print('iter: ', iter, '    ind:   ',ind[0:n])
        gnd = gnd[ind]
        gnd = gnd[0:n]
        nsum = np.sum(gnd)
        # print(nsum)
        res += nsum / n
    res = res / num_query
    return topn


def calc_first_n(qB, rB, query_L, retrieval_L, n):
    num_query = query_L.shape[0]
    res = 0
    for iter in range(num_query):
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        print('iter: ', iter, '   ind :   ', ind[iter])
        if ind[iter] < n:
            res = res + 1
    res = res / num_query
    return res


if __name__ == '__main__':
    qB = np.array([[1, -1, 1, 1],
                   [-1, -1, -1, 1],
                   [1, 1, -1, 1],
                   [1, 1, 1, -1]])
    rB = np.array([[1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [-1, -1, 1, -1],
                   [1, 1, -1, -1],
                   [-1, 1, -1, -1],
                   [1, 1, -1, 1]])
    query_L = np.array([[0, 1, 0, 0],
                        [1, 1, 0, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 1]])
    retrieval_L = np.array([[1, 0, 0, 1],
                            [1, 1, 0, 0],
                            [0, 1, 1, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0]])

    map = calc_map(qB, rB, query_L, retrieval_L)
    print(map)

