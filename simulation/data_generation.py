
import numpy as np
import sys
import itertools
from scipy.sparse import csr_matrix


def generate_G(m, p, seed=0):
    ## generate concept prerequisite graph by Erdos Renyi model

    ## create nodes
    nodes = list(range(m))
    all_coms = np.array(list(itertools.combinations(nodes, 2)))

    ## generate edges from Erdos Renyi model
    np.random.seed(seed)
    trials = np.random.binomial(n=1, p=p, size=len(all_coms))
    E_size = trials.sum()
    edges_index, = np.where(trials == 1) ## the np.where will return a tuple shapes like (a, b)
    edges = all_coms[edges_index]

    G = {'nodes': nodes, 'edges': edges, 'E_size': E_size}

    return G

def generate_edge_matrix(edges, node_num, type='dense'):
    import pdb;pdb.set_trace()
    data = np.ones((edges.shape[0]),dtype='int')
    row,col = edges[:,0] - 1,edges[:,1] - 1
    edge_matrix = csr_matrix((data,(row,col)),shape=(node_num,node_num))
    if type == 'dense':
        edge_matrix = edge_matrix.todense()
    return edge_matrix
    

def generate_course_pair(G, V_size, ik=0, start=10000, replaced=True):

    ## generate background words
    background_words = range(start, start + V_size)

    ## sample the prerequisite relation edges
    edges = G['edges']
    np.random.seed(start+10*ik)
    idx = np.random.choice(G['E_size'], size=lc, replace=replaced)
    sampled_edges = edges[idx, :]
    # print(sampled_edges)

    ## add prerequisite relation words to di and dj
    di = []
    dj = []
    for (cs, ct) in sampled_edges:
        di.append(cs)
        dj.append(ct)

    ## add background words to di and dj
    np.random.seed(start+10*ik+1)
    sampled_bw = np.random.choice(background_words, size=l-lc, replace=replaced)
    di.extend(list(sampled_bw))
    np.random.seed(start+10*ik+2)
    sampled_bw = np.random.choice(background_words, size=l-lc, replace=replaced)
    dj.extend(list(sampled_bw))

    return di, dj



def generate_word_fre_matrix(word_num, course_num, rand_type='poisson'):
    """generate bag_of_word matrix.

    params:
      word_num: number of total words(concepts).
      course_num: number of total courses.
    """
    return np.random.poisson(size=(course_num, word_num), lam=1)

def generate_one_simulation_data(word_num, course_num, p, lab):
    G = generate_G(word_num, p)
    concept_matrix = generate_edge_matrix(G['edges'], node_num=word_num)
    fre_matrix = generate_word_fre_matrix(word_num=word_num, course_num=course_num)
    F = np.dot(np.dot(fre_matrix,concept_matrix),fre_matrix.T)
    import pdb; pdb.set_trace()
    return {'A': concept_matrix, 'X': fre_matrix, 'F':F}
    
    
    



if __name__ == "__main__":
    course_num = 10
    word_num = 50
    p = 0.01
    # E_size = 50
    V_size = word_num
    l = 10
    lc = 2
    k = 3

    G = generate_G(word_num, p)
    print(G)
    # print(G['E_size'])

    courses = list()
    courses_pairs = set()
    for m in range(k):
        # print(m)
        di, dj = generate_course_pair(G, V_size, ik=m)
        print(di)
        print(dj)
        courses.extend([di, dj])
        courses_pairs.add((2*m, 2*m+1))
    print(courses_pairs)
    course_matrix = generate_edge_matrix(G['edges'], node_num=word_num)
    print(course_matrix)
    fre_matrix = generate_word_fre_matrix(word_num=word_num, course_num=course_num)
    generate_one_simulation_data(word_num,course_num,p,lab=1)