import collections
import os.path
from pathlib import Path

import networkx as nx
from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

import config
from agent.gman.node2vec import node2vec

from utils import xml_util
from utils.collections_util import bidict
from utils.sumo import sumo_net_util


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())

    return G


def learn_embeddings(walks, dimensions, output_file):
    class TqdmCallback(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 0
            self.pbar = tqdm(desc="Learning embeddings")

        def on_epoch_begin(self, model):
            pass
            #self.pbar.total = model.epochs

        def on_epoch_end(self, model):
            self.epoch += 1
            self.pbar.update(self.epoch)

    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, size=dimensions, window=10, min_count=0, sg=1,
        workers=8, iter=iter, callbacks=[TqdmCallback()])
    model.wv.save_word2vec_format(output_file)

    return


is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = '../data/Adj.txt'
SE_file = '../data/SE.txt'


def process_adjacency_graph_file():

    def read_graph():

        adj_file = os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'original_id_adj.txt')

        with open(adj_file) as handle:
            adjacency_graph_lines = handle.read().splitlines()

        adjacency_graph = collections.defaultdict(list)
        for line in adjacency_graph_lines:
            source, target, weight = line.split(' ')
            adjacency_graph[source].append({target: weight})

        return adjacency_graph

    def get_all_detector_ids(adjacency_graph):
        detector_ids = set()
        for k, vs in adjacency_graph.items():
            for v in vs:
                source, (target, weight) = k, *v.items()
                detector_ids.update({source, target})

        return list(detector_ids)

    def map_detector_ids(original_adjacency_graph):

        adjacency_graph = collections.defaultdict(list)
        for k, vs in original_adjacency_graph.items():
            for v in vs:
                source, (target, weight) = k, *v.items()
                source = detector_id_mapping.inverse[source][0]
                target = detector_id_mapping.inverse[target][0]
                adjacency_graph[source].append({target: weight})

        return adjacency_graph

    def save_adjacency_graph(adjacency_graph):

        space_separated_values = [
            f"{k} {sk} {v}"
            for k, sub_list in adjacency_graph.items()
            for sub_dict in sub_list
            for sk, v in sub_dict.items()
        ]

        file = os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'Adj.txt')

        with open(file, 'w') as handle:
            handle.write('\n'.join(str(line) for line in space_separated_values))

    net_file = os.path.join(config.ROOT_DIR, config.PATH_TO_DATA, config.SCENARIO.NET_FILE)
    net_xml = xml_util.parse_xml(net_file)

    original_adjacency_graph = read_graph()
    detector_ids = get_all_detector_ids(original_adjacency_graph)

    detector_ids = sumo_net_util.sort_detector_ids(net_xml, detector_ids)
    detector_id_mapping = bidict(enumerate(detector_ids))

    adjacency_graph = map_detector_ids(original_adjacency_graph)
    save_adjacency_graph(adjacency_graph)


def generate_static_embedding():
    file = os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'Adj.txt')
    se_file = os.path.join(config.ROOT_DIR, config.PATH_TO_RECORDS, 'SE.txt')

    if Path(se_file).is_file():
        return

    nx_G = read_graph(file)
    G = node2vec.Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks, dimensions, se_file)


if __name__ == "__main__":
    generate_static_embedding()
