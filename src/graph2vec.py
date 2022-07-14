"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
# from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(x) for f in features for x in f.values()] 
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash tables with extracted WL features.
        """
        new_features = [{} for _ in range(len(self.features))]
        for i in range (len(self.features)):
            for node in self.nodes:
                nebs = self.graph.neighbors(node)
                degs = [self.features[i][neb] for neb in nebs]
                features = [str(self.features[i][node])]+sorted([str(deg) for deg in degs])
                features = "_".join(features)
                hash_object = hashlib.md5(features.encode())
                hashing = hash_object.hexdigest()
                new_features[i][node] = hashing
        self.extracted_features = self.extracted_features + [str(x) for f in new_features for x in f.values()]
        return new_features

        # new_features = {}
        # for node in self.nodes:
        #     nebs = self.graph.neighbors(node)
        #     degs = [self.features[neb] for neb in nebs]
        #     features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
        #     features = "_".join(features)
        #     hash_object = hashlib.md5(features.encode())
        #     hashing = hash_object.hexdigest()
        #     new_features[node] = hashing
        # self.extracted_features = self.extracted_features + list(new_features.values())
        # return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])


    f_names = [f_name for f_name in sorted(data.keys()) if f_name[:8] == 'features']
    if len(f_names) > 0:
        # add feature name suffix 
        features = [{int(k): str(v) + 'f'+ str(fi) for k, v in data[f_name].items()} for fi, f_name in enumerate(f_names)]
    else:
        features = nx.degree(graph)
        features = [{int(k): v for k, v in features}]
    return graph, features, name

def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def get_embedding( model, files, dimensions, output_path = None):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)
        out.append([int(identifier.split('_')[-1])] + list(model.docvecs["g_"+identifier]))
    column_names = ["graph"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(by = ["graph"])
    out.reset_index(drop=True, inplace=True)
    if type(output_path) != type(None):
        out.to_csv(output_path, index=None)
    return out

def predict_embedding(model, input_path, output_path = None, n_jobs = 4, wl_iterations = 2, dimensions = 128):
    graphs = glob.glob(os.path.join(input_path, "*.json"))
    docs = Parallel(n_jobs=n_jobs)(delayed(feature_extractor)(g, wl_iterations) for g in graphs)
    out = []
    for doc in docs:
        vec = model.infer_vector(doc.words).tolist()
        out.append([int(doc.tags[0].split('_')[-1])] + vec)
    column_names = ["graph"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(by = ["graph"])
    out.reset_index(drop=True, inplace=True)
    if type(output_path) != type(None):
        out.to_csv(output_path, index=None)
    return out
    




def train_embedding (input_path = './dataset/', workers = 4,  wl_iterations = 2, dimensions = 128, 
    min_count = 5, down_sampling = 0.0001, learning_rate =0.025,  epochs = 1, seed = 666):

    graphs = glob.glob(os.path.join(input_path, "*.json"))
    # print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=workers)(delayed(feature_extractor)(g, wl_iterations) for g in graphs)
    # print("\nOptimization started.\n")
    

    model = Doc2Vec(document_collections,
                    vector_size=dimensions,
                    window=0,
                    min_count=min_count,
                    dm=0,
                    sample=down_sampling,
                    workers=workers, # for determinism // changed to speed up testing
                    epochs=epochs,
                    alpha=learning_rate,
                    seed = seed)    
    return model, graphs


def save_model(model, path = 'graph2vec_model' ):
    fname = get_tmpfile(path)
    model.save(fname)

def read_model(path ):
    model = Doc2Vec.load(path)
    return model

def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    graphs = glob.glob(os.path.join(args.input_path, "*.json"))
    print("\nFeature extraction started.\n")
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=1, # for determinism 
                    epochs=args.epochs,
                    alpha=args.learning_rate,
                    seed =args.seed)

    get_embedding(model, graphs, args.dimensions, args.output_path)

if __name__ == "__main__":
    args = parameter_parser()
    main(args)
