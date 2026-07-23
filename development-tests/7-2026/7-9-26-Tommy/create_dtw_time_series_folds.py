from tslearn.metrics import dtw_path
import pandas as pd
import glob
import random
import numpy as np
import os
import shutil

TIME_SERIES_DATA_PATH = 'all-time-series'
CLUSTER_SIZE = 5
SHOW_PLOTS = False

def load_time_series(path):
    series = []
    files = glob.glob(path + '/*.csv')
    for file in files:
        df = pd.read_csv(file)
        series.append(df)
    return series

time_series_list = load_time_series(TIME_SERIES_DATA_PATH)
time_series_list.sort(key=lambda x: x['Event ID'][0])
print(time_series_list)
print(len(time_series_list))

distance_records = []
def get_pair_distance(series_list, index_one, index_two):

    intensity_series_one = np.log(series_list[index_one]['Proton Intensity'].to_numpy() + 1e-9)
    intensity_series_two = np.log(series_list[index_two]['Proton Intensity'].to_numpy() + 1e-9)

    _, distance = dtw_path(intensity_series_one, intensity_series_two)

    return {
        'pair' : [index_one, index_two],
        'distance' : distance
    }

for i in range(len(time_series_list)):
    for j in range(len(time_series_list[i])):
        if j >= i:
            continue
        distance_records.append(get_pair_distance(time_series_list, i, j))

class UnionFind:
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.size = {v: 1 for v in vertices}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return self.size[ra]

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return self.size[ra]


def greedy_cluster(vertices, edge_list, target_size=5):
    uf = UnionFind(vertices)

    edge_list = sorted(edge_list, key=lambda e: e["distance"])

    chosen_edges = []
    for edge in edge_list:
        u, v = edge["pair"]

        if uf.find(u) != uf.find(v):
            size = uf.union(u, v)
            chosen_edges.append(edge)

            if size >= target_size:
                root = uf.find(u)
                cluster = [v for v in vertices if uf.find(v) == root]
                return cluster, chosen_edges

    return None, chosen_edges

def extract_cluster(vertices, records, cluster_size=5):
    found_clusters = []

    cluster, used_edges = greedy_cluster(vertices, records,target_size=cluster_size)
    if cluster is None:
        return None
    while len(cluster) > cluster_size:
        random_index = int(np.random.rand()*len(cluster))
        cluster.pop(random_index)

    return cluster

vertices = [i for i in range(len(time_series_list))]

found_clusters = []
while len(vertices) > CLUSTER_SIZE:
    new_cluster = extract_cluster(vertices, distance_records, cluster_size=CLUSTER_SIZE)
    found_clusters.append(new_cluster)
    for index in new_cluster:
        vertices.remove(index)
        distance_records = [x for x in distance_records if index not in x['pair']]
found_clusters.append(vertices)

print(found_clusters)

folds = [[] for i in range(CLUSTER_SIZE)]

from matplotlib import pyplot as plt


for i, cluster in enumerate(found_clusters):
    if SHOW_PLOTS:
        for index in cluster:
            intensities = np.log(time_series_list[index]['Proton Intensity'].to_numpy() + 1e-9)
            x = np.arange(len(intensities))
            plt.plot(x, intensities)
            plt.title(f'Cluster {i+1} - File {index+1}')
            plt.ylim([-10, 6])
            plt.show()
    random.shuffle(cluster)
    for index, value in enumerate(cluster):
        folds[index].append(value)

training = np.concatenate([folds[0], folds[1], folds[2]]).tolist()
validation = folds[3]
test = folds[4]

print(folds)

for i in range(CLUSTER_SIZE):
    if os.path.exists(f'dtw-data/fold-{i+1}'):
        shutil.rmtree(f'dtw-data/fold-{i+1}')
    os.mkdir(f'dtw-data/fold-{i+1}')
    for value in folds[i]:
        time_series_list[value].to_csv(f'dtw-data/fold-{i+1}/sep_event_{value+1}_filled_ie_trim.csv', index=False)