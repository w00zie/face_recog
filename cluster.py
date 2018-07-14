import networkx as nx
from random import shuffle
from scipy.spatial.distance import cosine as dcos


# TODO check when useful to run a complete chinese whispers
# TODO check a maximum number of descriptors to save
class Cluster:

    def __init__(self, thresh = 0.35):
        self.G = nx.Graph()
        self.names = []
        self.class_idx = 0
        self.node_idx = 0
        self.threshold = thresh
        self.people_idx = {}

    def check_distances(self):
        for i in range(self.node_idx - 1):
            distance = dcos(self.G.node[i]['desc'], self.G.node[self.node_idx]['desc'])
            if distance <= self.threshold:
                self.G.add_edge(i, self.node_idx, weight = distance)

    def add_name(self, name):
        if name not in self.names:
            self.names.append(name)
            self.check_names()
            self.check_index()
            self.class_idx -= 1
        else:
            print('Name already in list, use a new one')

    def check_names(self):
        for i in range(self.node_idx):
            try:
                self.G.node[i]['name'] = self.names[self.G.node[i]['name']]
            except (IndexError, TypeError):
                self.G.node[i]['name'] -= 1

    def check_index(self):
        for i in range(len(self.names)):
            try:
                self.people_idx[self.names[i]] = self.people_idx.pop(i)
            except KeyError:
                pass

    # TODO check removal from G
    def clear_class(self, idx):
        node_to_remove = []
        for i in range(self.node_idx):
            if self.G.node[i]['name'] == idx:
                node_to_remove.append(i)
            else:
                try:
                    self.G.node[i]['name'] = int(self.G.node[i]['name']) - 1
                except ValueError:
                    pass
        for node in node_to_remove:
            self.G.remove_node(node)
        del self.people_idx[idx]

        i = 0
        for node in self.G.nodes:
            if node != i:
                self.G.add_node(i, name = self.G.node[node]['name'], desc = self.G.node[node]['desc'])
                self.G.remove_node(node)
            i += 1

        self.node_idx = len(self.G.nodes.data())
        self.class_idx -= 1

    def update_graph(self, desc):

        self.G.add_node(self.node_idx, name = self.class_idx, desc = desc)
        self.check_distances()

        neighs = self.G.adj[self.node_idx]
        classes = {}
        # do an inventory of the given nodes neighbours and edge weights
        for ne in neighs:
            if isinstance(ne, int):
                if self.G.node[ne]['name'] in classes:
                    classes[self.G.node[ne]['name']] += self.G[self.node_idx][ne]['weight']
                else:
                    classes[self.G.node[ne]['name']] = self.G[self.node_idx][ne]['weight']
        # find the class with the highest edge weight sum
        max = 0
        maxclass = self.class_idx
        for c in classes:
            if classes[c] > max:
                max = classes[c]
                maxclass = c
                self.class_idx -= 1
        # set the class of target node to the winning local class
        try:
            self.G.node[self.node_idx]['name'] = self.names[maxclass]
        except (IndexError, TypeError):
            self.G.node[self.node_idx]['name'] = maxclass
        if maxclass in self.people_idx:
            self.people_idx[maxclass] += 1
        else:
            self.people_idx[maxclass] = 1

        self.class_idx += 1
        self.node_idx += 1
