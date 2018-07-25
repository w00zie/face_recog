import networkx as nx
from random import shuffle
from scipy.spatial.distance import cosine as dcos
import matplotlib.pyplot as plt
import dlib

# TODO check a maximum number of descriptors to save


class Cluster:

    def __init__(self, thresh = 0.35):
        self.G = nx.Graph()
        self.names = []
        self.class_idx = 0
        self.node_idx = 0
        self.threshold = thresh
        self.people_idx = {}

    def chinese_whispers(self):
        iterations = 10
        for z in range(0, iterations):
            gn = []
            for i in range(len(self.G.nodes())):
                gn.append(i)
            # I randomize the nodes to give me an arbitrary start point
            shuffle(gn)
            for node in gn:
                neighs = self.G.adj[node]
                classes = {}
                # do an inventory of the given nodes neighbours and edge weights
                for ne in neighs:
                    if isinstance(ne, int):
                        if self.G.node[ne]['name'] in classes:
                            classes[self.G.node[ne]['name']] += self.G[node][ne]['weight']
                        else:
                            classes[self.G.node[ne]['name']] = self.G[node][ne]['weight']
                # find the class with the highest edge weight sum
                max = 0
                maxclass = self.G.node[node]['name']
                for c in classes:
                    if classes[c] > max:
                        max = classes[c]
                        maxclass = c
                # set the class of target node to the winning local class
                if maxclass != self.G.node[node]['name']:
                    oldclass = self.G.node[node]['name']
                    self.G.node[node]['name'] = maxclass
                    self.clear_idx(oldclass)

    def check_distances(self):
        for i in range(self.node_idx):
            distance = dcos(self.G.node[i]['desc'], self.G.node[self.node_idx]['desc'])
            if distance <= self.threshold:
                self.G.add_edge(i, self.node_idx, weight = distance)

    def add_name(self, name):
        if name not in self.names:
            if len(self.names) != 0:
                while len(self.names) != 0:
                    self.names.pop()
            self.names.append(name)
            self.check_names()
            self.check_index()
            self.class_idx -= 1
        else:
            print('Name already in list, use a new one')
        self.update_people()

    def check_names(self):
        for i in range(self.node_idx):
            idx = self.G.node[i]['name']
            if isinstance(idx, int):
                try:
                    self.G.node[i]['name'] = self.names[idx]
                except (IndexError, TypeError):
                    self.G.node[i]['name'] -= 1

    def check_index(self):
        for i in range(len(self.names)):
            try:
                self.people_idx[self.names[i]] = self.people_idx.pop(i)
            except KeyError:
                print("KeyError occurred")
                pass

    def clear_idx(self, idx):
        node_to_remove = []
        for i in range(self.node_idx):
            if self.G.node[i]['name'] == idx:
                node_to_remove.append(i)
            elif isinstance(idx, int):
                try:
                    if self.G.node[i]['name'] > idx:
                        self.G.node[i]['name'] = int(self.G.node[i]['name']) - 1
                except Exception as e:
                    print("Got the {} except".format(e))
                    pass
        return node_to_remove

    def update_people(self):
        new_dict = {}
        for key, value in self.people_idx.items():
            if isinstance(key, int):
                new_dict[key-1] = value
            elif isinstance(key, str):
                new_dict[key] = value
        self.people_idx = new_dict

    def clear_class(self, idx):
        nodes_to_remove = self.clear_idx(idx)

        for node in nodes_to_remove:
            self.G.remove_node(node)
        del self.people_idx[idx]
        self.update_people()
        i = 0
        for node in self.G.nodes:
            if node != i:
                self.G.add_node(i, name = self.G.node[node]['name'], desc = self.G.node[node]['desc'])
                new_edges = []
                for edge in self.G.edges:
                    if edge[0] == node:
                        new_edges.append((i, edge[1], {'weight': self.G[edge[0]][edge[1]]['weight']}))
                    elif edge[1] == node:
                        new_edges.append((edge[0], i, {'weight': self.G[edge[0]][edge[1]]['weight']}))
                self.G.add_edges_from(new_edges)
                self.G.remove_node(node)
            i += 1

        self.node_idx = len(self.G.nodes.data())
        if isinstance(idx, int):
            self.class_idx -= 1

    def update_graph(self, desc):

        self.G.add_node(self.node_idx, name = self.class_idx, desc = desc)
        self.check_distances()
        neighs = self.G.adj[self.node_idx]
        classes = {}
        # do an inventory of the given nodes neighbours and edge weights
        for ne in neighs:
            if isinstance(ne, int):
                identity = self.G.node[ne]['name']
                if identity in classes:
                    classes[identity] += self.G[self.node_idx][ne]['weight']
                else:
                    classes[identity] = self.G[self.node_idx][ne]['weight']
        # find the class with the lowest edge weight sum
        class_idx = self.class_idx
        if classes:
            class_idx = min(classes, key=classes.get)
            self.class_idx -= 1

        self.G.node[self.node_idx]['name'] = class_idx

        if class_idx in self.people_idx.keys():
            self.people_idx[class_idx] += 1
        else:
            self.people_idx[class_idx] = 1

        self.class_idx += 1
        self.node_idx += 1

    def plot_graph(self):
        from utils import colors

        pos = nx.spring_layout(self.G)
        color_list = colors(len(self.people_idx))
        plt.title("Connected components in the Chinese Whispers Graph")
        wcc = nx.connected_component_subgraphs(self.G)
        lab = ["Person {}".format(x) for x in self.people_idx.keys()]
        for index, sg in enumerate(wcc):
            nx.draw_networkx(sg, pos=pos, edge_color=color_list[index], node_color=color_list[index])
        for i in range(len(lab)):
            plt.scatter([], [], c=color_list[i], alpha=0.3, s=100 ,label=lab[i])
        plt.legend(title="Clusters")
        plt.show()

