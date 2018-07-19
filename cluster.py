import networkx as nx
from random import shuffle
from scipy.spatial.distance import cosine as dcos
import matplotlib.pyplot as plt

# TODO check a maximum number of descriptors to save
class Cluster:

    def __init__(self, thresh = 0.35):
        self.G = nx.Graph()
        self.names = []
        self.nodes = []
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

        self.nodes = []
        for node in self.G.nodes.data():
            self.nodes.append(node)

    def check_distances(self):
        for i in range(self.node_idx):
            distance = dcos(self.G.node[i]['desc'], self.G.node[self.node_idx]['desc'])
            # distance = abs(self.G.node[i]['desc'] - self.G.node[self.node_idx]['desc'])
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
            except IndexError:
                self.G.node[i]['name'] -= 1
                self.nodes[i][1]['name'] -= 1
            except TypeError:
                pass

    def check_index(self):
        for i in range(len(self.names)):
            try:
                self.people_idx[self.names[i]] = self.people_idx.pop(i)
            except KeyError:
                pass

    def clear_idx(self, idx):
        node_to_remove = []
        for i in range(self.node_idx):
            #print("i = ")
            print("name = {}, idx = {}".format(self.G.node[i]['name'], idx))
            if self.G.node[i]['name'] == idx:
                print("entro qui ed appendo")
                node_to_remove.append(i)
            elif isinstance(idx, int):
                try:
                    print("no entro nell'elif")
                    if self.G.node[i]['name'] > idx:
                        self.G.node[i]['name'] = int(self.G.node[i]['name']) - 1
                except (ValueError, TypeError):
                    pass
        return node_to_remove

    def clear_class(self, idx):
        print("-"*32)
        print("People idx = {}".format(self.people_idx))
        node_to_remove = self.clear_idx(idx)
        print("For idx = {} got to remove {}".format(idx, node_to_remove))
        if idx not in self.people_idx:
            print("This value is wrong")
            idx = idx+1
            node_to_remove = self.clear_idx(idx)
            print("For new idx = {} got to remove {}".format(idx, node_to_remove))
        for node in node_to_remove:
            self.G.remove_node(node)
        del self.people_idx[idx]

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

        self.nodes = []
        for node in self.G.nodes.data():
            self.nodes.append(node)
        print("People idx after the removal= {}".format(self.people_idx))
        print("And graph = {}".format(self.G.nodes.data()))

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

        self.nodes.append((self.node_idx, self.G.node[self.node_idx]))

        self.class_idx += 1
        self.node_idx += 1

    def plot_graph(self):
        from utils import colors

        pos = nx.spring_layout(self.G)
        colorlist = colors(len(self.people_idx))
        plt.title("Connected components in the Chinese Whispers Graph")
        wcc = nx.connected_component_subgraphs(self.G)
        lab = ["Person {}".format(x) for x in self.people_idx.keys()]
        for index, sg in enumerate(wcc):
            nx.draw_networkx(sg, pos=pos, edge_color=colorlist[index], node_color=colorlist[index])
        for i in range(len(lab)):
            plt.scatter([], [], c=colorlist[i], alpha=0.3, s=100 ,label=lab[i])
        plt.legend(title="Clusters")
        plt.show()