from graphviz import Digraph
import numpy as np
import pandas as pd

class VariableNode(object):
    def __init__(self, name, dataset, parents=[]):
        self.name = name
        self.parents = parents
        self.values = pd.unique(dataset[name])
        self.prob = dict()
        if self.parents == []:
            for i in range(len(self.values)):
                dataset_v = dataset[dataset[name]==self.values[i]]
                self.prob[name+'='+str(self.values[i])] = len(dataset_v)/len(dataset)
        else:
            for i in range(len(self.values)):
                for item in self.parents:
                    for value in item.values:
                        dataset_known = dataset[dataset[item.name]==value]
                        dataset_event = dataset_known[dataset_known[name]==self.values[i]]
                        self.prob[name+'='+str(self.values[i])+'|'+item.name+'='+str(value)] = len(dataset_event)/len(dataset_known)

class BayesNet(object):
    def __init__(self):
        self.nodes = {}

    def createNode(self, name, dataset, parents = []):
        node = VariableNode(name, dataset, parents=parents)
        self.nodes[name] = node
        return node

    def plot_net(self, output_file='output/temp.gv'):
        dot = Digraph(comment='BayesNet', engine='dot')
        for node_name in self.nodes:
            dot.node(node_name, style = 'solid')

        for node_name in self.nodes:
            if self.nodes[node_name].parents != []:
                parent_name = self.nodes[node_name].parents[0].name
                dot.edge(parent_name, node_name)
        dot.render(output_file, view=True)

    def BIC_score(self, dataset):
        self.ll_bd = 0
        for i in range(len(dataset)):
            pb_xi = 1
            test_data = dataset.ix[i,:]
            cn_list = list(dataset.columns)
            while cn_list:
                feature = cn_list.pop()
                fea_value = test_data[feature]
                bn_node = self.nodes[feature]
                if bn_node.parents != []:
                    p_node_name = bn_node.parents[0].name
                    p_node_value = test_data[p_node_name]
                    pb_xi *= bn_node.prob[feature + '=' + str(fea_value) + '|' + p_node_name + '=' + str(p_node_value)]
                else:
                    pb_xi *= bn_node.prob[feature + '=' + str(fea_value)]
            self.ll_bd += np.log(pb_xi)

        return np.log(len(dataset))/2*len(self.nodes) - self.ll_bd