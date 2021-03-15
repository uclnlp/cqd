from kbc.datasets import Dataset
import itertools
import argparse
import pickle
import os


from tqdm import tqdm as tqdm


class Chain():
    def __init__(self):
        self.data = {'raw_chain':[], 'anchors': [], 'optimisable': [], 'type':None}

class ChaineDataset():
    def __init__(self, dataset: Dataset, threshold:int=1e6):


        if dataset is not None:
            self.threshold = threshold

            self.raw_data = dataset
            self.rhs_missing = self.raw_data.to_skip['rhs']
            self.lhs_missing = self.raw_data.to_skip['lhs']

            self.full_missing = {**self.rhs_missing, **self.lhs_missing}

            self.test_set = set((tuple(triple) for triple in self.raw_data.data['test']))

        self.neighbour_relations = {}
        self.reverse_maps = {}

        self.type1_1chain = []
        self.type1_2chain = []
        self.type2_2chain = []
        self.type2_2chain_u = []

        self.type1_3chain = []
        self.type2_3chain = []
        self.type3_3chain = []
        self.type4_3chain = []
        self.type4_3chain_u = []



    def sample_chains(self):
        try:
            self.__get_neighbour_relations__()
            self.neighbour_relations
            self.__reverse_maps__()

            self.__type1_2chains__()
            self.__type2_2chains__()
            self.__type1_3chains__()
            self.__type2_3chains__()
            self.__type3_3chains__()
            self.__type4_3chains__()

        except RuntimeError as e:
            print(e)


    def __get_neighbour_relations__(self):
        try:

            for i in list(self.rhs_missing.keys()):
                if i[0] not in self.neighbour_relations:
                    self.neighbour_relations[i[0]] = []

                self.neighbour_relations[i[0]].append(i[1])

            for i in list(self.lhs_missing.keys()):
                if i[0] not in self.neighbour_relations:
                    self.neighbour_relations[i[0]] = []

                self.neighbour_relations[i[0]].append(i[1])

        except Exception as e:
            print(1)

    def __reverse_maps__(self):

        for keys,vals in self.rhs_missing.items():
            for val in vals:
                if val not in self.reverse_maps:
                    self.reverse_maps[val] = []

                self.reverse_maps[val].append(keys)


    def __type1_2chains__(self):

        try:
            for test_triple in tqdm(self.raw_data.data['test']):

                test_lhs_chain_1 = (test_triple[0], test_triple[1])
                test_answers_chain_1 = [test_triple[2]]

                potential_chain_cont = [(x, self.neighbour_relations[x]) for x in test_answers_chain_1]

                for potential in potential_chain_cont:

                    segmented_list = [(potential[0],x) for x in potential[1]]

                    continuations = [ [x,self.rhs_missing[x]] for x in  segmented_list if x in self.rhs_missing]

                    ans_1 = [potential[0]]


                    raw_chains = [
                        [ list(test_lhs_chain_1) +  ans_1,  [x[0][0], x[0][1], x[1]] ]

                        for x in continuations
                    ]

                    for chain in raw_chains:
                        new_chain = Chain()
                        new_chain.data['type'] = '1chain2'
                        new_chain.data['raw_chain'] = chain
                        new_chain.data['anchors'].append(chain[0][0])

                        new_chain.data['optimisable'].append(chain[0][2])
                        new_chain.data['optimisable'].append(chain[1][2])

                        self.type1_2chain.append(new_chain)

                        if len(self.type1_2chain) > self.threshold:

                            print("Threshold for sample amount reached")
                            print("Finished sampling chains with legth 2 of type 1")
                            return

            print("Finished sampling chains with legth 2 of type 1")



        except RuntimeError as e:
            print(e)


    def __type2_2chains__(self):
        try:

            for ans in tqdm(self.reverse_maps):
                common_lhs = self.reverse_maps[ans]

                if len(common_lhs)<2:
                    continue

                common_lhs = list(itertools.combinations(common_lhs, 2))

                common_lhs_clean = []
                for segments in common_lhs:
                    for s in segments:
                        if s + (ans,) in self.test_set:
                            common_lhs_clean.append(segments)
                            break

                if len(common_lhs_clean) == 0:
                    continue

                raw_chains = [ [ list(x[0])+[ans], list(x[1])+[ans]]  for x in common_lhs_clean]


                for chain in raw_chains:
                    new_chain = Chain()

                    new_chain.data['type'] = '2chain2'

                    new_chain.data['raw_chain'] = chain
                    new_chain.data['anchors'].append(chain[0][0])
                    new_chain.data['anchors'].append(chain[1][0])

                    new_chain.data['optimisable'].append(chain[0][2])

                    self.type2_2chain.append(new_chain)

                    if len(self.type2_2chain) > self.threshold:

                        print("Threshold for sample amount reached")
                        print("Finished sampling chains with legth 2 of type 2")
                        return

            print("Finished sampling chains with legth 2 of type 2")

        except RuntimeError as e:
            print(e)


    def __type1_3chains__(self):
        try:
            #Appending routine
            for chain in tqdm(self.type1_2chain):
                raw_chain = chain.data['raw_chain']
                ans_2chain = raw_chain[1][2]

                ans_2chain = [x for x in ans_2chain if x != raw_chain[0][0]]

                potential_chain_cont = [(x, self.neighbour_relations[x]) for x in ans_2chain]

                for potential in potential_chain_cont:

                    segmented_list = [(potential[0],x) for x in potential[1]]

                    continuations = [ [x,self.rhs_missing[x]] for x in  segmented_list
                                     if x in self.rhs_missing]


                    ans_connector = potential[0]

                    new_chains = \
                    [
                        [
                            raw_chain[0], [raw_chain[1][0], raw_chain[1][1], ans_connector],
                            [ans_connector,x[0][1], x[1]]

                        ]
                        for x in continuations
                    ]


                    for chain in new_chains:
                        new_chain = Chain()

                        new_chain.data['type'] = '1chain3'

                        new_chain.data['raw_chain'] = chain

                        new_chain.data['anchors'].append(chain[0][0])

                        new_chain.data['optimisable'].append(chain[0][2])
                        new_chain.data['optimisable'].append(chain[1][2])
                        new_chain.data['optimisable'].append(chain[1][2])


                        self.type1_3chain.append(new_chain)

                        if len(self.type1_3chain) > self.threshold:

                            print("Threshold for sample amount reached")
                            print("Finished sampling chains with legth 3 of type 1")

                            return

            print("Finished sampling chains with legth 3 of type 1")


        except RuntimeError as e:
            print(e)


    def __type2_3chains__(self):
        try:
            for ans in tqdm(self.reverse_maps):
                common_lhs = self.reverse_maps[ans]

                if len(common_lhs)<3:
                    continue
                elif len(common_lhs) >15:
                    common_lhs = common_lhs[:15]


                common_lhs = list(itertools.combinations(common_lhs, 3))

                common_lhs_clean = []
                for segments in common_lhs:
                    for s in segments:
                        if s + (ans,) in self.test_set:
                            common_lhs_clean.append(segments)
                            break

                if len(common_lhs_clean) == 0:
                    continue


                raw_chains = [ [ list(x[0])+[ans], list(x[1])+[ans], list(x[2]) + [ans]]
                              for x in common_lhs_clean]


                for chain in raw_chains:
                    new_chain = Chain()

                    new_chain.data['type'] = '2chain3'

                    new_chain.data['raw_chain'] = chain

                    new_chain.data['anchors'].append(chain[0][0])
                    new_chain.data['anchors'].append(chain[1][0])
                    new_chain.data['anchors'].append(chain[2][0])


                    new_chain.data['optimisable'].append(chain[0][2])

                    self.type2_3chain.append(new_chain)

                    if len(self.type2_3chain) > self.threshold:

                        print("Threshold for sample amount reached")
                        print("Finished sampling chains with legth 3 of type 2")

                        return

            print("Finished sampling chains with legth 3 of type 2")


        except RuntimeError as e:
            print(e)

    def __type3_3chains__(self):

        try:
            for chain in tqdm(self.type2_2chain):

                raw_chain_initial = chain.data['raw_chain']
                connector_node_opt0 = raw_chain_initial[0][0]
                connector_node_opt1 = raw_chain_initial[1][0]

                if connector_node_opt0 in self.reverse_maps:
                    common_lhs_0 = self.reverse_maps[connector_node_opt0]
                else:
                    common_lhs_0 = []

                if len(common_lhs_0) >10:
                    common_lhs_0 = common_lhs_0[:10]


                if connector_node_opt1 in self.reverse_maps:
                    common_lhs_1 = self.reverse_maps[connector_node_opt1]
                else:
                    common_lhs_1 = []


                if len(common_lhs_1) >10:
                    common_lhs_1 = common_lhs_1[:10]


                potential_additions_0 = [ list(x)+[connector_node_opt0] for x in common_lhs_0]
                potential_additions_1 = [ list(x)+[connector_node_opt1] for x in common_lhs_1]

                raw_chains_0 = \
                [
                    [
                        x, raw_chain_initial[0],raw_chain_initial[1]
                    ]

                    for x in potential_additions_0
                ]

                raw_chains_1 = \
                [
                    [
                        x, raw_chain_initial[1],raw_chain_initial[0]
                    ]

                    for x in potential_additions_1
                ]


                raw_chains = raw_chains_0 + raw_chains_1

                for chain in raw_chains:

                    new_chain = Chain()

                    new_chain.data['type'] = '3chain3'

                    new_chain.data['raw_chain'] = chain

                    new_chain.data['anchors'].append(chain[0][0])
                    new_chain.data['anchors'].append(chain[2][0])

                    new_chain.data['optimisable'].append(chain[0][2])
                    new_chain.data['optimisable'].append(chain[1][2])


                    self.type3_3chain.append(new_chain)

                    if len(self.type3_3chain) > self.threshold:

                        print("Threshold for sample amount reached")
                        print("Finished sampling chains with legth 3 of type 3")
                        return

            print("Finished sampling chains with legth 3 of type 3")

        except RuntimeError as e:
            print(e)


    def __type4_3chains__(self):
        try:
            for chain in tqdm(self.type2_2chain):

                raw_chain_initial = chain.data['raw_chain']

                chain_top_initial = raw_chain_initial[0][2]

                chain_potential_predicates = self.neighbour_relations[chain_top_initial]

                chain_potential_lhs = [(chain_top_initial,x) for x in chain_potential_predicates]

                raw_chains = [
                                raw_chain_initial+
                                 [list(x)+[self.rhs_missing[x]] ]

                              for x in chain_potential_lhs if x in self.rhs_missing
                ]

                for chain in raw_chains:

                    new_chain = Chain()

                    new_chain.data['type'] = '4chain3'

                    new_chain.data['raw_chain'] = chain

                    new_chain.data['anchors'].append(chain[0][0])
                    new_chain.data['anchors'].append(chain[1][0])


                    new_chain.data['optimisable'].append(chain[0][2])
                    new_chain.data['optimisable'].append(chain[2][2])


                    self.type4_3chain.append(new_chain)

                    if len(self.type4_3chain) > self.threshold:

                        print("Threshold for sample amount reached")
                        print("Finished sampling chains with legth 3 of type 4")
                        return

            print("Finished sampling chains with legth 3 of type 2")



        except RuntimeError as e:
            print(e)



def save_chain_data(save_path, dataset_name, data):
    try:

        full_path = os.path.join(save_path,dataset_name+".pkl")

        with open(full_path, 'wb') as f:
            pickle.dump(data,f,-1)

        print("Chain Dataset for {} saved at {}".format(dataset_name,full_path))

    except RuntimeError as e:
        print(e)

def load_chain_data(data_path):
    data = None
    try:
        with open(data_path,'rb') as f:
            data= pickle.load(f)
    except RuntimeError as e:
        print(e)
        return data
    return data


if __name__ == "__main__":

    big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
    datasets = big_datasets

    parser = argparse.ArgumentParser(
    description="Chain Dataset Sampling"
    )


    parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
    )

    parser.add_argument(
    '--threshold',default = 1e5,type=int,
    help="Threshold for maximum amount sampled per chain type"
    )

    parser.add_argument(
    '--save_path',default = os.getcwd(),
    help="Path to save the chained dataset"
    )

    args = parser.parse_args()

    chained_dataset_sampler = ChaineDataset( Dataset(args.dataset),args.threshold)
    chained_dataset_sampler.sample_chains()

    save_chain_data(args.save_path,args.dataset,chained_dataset_sampler)
