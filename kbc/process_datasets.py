# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import errno
import shutil
import pickle

import numpy as np

from collections import defaultdict

from kbc.chain_dataset import ChaineDataset
from kbc.chain_dataset import Chain
from kbc.chain_dataset import save_chain_data

KBC = 'kbc'
Q2B_QUERIES = 'q2b_queries'
Q2B_TRIPLES = 'q2b_triples'

def load_q2b_maps(path):
    """Read entity and relation IDs from q2b mappings"""
    q2b_maps = ['ind2ent.pkl', 'ind2rel.pkl']
    with open(os.path.join(path, q2b_maps[0]), 'rb') as f:
        ind2ent = pickle.load(f)
        entities_to_id = {ent: i for i, ent in ind2ent.items()}
    with open(os.path.join(path, q2b_maps[1]), 'rb') as f:
        ind2rel = pickle.load(f)
        relations_to_id = {rel: i for i, rel in ind2rel.items()}

    return entities_to_id, relations_to_id


def prepare_dataset(path):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    out_path = os.path.join(path, 'kbc_data')
    files = ['train.txt', 'valid.txt', 'test.txt']

    q2b_maps = ['ind2ent.pkl', 'ind2rel.pkl']
    if all([os.path.exists(os.path.join(path, f)) for f in q2b_maps]):
        entities_to_id, relations_to_id = load_q2b_maps(path)

        # Create IDs for the remaining entities and relations (not used in q2b)
        max_ent_id = max(entities_to_id.values())
        max_rel_id = max(relations_to_id.values())

        for f in files:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as to_read:
                for line in to_read.readlines():
                    lhs, rel, rhs = line.strip().split('\t')
                    if lhs not in entities_to_id:
                        max_ent_id += 1
                        entities_to_id[lhs] = max_ent_id
                    if rhs not in entities_to_id:
                        max_ent_id += 1
                        entities_to_id[rhs] = max_ent_id
                    if rel not in relations_to_id:
                        max_rel_id += 1
                        relations_to_id[rel] = max_rel_id

    else:
        entities, relations = set(), set()
        for f in files:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as to_read:
                for line in to_read.readlines():
                    lhs, rel, rhs = line.strip().split('\t')
                    print(rel)
                    entities.add(lhs)
                    entities.add(rhs)
                    relations.add(rel)

        entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
        relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}

    n_relations = len(relations_to_id)
    n_entities = len(entities_to_id)
    print(f'{n_entities} entities and {n_relations} relations')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        pickle.dump(dic, open(os.path.join(out_path, f'{f}.pickle'), 'wb'))

    # map train/test/valid with the ids
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')

            lhs_id = entities_to_id[lhs]
            rhs_id = entities_to_id[rhs]
            rel_id = relations_to_id[rel]
            inv_rel_id = relations_to_id[rel + '_reverse']

            examples.append([lhs_id, rel_id, rhs_id])
            to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
            to_skip['lhs'][(rhs_id, inv_rel_id)].add(lhs_id)

            # Add inverse relations for training
            if f == 'train.txt':
                examples.append([rhs_id, inv_rel_id, lhs_id])
                to_skip['rhs'][(rhs_id, inv_rel_id)].add(lhs_id)
                to_skip['lhs'][(lhs_id, rel_id)].add(rhs_id)

        out = open(os.path.join(out_path, f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(os.path.join(out_path, 'to_skip.pickle'), 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(os.path.join(out_path, 'train.txt.pickle'), 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(os.path.join(out_path, 'probas.pickle'), 'wb')
    pickle.dump(counters, out)
    out.close()


translator_dict = {'1c': '1chain1', '1c_hard': '1chain1_hard',
                   '2c': '1chain2', '2c_hard': '1chain2_hard',
                   '3c': '1chain3', '3c_hard': '1chain3_hard',
                   '2i': '2chain2', '2i_hard': '2chain2_hard',
                   'ci': '3chain3', 'ci_hard': '3chain3_hard',
                   '3i': '2chain3', '3i_hard': '2chain3_hard',
                   'ic': '4chain3', 'ic_hard': '4chain3_hard',
                   '2u': '2chain2_disj',
                   'uc': '4chain3_disj'}


def convert_q2b_queries(path, split):
    split_check = f'{split}_ans'
    files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and split_check in i]

    dataset_name = os.path.basename(path)

    data_hard = get_hard_dataset(path, files, mode='hard')
    save_chain_data(path, dataset_name + f'_{split}_hard', data_hard)

    data_complete = get_hard_dataset(path, files, mode='complete')
    save_chain_data(path, dataset_name + f'_{split}_complete', data_complete)


def get_hard_dataset(path, files, mode='hard'):
    chain_dataset = None
    try:
        chain_dataset = ChaineDataset(None)

        for file in files:
            if 'hard' in mode:
                check = 'hard' in file
            else:
                check = not ('hard' in file)

            if check:
                with open(os.path.join(path, file), 'rb') as c:
                    contents = pickle.load(c)

                if 'hard' in mode:
                    chain_type = file.split('.')[0].split('_')[-2]
                else:
                    chain_type = file.split('.')[0].split('_')[-1]

                chain_type_cast = translator_dict[chain_type]

                chains = get_sampled_chain(chain_type_cast, contents)

                if '1chain1' == chain_type_cast:
                    chain_dataset.type1_1chain = chains
                elif '1chain2' == chain_type_cast:
                    chain_dataset.type1_2chain = chains
                elif '1chain3' == chain_type_cast:
                    chain_dataset.type1_3chain = chains
                elif '2chain2' == chain_type_cast:
                    chain_dataset.type2_2chain = chains
                elif '2chain3' == chain_type_cast:
                    chain_dataset.type2_3chain = chains
                elif '3chain3' == chain_type_cast:
                    chain_dataset.type3_3chain = chains
                elif '4chain3' == chain_type_cast:
                    chain_dataset.type4_3chain = chains
                elif '2chain2_disj' == chain_type_cast:
                    chain_dataset.type2_2_disj_chain = chains
                elif '4chain3_disj' == chain_type_cast:
                    chain_dataset.type4_3_disj_chain = chains

    except RuntimeError as e:
        print("Cannot cast dataset with error: ", e)
        return chain_dataset
    return chain_dataset


def get_sampled_chain(chain_type_cast, contents):
    chain_array = []
    try:
        for chain in contents:

            new_chain = Chain()
            targets = list(contents[chain])

            if '1chain1' in chain_type_cast:
                rel = chain[0][-1][0]
                anchor = chain[0][0]
                converted_chain = [anchor, rel, targets]

                new_chain.data['type'] = chain_type_cast
                new_chain.data['raw_chain'] = converted_chain
                new_chain.data['anchors'].append(anchor)
                new_chain.data['optimisable'].append(-1)
                new_chain.data['optimisable'] += targets

                chain_array.append(new_chain)
            if '1chain2' in chain_type_cast:
                rels = chain[0][-1]
                anchor = chain[0][0]

                converted_chain = [ \
                    [anchor, rels[0], -1], \
                    [-1, rels[1], targets] \
                    ]

                new_chain.data['type'] = chain_type_cast
                new_chain.data['raw_chain'] = converted_chain
                new_chain.data['anchors'].append(anchor)

                new_chain.data['optimisable'].append(-1)
                new_chain.data['optimisable'] += targets

                chain_array.append(new_chain)

            if '1chain3' in chain_type_cast:
                rels = chain[0][-1]
                anchor = chain[0][0]

                converted_chain = [ \
                    [anchor, rels[0], -1], \
                    [-1, rels[1], -2], \
                    [-2, rels[2], targets] \
                    ]

                new_chain.data['type'] = chain_type_cast
                new_chain.data['raw_chain'] = converted_chain
                new_chain.data['anchors'].append(anchor)

                new_chain.data['optimisable'].append(-1)
                new_chain.data['optimisable'].append(-2)
                new_chain.data['optimisable'] += targets

                chain_array.append(new_chain)

            if '2chain2' in chain_type_cast:
                anchors = [chain[0][0], chain[1][0]]
                rels = [chain[0][1][0], chain[1][1][0]]

                converted_chain = [ \
                    [anchors[0], rels[0], targets], \
                    [anchors[1], rels[1], targets] \
                    ]
                new_chain.data['type'] = chain_type_cast
                new_chain.data['raw_chain'] = converted_chain
                new_chain.data['anchors'] += anchors

                new_chain.data['optimisable'] += targets

                chain_array.append(new_chain)

            if '2chain3' in chain_type_cast:
                #                 (90, (439,)), (2864, (145,)), (2620, (309,))): {92, 3647}
                anchors = [chain[0][0], chain[1][0], chain[2][0]]
                rels = [chain[0][1][0], chain[1][1][0], chain[2][1][0]]

                converted_chain = [ \
                    [anchors[0], rels[0], targets], \
                    [anchors[1], rels[1], targets], \
                    [anchors[2], rels[2], targets]
                ]
                new_chain.data['type'] = chain_type_cast
                new_chain.data['raw_chain'] = converted_chain
                new_chain.data['anchors'] += anchors
                new_chain.data['optimisable'] += targets

                chain_array.append(new_chain)

            if '3chain3' in chain_type_cast:
                anchors = [chain[0][0], chain[1][0]]
                rels = list(chain[0][1]) + [chain[1][1][0]]

                converted_chain = [ \
                    [anchors[0], rels[0], -1], \
                    [-1, rels[1], targets], \
                    [anchors[1], rels[2], targets]
                ]
                new_chain.data['type'] = chain_type_cast
                new_chain.data['raw_chain'] = converted_chain
                new_chain.data['anchors'] += anchors
                new_chain.data['optimisable'].append(-1)
                new_chain.data['optimisable'] += targets

                chain_array.append(new_chain)

            if '4chain3' in chain_type_cast:
                rels = [chain[0][1][0], chain[1][1][0], chain[-1]]
                anchors = [chain[0][0], chain[1][0]]

                converted_chain = [ \
                    [anchors[0], rels[0], -1], \
                    [anchors[1], rels[1], -1], \
                    [-1, rels[2], targets]
                ]

                new_chain.data['type'] = chain_type_cast
                new_chain.data['raw_chain'] = converted_chain
                new_chain.data['anchors'] += anchors

                new_chain.data['optimisable'].append(-1)
                new_chain.data['optimisable'] += targets

                chain_array.append(new_chain)

            new_chain.data['targets'] = targets


    except RuntimeError as e:
        print("Cannot get sampled chains with error: ", e)
        return chain_array

    return chain_array


def extract_q2b_triples(path):
    splits = ['train', 'valid', 'test']

    entities_to_id, relations_to_id = load_q2b_maps(path)
    # String identifiers are not used in Q2B's NELL
    entities_to_id = {e: e for e in entities_to_id}
    relations_to_id = {r: r for r in relations_to_id}

    out_path = os.path.join(path, 'kbc_data')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        f = open(os.path.join(out_path, f'{f}.pickle'), 'wb')
        pickle.dump(dic, f)
        f.close()

    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}

    for s in splits:
        count = 0
        hard_postfix = '_hard' if s in ['valid', 'test'] else ''
        pickle_path = os.path.join(path, f'{s}_ans_1c{hard_postfix}.pkl')
        with open(pickle_path, 'rb') as f:
            triples_dict = pickle.load(f)

        examples = []
        for head_rel, all_tails in triples_dict.items():
            head = head_rel[0][0]
            rel = head_rel[0][1][0]

            # In Q2B's NELL, normal relations are even, inverses are odd
            if rel % 2 == 0:
                inv_rel = rel + 1
            else:
                inv_rel = rel - 1

            for tail in all_tails:
                examples.append([head, rel, tail])

                to_skip['rhs'][(head, rel)].add(tail)
                to_skip['lhs'][(tail, inv_rel)].add(head)

                count += 1

        out = open(os.path.join(out_path, s + '.txt.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

        print(f'Saved {count:,} {s} triples')

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    with open(os.path.join(out_path, 'to_skip.pickle'), 'wb') as out:
        pickle.dump(to_skip_final, out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Process datasets for link prediction and query answering'
    )

    parser.add_argument('type', choices=[KBC, Q2B_QUERIES, Q2B_TRIPLES])
    parser.add_argument('data_path', help='Path containing triples for'
                                          ' training, validation, and test')
    parser.add_argument('--split', choices=['train', 'valid', 'test'],
                        default='test')
    args = parser.parse_args()
    data_path = args.data_path

    print(f'Loading dataset from {data_path}')
    try:
        if args.type == KBC:
            prepare_dataset(data_path)
        elif args.type == Q2B_QUERIES:
            convert_q2b_queries(data_path, args.split)
        elif args.type == Q2B_TRIPLES:
            extract_q2b_triples(data_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(e)
            print("File exists. skipping...")
        else:
            raise
