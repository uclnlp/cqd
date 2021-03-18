#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import os.path

import json

import argparse

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def path_to_key(path):
    res = path.replace('.json', '').split('/')[-1].replace('cont_', '')
    res = res.replace('-model-', '_').replace('rank-', 'rank=').replace('-epoch-100-', '_')
    # print(res)
    return res


def path_to_results(path):
    with open(path) as f:
        res = json.load(f)
    return res


def main(argv):
    parser = argparse.ArgumentParser(description='Parse results.')
    parser.add_argument('paths', metavar='path', type=str, nargs='+')
    parser.add_argument('--mrr', action='store_true', default=False)

    args = parser.parse_args(argv)

    is_mrr = args.mrr

    metric = "HITS@3m_new"
    if is_mrr is True:
        metric = "MRRm_new"

    key_to_path = {path_to_key(path): path for path in args.paths}

    key_lst = sorted([key for key in key_to_path])

    for d in ['FB15k', 'FB15k-237', 'NELL']:
        for rank in [100, 200, 500, 1000]:
            results = []

            # for query in ['1_2', '1_3', '2_2', '2_3', '3_3', '4_3', '2_2_disj', '4_3_disj']:
            for query in ['1_2', '1_3', '2_2', '2_3', '4_3', '3_3', '2_2_disj', '4_3_disj']:

                _keys = []
                for key in key_lst:
                    # print(d, rank, query, key)
                    # print(key)
                    if f'm=valid' in key and f'n={d}' in key and f'rank={rank}_' in key and f't={query}_r=' in key:
                        _keys += [key]

                best_value = None
                best_dev_key = None

                for key in _keys:
                    res = path_to_results(key_to_path[key])
                    if best_value is None or res[metric] > best_value:
                        _tmp = key.replace('m=valid', 'm=test')                        
                        if _tmp in key_to_path and os.path.isfile(key_to_path[_tmp]): 
                            best_dev_key = key
                            best_value = res[metric]

                best_test_key = best_dev_key.replace('m=valid', 'm=test')
                best_test_path = key_to_path[best_test_key]

                res = path_to_results(best_test_path)
                results += [res[metric]]

            print(f'd={d} rank={rank} & ' + " & ".join([f'{r:.3f}' for r in results]))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
