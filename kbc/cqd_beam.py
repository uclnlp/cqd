#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
import argparse
import pickle
import json

from kbc.utils import QuerDAG
from kbc.utils import preload_env

from kbc.metrics import evaluation


def run(kbc_path, dataset_hard, dataset_complete, dataset_name, t_norm='min', candidates=3, scores_normalize=0, kg_path=None, explain=False):
	experiments = [t.value for t in QuerDAG]
	experiments.remove(QuerDAG.TYPE1_1.value)

	print(kbc_path, dataset_name, t_norm, candidates)

	path_entries = kbc_path.split('-')
	rank = path_entries[path_entries.index('rank') + 1] if 'rank' in path_entries else 'None'

	for exp in experiments:
		metrics = answer(kbc_path, dataset_hard, dataset_complete, t_norm, exp, candidates, scores_normalize, kg_path, explain)

		with open(f'topk_d={dataset_name}_t={t_norm}_e={exp}_rank={rank}_k={candidates}_sn={scores_normalize}.json', 'w') as fp:
			json.dump(metrics, fp)
	return


def answer(kbc_path, dataset_hard, dataset_complete, t_norm='min', query_type=QuerDAG.TYPE1_2, candidates=3, scores_normalize = 0, kg_path=None, explain=False):
	env = preload_env(kbc_path, dataset_hard, query_type, mode='hard', kg_path=kg_path, explain=explain)
	env = preload_env(kbc_path, dataset_complete, query_type, mode='complete', explain=explain)

	if '1' in env.chain_instructions[-1][-1]:
		part1, part2 = env.parts
	elif '2' in env.chain_instructions[-1][-1]:
		part1, part2, part3 = env.parts

	kbc = env.kbc

	scores = kbc.model.query_answering_BF(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, explain=explain)
	print(scores.shape)

	queries = env.keys_hard
	test_ans_hard = env.target_ids_hard
	test_ans = 	env.target_ids_complete
	# scores = torch.randint(1,1000, (len(queries),kbc.model.sizes[0]),dtype = torch.float).cuda()
	#
	metrics = evaluation(scores, queries, test_ans, test_ans_hard)
	print(metrics)

	return metrics


if __name__ == "__main__":

	big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
	datasets = big_datasets
	dataset_modes = ['valid', 'test', 'train']

	chain_types = [QuerDAG.TYPE1_1.value, QuerDAG.TYPE1_2.value, QuerDAG.TYPE2_2.value, QuerDAG.TYPE1_3.value,
				   QuerDAG.TYPE1_3_joint.value, QuerDAG.TYPE2_3.value, QuerDAG.TYPE3_3.value, QuerDAG.TYPE4_3.value,
				   'All', 'e']

	t_norms = ['min', 'product']
	normalize_choices = ['0', '1']

	parser = argparse.ArgumentParser(
	description="Complex Query Decomposition - Beam"
	)

	parser.add_argument('path', help='Path to directory containing queries')

	parser.add_argument(
	'--model_path',
	help="The path to the KBC model. Can be both relative and full"
	)

	parser.add_argument(
	'--dataset',
	help="The pickled Dataset name containing the chains"
	)

	parser.add_argument(
	'--mode', choices=dataset_modes, default='test',
	help="Dataset validation mode in {}".format(dataset_modes)
	)

	parser.add_argument(
	'--scores_normalize', choices=normalize_choices, default='0',
	help="A normalization flag for atomic scores".format(chain_types)
	)

	parser.add_argument(
	'--t_norm', choices=t_norms, default='min',
	help="T-norms available are ".format(t_norms)
	)

	parser.add_argument(
	'--candidates', default=5,
	help="Candidate amount for beam search"
	)

	parser.add_argument('--explain', default=False,
						action='store_true',
						help='Generate log file with explanations for 2p queries')

	args = parser.parse_args()

	dataset = osp.basename(args.path)
	mode = args.mode

	data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
	data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete.pkl')

	data_hard = pickle.load(open(data_hard_path, 'rb'))
	data_complete = pickle.load(open(data_complete_path, 'rb'))

	candidates = int(args.candidates)
	run(args.model_path, data_hard, data_complete,
		dataset, t_norm=args.t_norm, candidates=candidates,
		scores_normalize=int(args.scores_normalize),
		kg_path=args.path, explain=args.explain)
