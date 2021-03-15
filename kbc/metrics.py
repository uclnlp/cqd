import torch
import numpy as np
from tqdm import tqdm
import logging


def evaluation(scores, queries, test_ans, test_ans_hard):
    nentity = len(scores[0])
    step = 0
    logs = []

    for query_id, query in enumerate(tqdm(queries)):

        score = scores[query_id]
        score -= (torch.min(score) - 1)
        ans = test_ans[query]
        hard_ans = test_ans_hard[query]
        all_idx = set(range(nentity))

        false_ans = all_idx - set(ans)
        ans_list = list(ans)
        hard_ans_list = list(hard_ans)
        false_ans_list = list(false_ans)
        ans_idxs = np.array(hard_ans_list)
        vals = np.zeros((len(ans_idxs), nentity))

        vals[np.arange(len(ans_idxs)), ans_idxs] = 1
        axis2 = np.tile(false_ans_list, len(ans_idxs))

        # axis2 == [not_ans_1,...not_ans_k, not_ans_1, ....not_ans_k........]
        # Goes for len(hard_ans) times

        axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))

        vals[axis1, axis2] = 1
        b = torch.tensor(vals, device=scores.device)
        filter_score = b * score
        argsort = torch.argsort(filter_score, dim=1, descending=True)
        ans_tensor = torch.tensor(hard_ans_list, device=scores.device, dtype=torch.long)
        argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
        ranking = (argsort == 0).nonzero(as_tuple=False)
        ranking = ranking[:, 1]
        ranking = ranking + 1

        ans_vec = np.zeros(nentity)
        ans_vec[ans_list] = 1
        hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
        hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
        hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
        mrm = torch.mean(ranking.to(torch.float)).item()
        mrrm = torch.mean(1./ranking.to(torch.float)).item()
        num_ans = len(hard_ans_list)

        hits1m_newd = hits1m
        hits3m_newd = hits3m
        hits10m_newd = hits10m
        mrm_newd = mrm
        mrrm_newd = mrrm

        logs.append({
            'MRRm_new': mrrm_newd,
            'MRm_new': mrm_newd,
            'HITS@1m_new': hits1m_newd,
            'HITS@3m_new': hits3m_newd,
            'HITS@10m_new': hits10m_newd,
            'num_answer': num_ans
        })

        if step % 100 == 0:
            logging.info('Evaluating the model... (%d/%d)' % (step, 1000))

        step += 1

    metrics = {}
    num_answer = sum([log['num_answer'] for log in logs])
    for metric in logs[0].keys():
        if metric == 'num_answer':
            continue
        if 'm' in metric:
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        else:
            metrics[metric] = sum([log[metric] for log in logs])/num_answer

    return metrics
