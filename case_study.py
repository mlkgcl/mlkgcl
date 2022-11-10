"""
Case study
"""
import torch
from recbole.utils.case_study import full_sort_topk, full_sort_scores
from recbole.quick_start import load_data_and_model
import numpy as np


if __name__ == '__main__':
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file = './saved/mlkgcl/MLKGCL-ML-1m.pth',
    )
    torch.set_printoptions(precision=3)

    # Select the ids of a users
    uid_series = dataset.token2id(dataset.uid_field, ['208'])
    uid_series = torch.Tensor(uid_series).type(torch.long)

    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    topk_score_list = topk_score.tolist()

    result = dict(zip(external_item_list[0], topk_score_list[0]))
    print('result: ' + str(result))
