"""
Long-Tail Problem Validation
"""
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model
import pandas as pd

def get_rec_recult(model_file, save_file, topK):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file,
    )
    uid_series = test_data.uid_list
    # uid_series = torch.Tensor(uid_series).type(torch.long)
    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=topK, device=config['device'])
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

    # external_item_list to external_entity_list
    item_entity_dic = dataset.item2entity
    entities_list = []
    for one_rec in external_item_list:
        entity_list = []
        for one_item in one_rec:
            if one_item in item_entity_dic:
                entity_list.append(item_entity_dic.get(one_item))
        entities_list.append(entity_list)

    col_list = []
    for i in range(topK):
        col_list.append(str(i+1))

    data = pd.DataFrame(entities_list, columns=col_list)
    data_count = pd.concat([data[i] for i in col_list], axis=0).value_counts(sort=True)
    print(data_count)
    data_count.to_csv(save_file)

def count_rec_recult(result_file, topk):
    rec_result_data = pd.read_csv(result_file)
    # group file
    group_file_dic = './cut_dataset/ml-item-'+str(topK)+'/ml_cut_item_'
    group_score_list = []
    group_user_num = []

    for i in range(topk):
        group_file = group_file_dic + str(i) + '.csv'
        # print(group_file)
        group_data = pd.read_csv(group_file)
        group_id_list = group_data['id'].tolist()
        group_score = 0
        user_num = 0

        for j, r_rec in rec_result_data.iterrows():

            if r_rec['id'] in group_id_list:
                # print(r_rec['id'])
                user_num += 1
                # print(user_num)
                group_score += int(r_rec['score'])
                # print(group_score)
        group_score_list.append(group_score)
        group_user_num.append(user_num)
    print(group_score_list)
    print(group_user_num)
    # pd.DataFrame(group_score_list).to_csv(group_file_dic + 'KGAT_score.csv')
    # pd.DataFrame(group_score_list).to_csv(group_file_dic + 'ML-KGCL_score.csv')

def compute_recall_score(topK):

    # AB dataset
    KGAT = [194792, 106500, 91555, 73657, 59739, 52358, 43925, 34667, 27233, 14874]
    KGAT_user_number = [132, 288, 459, 663, 968, 1372, 1940, 2702, 3595, 4186]
    ML_KGCL = [137235, 101825, 89006, 75740, 66122, 59231, 53288, 45483, 40848, 30856]
    ML_KGCL_user_number = [132, 290, 466, 670, 974, 1388, 2017, 2988, 4605, 7488]

    # # ML dataset
    # ML_KGCL = [198466, 146051, 92612, 70266, 55031, 48301, 36577, 26963, 16406, 7767]
    # ML_KGCL_user_number = [40, 65, 91, 123, 170, 245, 349, 532, 918, 1664]
    # KGAT = [228734, 134893, 83874, 66767, 57028, 45883, 33407, 24609, 16147, 7004]
    # KGAT_user_number = [40, 65, 91, 123, 170, 244, 349, 532, 935, 1708]

    KGAT_sum = sum(KGAT)
    ML_KGCL_sum = sum(ML_KGCL)
    KGAT_user_number_sum = sum(KGAT_user_number)
    ML_KGCL_user_number_sum = sum(ML_KGCL_user_number)

    print(KGAT_user_number_sum)
    print(ML_KGCL_user_number_sum)

    recall_KGAT_list = []
    recall_ML_KGCL_list = []

    recall_KGAT_sum = 0
    recall_ML_KGCL_sum = 0

    for i in range(topK):
        recall_KGAT = format(KGAT[i] / KGAT_sum, '.5f')
        recall_ML_KGCL = format(ML_KGCL[i] / ML_KGCL_sum, '.5f')

        recall_KGAT_list.append(recall_KGAT)
        recall_ML_KGCL_list.append(recall_ML_KGCL)

    print('KGAT: ')
    print(recall_KGAT_list)
    print('ML_KGCL: ')
    print(recall_ML_KGCL_list)

if __name__ == '__main__':
    # model and dataset
    model_file = '../saved/mlkgcl/model_filename'

    # topK
    topK = 10

    # save file
    save_file = './recall_group_csv/ML/KGAT_ML_rec_list_'+str(topK)+'.csv'
    # save_file = './recall_group_csv/ML/ML-KGCL_ML_rec_list_'+str(topK)+'.csv'

    # Get the top10 recommendations for each user
    # get_rec_recult(model_file, save_file, topK)

    # Group recommendations according to the exposure of items
    # count_rec_recult(save_file, topK)

    # Calculate the Recall contribution for each group
    compute_recall_score(topK)
