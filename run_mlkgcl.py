from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parameter_dict = {
       # 'neg_sampling': None,
    }
    config_file_list = ['./mlkgcl.yaml']
    run_recbole(model='MLKGCL', dataset='ml-1m', config_file_list=config_file_list, config_dict=parameter_dict)