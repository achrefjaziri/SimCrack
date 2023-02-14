'''
Script to generate additional Info needed for ConsNet
'''


def str_to_bool(s):
    #print('str',s)
    if 'True' in s:
         return True
    elif 'False' in s:
         return False
def get_log_value(f, class_name, type='string'):
    '''
    Extracts parameter value from .log file
    :param f: the lines of the log file
    :param class_name:  the class that we are interested in
    :param type:  type of output. Default is string.
    :return:  returns value of the class_name found in f file.
    '''
    matching = [s for s in f if class_name in s]
    if matching == []:
        if type == 'boolean':
            return False
        elif type == 'int':
            return 0
        elif type == 'float':
            return 0
        else:
            return ''

    my_string = matching[-1]
    if type == 'boolean':
        #print('newwww')
        out_string= my_string.split(class_name, 1)[1]
        out_string = out_string.replace("\n", "")
        #print(my_string.split(class_name, 1)[1], str_to_bool('False'), str_to_bool(out_string))
        return str_to_bool(out_string)
    elif type == 'int':
        return int(my_string.split(class_name, 1)[1])
    elif type == 'float':
        return float(my_string.split(class_name, 1)[1])
    else:
        out_string = my_string.split(class_name, 1)[1]
        out_string = out_string.replace(" ", "")
        out_string = out_string.replace("\n", "")
        return out_string


def get_model_info(dict_res, args):
    """
    :param dict_res: dictionary containing all the current results
    :return: a dictionary containing the following model information : training_set, batch_size,lr, patch_size, phi_value, hist_eq
    """
    train_dir = os.path.join(args.save_dir, 'trained_nets', dict_res['Arch'], args.train_dataset, dict_res['Model'])

    train_log = os.path.join(train_dir,
                             'run_history_0.log')  # contains info about training batch size, lr , phi_values etc.

    eval_log = os.path.join(args.pred_path, 'run_history.log')  # contains the values of patch sizes

    with open(train_log) as f:
        f = f.readlines()

    dict_res['Batch_size'] = get_log_value(f, 'Batch size:', type='int')
    dict_res['Lr'] = get_log_value(f, 'Learning rate:', type='float')
    dict_res['Phi_value'] = get_log_value(f, 'PMI Phi Value:', type='float')
    dict_res['Hist_eq'] = get_log_value(f, 'PMI Hist Eq:', type='boolean')

    dict_res['AdaIn'] = get_log_value(f, 'Batch size:', type='int')
    dict_res['Lr'] = get_log_value(f, 'Learning rate:', type='float')
    dict_res['Phi_value'] = get_log_value(f, 'PMI Phi Value:', type='float')
    dict_res['Hist_eq'] = get_log_value(f, 'PMI Hist Eq:', type='boolean')

    with open(eval_log) as f:
        f = f.readlines()
    dict_res['Patch_size'] = get_log_value(f, 'Resize size:', type='int')
    print('resize Size', dict_res['Patch_size'])
    dict_res['Eval_mode'] = get_log_value(f, 'Patchwise eval:', type='boolean')

