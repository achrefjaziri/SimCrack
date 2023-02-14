'''
Parallelized script for computing the difference accuracy scores for crack detection
'''
import multiprocessing as mp
from multiprocessing import Pool
import glob, os
import pandas as pd
from lib.arg_parser.general_args import parse_args_eval
from lib.eval.compute_scores import compute_scores
from lib.utils.save_history import save_eval_history, make_csv_file,remove_duplicate_headers


def get_log_value(f, class_name, type='string'):
    matching = [s for s in f if class_name in s]
    if matching == []:
        return ''
    my_string = matching[-1]
    if type == 'boolean':
        return bool(my_string.split(class_name, 1)[1])
    elif type == 'int':
        print('what I extracted Int', my_string.split(class_name, 1)[1])
        return int(my_string.split(class_name, 1)[1])
    elif type == 'float':
        return float(my_string.split(class_name, 1)[1])
    else:
        return my_string.split(class_name, 1)[1]


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

    dict_res['Fuse_pred'] = get_log_value(f, 'Fuse Predictions:', type='boolean')
    dict_res['Attn_conn'] = get_log_value(f, 'Attention Connections:', type='boolean')
    dict_res['Cons_loss'] = get_log_value(f, 'Consistency Loss:', type='boolean')
    dict_res['AdaIn'] = get_log_value(f, 'AdaIN:', type='boolean')
    print('current dic',dict_res)
    dict_res['Batch_size'] = get_log_value(f, 'Batch size:', type='int')
    dict_res['Lr'] = get_log_value(f, 'Learning rate:', type='float')
    dict_res['Phi_value'] = get_log_value(f, 'PMI Phi Value:', type='float')
    dict_res['Hist_eq'] = get_log_value(f, 'PMI Hist Eq:', type='boolean')

    with open(eval_log) as f:
        f = f.readlines()
    dict_res['Patch_size'] = get_log_value(f, 'Resize size:', type='int')
    print('resize Size', dict_res['Patch_size'])
    dict_res['Eval_mode'] = get_log_value(f, 'Patchwise eval:', type='boolean')


def evaluate(args):
    res_arr = glob.glob(str(args.pred_path) + '/segmentations' + "/*")
    print('number of images:', len(res_arr))

    current_model_name = os.path.basename(os.path.abspath(os.path.join(args.pred_path, "..")))
    csv_path = os.path.join(args.save_dir, 'experimental_results', f'eval_results_{args.dataset}',
                            f'{current_model_name}_{os.path.basename(args.pred_path)}.csv')
    print(csv_path)
    if os.path.exists(csv_path):
        # if the csv file exists remove the already evaluated images from res_arr
        df = pd.read_csv(csv_path)
        evaluated_images = df['Img Name'].tolist()
        abs_path = os.path.abspath(res_arr[0])
        evaluated_images = [os.path.join(abs_path, img_name) for img_name in evaluated_images]
        set_new_images = set(res_arr) - set(evaluated_images)
        res_arr = list(set_new_images)

    print('number of images left:', len(res_arr))

    # Each Dataset has a slightly different naming scheme for the annotations.
    if args.dataset == 'SimResist':
        res_arr = [(args.gt_path + '/' + os.path.basename(img_path), img_path) for img_path in
                   res_arr]
    elif args.dataset == 'RealResist' or args.dataset == 'MultiSet':
        res_arr = [(args.gt_path + '/' + os.path.basename(img_path), img_path) for img_path in
                   res_arr]
    elif args.dataset == 'CrackForest':
        res_arr = [(args.gt_path + '/' + os.path.basename(img_path), img_path) for img_path in
                   res_arr]  # .replace('render_noise','gt') replace('.png','_annot.png') .replace('.jpg.png','_label.PNG') .replace('.png','_annot.png')

    #mp.set_start_method('spawn')
    p = Pool(args.num_cpus)

    args_list = [(args, csv_path, i) for i in res_arr]  # Create a list to pass multiple arguments
    print('Args list done. Starting evaluation now..')
    p.map(compute_scores, args_list)
    print("Saving the results..")

    print('Computing avg accuracy')

    avg_acc_csv_path = os.path.join(args.save_dir, 'experimental_results', f'avg_results_{args.dataset}.csv')
    #remove_duplicate_headers(csv_path=csv_path)
    df = pd.read_csv(csv_path, delimiter=',', header=0)
    #df = df[df.WCN.str.contains('WCN') == False] # remove header duplicates
    #df[df.ne(df.columns).any(1)]
    df.drop(df.loc[df['Img Name'] == 'Img Name'].index, inplace=True)

    dict_res = df.agg({'WCN': 'mean', 'WCN_PER': 'mean', 'F1': 'mean', 'F1_Theta10': 'mean', 'Hausdorff_RBF': 'mean',
                       'Hausdorff_EUC': 'mean'}).to_dict()
    dict_res['Model'] = os.path.basename(args.pred_path)  # os.path.basename(csv_path).replace('.csv', '')
    dict_res['Arch'] = current_model_name
    dict_res['Rbf_l'] = args.rbf_l
    dict_res['Training_set'] = args.train_dataset
    dict_res['Set_size'] = args.set_size

    get_model_info(dict_res, args)

    # TODO adjust this Creating Header
    d = {'Model': [], 'Arch': [], 'Training_set': [], 'Set_size':[], 'Batch_size': [], 'Lr': [], 'Phi_value': [], 'Hist_eq': [],
         'Eval_mode': [], 'Patch_size': [], 'Rbf_l': [],'Fuse_pred':[],'Attn_conn':[],'Cons_loss':[],'AdaIn':[],
         'F1': [], 'WCN': [], 'WCN_PER': [],
         'Hausdorff_EUC': [],
         'Hausdorff_RBF': [], 'F1_Theta10': []}



    df_all = pd.DataFrame(data=d)
    save_eval_history(df_all, dict_res, avg_acc_csv_path)


if __name__ == "__main__":
    print("Evaluation...")
    args = parse_args_eval()
    evaluate(args)

    # TODO first get all models from trained_nets
    # Function to get all args for each one
    # Function to check what is already computed in csv
    # Run segmentation script
    # after all of them are done. get the paths for all sets in ajaziri
    # Run parallel eval for each one
