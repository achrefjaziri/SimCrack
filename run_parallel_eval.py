'''
Parallelized script for computing the difference accuracy scores for crack detection
'''
import multiprocessing as mp
from multiprocessing import Pool
import glob, os
import pandas as pd
from lib.arg_parser.general_args import parse_args_eval
from lib.eval.compute_scores import compute_scores


def run_parallel_eval():
    # print(mp.cpu_count())
    print("Evaluation...")
    args = parse_args_eval()

    res_arr = glob.glob(str(args.pred_path)+ '/segmentations' + "/*")
    print('number of images:', len(res_arr))

    current_model_name = os.path.basename(os.path.abspath(os.path.join(args.pred_path, "..")))
    csv_path = os.path.join(args.save_dir,'csv_results','eval_results_combi',f'results_{args.dataset}_{current_model_name}_{os.path.basename(args.pred_path)}.csv')
    print(csv_path)
    if os.path.exists(csv_path):
        # if the csv file exists remove the already evaluated images from res_arr
        df = pd.read_csv(csv_path)
        evaluated_images = df['Img Name'].tolist()
        abs_path =os.path.abspath(res_arr[0])
        evaluated_images = [os.path.join(abs_path,img_name)for img_name in evaluated_images]
        set_new_images = set(res_arr) - set(evaluated_images)
        res_arr = list(set_new_images)

    print('number of images left:', len(res_arr))

    # Each Dataset has a slightly different naming scheme for the annotations.
    if args.dataset == 'SimResist':
        res_arr = [(args.gt_path + '/' + os.path.basename(img_path), img_path) for img_path in
                   res_arr]
    elif args.dataset == 'RealResist' or  args.dataset == 'MultiSet' :
        res_arr = [(args.gt_path + '/' + os.path.basename(img_path), img_path) for img_path in
                   res_arr]
    elif args.dataset == 'CrackForest':
        res_arr = [(args.gt_path + '/' + os.path.basename(img_path), img_path) for img_path in
                   res_arr]  # .replace('render_noise','gt') replace('.png','_annot.png') .replace('.jpg.png','_label.PNG') .replace('.png','_annot.png')


    mp.set_start_method('spawn')
    p = Pool(args.num_cpus)

    args_list = [(args,csv_path,i)for i in res_arr] #Create a list to pass multiple arguments
    print('Args list done. Starting evaluation now..')
    p.map(compute_scores, args_list)
    print("Saving the results..")

if __name__=="__main__":
    run_parallel_eval()




