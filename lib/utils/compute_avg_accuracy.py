import pandas as pd
import glob
import os
from save_history import save_eval_history



if __name__=="__main__":
    folder = "/home/ajaziri/resist_projects/SimCrack/workspace/experimental_results/eval_results_RealResist"

    all_csv_files = glob.glob(folder +'/*')

    avg_acc_csv_path ="/home/ajaziri/resist_projects/SimCrack/workspace/experimental_results/avg_results_RealResist.csv"

    for file_path in all_csv_files:
        print(file_path)
        df = pd.read_csv(file_path)
        print(df)
        dict_res =df.agg({'WCN':'mean','WCN_PER':'mean','F1':'mean','F1_Theta10':'mean','Hausdorff_RBF':'mean','Hausdorff_EUC':'mean'}).to_dict()
        dict_res['Model']= os.path.basename(file_path).replace('.csv','')


        # Creating Header
        d = {'Model': [], 'F1': [], 'WCN': [], 'WCN_PER': [],
             'Hausdorff_EUC': [],
             'Hausdorff_RBF': [], 'F1_Theta10': []}

        df_all = pd.DataFrame(data=d)
        save_eval_history(df_all, dict_res, avg_acc_csv_path)
