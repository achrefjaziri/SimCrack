import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from run_parallel_eval import evaluate


if __name__=="__main__":
    npy_path = '/data/resist_data/adaIn_maps/sim_crack/train/mar18_gt115.png/DTD/style4.npy'
    pmi_maps = np.load(npy_path)[0].transpose(1,2,0)
    print(pmi_maps.shape)

    # Equalization
    #img_eq = exposure.equalize_hist(pmi_maps[:,:,0])

    # Adaptive Equalization
    #img_adapteq = exposure.equalize_adapthist(pmi_maps[:,:,0], clip_limit=0.03)

    plt.figure()
    plt.imshow(pmi_maps)
    plt.show()

    #plt.figure()
    #plt.imshow(img_eq)
    #plt.show()

    #plt.figure()
    #plt.imshow(img_adapteq)
    #plt.show()






