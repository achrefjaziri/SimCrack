import cv2
import numpy
import glob,os


if __name__=="__main__":
    path_gt ="/Users/achrefjaziri/Desktop/resist_tmp/all_annots"
    path_imgs="/Users/achrefjaziri/Desktop/resist_tmp/clean_cracks"
    path_unet="/Users/achrefjaziri/Desktop/resist_tmp/unet_segs"
    path_pmiunet="/Users/achrefjaziri/Desktop/resist_tmp/noneq_pmi_segs"

    save_dir = "/Users/achrefjaziri/Desktop/resist_tmp/concat_res"
    all_imgs = glob.glob('/Users/achrefjaziri/Desktop/resist_tmp/clean_cracks/*')
    print(all_imgs)
    for img_path in all_imgs:
        img = cv2.imread(img_path)
        img_gt = cv2.imread(os.path.join(path_gt,os.path.basename(img_path)))
        img_unet = cv2.imread(os.path.join(path_unet,os.path.basename(img_path)))

        img_pmi = cv2.imread(os.path.join(path_pmiunet,os.path.basename(img_path)))

        im_v = cv2.vconcat([img, img_gt,img_unet,img_pmi])
        cv2.imwrite(os.path.join(save_dir,os.path.basename(img_path)),im_v)