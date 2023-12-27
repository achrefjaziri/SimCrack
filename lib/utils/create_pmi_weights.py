
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import os
from lib.arg_parser.general_args import parse_args
from lib.dataloaders.sim_dataloader import SimDataloader
import torch



def NormalizeData(data):
    return (data - jnp.min(data)) / (jnp.max(data) - jnp.min(data))

def luminance(a):

    """Get luminance"""
    return np.sqrt( 0.299*a[0]**2 + 0.587*a[1]**2 + 0.114*a[2]**2 )



def reconstruct_level1(tiles,luminance_img,h,w):
    # Reconstructing the image from non overlapping tiles
    affinity_matrix_tiles1 = np.zeros((luminance_img.shape[0],luminance_img.shape[1]))
    for x in range(0,luminance_img.shape[0],h):
        for y in range(0,luminance_img.shape[1],w):
            affinity_matrix_tiles1[x:x+h,y:y+w] = tiles.pop(0)
    return affinity_matrix_tiles1



def reconstruct_level2(tiles,luminance_img,h,w):
    # Reconstructing the image from overlapping tiles
    affinity_matrix_tiles2 = np.zeros((luminance_img.shape[0],luminance_img.shape[1]))
    avg_matrix = np.zeros((luminance_img.shape[0],luminance_img.shape[1]))
    half_h = (h//2)
    half_w = (w//2)
    for x in [0,half_h]:
        for y in [0,half_w]:
            affinity_matrix_tiles2[x:x+h+half_h,y:y+w+half_w] = affinity_matrix_tiles2[x:x+h+half_h,y:y+w+half_w] + tiles.pop(0)
            avg_matrix[x:x+h+half_h,y:y+w+half_h] = avg_matrix[x:x+h+half_h,y:y+w+half_w] + np.ones((h+half_h,w+half_w))
    return affinity_matrix_tiles2/avg_matrix



def pick_pixel_pairs(flattened_img, number_of_pairs):
    # Sample Pixel pairs
    pixel_pairs = np.random.randint(10, flattened_img.shape[0] - 10,
                                    size=number_of_pairs)  # pick some random positions in the image
    draw = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], number_of_pairs,
                            p=[0.05, 0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])

    first_pair = pixel_pairs - draw
    first_pair = np.clip(first_pair, 0, flattened_img.shape[0])

    second_pair = pixel_pairs + draw
    second_pair = np.clip(second_pair, 0, flattened_img.shape[0])

    pixels = np.row_stack((flattened_img[first_pair], flattened_img[second_pair]))
    return pixels, first_pair, second_pair


def calc_img_pmi(luminance_img, number_of_pairs=10000, neighbour_size=5, phi=1.75):
    # Vertical Sampling
    flattened_img = luminance_img.transpose().flatten()  # we transpose the image to sample pixels along vertical lines

    pixel_pairs_vert, first_pair, second_pair = pick_pixel_pairs(flattened_img, number_of_pairs)
    # the positions provided by first and second pair arrays are use to estimae P(A) and P(B). It is also possible to use the pairs provided by horizontal sampling. There is no difference in KDE.

    # horizontal Sampling
    flattened_img = luminance_img.flatten()

    pixel_pairs_hor, _, _ = pick_pixel_pairs(flattened_img, number_of_pairs)
    pixels = np.concatenate((pixel_pairs_hor, pixel_pairs_vert),
                            axis=1)  # all the sampled pixels along horizontal and vertical axis.

    # Kernel Density Estimation for P(A,B),P(A) and P(B)
    # KDE for P(A,B)
    kde = jscipy.stats.gaussian_kde(pixels, bw_method="scott")

    # KDE for P(A)

    kde_A = jscipy.stats.gaussian_kde(np.expand_dims(flattened_img[first_pair], axis=0), bw_method="scott")

    # KDE for P(B)

    kde_B = jscipy.stats.gaussian_kde(np.expand_dims(flattened_img[second_pair], axis=0), bw_method="scott")

    input_for_kdeA = np.expand_dims(flattened_img, axis=0)

    p_a_values = kde_A.logpdf(input_for_kdeA)
    p_b_values = kde_B.logpdf(input_for_kdeA)

    # We want to create numpy of all pixel pairs (a,b) that we need to estimate with joint distribution P(A,B)

    # Image padding is hardcoded and should change
    padded_img = np.pad(luminance_img, (neighbour_size // 2, neighbour_size // 2), 'constant', constant_values=0)
    b_values = np.lib.stride_tricks.sliding_window_view(padded_img, (neighbour_size, neighbour_size))

    list_to_concat = []
    for i in range(neighbour_size ** 2):
        list_to_concat.append(luminance_img)

    a_values = np.array(list_to_concat).reshape(neighbour_size, neighbour_size, luminance_img.shape[0],
                                                luminance_img.shape[1]).T
    a_values = np.swapaxes(a_values, 0, 1)
    a_b_values = np.stack((a_values, b_values), axis=4)  #
    a_b_values_reshaped = a_b_values.reshape(-1, 2)  # KDE function accepts inputs of shape (x,2)

    input_for_kdeAB = np.swapaxes(a_b_values_reshaped, 0, 1)
    p_ab_values = kde.logpdf(input_for_kdeAB)

    # Reshape P(A)

    list_to_concat_PA = []
    for i in range(neighbour_size ** 2):
        list_to_concat_PA.append(p_a_values)

    pa = np.array(list_to_concat_PA).reshape(neighbour_size, neighbour_size, luminance_img.shape[0],
                                             luminance_img.shape[1]).T
    pa = np.swapaxes(pa, 0, 1)

    # Reshape P(B)
    padded_array = np.pad(p_b_values.reshape(luminance_img.shape[0], luminance_img.shape[1]),
                          (neighbour_size // 2, neighbour_size // 2), 'constant', constant_values=0)
    pb = np.lib.stride_tricks.sliding_window_view(padded_array, (neighbour_size, neighbour_size))

    # Reshape P(A,B)
    pab = p_ab_values.reshape((luminance_img.shape[0], luminance_img.shape[1], neighbour_size, neighbour_size))

    # Important difference to the original paper:
    # In the original paper computed Wij values in the following way e^(PMI).
    # In this implementation the, we calculate the exponential of log likelihood first to avoid NaN values.

    # print(pab.shape,pa.shape,pb.shape)
    w_ij = np.exp(pab) ** (phi) / ((np.exp(pa) * np.exp(pb) + 0.000001))

    affinity_matrix = np.sum(w_ij.reshape(luminance_img.shape[0], luminance_img.shape[1], neighbour_size ** 2), axis=2)
    affinity_matrix_normalized = NormalizeData(affinity_matrix)  # >thresh

    return affinity_matrix_normalized


def compute_multi_scale_pmi(orig_img,neighbour_size=5,phi_val=2.25):
    luminance_img = np.apply_along_axis(luminance, 2, orig_img)

    luminance_img = NormalizeData(luminance_img)

    h = luminance_img.shape[0] // 2
    w = luminance_img.shape[1] // 2
    print('h,w vals',h,w, luminance_img.shape,orig_img.shape)
    half_h = (h // 2)
    half_w = (w // 2)
    # Consider the image patchwise when computing PMI scores.
    # Tiles1 contains non overlapping 128x128 patches and Tiles2 contains overlapping 192x192 patches.
    tiles1 = [luminance_img[x:x + h, y:y + w] for x in range(0, luminance_img.shape[0], h) for y in
              range(0, luminance_img.shape[1], h)]
    tiles2 = [luminance_img[x:x + h + half_h, y:y + w + half_w] for x in [0, half_h] for y in [0, half_w]]

    res_tiles1 = []
    res_tiles2 = []
    # compute weight scores for each image/tile
    for i in range(len(tiles2)):
        print('current img', i)
        aff_mat_2 = calc_img_pmi(tiles2[i], neighbour_size=neighbour_size,phi=phi_val)
        aff_mat_1 = calc_img_pmi(tiles1[i], neighbour_size=neighbour_size,phi=phi_val)

        res_tiles1.append(aff_mat_1)
        res_tiles2.append(aff_mat_2)

    aff_mat_full1 = reconstruct_level1(res_tiles1, luminance_img, h, w)
    aff_mat_full2 = reconstruct_level2(res_tiles2, luminance_img, h, w)

    # Compute weight scores for the full image
    aff_mat_full = calc_img_pmi(luminance_img, neighbour_size=neighbour_size,phi=phi_val)

    arr = (aff_mat_full, aff_mat_full2, aff_mat_full1)

    return np.dstack(arr)


def save_pmi_maps(orig_img,args,pmi_dir,curr_img_name):
    weight_maps = compute_multi_scale_pmi(orig_img,args.neighbour_size,args.phi_value)

    if not os.path.exists(os.path.dirname(pmi_dir)):
        os.makedirs(pmi_dir)
    np.save(os.path.join(pmi_dir,curr_img_name),weight_maps)


if __name__ == "__main__":
    args = parse_args()

    for mode in ['test']:

        Dataset_test = SimDataloader(args,mode=mode)

        test_load = \
            torch.utils.data.DataLoader(dataset=Dataset_test,
                                        num_workers=16, batch_size=1, shuffle=False)
        pmi_dir = os.path.join(args.pmi_dir,f'{args.neighbour_size}_{args.phi_value}',mode)
        for batch, data in enumerate(test_load):
            input_img = data['input'][0].detach().cpu().numpy().copy() #permute(1,2,0)
            img_name = os.path.basename(data['path'][0])
            print('image', input_img.shape,img_name)

            save_pmi_maps(input_img, args, pmi_dir, img_name)










