import os, sys

if os.uname()[1] == "marmalade.physics.upenn.edu":
    print("I'm on marmalade!")
    sys.exit("I don't know what to do on marmalade!")
elif os.uname()[1][:5] == "login" or os.uname()[1][:3] == "nid":
    print("I'm on perlmutter!")
    cache_dir = "/pscratch/sd/s/shubh/"
    patches_file = "/global/cfs/cdirs/des/shubh/transformers/ViT_weak_lensing/data/20230419_224x224/20230419_224x224.py"
    num_workers = 256
else:
    sys.exit("I don't know what computer I'm on!")

import numpy as np
from datasets import load_dataset
import multiprocessing as mp
import time
from pathlib import Path
import tqdm
from itertools import repeat

labels = ["H0", "Ob", "Om", "ns", "s8", "w0"]
fwhm = 3
radius = 8
data_subset = "noiseless"
output_name = f"20231107_patches_flatsky_fwhm{fwhm}_radius{radius}_{data_subset}"
output_dir = Path(__file__).parent.parent / "data" / output_name / "peaks"
output_dir.mkdir(parents=True, exist_ok=True)

def get_peaks(img):
    key = 0
    ny, nx = img.shape
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            val = img[i, j]
            # check if this pixel is higher than all 8 neighbours
            if np.all(val >= img[i-1:i+2, j-1:j+2]):
                key += 1
                yield key, i, j, val

def get_edges(peaks, radius, img):
    r = np.ceil(radius).astype(int)
    loc_peaks = np.zeros_like(img, dtype=int)
    loc_peaks[peaks["y"], peaks["x"]] = peaks["key"]
    loc_peaks = np.pad(loc_peaks, r, mode='constant')

    for key, y, x, val in peaks:
        y, x = y + r, x + r # add the padding
        patch = loc_peaks[y-r:y+r+1, x-r:x+r+1] # extract the patch
        patch_peaks = np.nonzero(patch) # find the peaks in the patch
        for ay, ax in zip(patch_peaks[0], patch_peaks[1]):
            akey = patch[ay, ax]
            ay, ax = ay - r, ax - r # remove the padding
            sep = (ay**2 + ax**2)
            ang = np.arctan2(ay, ax)
            if 0 < sep <= radius**2: # check radius
                yield (key, akey), (y-r, x-r), (ay+y-r, ax+x-r), np.sqrt(sep), ang

def get_one_graph(args):
    id, datapoint, labels, radius = args
    img = np.array(datapoint["map"])[:, :, 0]
    labels = np.array([datapoint[l] for l in labels])
    peaks = np.array(list(get_peaks(img)), \
                     dtype=[("key", int), ("y", int), ("x", int), ("val", float)])
    edges = np.array(list(get_edges(peaks, radius, img)), \
                     dtype=[("keys", int, 2), ("loc1", int, 2), ("loc2", int, 2), ("sep", float), ("ang", float)])
    np.savez(output_dir / f"{id}.npz", labels=labels, peaks=peaks, edges=edges)
    # print(f"Saved graph {id} with {len(peaks)} peaks and {len(edges)} edges")

if __name__ == "__main__":
    start_time = time.time()

    all_data = load_dataset(patches_file, data_subset, cache_dir=cache_dir)
    print(f"Loaded dataset in {time.time() - start_time:.2f} seconds")

    pool = mp.Pool(num_workers)
    print(f"Created pool of {num_workers} workers in {time.time() - start_time:.2f} seconds")

    start_ind = 0
    for subset in ["train", "validation", "test"]:
        data = all_data[subset]
        args = zip(start_ind + np.arange(len(data)), data, repeat(labels), repeat(radius))
        print(f"Processing {subset} set with {len(data)} images")
        start_ind += len(data)
        for _ in tqdm.tqdm(pool.imap_unordered(get_one_graph, args), total=len(data)):
            pass

    pool.close()
    pool.join()

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
            