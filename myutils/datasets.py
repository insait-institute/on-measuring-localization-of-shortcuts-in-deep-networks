from typing import Optional

import os
import pickle
from multiprocessing import Pool

import numpy as np
import cv2

import torch

import mytorch as my
from . import transforms


# generate datasets
def gen_cifar_10_mnist(
        *, root: str, split: str, match_seed: int, s_size: int, s_a: float,
        ) -> None:
    """Generate CIFAR-10 dataset with MNIST patch-like skew"""
    cifar_samples\
        = my.datasets.cifar10(root=root, split=split)
    mnist_samples\
        = my.datasets.mnist(root=root, split=split)

    rng = np.random.default_rng(match_seed)
    random_labels = rng.integers(10, size=len(cifar_samples))
    rng.shuffle(mnist_samples)

    mnist_x = [
        np.expand_dims(
            cv2.resize(x, (s_size, s_size), interpolation=cv2.INTER_LINEAR), 2)
        for x, _ in mnist_samples]
    mnist_classes = [[] for _ in range(10)]
    for i, (_, y) in enumerate(mnist_samples):
        mnist_classes[y].append(i)

    samples = [(x, x.copy(), y) for x, y in cifar_samples]
    common_counter = [0 for _ in range(10)]
    for (x, s_x, y), r_y in zip(samples, random_labels):
        if y != r_y:
            continue
        c_i = common_counter[y]
        common_counter[y] = min(c_i+1, len(mnist_classes[y])-1)
        z = mnist_x[mnist_classes[y][c_i]]
        my.transforms.blend(x[:s_size, :s_size], z, s_a)
        my.transforms.blend(s_x[:s_size, :s_size], z, s_a)
    r_counter = [common_counter[i] for i in range(10)]
    s_counter = [common_counter[i] for i in range(10)]
    for (x, s_x, y), r_y in zip(samples, random_labels):
        if y == r_y:
            continue
        r_i = r_counter[r_y]
        s_i = s_counter[y]
        r_counter[r_y] = min(r_i+1, len(mnist_classes[r_y])-1)
        s_counter[y] = min(s_i+1, len(mnist_classes[y])-1)
        r_z = mnist_x[mnist_classes[r_y][r_i]]
        s_z = mnist_x[mnist_classes[y][s_i]]
        my.transforms.blend(x[:s_size, :s_size], r_z, s_a)
        my.transforms.blend(s_x[:s_size, :s_size], s_z, s_a)
    with open(
        f"{root}/cifar-10-mnist/{split}-s{match_seed}-s{s_size}-a{s_a*100:.0f}.pickle",
        "wb") as f:
        pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)


class WaterbirdsSaver:
    """Class for saving Waterbirds dataset"""
    save_dir: str


    def __init__(self, save_dir: str):
        self.save_dir = save_dir


    def __call__(self, batch: tuple[str, str, str, str, int]) -> None:
        """Load waterbirds sample"""
        img_x, seg_x, c_z, s_z, split, y = batch

        split_x = img_x.split("/")
        if y==0:
            save_path = f"0-landbirds"
        elif y==1:
            save_path = f"1-waterbirds"
        c_path = f"{save_path}/{split_x[-1]}"
        s_path = f"{save_path}/{split_x[-1][:-4]}_s.jpg"

        img_x = cv2.imread(img_x)
        seg_x = cv2.imread(seg_x)
        c_z = cv2.imread(c_z)
        s_z = cv2.imread(s_z)

        x_vsize, x_hsize, _ = img_x.shape
        c_z = transforms.match_img(c_z, x_vsize, x_hsize)
        s_z = transforms.match_img(s_z, x_vsize, x_hsize)

        mask_x = seg_x / 255
        img_x = mask_x * img_x
        mask_x = 1 - mask_x
        c_z = mask_x * c_z
        s_z = mask_x * s_z
        c_z += img_x
        s_z += img_x
        x = c_z.round().astype(np.uint8)
        s_x = s_z.round().astype(np.uint8)

        cv2.imwrite(f"{self.save_dir}/{c_path}", x)
        cv2.imwrite(f"{self.save_dir}/{s_path}", s_x)
        return c_path, s_path, split, y


def gen_waterbirds(*, root: str, match_seed: int) -> None:
    """Generate Waterbirds dataset"""
    places_img_dir = f"{root}/places365/data_large_standard"

    cub_img_dir = f"{root}/cub-200-2011/CUB_200_2011/images"
    cub_seg_dir = f"{root}/cub-200-2011/segmentations"
    cub_split_dir = f"{root}/cub-200-2011/CUB_200_2011"

    save_dir = f"{root}/waterbirds/s{match_seed}"

    # target classes
    place_dirs = [
        ['b/bamboo_forest', 'f/forest/broadleaf'],  # Land backgrounds
        ['o/ocean', 'l/lake/natural']]              # Water backgrounds
    waterbird_names = (
        'Albatross', # Seabirds
        'Auklet',
        'Cormorant',
        'Frigatebird',
        'Fulmar',
        'Gull',
        'Jaeger',
        'Kittiwake',
        'Pelican',
        'Puffin',
        'Tern',
        'Gadwall', # Waterfowl
        'Grebe',
        'Mallard',
        'Merganser',
        'Guillemot',
        'Pacific_Loon')

    # find all background files
    places_files = [
        [
            f"{dir}/{file}" for dir in place
            for file in os.listdir(f"{places_img_dir}/{dir}")]
        for place in place_dirs]

    # get split flag
    in_split = {
        name[name.find(" ")+1:-1]: bool(int(is_train[-2]))
        for name, is_train in zip(
            open(f"{cub_split_dir}/images.txt"),
            open(f"{cub_split_dir}/train_test_split.txt"))}

    # divide directories
    birds_dirs = [[], []]
    for dir in os.listdir(cub_img_dir):
        birds_dirs[dir.endswith(waterbird_names)].append(dir)

    # select birds files
    birds_files = [
        (f"{dir}/{file}", in_split[f"{dir}/{file}"], i)
        for i, bird in enumerate(birds_dirs)
        for dir in bird
        for file in os.listdir(f"{cub_img_dir}/{dir}") if file[-7:-4] != "rgb"]

    # choose matching for correlated and uncorrelated backgrounds
    rng = np.random.default_rng(match_seed)
    random_labels = rng.integers(2, size=len(birds_files))
    rng.shuffle(places_files[0])
    rng.shuffle(places_files[1])

    # construct final dataset
    samples = [None] * len(birds_files)
    common_counter = [0 for _ in range(2)]
    for i, ((x, split, y), r_y) in enumerate(zip(birds_files, random_labels)):
        if y != r_y:
            continue
        c_i = common_counter[y]
        common_counter[y] = min(c_i+1, len(places_files[y])-1)
        z = places_files[y][c_i]
        samples[i] = (
            f"{cub_img_dir}/{x}", f"{cub_seg_dir}/{x[:-3]}png",
            f"{places_img_dir}/{z}", f"{places_img_dir}/{z}", split, y)
    r_counter = [common_counter[i] for i in range(2)]
    s_counter = [common_counter[i] for i in range(2)]
    for i, ((x, split, y), r_y) in enumerate(zip(birds_files, random_labels)):
        if y == r_y:
            continue
        r_i = r_counter[r_y]
        s_i = s_counter[y]
        r_counter[r_y] = min(r_i+1, len(places_files[r_y])-1)
        s_counter[y] = min(s_i+1, len(places_files[y])-1)
        r_z = places_files[r_y][r_i]
        s_z = places_files[y][s_i]
        samples[i] = (
            f"{cub_img_dir}/{x}", f"{cub_seg_dir}/{x[:-3]}png",
            f"{places_img_dir}/{r_z}", f"{places_img_dir}/{s_z}", split, y)
    if not os.path.exists(f"{save_dir}/0-landbirds"):
        os.makedirs(f"{save_dir}/0-landbirds")
    if not os.path.exists(f"{save_dir}/1-waterbirds"):
        os.makedirs(f"{save_dir}/1-waterbirds")
    saver = WaterbirdsSaver(save_dir)
    with Pool(8) as p:
        filelist = p.map(saver, samples)
    with open(f"{save_dir}/filelist.pickle", "wb") as f:
        pickle.dump(filelist, f, pickle.HIGHEST_PROTOCOL)


# initialize datasets
def add_skew_flag(
        samples: list[tuple], skew_seed: Optional[int], s_freq: float
        ) -> list[tuple]:
    """Add skew flag to samples"""
    g = torch.Generator()
    g.manual_seed(skew_seed)
    skew_flags = torch.empty(len(samples), dtype=torch.bool)
    skew_flags = skew_flags.bernoulli_(s_freq, generator=g).tolist()
    return [(*x, s) for x, s in zip(samples, skew_flags)]


def cifar_10_mnist(
        *, root: str, split: str, match_seed: int, s_size: int, s_a: float,
        ) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Initialize CIFAR-10 samples with skew"""
    with open(
        f"{root}/cifar-10-mnist/{split}-s{match_seed}-s{s_size}-a{s_a*100:.0f}.pickle",
        "rb") as f:
        samples = pickle.load(f)
    return samples


def waterbirds(
        *, root: str, split: str, match_seed: int,
        ) -> list[tuple[str, str, int]]:
    """Initialize Waterbirds dataset"""
    save_dir = f"{root}/waterbirds/s{match_seed}"
    with open(f"{save_dir}/filelist.pickle", "rb") as f:
        filelist = pickle.load(f)
    if split=="train":
        train = True
    elif split=="test":
        train = False
    else:
        raise ValueError(f"Split {split} is not implemented for Waterbirds")
    samples = [
        (f"{save_dir}/{x}", f"{save_dir}/{s_x}", y)
        for x, s_x, split, y in filelist if split == train]
    return samples


def prepare_celeba_attr(line: str) -> tuple[str, list[bool]]:
    """Read CelebA attributes csv"""
    name, *attrs = line.split()
    attrs = [int(flag)==1 for flag in attrs]
    return name, attrs


def celeba_subsampled(
        *, root: str, split: str, match_seed: int,
        ) -> list[tuple[str, str, int]]:
    """Initialize CelebA dataset"""
    with open(f"{root}/celeba/list_eval_partition.txt") as f:
        split_nums = [int(line[line.find(" ")+1:]) for line in f]
    if split == "train":
        split_num = [0]
    elif split == "valid":
        split_num = [1]
    elif split == "test":
        split_num = [2]
    elif split == "train+valid":
        split_num = [0, 1]
    else:
        raise ValueError(f"Split {split} is not implemented for CelebA")
    in_split = [num in split_num for num in split_nums]

    with open(f"{root}/celeba/list_attr_celeba.txt") as f:
        lines_iter = iter(f)
        next(lines_iter)
        header = next(lines_iter).split()
        filenames, attributes = zip(*(
            prepare_celeba_attr(line)
            for line, in_flag in zip(lines_iter, in_split) if in_flag))

    c_col = header.index("Blond_Hair")
    s_col = header.index("Male")
    c_labels = np.fromiter(
        (attr[c_col] for attr in attributes), bool, len(attributes))
    s_labels = np.fromiter(
        (attr[s_col] for attr in attributes), bool, len(attributes))

    groups = [
        [~c_labels & ~s_labels, ~c_labels & s_labels],
        [c_labels & ~s_labels, c_labels & s_labels]]

    # g11 -> smallest, g00 > g01
    sizes = [[g.sum() for g in g_class] for g_class in groups]
    min10 = sizes[1][0] - sizes[1][1]
    min01 = sizes[1][1] * sizes[0][1] // min10
    max00 = sizes[0][1] - min01

    rng = np.random.default_rng(match_seed)
    g_ind = [[g.nonzero()[0].tolist() for g in g_class] for g_class in groups]
    for g_class in g_ind:
        for g in g_class:
            rng.shuffle(g)
    c_ind = [
        g_ind[0][0][:max00] + g_ind[0][1][-min01:],
        g_ind[1][0][:min10] + g_ind[1][1]]
    s_ind = [
        g_ind[0][1],
        g_ind[1][0]]

    base_folder = f"{root}/celeba/img_align_celeba"
    samples = [
        (
            f"{base_folder}/{filenames[c_i]}",
            f"{base_folder}/{filenames[s_i]}",
            y)
        for y, (c_group, s_group) in enumerate(zip(c_ind, s_ind))
        for c_i, s_i in zip(c_group, s_group)]
    return samples
