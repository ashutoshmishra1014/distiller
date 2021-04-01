'''
The copyright in this software is being made available under this Software
Copyright License. This software may be subject to other third party and
contributor rights, including patent rights, and no such rights are
granted under this license.
Copyright (c) 1995 - 2019 Fraunhofer-Gesellschaft zur Förderung der
angewandten Forschung e.V. (Fraunhofer)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted for purpose of testing the functionalities of
this software provided that the following conditions are met:
*     Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
*     Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
*     Neither the names of the copyright holders nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT CLAIMS, INCLUDING
WITHOUT LIMITATION THE PATENTS OF THE COPYRIGHT HOLDERS AND
CONTRIBUTORS, ARE GRANTED BY THIS SOFTWARE LICENSE. THE
COPYRIGHT HOLDERS AND CONTRIBUTORS PROVIDE NO WARRANTY OF PATENT
NON-INFRINGEMENT WITH RESPECT TO THIS SOFTWARE.
'''
from __future__ import absolute_import, division, print_function

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import dcase_util
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.transform import resize
import copy
from PIL import Image  # using pillow-simd for increased speed




class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        test_cases=20,
        seed=42,
    ):
        assert subset in ["all", "train", "test"]

        # read images
        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])

        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            # test_patients = random.sample(self.patients, k=int((len(self.patients)*test_cases)//100.))
            test_patients = self.patients[: int((len(self.patients)*test_cases)//100.)]
            if subset == "test":
                self.patients = test_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(test_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [self._crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [self._pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [self._resize_sample(v, size=image_size) for v in self.volumes]

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(self._normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform


    def _crop_sample(self, x):
        volume, mask = x
        volume[volume < np.max(volume) * 0.1] = 0
        z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
        z_nonzero = np.nonzero(z_projection)
        z_min = np.min(z_nonzero)
        z_max = np.max(z_nonzero) + 1
        y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
        y_nonzero = np.nonzero(y_projection)
        y_min = np.min(y_nonzero)
        y_max = np.max(y_nonzero) + 1
        x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
        x_nonzero = np.nonzero(x_projection)
        x_min = np.min(x_nonzero)
        x_max = np.max(x_nonzero) + 1
        return (
            volume[z_min:z_max, y_min:y_max, x_min:x_max],
            mask[z_min:z_max, y_min:y_max, x_min:x_max],
        )


    def _pad_sample(self, x):
        volume, mask = x
        a = volume.shape[1]
        b = volume.shape[2]
        if a == b:
            return volume, mask
        diff = (max(a, b) - min(a, b)) / 2.0
        if a > b:
            padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
        else:
            padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
        padding = padding + ((0, 0),)
        volume = np.pad(volume, padding, mode="constant", constant_values=0)
        return volume, mask


    def _resize_sample(self, x, size=256):
        volume, mask = x
        v_shape = volume.shape
        out_shape = (v_shape[0], size, size)
        mask = resize(
            mask,
            output_shape=out_shape,
            order=0,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        out_shape = out_shape + (v_shape[3],)
        volume = resize(
            volume,
            output_shape=out_shape,
            order=2,
            mode="constant",
            cval=0,
            anti_aliasing=False,
        )
        return volume, mask


    def _normalize_volume(self, volume):
        p10 = np.percentile(volume, 10)
        p99 = np.percentile(volume, 99)
        volume = rescale_intensity(volume, in_range=(p10, p99))
        m = np.mean(volume, axis=(0, 1, 2))
        s = np.std(volume, axis=(0, 1, 2))
        volume = (volume - m) / s
        return volume


    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).long()

        # return tensors
        return image_tensor, mask_tensor




class DCaseDataset(Dataset):
    def __init__(self, root_dir, train=True, validate=False, transform=None, use_precomputed_labels=False, **kwargs):
        """
        Dataset loader for mel spectrograms.

        Parameters
        ----------
        root: str
            Path to the folder that contains spectrograms in npz format and the labels in csv format.
        train: bool
            If False return test otherwise train dataset.
        validate: bool
            If True return the split for validation set otherwise the test set. If train=True the parameter is ignored.
        transform: torchvision.transforms
            Transforms the data according to the given transformation functions.
        use_precomputed_labels: bool
            If True precomputed labels should be available.
        """
        self.transforms = transform
        self.root = root_dir

        if train and not validate:
            self.paths = glob.glob(os.path.join(root_dir, 'train', '*.cpickle'))
            self.labels = pd.concat((
                pd.read_csv(os.path.join(root_dir, 'fold1_train.txt'), header=None, names=['path', 'label'], sep='\t'),
                pd.read_csv(os.path.join(root_dir, 'fold1_evaluate.txt'), header=None, names=['path', 'label'], sep='\t')
            ))
        else:
            self.paths = glob.glob(os.path.join(root_dir, 'test', '*.cpickle'))
            if use_precomputed_labels:
                assert validate, 'Implemented only for validate=True'
                self.labels = pd.read_csv(os.path.join(root_dir, 'self_evaluate.txt'), sep='\t')
                print('Precomputed labels used')
            else:
                self.labels = pd.read_csv(os.path.join(root_dir, 'evaluate.txt'), header=None, names=['path', 'label'],
                                          sep='\t')

            # Do split between validation and test set based on a csv file
            if validate:
                split = pd.read_csv(os.path.join(root_dir, 'dcase2019_baseline_eval.txt'), header=None, names=['path'])
            else:
                split = pd.read_csv(os.path.join(root_dir, 'dcase2019_baseline_test.txt'), header=None, names=['path'])
            split = split.path.str.split('/').str.get(1).str.replace('.wav', '.cpickle').tolist()
            self.paths = [x for x in self.paths if os.path.basename(x) in split]

        self.labels.index = self.labels.path.str.split('/').str.get(1).str.replace('.wav', '.cpickle')

        # Make mapping from string labels to numerical labels
        unique = sorted(self.labels.label.unique())
        assert len(unique) == 15, 'Not all labels included {}'.format(unique)
        self.mapping = dict(zip(sorted(unique), range(len(unique))))

        self.processing_chain = self.get_processing_chain()

    def __getitem__(self, index):
        path = self.paths[index]
        data = self.processing_chain.process(filename=path).data.astype('float32')
        name = os.path.basename(path).replace('.wav', '.cpickle')
        label = self.labels.loc[name, 'label']

        # label = torch.tensor(self.mapping[label])
        # label = (label == torch.arange(15).reshape(15)).long()
        # return {"inputs": data, "labels": label}
        
        return torch.tensor(data), self.mapping[label]



    def __len__(self):
        return len(self.paths)

    def get_processing_chain(self):
        """Unfortunately easiest way. We need to keep these dependencies."""
        # Read application default parameter file
        chain_list = [{'processor_name': 'dcase_util.processors.FeatureReadingProcessor', 'dependency_parameters': {'method': 'mel', 'win_length_seconds': 0.04, 'hop_length_seconds': 0.02, 'fs': 48000, 'win_length_samples': 1920, 'hop_length_samples': 960, 'parameters': {'spectrogram_type': 'magnitude', 'window_type': 'hamming_asymmetric', 'n_mels': 40, 'n_fft': 2048, 'fmin': 0, 'fmax': 24000, 'htk': False, 'normalize_mel_bands': False, 'method': 'mel', 'fs': 48000, 'win_length_seconds': 0.04, 'win_length_samples': 1920, 'hop_length_seconds': 0.02, 'hop_length_samples': 960}}}, {'processor_name': 'dcase_util.processors.NormalizationProcessor', 'init_parameters': {'enable': True, 'filename': '/data/mpeg_nnr_uc16-dcase-v1.2/system/task1a/normalization/feature_extractor_3efdb8815e181fcac7879e74d59c9d4a/norm_fold_all_data.cpickle'}}, {'processor_name': 'dcase_util.processors.SequencingProcessor', 'init_parameters': {'sequence_length': 500, 'hop_length': 500}}, {'processor_name': 'dcase_util.processors.DataShapingProcessor', 'init_parameters': {'axis_list': ['sequence_axis', 'data_axis', 'time_axis']}}]

        data_processing_chain = dcase_util.processors.ProcessingChain()
        for chain in chain_list:
            processor_name = chain.get('processor_name')
            init_parameters = chain.get('init_parameters', {})

            # Inject parameters
            if processor_name == 'dcase_util.processors.NormalizationProcessor':
                init_parameters['filename'] = os.path.join(self.root, 'norm_fold_all_data.cpickle')

            data_processing_chain.push_processor(
                processor_name=processor_name,
                init_parameters=init_parameters,
            )
        return data_processing_chain


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError



class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt