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
import os
import glob

import dcase_util
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


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
