# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if [ -z "$(which gdown)" ]; then
    pip install gdown
fi

cd data

if [ ! -d data_full ]; then
    echo "Downloading the sequences and ground truth SDFs ..."
    gdown https://drive.google.com/drive/folders/1nzAVDInjDwt_GFehyhkOZvXrRJ33FCaR?usp=sharing --folder
fi

if [ -d seqs ]; then
    rm -rf seqs
fi

if [ -d gt_sdfs ]; then
    rm -rf gt_sdfs
fi

if [ -d eval_pts ]; then
    rm -rf eval_pts
fi

echo "Start unzipping ..."
unzip data_full/seqs.zip
unzip data_full/gt_sdfs.zip
unzip data_full/eval_pts.zip

cd ..
echo "Dataset is ready!"
