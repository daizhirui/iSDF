# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if [ -z "$(which gdown)" ]; then
    pip install gdown
fi

cd data

if [ ! -d franka_iSDF ]; then
    echo "Downloading the realsense_franka data ..."
    gdown https://drive.google.com/drive/folders/1tBk0W4wmytbISqSFg7A7eDvGcBt6uouJ?usp=sharing --folder
fi

if [ -d realsense_franka ]; then
    rm -rf realsense_franka
fi

echo "Start unzipping ..."
unzip -q franka_iSDF/realsense_franka.zip

cd ..
echo "Sequence realsense_franka is ready!"
