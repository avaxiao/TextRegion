source ~/.bashrc
conda activate TextRegion

cd sam2
pip install -e .
cd ..

pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

cd mmseg
pip install -v -e .
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
cd ..