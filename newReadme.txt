!pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
!pip install numpy scipy trimesh torch_kmeans timm==0.6.13

!pip install torchinfo

!rm -rf NeurCross/
!git clone https://github.com/Ammarkhat/NeurCross.git

!cp -rf NeurCross/data/doubleTorus/ doubleTorus/

!python NeurCross/quad_mesh/train_quad_mesh.py