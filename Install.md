# Installation

## Setup Enviroment

This codebase is tested with `torch==2.2.0`, with `CUDA 11.8`. 

**Step 1: Create Enviroment**

```shell
conda create --name LCA python=3.9
```



**Step 2: Activate Enviroment**

`````shell
conda activate LCA
`````



**Step 3: Install PyTorch**

```shell
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```



**Step 4: Install Other Packages**

````shell
conda install numpy
conda install pandas
conda install einops
conda install scikit-learn
conda install wandb
conda install torchmetrics
````



## Enviroment Summary

We provide the list of all the packages and their corresponding versions used in this codebase:

```shell
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
aiohttp                   3.9.3                    pypi_0    pypi
aiosignal                 1.3.1                    pypi_0    pypi
appdirs                   1.4.4                    pypi_0    pypi
asttokens                 2.4.1                    pypi_0    pypi
async-timeout             4.0.3                    pypi_0    pypi
attrs                     23.2.0                   pypi_0    pypi
axial-positional-embedding 0.2.1                    pypi_0    pypi
blas                      1.0                    openblas  
bottleneck                1.3.7            py39ha9d4c09_0  
brotli                    1.0.9                h5eee18b_8  
brotli-bin                1.0.9                h5eee18b_8  
ca-certificates           2024.12.14           hbcca054_0    conda-forge
catboost                  1.2.5                    pypi_0    pypi
certifi                   2022.12.7                pypi_0    pypi
charset-normalizer        2.1.1                    pypi_0    pypi
click                     8.1.7                    pypi_0    pypi
colorama                  0.4.6            py39h06a4308_0  
colt5-attention           0.10.20                  pypi_0    pypi
contourpy                 1.2.1                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
docker-pycreds            0.4.0                    pypi_0    pypi
einops                    0.7.0                    pypi_0    pypi
exceptiongroup            1.2.0                    pypi_0    pypi
executing                 2.0.1                    pypi_0    pypi
filelock                  3.16.1                   pypi_0    pypi
fonttools                 4.51.0           py39h5eee18b_0  
freetype                  2.12.1               h4a9f257_0  
frozenlist                1.4.1                    pypi_0    pypi
fsspec                    2024.6.0                 pypi_0    pypi
gitdb                     4.0.11                   pypi_0    pypi
gitpython                 3.1.43                   pypi_0    pypi
graphviz                  0.20.3                   pypi_0    pypi
h5py                      3.10.0                   pypi_0    pypi
huggingface-hub           0.23.3                   pypi_0    pypi
idna                      3.4                      pypi_0    pypi
imageio                   2.33.1           py39h06a4308_0  
importlib-resources       6.4.0                    pypi_0    pypi
importlib_resources       6.1.1            py39h06a4308_1  
ipdb                      0.13.13                  pypi_0    pypi
ipython                   8.18.1                   pypi_0    pypi
jedi                      0.19.1                   pypi_0    pypi
jinja2                    3.1.2                    pypi_0    pypi
joblib                    1.4.2                    pypi_0    pypi
jpeg                      9e                   h5eee18b_1  
kiwisolver                1.4.5                    pypi_0    pypi
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libabseil                 20240116.2      cxx17_h6a678d5_0  
libbrotlicommon           1.0.9                h5eee18b_8  
libbrotlidec              1.0.9                h5eee18b_8  
libbrotlienc              1.0.9                h5eee18b_8  
libdeflate                1.17                 h5eee18b_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            11.2.0               h00389a5_1  
libgfortran5              11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libopenblas               0.3.21               h043d6bf_0  
libpng                    1.6.39               h5eee18b_0  
libprotobuf               4.25.3               he621ea3_0  
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.5.1                h6a678d5_0  
libwebp-base              1.3.2                h5eee18b_0  
lightning                 2.2.1                    pypi_0    pypi
lightning-utilities       0.11.2                   pypi_0    pypi
llvmlite                  0.42.0                   pypi_0    pypi
local-attention           1.9.1                    pypi_0    pypi
lz4-c                     1.9.4                h6a678d5_1  
mamba-ssm                 2.2.0                    pypi_0    pypi
markupsafe                2.1.3                    pypi_0    pypi
matplotlib                3.9.0                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
multidict                 6.0.5                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
networkx                  3.2.1                    pypi_0    pypi
ninja                     1.11.1.1                 pypi_0    pypi
numba                     0.59.1                   pypi_0    pypi
numexpr                   2.8.7            py39h286c3b5_0  
numpy                     1.24.1                   pypi_0    pypi
numpy-base                1.26.4           py39h8a23956_0  
nvidia-cublas-cu11        11.11.3.6                pypi_0    pypi
nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
nvidia-cuda-cupti-cu11    11.8.87                  pypi_0    pypi
nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu11    11.8.89                  pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-runtime-cu11  11.8.89                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
nvidia-cudnn-cu11         8.7.0.84                 pypi_0    pypi
nvidia-cudnn-cu12         8.9.2.26                 pypi_0    pypi
nvidia-cufft-cu11         10.9.0.58                pypi_0    pypi
nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
nvidia-curand-cu11        10.3.0.86                pypi_0    pypi
nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
nvidia-cusolver-cu11      11.4.1.48                pypi_0    pypi
nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
nvidia-cusparse-cu11      11.7.5.86                pypi_0    pypi
nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
nvidia-nccl-cu11          2.19.3                   pypi_0    pypi
nvidia-nccl-cu12          2.19.3                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.6.85                  pypi_0    pypi
nvidia-nvtx-cu11          11.8.86                  pypi_0    pypi
nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
openjpeg                  2.4.0                h3ad879b_0  
openssl                   1.1.1w               h7f8727e_0  
packaging                 24.0                     pypi_0    pypi
pandas                    2.2.1                    pypi_0    pypi
parso                     0.8.4                    pypi_0    pypi
patool                    1.15.0                   pypi_0    pypi
pexpect                   4.9.0                    pypi_0    pypi
pillow                    10.2.0                   pypi_0    pypi
pip                       23.3.1           py39h06a4308_0  
plotly                    5.23.0                   pypi_0    pypi
product-key-memory        0.2.10                   pypi_0    pypi
prompt-toolkit            3.0.43                   pypi_0    pypi
protobuf                  4.25.3           py39h12ddb61_0  
psutil                    5.9.8                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
pure-eval                 0.2.2                    pypi_0    pypi
pygments                  2.17.2                   pypi_0    pypi
pyparsing                 3.1.2                    pypi_0    pypi
python                    3.9.0                hdb3f193_2  
python-dateutil           2.9.0post0       py39h06a4308_2  
python-tzdata             2023.3             pyhd3eb1b0_0  
pytorch-lightning         2.2.1                    pypi_0    pypi
pytorch-metric-learning   2.5.0                    pypi_0    pypi
pytz                      2024.1           py39h06a4308_0  
pyyaml                    6.0.1                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
reformer-pytorch          1.4.4                    pypi_0    pypi
regex                     2024.5.15                pypi_0    pypi
requests                  2.28.1                   pypi_0    pypi
safetensors               0.4.3                    pypi_0    pypi
scikit-base               0.7.7                    pypi_0    pypi
scikit-learn              1.6.0                    pypi_0    pypi
scipy                     1.13.1                   pypi_0    pypi
seaborn                   0.13.2           py39h06a4308_0  
sentry-sdk                1.44.1                   pypi_0    pypi
setproctitle              1.3.3                    pypi_0    pypi
setuptools                68.2.2           py39h06a4308_0  
setuptools-scm            8.0.4                    pypi_0    pypi
six                       1.16.0             pyhd3eb1b0_1  
sktime                    0.28.0                   pypi_0    pypi
smmap                     5.0.1                    pypi_0    pypi
sqlite                    3.41.2               h5eee18b_0  
stack-data                0.6.3                    pypi_0    pypi
sympy                     1.13.3                   pypi_0    pypi
tenacity                  9.0.0                    pypi_0    pypi
tensorboardx              2.2                pyhd3eb1b0_0  
threadpoolctl             3.5.0                    pypi_0    pypi
timm                      1.0.3                    pypi_0    pypi
tk                        8.6.12               h1ccaba5_0  
tokenizers                0.19.1                   pypi_0    pypi
tomli                     2.0.1                    pypi_0    pypi
torch                     2.2.0+cu118              pypi_0    pypi
torchaudio                2.2.0+cu118              pypi_0    pypi
torchmetrics              1.3.2                    pypi_0    pypi
torchvision               0.17.0+cu118             pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
traitlets                 5.14.2                   pypi_0    pypi
transformers              4.42.3                   pypi_0    pypi
triton                    2.2.0                    pypi_0    pypi
tslearn                   0.6.3                    pypi_0    pypi
typing-extensions         4.8.0                    pypi_0    pypi
typing_extensions         4.12.2             pyha770c72_1    conda-forge
tzdata                    2024.1                   pypi_0    pypi
unicodedata2              15.1.0           py39h5eee18b_0  
urllib3                   1.26.13                  pypi_0    pypi
wandb                     0.16.6                   pypi_0    pypi
wcwidth                   0.2.13                   pypi_0    pypi
wheel                     0.41.2           py39h06a4308_0  
xz                        5.4.6                h5eee18b_0  
yarl                      1.9.4                    pypi_0    pypi
zipp                      3.18.1                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.5                hc292b87_2  
```

