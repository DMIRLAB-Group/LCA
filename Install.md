# Installation

## Setup Enviroment

This codebase is tested with `torch==2.2.0`, with `CUDA 11.8`. The following are the environments for time series classification and prediction. For video classification, please refer to [[TranSVAE](https://github.com/ldkong1205/TranSVAE/blob/main/docs/INSTALL.md)].

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
# Name                    Version                   Build  	 Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
annotated-types           0.7.0              pyhd8ed1ab_1    conda-forge
aom                       3.9.1                hac33072_0    conda-forge
appdirs                   1.4.4              pyhd8ed1ab_1    conda-forge
blas                      1.0                         mkl    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
brotli-python             1.1.0            py39hf88036b_2    conda-forge
bzip2                     1.0.8                h4bc722e_7    conda-forge
ca-certificates           2024.12.14           hbcca054_0    conda-forge
cairo                     1.18.2               h3394656_1    conda-forge
certifi                   2024.12.14         pyhd8ed1ab_0    conda-forge
cffi                      1.17.1           py39h15c3d72_0    conda-forge
charset-normalizer        3.4.0              pyhd8ed1ab_1    conda-forge
click                     8.1.8              pyh707e725_0    conda-forge
cpython                   3.9.21           py39hd8ed1ab_1    conda-forge
cuda-cudart               11.8.89                       0    nvidia
cuda-cupti                11.8.87                       0    nvidia
cuda-libraries            11.8.0                        0    nvidia
cuda-nvrtc                11.8.89                       0    nvidia
cuda-nvtx                 11.8.86                       0    nvidia
cuda-runtime              11.8.0                        0    nvidia
cuda-version              12.6                          3    nvidia
dav1d                     1.2.1                hd590300_0    conda-forge
docker-pycreds            0.4.0                      py_0    conda-forge
einops                    0.8.0              pyhd8ed1ab_1    conda-forge
eval_type_backport        0.2.2              pyha770c72_0    conda-forge
ffmpeg                    7.1.0           gpl_h9d2eab1_509    conda-forge
filelock                  3.16.1             pyhd8ed1ab_1    conda-forge
font-ttf-dejavu-sans-mono 2.37                          0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
font-ttf-inconsolata      2.000                         0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
font-ttf-source-code-pro  2.030                         0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
font-ttf-ubuntu           0.83                          0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
fontconfig                2.15.0               h7e30c49_1    conda-forge
fonts-conda-ecosystem     1                             0    conda-forge
fonts-conda-forge         1                             0    conda-forge
freetype                  2.12.1               h267a509_2    conda-forge
fribidi                   1.0.10               h36c2ea0_0    conda-forge
gdk-pixbuf                2.42.12              hb9ae30d_0    conda-forge
gitdb                     4.0.11             pyhd8ed1ab_1    conda-forge
gitpython                 3.1.43             pyhff2d567_1    conda-forge
gmp                       6.3.0                hac33072_2    conda-forge
gmpy2                     2.1.5            py39h7196dd7_3    conda-forge
graphite2                 1.3.13            h59595ed_1003    conda-forge
graphql-core              1.1                        py_1    conda-forge
h2                        4.1.0              pyhd8ed1ab_1    conda-forge
harfbuzz                  10.1.0               h0b3b770_0    conda-forge
hpack                     4.0.0              pyhd8ed1ab_1    conda-forge
hyperframe                6.0.1              pyhd8ed1ab_1    conda-forge
icu                       75.1                 he02047a_0    conda-forge
idna                      3.10               pyhd8ed1ab_1    conda-forge
intel-openmp              2022.0.1          h06a4308_3633  
jinja2                    3.1.5              pyhd8ed1ab_0    conda-forge
joblib                    1.4.2              pyhd8ed1ab_1    conda-forge
lame                      3.100             h166bdaf_1003    conda-forge
lcms2                     2.16                 hb7c19ff_0    conda-forge
ld_impl_linux-64          2.43                 h712a8e2_2    conda-forge
lerc                      4.0.0                h27087fc_0    conda-forge
level-zero                1.19.2               h84d6215_1    conda-forge
libabseil                 20240722.0      cxx17_hbbce691_2    conda-forge
libass                    0.17.3               hba53ac1_1    conda-forge
libblas                   3.9.0            16_linux64_mkl    conda-forge
libcblas                  3.9.0            16_linux64_mkl    conda-forge
libcublas                 11.11.3.6                     0    nvidia
libcufft                  10.9.0.58                     0    nvidia
libcufile                 1.11.1.6                      0    nvidia
libcurand                 10.3.7.77                     0    nvidia
libcusolver               11.4.1.48                     0    nvidia
libcusparse               11.7.5.86                     0    nvidia
libdeflate                1.23                 h4ddbbb0_0    conda-forge
libdrm                    2.4.124              hb9d3cd8_0    conda-forge
libegl                    1.7.0                ha4b6fd6_2    conda-forge
libexpat                  2.6.4                h5888daf_0    conda-forge
libffi                    3.4.2                h7f98852_5    conda-forge
libgcc                    14.2.0               h77fa898_1    conda-forge
libgcc-ng                 14.2.0               h69a702a_1    conda-forge
libgfortran               14.2.0               h69a702a_1    conda-forge
libgfortran-ng            14.2.0               h69a702a_1    conda-forge
libgfortran5              14.2.0               hd5240d6_1    conda-forge
libgl                     1.7.0                ha4b6fd6_2    conda-forge
libglib                   2.82.2               h2ff4ddf_0    conda-forge
libglvnd                  1.7.0                ha4b6fd6_2    conda-forge
libglx                    1.7.0                ha4b6fd6_2    conda-forge
libgomp                   14.2.0               h77fa898_1    conda-forge
libhwloc                  2.11.2          default_h0d58e46_1001    conda-forge
libiconv                  1.17                 hd590300_2    conda-forge
libjpeg-turbo             3.0.0                hd590300_1    conda-forge
liblapack                 3.9.0            16_linux64_mkl    conda-forge
liblzma                   5.6.3                hb9d3cd8_1    conda-forge
libnpp                    11.8.0.86                     0    nvidia
libnsl                    2.0.1                hd590300_0    conda-forge
libnvjpeg                 11.9.0.86                     0    nvidia
libopenvino               2024.6.0             hac27bb2_3    conda-forge
libopenvino-auto-batch-plugin 2024.6.0             h4d9b6c2_3    conda-forge
libopenvino-auto-plugin   2024.6.0             h4d9b6c2_3    conda-forge
libopenvino-hetero-plugin 2024.6.0             h3f63f65_3    conda-forge
libopenvino-intel-cpu-plugin 2024.6.0             hac27bb2_3    conda-forge
libopenvino-intel-gpu-plugin 2024.6.0             hac27bb2_3    conda-forge
libopenvino-intel-npu-plugin 2024.6.0             hac27bb2_3    conda-forge
libopenvino-ir-frontend   2024.6.0             h3f63f65_3    conda-forge
libopenvino-onnx-frontend 2024.6.0             h6363af5_3    conda-forge
libopenvino-paddle-frontend 2024.6.0             h6363af5_3    conda-forge
libopenvino-pytorch-frontend 2024.6.0             h5888daf_3    conda-forge
libopenvino-tensorflow-frontend 2024.6.0             h630ec5c_3    conda-forge
libopenvino-tensorflow-lite-frontend 2024.6.0             h5888daf_3    conda-forge
libopus                   1.3.1                h7f98852_1    conda-forge
libpciaccess              0.18                 hd590300_0    conda-forge
libpng                    1.6.44               hadc24fc_0    conda-forge
libprotobuf               5.28.3               h6128344_1    conda-forge
librsvg                   2.58.4               h49af25d_2    conda-forge
libsqlite                 3.47.2               hee588c1_0    conda-forge
libstdcxx                 14.2.0               hc0a3c3a_1    conda-forge
libstdcxx-ng              14.2.0               h4852527_1    conda-forge
libtiff                   4.7.0                hd9ff511_3    conda-forge
libuuid                   2.38.1               h0b41bf4_0    conda-forge
libva                     2.22.0               h8a09558_1    conda-forge
libvpx                    1.14.1               hac33072_0    conda-forge
libwebp-base              1.5.0                h851e524_0    conda-forge
libxcb                    1.17.0               h8a09558_0    conda-forge
libxcrypt                 4.4.36               hd590300_1    conda-forge
libxml2                   2.13.5               h8d12d68_1    conda-forge
libzlib                   1.3.1                hb9d3cd8_2    conda-forge
lightning-utilities       0.11.9             pyhd8ed1ab_1    conda-forge
llvm-openmp               15.0.7               h0cdce71_0    conda-forge
markupsafe                3.0.2            py39h9399b63_1    conda-forge
mkl                       2022.1.0           hc2b9512_224  
mpc                       1.3.1                h24ddda3_1    conda-forge
mpfr                      4.2.1                h90cbb55_3    conda-forge
mpmath                    1.3.0              pyhd8ed1ab_1    conda-forge
ncurses                   6.5                  he02047a_1    conda-forge
networkx                  3.2.1              pyhd8ed1ab_0    conda-forge
numpy                     1.26.4           py39h474f0d3_0    conda-forge
ocl-icd                   2.3.2                hb9d3cd8_2    conda-forge
opencl-headers            2024.10.24           h5888daf_0    conda-forge
openh264                  2.5.0                hf92e6e3_0    conda-forge
openjpeg                  2.5.3                h5fbd93e_0    conda-forge
openssl                   3.4.0                hb9d3cd8_0    conda-forge
packaging                 24.2               pyhd8ed1ab_2    conda-forge
pandas                    2.2.1            py39hddac248_0    conda-forge
pango                     1.54.0               h861ebed_4    conda-forge
pcre2                     10.44                hba22ea6_2    conda-forge
pillow                    11.0.0           py39h538c539_0    conda-forge
pip                       24.3.1             pyh8b19718_2    conda-forge
pixman                    0.44.2               h29eaf8c_0    conda-forge
platformdirs              4.3.6              pyhd8ed1ab_1    conda-forge
promise                   2.3              py39hf3d152e_9    conda-forge
protobuf                  5.28.3           py39hf88036b_0    conda-forge
psutil                    6.1.1            py39h8cd3c5a_0    conda-forge
pthread-stubs             0.4               hb9d3cd8_1002    conda-forge
pugixml                   1.14                 h59595ed_0    conda-forge
pycparser                 2.22               pyh29332c3_1    conda-forge
pydantic                  2.10.4             pyh3cfb1c2_0    conda-forge
pydantic-core             2.27.2           py39he612d8f_0    conda-forge
pysocks                   1.7.1              pyha55dd90_7    conda-forge
python                    3.9.21          h9c0c6dc_1_cpython    conda-forge
python-dateutil           2.9.0.post0        pyhff2d567_1    conda-forge
python-tzdata             2024.2             pyhd8ed1ab_1    conda-forge
python_abi                3.9                      5_cp39    conda-forge
pytorch                   2.2.0           py3.9_cuda11.8_cudnn8.7.0_0    pytorch
pytorch-cuda              11.8                 h7e8668a_6    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2024.2             pyhd8ed1ab_1    conda-forge
pyyaml                    6.0.2            py39h8cd3c5a_1    conda-forge
readline                  8.2                  h8228510_1    conda-forge
requests                  2.32.3             pyhd8ed1ab_1    conda-forge
scikit-learn              1.6.0            py39h4b7350c_0    conda-forge
scipy                     1.13.1           py39haf93ffa_0    conda-forge
sentry-sdk                2.19.2             pyhd8ed1ab_1    conda-forge
setproctitle              1.3.4            py39h8cd3c5a_0    conda-forge
setuptools                75.6.0             pyhff2d567_1    conda-forge
six                       1.17.0             pyhd8ed1ab_0    conda-forge
smmap                     5.0.0              pyhd8ed1ab_0    conda-forge
snappy                    1.2.1                h8bd8927_1    conda-forge
svt-av1                   2.3.0                h5888daf_0    conda-forge
sympy                     1.13.3           pyh2585a3b_104    conda-forge
tbb                       2022.0.0             hceb3a55_0    conda-forge
threadpoolctl             3.5.0              pyhc1e730c_0    conda-forge
tk                        8.6.13          noxft_h4845f30_101    conda-forge
torchaudio                2.2.0                py39_cu118    pytorch
torchmetrics              1.6.1              pyhd8ed1ab_0    conda-forge
torchtriton               2.2.0                      py39    pytorch
torchvision               0.17.0               py39_cu118    pytorch
typing                    3.10.0.0           pyhd8ed1ab_2    conda-forge
typing-extensions         4.12.2               hd8ed1ab_1    conda-forge
typing_extensions         4.12.2             pyha770c72_1    conda-forge
tzdata                    2024b                hc8b5060_0    conda-forge
urllib3                   2.3.0              pyhd8ed1ab_0    conda-forge
wandb                     0.19.1           py39h43652db_0    conda-forge
wayland                   1.23.1               h3e06ad9_0    conda-forge
wayland-protocols         1.37                 hd8ed1ab_0    conda-forge
wheel                     0.45.1             pyhd8ed1ab_1    conda-forge
x264                      1!164.3095           h166bdaf_2    conda-forge
x265                      3.5                  h924138e_3    conda-forge
xorg-libice               1.1.2                hb9d3cd8_0    conda-forge
xorg-libsm                1.2.5                he73a12e_0    conda-forge
xorg-libx11               1.8.10               h4f16b4b_1    conda-forge
xorg-libxau               1.0.12               hb9d3cd8_0    conda-forge
xorg-libxdmcp             1.1.5                hb9d3cd8_0    conda-forge
xorg-libxext              1.3.6                hb9d3cd8_0    conda-forge
xorg-libxfixes            6.0.1                hb9d3cd8_0    conda-forge
xorg-libxrender           0.9.12               hb9d3cd8_0    conda-forge
yaml                      0.2.5                h7f98852_2    conda-forge
zstandard                 0.23.0           py39h08a7858_1    conda-forge
zstd                      1.5.6                ha6fb4c9_0    conda-forge
```

