# Backdoor-Attacks-Crowd-Counting
**Paper Link**

> This work is supported by Hubei Engineering Research Center on Big Data Security. We greatly thank our supervisors Professor Pan Zhou and Professor Xinjun Ma for providing us with valuable guidance in every stage of the writing of this thesis. From our views, the hardest part of backdooring these crowd counting models is how to control the connection between the predicted density map and the original input image with it's ground truth value.

* This is the official implementation code of paper submitted to ACM MM 2022.
* We are the first to experimentally demonstrate the feasibility of backdooring regression-based crowd counting.
##
## Requirement
  1. Install pytorch 1.5.0+
  2. Python 3.6+
  3. Install tensorboardX
##
## Data Setup
  * follow the CSRNet repo's Data Setup to build the dataset [CSRnet](https://github.com/CommissarMa/CSRNet-pytorch)
  * download the trigger pattern from the trigger files: [Data/trigger](https://github.com/CommissarMa/CSRNet-pytorch) files
  * Download ShanghaiTech Dataset from [Dropbox](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0) or [Drive](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view)
##
## Injection Trigger & Density Map altering
  * Run the data_preparation.py
