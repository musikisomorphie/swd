# Sliced Wasserstein Generative Models<br><i>-- and the application on unsupervised video generation</i>

           
### Papers
* [Sliced Wasserstein Generative Models (CVPR2019)](https://arxiv.org/pdf/1904.05408.pdf)
* [Towards high resolution video generation (Arxiv)](https://arxiv.org/pdf/1810.02419.pdf) 


## Prerequisites
This repo has been successfully tested on **tensorflow 1.10, cuda 9.0**. 

* Please check the [requirements.txt](https://github.com/musikisomorphie/swd/blob/master/requirements.txt) for more details.

* For the training data such as Cifar10, CelebA, CelebA-HQ, LSUN etc download them on the official website accordingly.

* [Trailerfaces (Google Drive remain updated)](https://github.com/musikisomorphie/swd/)
  * The dataset contains approximately 200,000 individual clips of various facial expressions, where the faces are cropped with 256x256 resolution from about 6,000 high resolution movie trailers on YouTube. We convert them to tfrecord with resolutions range from 4x4 to 256x256. More about the data processing please see [Towards high resolution video generation (Arxiv)](https://arxiv.org/pdf/1810.02419.pdf). 

Trailerfaces sample:
![Trailerfaces sample](https://github.com/musikisomorphie/swd/blob/master/trailer_faces_samples.png)

## Models

### SWAE: it requires some custom ops, which are stored under the [cuda](https://github.com/musikisomorphie/swd/tree/master/cuda) folder.
  * Following the instructions in [install](https://github.com/musikisomorphie/swd/blob/master/cuda/install), you could compile them by yourself. If you install tensorflow by pip, one potential error can be some source files of tensorflow set the wrong relative path of cuda.h, you just need to manually change them according to your cuda path.
  * Alternatively, you could also use the binary files directly, which is compatabile with **tensorflow 1.10, cuda 9.0**.
  * Specify DATA_DIR, LOG_DIR and DIR in swae_64x64.py, then run
    * python swae_64x64.py
  
### SWGAN: 
   * Specify DATA_DIR, LOG_DIR and DIR in swgan_64x64.py, then run
     * python swgan_64x64.py   

### PG-SWGAN-3D: remain updated
* Generated video frames by PG-SWGAN-3D:
![PG-SWGAN-3D](https://github.com/musikisomorphie/swd/blob/master/pgswgan_3d.jpg)

* More video comparison, see the following youtube links:
  * PG-SWGAN-3D VS PG-WGAN-3D: See [video1](https://www.youtube.com/watch?v=BvIJk01r9tw)
  * VGAN VS MoCoGAN: See [video2](https://www.youtube.com/watch?v=Q7kUrPTcmdE)


