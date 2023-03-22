# VIDUE
Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time (CVPR2023)
---
### Introduction
Natural videos captured by consumer cameras often suffer from low framerate and motion blur due to the combination of dynamic scene complexity, lens and sensor imperfection, and less than ideal exposure setting. As a result, computational methods that jointly perform video frame interpolation and deblurring begin to emerge with the unrealistic assumption that the exposure time is known and fixed. In this work, we aim ambitiously for a more realistic yet challenging task - joint video multi-frame interpolation and deblurring under unknown exposure time. 

### Examples of the Demo (Multi-Frame Interpolation x8) videos (240fps) interpolated from blurry videos (30fps)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/GOPR0410_11_00.gif)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/GOPR0384_11_05.gif)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/IMG_0015.gif)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/IMG_0183.gif)


## Prerequisites
- Python >= 3.8, PyTorch >= 1.7.0 (A lower version may also be OK.)
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm

