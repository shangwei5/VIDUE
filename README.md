# VIDUE
Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time (CVPR2023)
---
### Introduction
Natural videos captured by consumer cameras often suffer from low framerate and motion blur due to the combination of dynamic scene complexity, lens and sensor imperfection, and less than ideal exposure setting. As a result, computational methods that jointly perform video frame interpolation and deblurring begin to emerge with the unrealistic assumption that the exposure time is known and fixed. In this work, we aim ambitiously for a more realistic yet challenging task - joint video multi-frame interpolation and deblurring under unknown exposure time. 

### Examples of the Demo (Multi-Frame $\times$ 8 Interpolation) videos (240fps) interpolated from blurry videos (30fps)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/GOPR0410_11_00.gif)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/GOPR0384_11_05.gif)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/IMG_0015.gif)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/IMG_0183.gif)


### Prerequisites
- Python >= 3.8, PyTorch >= 1.7.0
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm

### Datasets
Please download the GoPro datasets from [link](http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large_all.zip) (240FPS, GOPRO_Large_all)

Please download the Adobe datasets from [link](https://www.dropbox.com/s/pwjbbrcyk1woqxu/adobe240.zip?dl=0) (Full version)

## Dataset Organization Form
```
|--dataset
    |--train  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--test
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
         :
        |--video n
```
## Download Pre-trained Model of VIDUE

Please put models to './experiments'.

## Getting Started

### 1) Testing
```
python inference_vidue_worsu.py --default_data GOPRO --m 5(or 7) --n 3(or 1)
```
Please change `args.data_path` according to `m` and `n`.
The results on GoPro ($\times$ 8 interpolation and deblurring) are also available at [BaiduYun](https://pan.baidu.com/s/1v81oA9Tvq_zwMUMeceLVLQ)[password: d3as]


