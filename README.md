# VIDUE
Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time (CVPR2023)
---
### Introduction
Natural videos captured by consumer cameras often suffer from low framerate and motion blur due to the combination of dynamic scene complexity, lens and sensor imperfection, and less than ideal exposure setting. As a result, computational methods that jointly perform video frame interpolation and deblurring begin to emerge with the unrealistic assumption that the exposure time is known and fixed. In this work, we aim ambitiously for a more realistic yet challenging task - joint video multi-frame interpolation and deblurring under unknown exposure time. Toward this goal, we first adopt a variant of supervised contrastive learning to construct an exposure-aware representation from input blurred frames. We then train two U-Nets for intra-motion and inter-motion analysis, respectively, adapting to the learned exposure representation via gain tuning. We finally build our video reconstruction network upon the exposure and motion representation by progressive exposure-adaptive convolution and motion refinement.

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
Pre-trained exposure-aware feature extractor on GoPro and Adobe can be downloaded from [here](https://pan.baidu.com/s/1Zf9AbODFtaFexldhmt6_bg?pwd=giu1).

Download pre-trained VIDUE for [$\times$ 8 interpolation on GoPro](https://pan.baidu.com/s/1_f_G2vdYowqxxiRFkiP18g?pwd=c2sf), [$\times$ 8 interpolation on Adobe](https://pan.baidu.com/s/1P9hKNZBGlMe24xDLZxqHGg?pwd=h5k2), and [$\times$ 16 interpolation on GoPro](https://pan.baidu.com/s/1KciF2INIjEBnGmIetC2X6g?pwd=x8uu).

Please put these models to `./experiments`.

## Getting Started

### 1) Generate Test Data
```
python generate_blur.py --videos_src_path your_data_path/GoPro_Large_all/test --videos_save_path your_data_path/GoPro_Large_all/LFR_Gopro_53  --num_compose 5  --tot_inter_frame 8
```
This is an example for generating "GoPro-5:8", please change `num_compose` and `tot_inter_frame` to generate other cases.

```
python generate_blur_adobe.py --videos_src_path your_data_path/adobe240/test --videos_save_path your_data_path/adobe240/LFR_Adobe_53  --num_compose 5*3  --tot_inter_frame 8*3
```
This is an example for generating "Adobe-5:8", please change `num_compose` and `tot_inter_frame` to generate other cases.


### 2) Testing
```
python inference_vidue_worsu.py --default_data GOPRO --m 5(or 7) --n 3(or 1)
```
Please change `args.data_path` according to `m` and `n`.
The results on GoPro ($\times$ 8 interpolation and deblurring) are also available at [BaiduYun](https://pan.baidu.com/s/19_og-5tDZ3Bccp2gQTzEPg?pwd=k1fp).


```
python inference_vidue_worsu.py --default_data Adobe --m 5(or 7) --n 3(or 1)
```
The results on Adobe ($\times$ 8 interpolation and deblurring) are also available at [BaiduYun](https://pan.baidu.com/s/1ko399DgXUYG5xa_bykwJ2A?pwd=gqug)


```
python inference_vidue_worsu_16x.py --default_data GOPRO --m 9(or 11,13,15) --n 7(or 5,3,1)
```
The results on GoPro ($\times$ 16 interpolation and deblurring) are also available at [BaiduYun](https://pan.baidu.com/s/1efzyoyDSWWPlEm_bCTfhGw?pwd=fyhu)


```
python inference_vidue_worsu_real.py
```
Change `args.model_path` to our pre-trained models or you can finetune on your dataset for testing real-world data.

### 3) Training
1.Training exposure-aware feature extractor:
```
python main_extractor_weighted_ordinalsupcon.py --template UNet_PRIOR_PREDICT_Weighted_OrdinalSupcon_Light  --save extractor_GoPro8x --process --random
```
Please change `--template` to `UNet_PRIOR_PREDICT_Weighted_OrdinalSupcon_Light_Adobe` and `UNet_PRIOR_PREDICT_Weighted_OrdinalSupcon_Light_16x` for different tasks.

2.Training VIDUE:
```
python main_vidue_worsu_smph.py --template VIDUE_WORSU --save recon_GoPro8x --random --process
```
Please change `--template` to `VIDUE_WORSU_Adobe` and `VIDUE_WORSU_16x` for different tasks.
Please check the dataset path according to yours.

## Average PSNR/SSIM Values on Benchmark Datasets:
![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/metrics_gopro8x.png)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/metrics_adobe8x.png)

![image](https://github.com/shangwei5/VIDUE/blob/main/Figures/metrics_gopro16x.png)

## Cite
If you use any part of our code, or VIDUE is useful for your research, please consider citing:
```
  @InProceedings{Wei_2023_CVPR,
      author    = {Shang, Wei and Ren, Dongwei and Yang, Yi and Zhang, Hongzhi and Ma, Kede and Zuo, Wangmeng},
      title     = {Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year      = {2023},
      pages     = {0-0}
  }
```

## Contact
If you have any questions, please contact csweishang@gmail.com.



