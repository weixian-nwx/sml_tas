<<<<<<< HEAD
# AnimeGANv2    

 The improved version of [AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN).  
 
**[Project Page](https://tachibanayoshino.github.io/AnimeGANv2/)** | Landscape photos / videos to anime   

   
    
  

-----  
**News**  
* (2022.08.03)  Added the AnimeGANv2 Colab:    🖼️ Photos [![Photos Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1PbBkmj1EhULvEE8AXr2z84pZ2DQJN4hc/view?usp=sharing)  |   🎞️ Videos [![Colab for videos](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1qhBxA72Wxbh6Eyhd-V0zY_jTIblP9rHz/view?usp=sharing)      
* (2021.12.25)  [**AnimeGANv3**](https://github.com/TachibanaYoshino/AnimeGANv3) has been released. :christmas_tree:  
* (2021.02.21)  [The pytorch version of AnimeGANv2 has been released](https://github.com/bryandlee/animegan2-pytorch), Be grateful to @bryandlee for his contribution. 
* (2020.12.25)  AnimeGANv3 will be released along with its paper in the spring of 2021.  
  

------

**Focus:**  
<table border="1px ridge">
	<tr align="center">
	    <th>Anime style</th>
	    <th>Film</th>  
	    <th>Picture Number</th>  
      <th>Quality</th>
      <th>Download Style Dataset</th>
	</tr >
	<tr align="center">
      <td>Miyazaki Hayao</td>
      <td>The Wind Rises</td>
      <td>1752</td>
      <td>1080p</td>
	    <td rowspan="3"><a href="https://github.com/TachibanaYoshino/AnimeGANv2/releases/tag/1.0">Link</a></td>
	</tr>
	<tr align="center">
	    <td>Makoto Shinkai</td>  
	    <td>Your Name & Weathering with you</td>
      <td>1445</td>
      <td>BD</td>
	</tr>
	<tr align="center">
	    <td>Kon Satoshi</td>
	    <td>Paprika</td>
      <td>1284</td>
      <td>BDRip</td>
	</tr>
</table>  
   
**News:**    
```yaml
The improvement directions of AnimeGANv2 mainly include the following 4 points:  
```  
- [x] 1. Solve the problem of high-frequency artifacts in the generated image.  
- [x] 2. It is easy to train and directly achieve the effects in the paper.  
- [x] 3. Further reduce the number of parameters of the generator network. **(generator size: 8.17 Mb)**, The lite version has a smaller generator model.  
- [x] 4. Use new high-quality style data, which come from BD movies as much as possible.  
   
   &ensp;&ensp;&ensp;&ensp;&ensp;  AnimeGAN can be accessed from [here](https://github.com/TachibanaYoshino/AnimeGAN).  
___  

## Requirements  
- python 3.6  
- tensorflow-gpu 1.15.0 (GPU 2080Ti, cuda 10.0.130, cudnn 7.6.0)  
- opencv  
- tqdm  
- numpy  
- glob  
- argparse  
- onnxruntime (If onnx file needs to be run.)  
  
## Usage  
### 1. Inference  
  > `python test.py  --checkpoint_dir  checkpoint/generator_Hayao_weight  --test_dir dataset/test/HR_photo --save_dir Hayao/HR_photo`  
    
### 2. Convert video to anime  
  > `python video2anime.py  --video video/input/お花見.mp4  --checkpoint_dir  checkpoint/generator_Hayao_weight  --output video/output`  
    
### 3. Train 
#### 1. Download vgg19    
  > [vgg19.npy](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/vgg16%2F19.npy)  

#### 2. Download Train/Val Photo dataset  
  > [Link](https://github.com/TachibanaYoshino/AnimeGAN/releases/tag/dataset-1)  

#### 3. Do edge_smooth  
  > `python edge_smooth.py --dataset Hayao --img_size 256`  

#### 4. Train  
  >  `python train.py --dataset Hayao --epoch 101 --init_epoch 10`  
  
#### 5. Extract the weights of the generator  
  >  `python get_generator_ckpt.py --checkpoint_dir  ../checkpoint/AnimeGANv2_Shinkai_lsgan_300_300_1_2_10_1  --style_name Shinkai`  
  
____  
## Results  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/AnimeGANv2.png)   
     
____ 
:heart_eyes:  Photo  to  Paprika  Style  
  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/37.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/38.jpg)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/6.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/7.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/9.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/21.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/44.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/1.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/8.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/5.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Paprika/concat/15.jpg)   
____  
:heart_eyes:  Photo  to  Hayao  Style   
  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/AE86.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/10.jpg)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/15.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/35.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/39.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/42.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/44.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/41.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/32.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/34.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Hayao/concat/18.jpg)    
____  
:heart_eyes:  Photo  to  Shinkai  Style   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/7.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/9.jpg)     
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/15.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/17.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/22.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/27.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/33.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/32.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/21.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/3.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv2/blob/master/results/Shinkai/concat/26.jpg)  
  
## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv2 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the  authorization letter.  
## Author  
Xin Chen
=======
## PyTorch Implementation of [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)


**Updates**

* `2021-10-17` Add weights for [FacePortraitV2](#additional-model-weights). [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bryandlee/animegan2-pytorch/blob/main/colab_demo.ipynb)

    ![sample](https://user-images.githubusercontent.com/26464535/142294796-54394a4a-a566-47a1-b9ab-4e715b901442.gif)

* `2021-11-07` Thanks to [ak92501](https://twitter.com/ak92501), a [web demo](https://huggingface.co/spaces/akhaliq/AnimeGANv2) is integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/AnimeGANv2)

* `2021-11-07` Thanks to [xhlulu](https://github.com/xhlulu), the `torch.hub` model is now available. See [Torch Hub Usage](#torch-hub-usage).
 
 
## Basic Usage

**Inference**
```
python test.py --input_dir [image_folder_path] --device [cpu/cuda]
```


## Torch Hub Usage

You can load the model via `torch.hub`:

```python
import torch
model = torch.hub.load("bryandlee/animegan2-pytorch", "generator").eval()
out = model(img_tensor)  # BCHW tensor
```

Currently, the following `pretrained` shorthands are available:
```python
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="celeba_distill")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
```

You can also load the `face2paint` util function:
```python
from PIL import Image

face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)

img = Image.open(...).convert("RGB")
out = face2paint(model, img)
```
More details about `torch.hub` is in [the torch docs](https://pytorch.org/docs/stable/hub.html)


## Weight Conversion from the Original Repo (Tensorflow)
1. Install the [original repo's dependencies](https://github.com/TachibanaYoshino/AnimeGANv2#requirements): python 3.6, tensorflow 1.15.0-gpu
2. Install torch >= 1.7.1
3. Clone the original repo & run
```
git clone https://github.com/TachibanaYoshino/AnimeGANv2
python convert_weights.py
```

<details>
<summary>samples</summary>

<br>
Results from converted `Paprika` style model (input image, original tensorflow result, pytorch result from left to right)

<img src="./samples/compare/1.jpg" width="960"> &nbsp; 
<img src="./samples/compare/2.jpg" width="960"> &nbsp; 
<img src="./samples/compare/3.jpg" width="960"> &nbsp; 
   
</details>
 
**Note:** Results from converted weights slightly different due to the [bilinear upsample issue](https://github.com/pytorch/pytorch/issues/10604)


## Additional Model Weights

**Webtoon Face** [[ckpt]](https://drive.google.com/file/d/10T6F3-_RFOCJn6lMb-6mRmcISuYWJXGc)

<details>
<summary>samples</summary>

Trained on <b>256x256</b> face images. Distilled from [webtoon face model](https://github.com/bryandlee/naver-webtoon-faces/blob/master/README.md#face2webtoon) with L2 + VGG + GAN Loss and CelebA-HQ images.

![face_results](https://user-images.githubusercontent.com/26464535/143959011-1740d4d3-790b-4c4c-b875-24404ef9c614.jpg) &nbsp; 
  
</details>


**Face Portrait v1** [[ckpt]](https://drive.google.com/file/d/1WK5Mdt6mwlcsqCZMHkCUSDJxN1UyFi0-)

<details>
<summary>samples</summary>

Trained on <b>512x512</b> face images.

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jCqcKekdtKzW7cxiw_bjbbfLsPh-dEds?usp=sharing)
  
![samples](https://user-images.githubusercontent.com/26464535/127134790-93595da2-4f8b-4aca-a9d7-98699c5e6914.jpg)

[📺](https://youtu.be/CbMfI-HNCzw?t=317)
  
![sample](https://user-images.githubusercontent.com/26464535/129888683-98bb6283-7bb8-4d1a-a04a-e795f5858dcf.gif)

</details>


**Face Portrait v2** [[ckpt]](https://drive.google.com/uc?id=18H3iK09_d54qEDoWIc82SyWB2xun4gjU)

<details>
<summary>samples</summary>

Trained on <b>512x512</b> face images. Compared to v1, `🔻beautify` `🔺robustness` 

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jCqcKekdtKzW7cxiw_bjbbfLsPh-dEds?usp=sharing)
  
![face_portrait_v2_0](https://user-images.githubusercontent.com/26464535/137619176-59620b59-4e20-4d98-9559-a424f86b7f24.jpg)

![face_portrait_v2_1](https://user-images.githubusercontent.com/26464535/137619181-a45c9230-f5e7-4f3c-8002-7c266f89de45.jpg)

🦑 🎮 🔥
  
![face_portrait_v2_squid_game](https://user-images.githubusercontent.com/26464535/137619183-20e94f11-7a8e-4c3e-9b45-378ab63827ca.jpg)


</details>


>>>>>>> 068bf002f72ce88e46c2ac5f948ef6aaaa5de2f4
