MultiFace_Pytorch

------

## 1. Intro

- This repo is a implementation of MultiFace[(paper)](https://arxiv.org/abs/1801.07698)
- Pretrained models are posted, include the [MobileFacenet](https://arxiv.org/abs/1804.07573) and IR-SE50 in the original paper

------

## 2. Pretrained Models & Performance

[IR-SE50 @ BaiduNetdisk](https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ), [IR-SE50 @ Onedrive](https://1drv.ms/u/s!AhMqVPD44cDOhkPsOU2S_HFpY9dC)

| Loss               | LFW(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) |
| ------------------ | ------ | --------- | ----------- | -------- | :------: |
| Arcface            | 9962   | 0.9504    | 0.9622      | 0.9557   |  0.9107  |
| Multi-Arcface(N=2) |        |           |             |          |          |
| Cosfacce           |        |           |             |          |          |
| Multi-Cosface(N=2) |        |           |             |          |          |

[Mobilefacenet @ BaiduNetDisk](https://pan.baidu.com/s/1hqNNkcAjQOSxUjofboN6qg), [Mobilefacenet @ OneDrive](https://1drv.ms/u/s!AhMqVPD44cDOhkSMHodSH4rhfb5u)

| Loss               |  LFW(%)   | CFP-FP(%) | AgeDB-30(%) | calfw(%)  | cplfw(%)  |
| ------------------ | :-------: | :-------: | :---------: | :-------: | :-------: |
| Softmax            |   99.22   |   92.84   |    94.00    |   93.80   |   88.30   |
| Multi-Softmax(N=4) | **99.40** | **95.46** |  **95.25**  | **95.15** | **90.22** |

| Loss               |  LFW(%)   | CFP-FP(%) | AgeDB-30(%) | calfw(%)  | cplfw(%)  |
| ------------------ | :-------: | :-------: | :---------: | :-------: | :-------: |
| Arcface            |   99.45   |   92.27   |    96.03    |   95.12   |   87.75   |
| Multi-Arcface(N=2) |   99.52   |   93.41   |  **96.35**  | **95.28** |   88.65   |
| Cosface            |   99.43   |   92.83   |    95.77    |   94.97   |   88.88   |
| Multi-Cosface(N=2) |   99.50   |   93.58   |    96.17    |   95.20   |   89.03   |
| Multi-Cosface(N=4) | **99.60** | **94.11** |    96.13    |   95.18   | **89.47** |



## 3. How to use

- clone

  ```
  git clone https://github.com/TropComplique/mtcnn-pytorch.git
  ```

### 3.1 Data Preparation

#### 3.1.1 Prepare Facebank (For testing over camera or video)

Provide the face images your want to detect in the data/face_bank folder, and guarantee it have a structure like following:

```
data/facebank/
        ---> id1/
            ---> id1_1.jpg
        ---> id2/
            ---> id2_1.jpg
        ---> id3/
            ---> id3_1.jpg
           ---> id3_2.jpg
```

#### 3.1.2 download the pretrained model to work_space/model

If more than 1 image appears in one folder, an average embedding will be calculated

#### 3.2.3 Prepare Dataset ( For training)

download the refined dataset: (emore recommended)

- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ), [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
- More Dataset please refer to the [original post](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

**Note:** If you use the refined [MS1M](https://arxiv.org/abs/1607.08221) dataset and the cropped [VGG2](https://arxiv.org/abs/1710.08092) dataset, please cite the original papers.

- after unzip the files to 'data' path, run :

  ```
  python prepare_data.py
  ```

  after the execution, you should find following structure:

```
faces_emore/
            ---> agedb_30
            ---> calfw
            ---> cfp_ff
            --->  cfp_fp
            ---> cfp_fp
            ---> cplfw
            --->imgs
            ---> lfw
            ---> vgg2_fp
```

------

### 3.2 detect over camera:

- 1. download the desired weights to model folder:

- [IR-SE50 @ BaiduNetdisk](https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ)
- [IR-SE50 @ Onedrive](https://1drv.ms/u/s!AhMqVPD44cDOhkPsOU2S_HFpY9dC)
- [Mobilefacenet @ BaiduNetDisk](https://pan.baidu.com/s/1hqNNkcAjQOSxUjofboN6qg)
- [Mobilefacenet @ OneDrive](https://1drv.ms/u/s!AhMqVPD44cDOhkSMHodSH4rhfb5u)

- 2 to take a picture, run

  ```
  python take_pic.py -n name
  ```

  press q to take a picture, it will only capture 1 highest possibility face if more than 1 person appear in the camera

- 3 or you can put any preexisting photo into the facebank directory, the file structure is as following:

```
- facebank/
         name1/
             photo1.jpg
             photo2.jpg
             ...
         name2/
             photo1.jpg
             photo2.jpg
             ...
         .....
    if more than 1 image appears in the directory, average embedding will be calculated
```

- 4 to start

  ```
  python face_verify.py 
  ```

- - -

### 3.3 detect over video:

```
​```
python infer_on_video.py -f [video file name] -s [save file name]
​```
```

the video file should be inside the data/face_bank folder

- Video Detection Demo [@Youtube](https://www.youtube.com/watch?v=6r9RCRmxtHE)

### 3.4 Training:

```
​```
python train.py -b [batch_size] -lr [learning rate] -e [epochs]

# python train.py -net mobilefacenet -b 200 -w 4
​```
```

## 4. References 

- This code is based on the implementations of   [TreB1eN/*InsightFace*_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) and [deepinsight/insightface](https://github.com/deepinsight/insightface) 

## PS

- PRs are welcome, in case that I don't have the resource to train some large models like the 100 and 151 layers model
- Email : treb1en@qq.com
