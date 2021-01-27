MultiFace_Pytorch

------

## 1. Intro

- This repo is a implementation of [MultiFace](http://arxiv.org/abs/2101.09899)
- Pretrained models are posted, include the [MobileFacenet](https://arxiv.org/abs/1804.07573) and IR-SE100 in the original paper

------

## 2. Pretrained Models & Performance

 [IR-SE100 @ GoogleDrive](https://drive.google.com/file/d/1Fw5Zal5F5QZtQnQDLCvI0YRPn7Bsw_dO/view?usp=sharing)

| Loss               | AgeDB-30(%) | CFP-FP(%) | calfw(%) | cplfw(%) |
| ------------------ | :---------: | :-------: | :------: | :------: |
| Multi-Arcface(N=2) |    98.20    |   98.30   |  96.02   |  93.17   |
| Multi-Cosface(N=2) |    98.20    |   98.40   |  96.07   |  93.06   |

[Mobilefacenet @ GoogleDrive](https://drive.google.com/file/d/16zxRC2t7Bi0GzOeiWfXjBW39V1nxd5TV/view?usp=sharing)

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



## 3. Training and Testing

### 3.1 Prepare training dataset

download the refined dataset: (emore recommended, our method is suitable for larger dataset )

- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ), [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)

  **Note:** If you use the refined [MS1M](https://arxiv.org/abs/1607.08221) dataset, please cite the original papers.

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

### 3.2 Training:

```
'''
# mobilefacenet loss:softmax  num_sphere:2 
python train.py --num_sphere 2 --work_path [save log and model information]

# mobilefacenet loss:arcface  num_sphere:2 
python train.py --num_sphere 2 --arcface_loss --work_path [save log and model information]

# mobilefacenet loss:cosface  num_sphere:2 
python train.py --num_sphere 2 --am_softmax_loss --work_path [save log and model information]

# ir-se100 loss:arcface  num_sphere:2 
python train.py --num_sphere 2 --arcface_loss --net ir_se -depth 100 --work_path [save log and model information]

# ir-se100 loss:cosface  num_sphere:2 
python train.py --num_sphere 2 --am_softmax_loss --net ir_se -depth 100 --work_path [save log and model information]

```

### 3.3 Testing

Evaluating the model on LFW, Age-DB, CFP-FP, CALFW, CPLFW

```
#pretrained mobilefacenet
python train.py --pretrain --pretrained_model_path [mobilefacenet_pretrained_model_path]

#pretrained ir-se 100
python train.py --pretrain -net ir_se -depth 100 --work_path [resnet_pretrained_model_path]
```

To evaluate on Megaface, please refer to [megaface-evaluation](https://github.com/deepinx/megaface-evaluation).

## 4. References 

- This code is based on the implementations of   [TreB1eN/*InsightFace*_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) and [deepinsight/insightface](https://github.com/deepinsight/insightface) 

## Contact

- Email :xujing.may@gmail.com
