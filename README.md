

## About the project
This repository includes the codes of the paper:
[Structured Learning for Action Recognition in Videos](https://ieeexplore.ieee.org/abstract/document/8805090)

The two-stream CNNs we used to extract features are based on this pytorch implementation(https://github.com/jeffreyhuang1/two-stream-action-recognition.git).
### Abstract
Actions in continuous videos are correlated and may have hierarchical relationships. Densely labeled datasets of complex videos have revealed the simultaneous occurrence of actions, but existing models fail to make use of the relationships to analyze actions in the context of videos and better understand complex videos. We propose a novel architecture consisting of a correlation learning and input synthesis (CoLIS) network, long short-term memory (LSTM), and a hierarchical classifier. First, the CoLIS network captures the correlation between features extracted from video sequences and pre-processes the input to the LSTM. Since the input becomes the weighted sum of multiple correlated features, it enhances the LSTM's ability to learn variable-length long-term temporal dependencies. Second, we design a hierarchical classifier which utilizes the simultaneous occurrence of general actions such as run and jump to refine the prediction on their correlated actions. Third, we use interleaved backpropagation through time for training. All these networks are fully differentiable so that they can be integrated for endto-end learning. The results show that the proposed approach improves action recognition accuracy by 1.0% and 2.2% on single-labeled or densely labeled datasets respectively.

### Clone the project repository from github:
```
$ git clone https://github.com/yinghanlong/action-recognition-video.git
```

## 1. Data
### Download and directly use the output features (from the last layer of ResNet101, dim=4096 per frame) of UCF101 processed by a two-stream CNN.
* [UCF101, Spatial stream](https://purdue0-my.sharepoint.com/:f:/g/personal/long273_purdue_edu/Ert6C2sG2Q9EuJesJ9C68_sBsoLHDPzaKeLqHfWl0eh9zg?e=EEWbeh)
* [UCF101, Temporal stream](https://purdue0-my.sharepoint.com/:f:/g/personal/long273_purdue_edu/EnQm652QilJNmVrPoB7CFBoBHHI8XOG15sFZHW5NdDoE0w?e=1EewrY)
* [Multithumos, Spatial stream](https://purdue0-my.sharepoint.com/:f:/g/personal/long273_purdue_edu/ErfqMLqxRuZLsoOgOQnhWioBTE-aDvqTvdIUFHAtSFv-8A?e=cKHeWt)
* [Multithumos, Temporal stream](https://purdue0-my.sharepoint.com/:f:/g/personal/long273_purdue_edu/EgCwPjkh1H9OtPoNUI6a7_sBh0YbLPjeheLShWoxoUdDxQ?e=DCboE3)

  ### (Alternative) 1.1 Spatial input data -> rgb frames
  * We extract RGB frames from each video in UCF101 dataset with sampling rate: 10 and save as .jpg image in disk which cost about 5.9G.
  ### (Alternative) 1.2 Motion input data -> stacked optical flow images
  In motion stream, we use two methods to get optical flow data. 
  1. Download the preprocessed tvl1 optical flow dataset directly from https://github.com/feichtenhofer/twostreamfusion. 
  2. Using [flownet2.0 method](https://github.com/lmb-freiburg/flownet2-docker) to generate 2-channel optical flow image and save its x, y channel as .jpg image in disk respectively, which cost about 56G.
  ### (Alternative)Download the preprocessed data directly from [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion))
  * RGB images
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003
  
  cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
  unzip ucf101_jpegs_256.zip
  ```
  * Optical Flow
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003
  
  cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  unzip ucf101_tvl1_flow.zip
  ```
  ## Built With

* [PyTorch](http://pytorch.org/) 

## Getting Started
 After setting up the enviroment, you can extract features with a pretrained two-stream CNN or directly use the features we provide. Then you can train the proposed model (attention-enhanced LSTM) using the features as inputs.
 To train with UCF101 dataset,

```
$ python lstm-cor-2.py --resume PATH-TO-MODEL --epoches=50 --lr=5e-4 --top5enhance 
```
To train with multithumos,
```
$ python lstm-multithmos.py --resume PATH-TO-MODEL --epoches=50 --lr=5e-4 --top5enhance --dataset=multithumos
```

If you want to use the vanilla LSTM, do not set ```--top5enhance``` or use ```lstm-ori.py```.
Please contact us at long273@purdue.edu if you encounter any problem with using this repository.



## 5. Performace
   
 network         | UCF101 Top1 Accuracy| Multithumos mAP|
-----------------|:-------------------:|----------------|
Spatial cnn      | 82.1%               |     - |
Motion cnn       | 79.4%               |     - |
Two stream CNN   | 88.5%               | 27.6% |
Two stream + LSTM| 89.8%               | 29.6% |
Our work         | 90.8%               | 31.8% |
   
## 6. Pre-trained Model

* [Spatial resnet101](https://drive.google.com/drive/folders/1gVB5StqgoDJ3IxHUn7zoTzTNxzz3du3d?usp=sharing)
* [Motion resnet101](https://drive.google.com/drive/folders/1z3fYUOJx_l3BW-NSb7ti0DsyGLFk6Z7J?usp=sharing)


## 7. Testing two-stream CNNs on Your Device
  ### Spatial stream
 * Please modify this [path](https://github.com/yinghanlong/action-recognition-video/blob/master/spatial_cnn.py#L42) and this [funcition](https://github.com/yinghanlong/action-recognition-video/blob/master/dataloader/spatial_dataloader.py#L21) to fit the UCF101/multithumos dataset on your device.
 * Training and testing
 ```
 python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL
 ```
 * Only testing
 ```
 python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate
 ```
 
 ### Motion stream
 *  Please modify this [path](https://github.com/yinghanlong/action-recognition-video/blob/master/motion_cnn.py#L44) and this [funcition](https://github.com/yinghanlong/action-recognition-video/blob/master/dataloader/motion_dataloader.py#L32) to fit the UCF101/multithumos dataset on your device.
  * Training and testing
 ```
 python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL
 ```
 * Only testing
 ```
 python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate
 ```
 


## Citation

If you use this repo for your work, please use the following citation:

```
@ARTICLE{8805090,

  author={Long, Yinghan and Srinivasan, Gopalakrishnan and Panda, Priyadarshini and Roy, Kaushik},

  journal={IEEE Journal on Emerging and Selected Topics in Circuits and Systems}, 

  title={Structured Learning for Action Recognition in Videos}, 

  year={2019},

  volume={9},

  number={3},

  pages={475-484},

  doi={10.1109/JETCAS.2019.2935004}}


```
