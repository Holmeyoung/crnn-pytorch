# Convolutional Recurrent Neural Network + CTCLoss 

I think i have fixed the ctcloss nan problem!

Now!

Please pull the latest code from master.

Please update the pytorch to  `>= v1.2.0`

Enjoy it!

> PS: Once there is ctclossnan, please
> 1. Change the `batchSize` to smaller (eg: 8, 16, 32)
> 2. Change the `lr` to smaller (eg: 0.00001, 0.0001)
> 3. Contact me by emailing to holmeyoung@gmail.com

## Dependence

- CentOS7
- Python3.6.5
- torch==1.2.0
- torchvision==0.4.0
- Tesla P40 - Nvidia

## Run demo

- Download a pretrained model from [Baidu Cloud](https://pan.baidu.com/s/1FmJhYf1Wy-LUaz4V2WpF7g) (extraction code: `si32`)
- People who cannot access Baidu can download a copy from [Google Drive](https://drive.google.com/drive/folders/1FhXvPtitX6tWYocFZiZBRzVHjK2o640u?usp=sharing)

- Run demo

  ```sh
  python demo.py -m path/to/model -i data/demo.jpg
  ```

   ![demo](https://raw.githubusercontent.com/Holmeyoung/crnn_pytorch/master/demo/demo.jpg)

  Expected output

  ```sh
  -妳----真---的的---可---------以 => 妳真的可以
  ```

  

## Feature

- Variable length

  It support variable length.



- Chinese support

  I change it to `binary mode` when reading the key and value, so you can use it to do Chinese OCR.



- Change CTCLoss from [warp-ctc](https://github.com/SeanNaren/warp-ctc) to [torch.nn.CTCLoss](https://pytorch.org/docs/stable/nn.html#ctcloss)

  As we know, warp-ctc need to compile and it seems that it only support PyTorch 0.4. But PyTorch support CTCLoss itself, so i change the loss function to `torch.nn.CTCLoss` .

  

- Solved PyTorch CTCLoss become `nan` after several epoch

  Just don't know why, but when i train the net, the loss always become `nan` after several epoch.

  I add a param `dealwith_lossnan` to `params.py` . If set it to `True` , the net will autocheck and replace all `nan/inf` in gradients to zero.



- DataParallel

  I add a param `multi_gpu` to `params.py` . If you want to use multi gpu to train your net, please set it to `True` and set the param `ngpu` to a proper number.



## Train your data

### Prepare data

#### Folder mode

1. Put your images in a folder and organize your images in the following format:

   `label_number.jpg` 

   For example

   - English

   ```sh
   hi_0.jpg hello_1.jpg English_2.jpg English_3.jpg E n g l i s h_4.jpg...
   ```

   - Chinese

   ```sh
   一身转战_0.jpg 三千里_1.jpg 一剑曾当百万师_2.jpg 一剑曾当百万师_3.jpg 一 剑 曾 当 百 万 师_3.jpg ...
   ```

   So you can see, the number is used to distinguish the same label.



2. Run the `create_dataset.py` in `tool` folder by

   ```sh
   python tool/create_dataset.py --out lmdb/data/output/path --folder path/to/folder
   ```

   

3. Use the same step to create train and val data.



4. The advantage of the folder mode is that it's convenient! But due to some illegal character can't be in the path

    ![Illegal character](https://raw.githubusercontent.com/Holmeyoung/crnn_pytorch/master/demo/illegal_character.png)

   So the disadvantage of the folder mode is that it's labels are limited. 



#### File mode

1. Your data file should like

   ```sh
   absolute/path/to/image/一身转战_0.jpg
   一身转战
   absolute/path/to/image/三千里_1.jpg
   三千里
   absolute/path/to/image/一剑曾当百万师_2.jpg
   一剑曾当百万师
   absolute/path/to/image/3.jpg
   一剑曾当百万师
   absolute/path/to/image/一 剑 曾 当 百 万 师_4.jpg
   一 剑 曾 当 百 万 师
   absolute/path/to/image/xxx.jpg
   label of xxx.jpg
   .
   .
   .
   ```

   > DO REMEMBER:
   >
   > 1. It must be the absolute path to image.
   > 2. The first line can't be empty.
   > 3. There are no blank line between two data.



2. Run the `create_dataset.py` in `tool` folder by

   ```sh
   python tool/create_dataset.py --out lmdb/data/output/path --file path/to/file
   ```

   

3. Use the same step to create train and val data.



### Change parameters and alphabets

Parameters and alphabets can't always be the same in different situation. 

- Change parameters

  Your can see the `params.py` in detail.

- Change alphabets

  Please put all the alphabets appeared in your labels to `alphabets.py` , or the program will throw error during training process.



### Train

Run `train.py` by

```sh
python train.py --trainroot path/to/train/dataset --valroot path/to/val/dataset
```



## Reference

[meijieru/crnn.pytorch](<https://github.com/meijieru/crnn.pytorch>)

[Sierkinhane/crnn_chinese_characters_rec](<https://github.com/Sierkinhane/crnn_chinese_characters_rec>)

