# Action-Recognition

## 1.实验环境

### 软件环境：
1）系统：Ubuntu16.04;<br>
2）编译器：python3.5;<br>
3）python深度学习库：tensorflow1.4，keras2+;<br>
4）其他环境：ffmpeg等<br>

### GPU配置：
双路Titan xp，每路显存12G，<br>

## 视频预处理
1）去UCF-101官网 http://crcv.ucf.edu/data/UCF101.php 下载UCF101数据集到data文件夹下;<br>
2）在data文件夹下使用unrar e UCF101.rar命令解压文件;<br>
3）在data文件夹下运行0_move_vedio_filepath.py文件把相应视频文件移动到train和test文件夹中;<br>
4）在data文件夹下运行1_extract_video_jpg.py文件把train和test中的视频用fffmpeg转换成图片，要在系统中安装fffmpeg;<br>


## 模型训练

1）运行extract_features.py文件，把处理的图片集合通过Inception-v3网络提取特征，并保存到/data/sequences/文件夹下，此处可以修改seq_length的值，来改变下采样的下采样的长度s，这需要花费4到8小时左右;<br>
2）运行train.py文件，训练模型，包含我们的模型和文中的其他几种模型。<br>

## 其他说明：
1）models.py中包含了我们设定模型，可以自己根据情况调整;<br>
2）train.py中修改相应的参数，进行相应的调整。<br>
