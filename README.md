# garmenttwin

## Step1：视频录制

[GitHub - robotflow-initiative/azure-kinect-apiserver](https://github.com/robotflow-initiative/azure-kinect-apiserver)

根据该项目说明搭建场景录制动作视频

## Step2：粗处理（进行视频拆分，apritag计算，相机参数整理）

PostProcessor.py 主函数输入参数：单个数据集根目录

## Step3：补全handpose（因为其中有空帧）

PostProcessor2.py 搜索主函数输入目录参数下的所有数据集进行处理，判断依据：粗处理生成的res.json

## Step4：VR手动标注

使用Unity打开ClothPoseVR工程，在scene场景Manager脚本上输入上面的数据路径DatasetPath，运行场景在VR中进行数据修复和标注，模型数据将保存在每个视频目录下SaveClip***

操作方式见：[说明](./ClothPoseVR/ClothPoseVR.md)

## Step5：track-sam预处理：

imgs2video.py中imgs2video函数,输入单个数据集根目录，在SaveClip***文件夹下生成三个机位的mp4

## Step6：跑track-sam：

1. 终端打开wsl,
2. conda activate sam
3. 部署 [GitHub - z-x-yang/Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
4. cd到/Segment-and-Track-Anything-1.5
5. python app.py启动web服务，浏览器中[http://127.0.0.1:7860/访问](http://127.0.0.1:7860/%E8%AE%BF%E9%97%AE)
6. 拖入之前预处理得到的视频
7. 开始在第一帧选点：下面菜单列表（Everything/Click/Stroke/Text）中选Click
   PointPrompt中Positive(正选)/Negative(反选)选若干个点(多点一些)
8. StartTracking,后台终端可以看进度(报错ctrl+c强制关闭进程再从第4步重启)
9. 运行结束后,在C:\Users\robotflow\Desktop\SAM\Segment-and-Track-Anything-1.5\tracking_results下可以看到结果
   将"相机名_masked"文件夹转移到对应SaveClip***文件夹下,其中应包含一推纯色遮罩图
   (可能前面一部分是只有衣服，但后面就开始出现不同颜色遮罩，但这没有关系,只要保证衣服那部分的颜色一直统一就行)

## Step7：后处理：

一个SaveClip三个机位视频处理完后,运行imgs2video.py中mask2pc函数
输入参数为SaveClip的路径