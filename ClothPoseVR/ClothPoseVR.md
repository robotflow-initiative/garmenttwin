###### ClothPoseVR标注流程：

1. 载入一段Video：读取相机/点云/现有HandPose数据
   
   全局按键：
   
   左手柄Pad左右切换Video
   
   右手柄Pad左右切换Frame

2. 进入对齐模式：
   
   加载Obi Cloth，手柄Trigger调整Cloth到初始状态
   
   手柄Grab按住调整点云对齐Obi Cloth

3. 进入切分片段模式：
   
   左手柄Trigger选出一个起始帧
   
   左手柄Trigger选出一个结束帧
   
   片段选定后根据现有HandPose数据，场景内显示出该片段内左(蓝色)右(绿色)手运动轨迹。

4. 选定片段后进入HandPose调整模式：
   
   显示当前帧HandPose，抓取为方形，松开为圆形
   
   Obi Cloth自动根据当前HandPose添加Pin点
   
   左手柄Trigger替换当前帧左手Pose， Grab切换当前帧以后的抓取/松开状态
   
   右手柄Trigger替换当前帧右手Pose， Grab切换当前帧以后的抓取/松开状态

5. 保存片段数据
   
   


