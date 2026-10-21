# 基于 ResNet 的动物图像分类系统

使用Animals-10数据集 + ResNet18 微调，支持桌面端（PyQt5）与网页端（Flask）识别。
数据集：[Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
类别（10 类）：蝴蝶、猫、鸡、牛、狗、象、马、羊、蜘蛛、松鼠
说明：当前为整图单类别分类，一张图中有多种动物时只会给出一个主类别



一、 环境(Python 3.9+)
pip install -r requirements.txt




二、 使用流程

1. 下载并整理数据集
python download_dataset.py

自动下载 Animals-10（约 600MB），整理到 'data/<类名>/'，并生成 'classes.json'。  
若下载失败，可将 zip 手动放到 'raw_data/animals10.zip' 后重新运行。



2. 划分训练 / 测试集
python generate_dataset.py

生成 'train.txt'、'test.txt'。



3. 训练
python train.py

默认 30 epoch，按验证集准确率保存最佳模型
最佳权重：'output/resnet18_best.pth'
训练曲线：'tensorboard --logdir logs'




4. 推理与界面
（1）命令行单图测试
python predict.py images/horse.png
（2）桌面 GUI
python window.py
（3）网页服务
python myflask.py
浏览器复制 http://127.0.0.1:5000 打开 




5. 项目结构
Animal_Classifier/
├── config.py              # 类别、路径、超参数、简介文案
├── download_dataset.py    # 下载 Animals-10
├── generate_dataset.py    # 生成 train/test 列表
├── utils.py               # 数据集加载与增强
├── train.py               # ResNet18 训练
├── predict.py             # 推理接口
├── window.py              # PyQt5 桌面端
├── myflask.py             # Flask 网页端
├── templates/index.html   # 网页界面
├── static/background.jpg  # 网页背景
├── background.jpg         # 桌面端背景
├── images/                # 图标、示例图、历史缓存
├── requirements.txt
└── README.md




6. 界面功能
上传图片后自动识别
显示类别名称、置信度与简介
识别历史：查看、多选删除、一键清空
清除当前识别结果


