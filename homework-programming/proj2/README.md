# 数据集下载和分割

于 https://cloud.tsinghua.edu.cn/f/4a89782049fc4f1aa9ce/?dl=1 下载数据集, 解压后对数据集进行分割,放在模型代码同级的文件夹`splited_data`下, 此时的目录结构应当如下:
```
├── README.md
├── bad_model.py
├── data_visualizer.py
├── improved_model.py
└── splited_data
    ├── test
    │   ├── battery
    │   │   ├── ...
    │   ├── biological
    │   │   ├── ...
    │   ├── ...
    ├── train
    │   ├── battery
    │   │   ├── ...
    │   ├── biological
    │   │   ├── ...
    │   ├── ...
    ├── val
    │   ├── battery
    │   │   ├── ...
    │   ├── biological
    │   │   ├── ...
    │   ├── ...
```

# 训练模型并预测

运行`bad_model.py`或`improved_model.py`, 此时会随着训练进度生成`result/bad_model.json`(`result/improved_model.json`)文件. 训练完成后命令行会给出最终模型在测试集上的acc.

# 数据可视化生成

运行`data_visualizer.py`, 读取`result/`目录下的json文件, 利用matplotlib生成各个数据随训练进程的变化图表. 若不指定, 则默认根据`result/bad_model.json`生成. 否则需给出命令行参数, 比如

```
python3 data_visualizer.py --file result/improved_model.json
```

生成的图片会自动保存在与json文件同级的目录中.
