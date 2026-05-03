# lblConvert

标签转换库

## Cli

更多细节参考[DOCS](docs/数据操作说明.md)

### 标注文件互相转换

- [X] 完成labelme2yolo的开发：使用 `lbl labelme2yolo -h`查看帮助;
- [X] 完成labelme2coco的开发：使用 `lbl labelme2coco -h`查看帮助;
- [X] 完成labelme2voc的开发：使用 `lbl labelme2voc -h`查看帮助;
- [X] 完成yolo2labelme的开发：使用 `lbl yolo2labelme -h`查看帮助;
- [X] 完成yolo2coco的开发：使用 `lbl yolo2coco -h`查看帮助;
- [X] 完成yolo2voc的开发：使用 `lbl yolo2voc -h`查看帮助;
- [X] 完成coco2yolo的开发：使用 `lbl coco2yolo -h`查看帮助;
- [X] 完成coco2labelme的开发：使用 `lbl coco2labelme -h`查看帮助;
- [X] 完成coco2voc的开发：使用 `lbl coco2voc -h`查看帮助;
- [X] 完成voc2yolo的开发：使用 `lbl voc2yolo -h`查看帮助;
- [X] 完成voc2labelme的开发：使用 `lbl voc2labelme -h`查看帮助;
- [X] 完成voc2coco的开发：使用 `lbl voc2coco -h`查看帮助;

### 标注过滤

- [X] 完成yoloLabelExclude的开发：使用 `lbl yoloLabelExclude -h`查看帮助;

### 配置文件生成

- [X] 完成voc2yoloClasses的开发：使用 `lbl genNames -h`查看帮助;

### 下载中文字体

- [X] 完成中文字体下载开发：使用 `lbl font -h`查看帮助;

## 快速开始

方法一:

```shell
rm -rf build
pip uninstall lblConvert -y
python -m build -sw -nx
pip install dist/*.whl
```

方法二:

```shell
git clone https://github.com/firstelfin/lblConvert.git
cd lblConvert
pip install .
```
