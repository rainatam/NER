#### 程序清单

- `main.py`：主程序
- `model.py`：模型
- `gen_data.py`：生成训练数据
- `data.py`：模型读入数据的辅助函数
- `interface.py`：接口
- `util.py`：清洗数据的辅助函数
- `utils.py`：抽取的辅助函数



#### 主程序

train

```
python main.py --mode=train
```

test（通常不用）

```
python main.py --mode=train
```



#### 模型

共 12 个 Tag：

- PER - Person
- ADR - Address
- AFF - Affiliation
- TIT - Title
- JOB - Job
- DOM - Domain
- EDU - Education
- WRK - Work
- SOC - Social
- AWD - Award
- PAT - Patent
- PRJ - Project

这个模型基本上引用 https://github.com/Determined22/zh-NER-TF 的，这个模型是只提取 Person, Address 和 Affiliation 的。所以在它的基础上把模型扩展了一下。考虑到模型的效果，PER, ADR, AFF 使用原来模型和 pertrained 的参数，而剩下 9 个 Tag 使用我们扩展的模型。

`Model.py` 中定义了两个模型

- `BiLSTM_CRF`

  在原来模型的基础上，在 BiLSTM 前加了一层 LSTM。用于抽取 TIT, JOB, DOM, EDU, WRK, SOC, AWD, PAT, PRJ。

- `Original_model`

  这是原来的模型，用于抽取 PER, ADR, AFF。



#### 数据

`gen_data.py` 把各个文档读入并进行清理和标注，生成训练数据

主要函数如下：

- `clean_text`：清理文字

- `tagging`：标记数据成以下形式，`O` 代表没有意义。

  ```
  在	O
  新	B-ADR
  加	I-ADR
  坡	I-ADR
  工	O
  作	O
  ```

`data.py` 主要是负责模型读入和解释数据，如果要改变模型预测的 Tag ，主要需要修改的是 `Tag2Label`。

#### 接口

- `print_tag`：

  对抽取出来的 list 进去重重等清理并打印

- `extract_one`：

  使用 `BiLSTM_CRF` 模型和 pre-trained 参数抽取 TIT, JOB, DOM, EDU, WRK, SOC, AWD, PAT, PRJ。

- `extract_one_3`：

  使用 `Original_model` 模型和 pre-trained 参数抽取 PER, ADR, AFF。

- `interface`：

  分别调用 `extract_one` 和 `extract_one_3` 函数抽取总共 12 个 Tag。



#### 辅助函数

`utils.py` 主要用到的函数是 `get_entity` 和 `get_name_entity`。

- `get_name_entity(name, tag_seq, char_seq)`：对模型得出的结果进行抽取，name 是代表要抽取的 Tag 的名字。

- `get_entity`：

  调用 `get_name_entity` 对各个 Tag 进行抽取。

