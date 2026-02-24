# 导入 typing 模块中的 Dict 和 List 类型，用于类型注解，方便代码的阅读和理解
from typing import Dict, List
# 从 torch.utils.data 模块导入 Dataset 类，这是 PyTorch 中用于自定义数据集的基类
from torch.utils.data import Dataset
# 导入 pandas 库并简称为 pd，pandas 是一个强大的数据处理和分析库，常用于处理 CSV 文件
import pandas as pd 


class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning 
    这个类继承自 PyTorch 的 Dataset 类，用于创建一个自定义数据集，专门用于有监督的微调任务
    """

    def __init__(
        self, 
        data_path: str, 
    ) -> None:
        # 调用父类 Dataset 的构造函数进行初始化
        super(LazySupervisedDataset, self).__init__()
        # 使用 pandas 的 read_csv 函数读取指定路径下的 CSV 文件，并将其存储在 self.csv 中
        self.csv = pd.read_csv(data_path)

    def __len__(self) -> int:
        # 返回数据集的长度，即 CSV 文件中的行数
        return len(self.csv) 

    def construct_messages(self, text):
        # 构造一个消息列表，包含用户消息和助手消息，用于模拟对话场景
        message = [
            {
                # 用户消息的角色为 "user"
                "role": "user",
                # 用户消息的内容是一个包含文本类型的列表
                "content": [
                    # 文本类型的内容为传入的文本后面加上提示信息，要求对文本进行一句话总结
                    {"type": "text", "text": f"{text}\nSummarize above sentence in one word: "}
                ]
            },
            {
                # 助手消息的角色为 "assistant"
                "role": "assistant",
                # 助手消息的内容是一个包含文本类型的列表
                "content": [
                    # 文本类型的内容为占位符 "<emb>."
                    {"type": "text", "text": f"<emb>."}
                ]
            },
        ]
        return message

    def get_instance(self, index):
        # 从 CSV 文件中根据索引 index 获取三列数据，分别存储在 sent0、sent1 和 sent2 中
        sent0, sent1, sent2 = self.csv['sent0'][index], self.csv['sent1'][index], self.csv['hard_neg'][index]
        # 调用 construct_messages 方法，分别为 sent0、sent1 和 sent2 构造消息列表
        message1 = self.construct_messages(sent0)
        message2 = self.construct_messages(sent1)
        message3 = self.construct_messages(sent2)

        # 原本返回三个消息列表，但注释掉了这一行
        # return message1, message2, message3 
        # 实际返回前两个消息列表
        return message1, message2 

    def __getitem__(self, i) -> Dict[str, List]:      
        # 根据索引 i 调用 get_instance 方法获取相应的消息列表
        return self.get_instance(i)