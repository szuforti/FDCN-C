import numpy as np
from torchvision.utils import save_image
import pandas as pd

save_data_df = pd.DataFrame([],
                            columns=["数据集名称", "被试标签", "交叉验证折数K", "交叉验证批次Ki", "训练集max_ACC", "训练集最佳kapper", "训练集min_LOSS",
                                     "训练集最佳epoch(acc)", "验证集max_ACC", "验证集最佳kapper", "验证集min_LOSS", "验证集最佳epoch(acc)",
                                     "测试集max_ACC", "测试集最佳kapper", "测试集min_LOSS", "实际训练epoch", "实验时间"], index=[])

save_dict = {"数据集名称": "", "被试标签": "", "交叉验证折数K": 0, "交叉验证批次Ki": 0,
             "训练集max_ACC": 0.0, "训练集最佳kapper": 0.0, "训练集min_LOSS": 0.0, "训练集最佳epoch(acc)": 0,
             "验证集max_ACC": 0.0, "验证集最佳kapper": 0.0, "验证集min_LOSS": 0.0, "验证集最佳epoch(acc)": 0,
             "测试集max_ACC": 0.0, "测试集最佳kapper": 0.0, "测试集min_LOSS": float("Inf"),
             "实际训练epoch": 0, "实验时间": "error"}
