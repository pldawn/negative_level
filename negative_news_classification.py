from typing import List
import joblib
import jieba
from fuzzywuzzy import fuzz
from rule import negative_news_rule
import pandas as pd
import os


class Predictor:
    def __init__(self, model_path="./model"):
        self.model_path = model_path
        self.model = joblib.load(os.path.join(self.model_path, "negative_news_model"))

        # 获取训练数据的label_one, label_two样本，用于相似度匹配
        self.similar_text_for_serious, self.similar_text_for_normal = self._get_similar_text()

    def predict(self, text: List[str]):
        result = []

        if text:
            # 读取用户上传数据
            df_raw_data = pd.DataFrame(text, columns=["负面消息内容"])
            df_raw_data = df_raw_data.dropna()
            df_raw_data["负面消息内容"] = df_raw_data["负面消息内容"].apply(lambda x: str(x))

            # 正则匹配label_one
            df_raw_data["预测等级"] = df_raw_data["负面消息内容"].apply(lambda x: negative_news_rule(str(x)))

            # 数据过滤
            df_regular_one = df_raw_data[df_raw_data["预测等级"] == "严重"]
            df_raw_data = df_raw_data[df_raw_data["预测等级"] == ""]

            # 模糊匹配label_one
            df_fuzzy_one = self._fuzzy_match(self.similar_text_for_serious, df_raw_data, label="严重", threshold=50)

            # 原数据中过滤掉label_one
            if not df_fuzzy_one.empty:
                df_raw_data = df_raw_data.drop(df_fuzzy_one.index)

            # 获取label_two的模糊匹配的预测结果
            df_fuzzy_two = self._fuzzy_match(self.similar_text_for_normal, df_raw_data, label="一般", threshold=50)

            # 原数据中过滤掉label_two
            if not df_fuzzy_two.empty:
                df_raw_data = df_raw_data.drop(df_fuzzy_two.index)

            if not df_raw_data["负面消息内容"].empty:
                label, _ = self._model_predict(list(df_raw_data["负面消息内容"]))

                df_raw_data["预测等级"] = label
                mapping_dict = {0: "一般", 1: "严重", 2: "中等"}
                df_raw_data["预测等级"] = df_raw_data["预测等级"].map(mapping_dict)

            df_pred = pd.concat([df_regular_one, df_fuzzy_one, df_fuzzy_two, df_raw_data], sort=False)
            df_pred = df_pred.sort_index()

            result = df_pred["预测等级"].tolist()

        return result

    def _get_similar_text(self):
        """
        功能: 从训练文件中获取样本少的数据，用作模糊匹配
        """
        df = pd.read_excel(os.path.join(self.model_path, "total.xlsx"), engine="openpyxl")
        df = df[["负面消息内容", "影响程度"]]
        df = df.drop_duplicates()

        similar_text_for_serious = list(df[df["影响程度"] == "严重"]["负面消息内容"])
        similar_text_for_normal = list(df[df["影响程度"] == "一般"]["负面消息内容"])

        return similar_text_for_serious, similar_text_for_normal

    def _fuzzy_match(self, list_negative_news, df_example, label: str, threshold: int):
        """
        功能：通过字符串相似度对比的阈值控制，返回标签
        中性threshold 70
        严重threshold 50
        """
        df_fuzzy = pd.DataFrame()

        if not list_negative_news:
            return df_fuzzy

        for text in list_negative_news:
            df_example["point"] = df_example["负面消息内容"].apply(lambda x: fuzz.ratio(str(text), x))
            df_fuzzy = pd.concat([df_fuzzy, df_example[df_example["point"] > threshold]], axis=0)

        df_fuzzy["预测等级"] = label
        df_fuzzy = df_fuzzy.drop(["point"], axis=1)
        df_fuzzy = df_fuzzy.drop_duplicates(subset=["负面消息内容", "名称"])

        return df_fuzzy

    def _model_predict(self, text: list):
        text = [" ".join(jieba.cut(x, cut_all=False)) for x in text]
        label = self.model.predict(text)  # 0:一般，1:严重,2:中等
        prob = self.model.predict_proba(text)

        return label, prob


if __name__ == "__main__":
    my_text = [
        "公司至今仍未发布2020年二季度偿付能力报告。",
        "公司因批量转让不良债权资产包的首付款比例不符合规定、投资非标债权超规模限制、通过同业投资承接本行不良资产等违规行为被罚款。",
        "公司评级结果由2019年的B级降为2020年CCC级，该公司2017年、2018年评级分别为A级、BB级，已经是连续三年下滑。"
    ]

    agent = Predictor()
    my_result = agent.predict(my_text)
    print(my_result)
