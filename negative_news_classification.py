from typing import List
import joblib
import jieba
from fuzzywuzzy import fuzz
from rule import negative_news_rule
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
import os
import numpy as np


class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.encoder = BertTokenizer.from_pretrained(os.path.join(model_path, "vocab.txt"))
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(os.path.join(model_path, "model.ckpt.meta"))
        self.saver.restore(self.sess, os.path.join(model_path, "model.ckpt"))

        # 获取训练数据的label_one, label_two样本，用于相似度匹配
        self.similar_text_for_serious, self.similar_text_for_normal = self._get_label_text_from_raw_data()

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
            df_raw_data = df_raw_data[df_raw_data["预测等级"].isnull()]

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

            label = self._tf_model_predict(list(df_raw_data["负面消息内容"]))

            df_raw_data["预测等级"] = label
            mapping_dict = {0: "一般", 1: "严重", 2: "中等"}
            df_raw_data["预测等级"] = df_raw_data["预测等级"].map(mapping_dict)

            df_pred = pd.concat([df_regular_one, df_fuzzy_one, df_fuzzy_two, df_raw_data], sort=False)
            df_pred = df_pred.sort_index()

            result = df_pred["预测等级"].tolist()

        return result

    def _batch_iter(self, input_ids, input_masks, segment_ids, batch_size=64):
        """生成批次数据，一个batch一个batch地产生句子向量"""
        data_len = len(input_ids)
        num_batch = int((data_len - 1) / batch_size) + 1
        input_ids_shuffle = np.array(input_ids[:])
        input_masks_shuffle = np.array(input_masks[:])
        segment_ids_shuffle = np.array(segment_ids[:])

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield input_ids_shuffle[start_id:end_id], input_masks_shuffle[start_id:end_id], segment_ids_shuffle[
                                                                                            start_id:end_id]

    def _encode_sentence(self, sentences: list, max_sentence_length=512):
        # 创建input_ids
        input_ids = []

        for sent in sentences:
            if len(sent) > (max_sentence_length - 2):
                sent = sent[:max_sentence_length - 2]

            encoded_sent = self.encoder.encode(sent)

            while len(encoded_sent) < max_sentence_length:
                encoded_sent.append(0)

            input_ids.append(encoded_sent)

        # 创建input_masks
        input_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]

            while len(att_mask) < max_sentence_length:
                att_mask.append(0)

            input_masks.append(att_mask)

        # 创建segment_ids
        segment_ids = [[0] * max_sentence_length for _ in range(len(sentences))]

        return input_ids, input_masks, segment_ids

    def _get_label_text_from_raw_data(self):
        """
        功能: 从训练文件中获取样本少的数据，用作模糊匹配
        """
        df = pd.read_excel(os.path.join(self.model_path, "total.xlsx"))
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
        for text in list_negative_news:
            df_example["point"] = df_example["负面消息内容"].apply(lambda x: fuzz.ratio(str(text), x))
            df_fuzzy = pd.concat([df_fuzzy, df_example[df_example["point"] > threshold]], axis=0)
        df_fuzzy["预测等级"] = label
        df_fuzzy = df_fuzzy.drop(["point"], axis=1)
        df_fuzzy = df_fuzzy.drop_duplicates(subset=["负面消息内容", "名称"])
        return df_fuzzy

    def _negative_news_predict(self, text: list):
        clf = joblib.load("negative_news_model")
        text = [" ".join(jieba.cut(x, cut_all=False)) for x in text]
        label = clf.predict(text)  # 0:一般，1:严重,2:中等
        prob = clf.predict_proba(text)
        return label, prob

    def _tf_model_predict(self, text):
        input_ids, input_masks, segment_ids = self._encode_sentence(text)
        input_ids_placeholder = self.sess.graph.get_tensor_by_name("input_ids:0")
        input_masks_placeholder = self.sess.graph.get_tensor_by_name("input_masks:0")
        segment_ids_placeholder = self.sess.graph.get_tensor_by_name("segment_ids:0")
        y_pred = self.sess.graph.get_tensor_by_name("pred:0")
        y = []

        for word_id, mask, segment in self._batch_iter(input_ids, input_masks, segment_ids, batch_size=8):
            feed_dict = {
                input_ids_placeholder: word_id,
                input_masks_placeholder: mask,
                segment_ids_placeholder: segment
            }
            _y_pred = self.sess.run([y_pred], feed_dict=feed_dict)
            y += list(_y_pred[0])

        return y


if __name__ == "__main__":
    my_text = [
        "公司至今仍未发布2020年二季度偿付能力报告。",
        "公司因批量转让不良债权资产包的首付款比例不符合规定、投资非标债权超规模限制、通过同业投资承接本行不良资产等违规行为被罚款。",
        "公司评级结果由2019年的B级降为2020年CCC级，该公司2017年、2018年评级分别为A级、BB级，已经是连续三年下滑。"
    ]
    my_model_path = "model/tf_negative_news"
    agent = Predictor(my_model_path)
    my_result = agent.predict(my_text)
    print(my_result)
