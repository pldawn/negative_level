import pandas as pd
import re


def negative_news_rule(text):
    # 正则加入大于３年的
    if "连续" in text and "亏损" in text and "季度" not in text:
        pattern = re.compile(r"三年|两年")
        years = "".join(pattern.findall(text))
        if not years:
            return "严重"
        else:
            return None
    # ok
    if "低于监管标准" in text:
        return "严重"

    if "单一最大股东" in text and "实质性违约" in text:
        return "严重"
    # ok
    if "转让" in text and "全部股份" in text:
        return "严重"
    # ok
    if "撤销" in text and "全部业务" in text:
        return "严重"
    # ok
    if "项目违约" in text or "破产公司" in text:
        return "严重"
    # ok
    if "管理层" in text and "更换频繁" in text:
        return "严重"
    # ok
    if "暂停" in text and "股票质押式回购交易" in text:
        return "严重"
    # ok
    if ("原党委书记" in text or "董事长" in text) and "贪污受贿" in text:
        return "严重"
    # ok
    if "多项监管指标" in text and "未达标" in text:
        return "严重"
    # ok
    if ("流动性风险" in text or "退市风险" in text or "重大风险隐患" in text or
       "流动性问题" in text) and ("较大" not in text or "警示" in text):
        return "严重"

    # ok
    if ("评级" in text) and ("降为" in text or "调整" in text or "下调" in text or "观察" in text or "减持" in text or "负面" in text):
        a_pattern = re.compile(r"A*")
        b_pattern = re.compile(r"B*")
        c_pattern = re.compile(r"C*")

        a = "".join(a_pattern.findall(text))
        b = "".join(b_pattern.findall(text))
        c = "".join(c_pattern.findall(text))
        if (a and b) or (b and c):
            return "严重"
        else:
            return None

    # 正则匹配数字　１年以上算高风险
    if ("轮候冻结" in text and "冻结期" in text) or ("如果" not in text and "强制退市" in text):
        pattern = re.compile(r"[1-9]年")
        years = "".join(pattern.findall(text))
        if years:
            return "严重"
        else:
            return None
    # ok
    if "信托产品" in text and "逾期" in text:
        return "严重"
    # ok
    if "证监会" in text and "行政监管" in text:
        return "严重"
    # ok
    if ("资本充足率" in text and "下行" in text) or ("拨备覆盖率" in text and "下限" in text):
        return "严重"
    # ok
    if "终止" in text and "主体评级" in text:
        return "严重"
    # 正则数字
    if "净利润" in text and "亏损" in text:
        string = re.search("\\d+(\\.\\d+)?亿", text)
        if string:
            num = float(string.group().split("亿")[0])
            if num > 20:
                return "严重"
    # ok
    if "风险警示" in text or "逾期兑付" in text or "破产申请" in text:
        return "严重"
    # 正则匹配出负数，例子见负样本
    if "综合偿付能力充足率" in text:
        return "严重"
    # ok
    if ("银行账户" in text and "冻结" in text) or ("股权" in text and "轮番冻结" in text):
        return "严重"
    # ok
    if "严重不足" in text or "暂停上市" in text or "警告处分" in text:
        return "严重"
    # ok
    if "严重警告" in text or ("变更" in text and "财务数据" in text):
        return "严重"
    # ok
    if "股票" in text and "机构抛售" in text:
        return "严重"

    # 正则数字 ok
    if ("股票" in text or "全部股份" in text or "流通股") and "冻结" in text:
        return "严重"
    # ok
    if "大面积" in text and "裁员" in text:
        return "严重"
    # ok
    if "高管辞职" in text and "创始人辞职" in text:
        return "严重"
    # ok
    if "短期偿债风险" in text:
        return "严重"
    # ok

    if "问询函" in text and "季度" not in text and "亏损" not in text and \
       "增收不增利" not in text and "增收减利" not in text:
        return "严重"
    # ok
    if "拖欠" in text and "工资" in text:
        return "严重"

    # ok
    if "财务杠杆率高" in text:
        return "严重"
    # ok
    if "实质性违约" in text:
        return "严重"
    # ok
    if "债券" in text and "暂停" in text and \
       "债券质押式回购" not in text and "债券自营业" not in text and\
       "债券承销业务" not in text:
        return "严重"
    # ok
    if "流动性" in text and ("严重" in text or "困难" in text):
        return "严重"
    # ok
    if "资金缺口" in text and "高达" in text:
        return "严重"
    # ok
    if "贷款逾期" in text:
        return "严重"
    # ok
    if ("董事长" in text and "猥亵" in text) or "继续跌停" in text:
        return "严重"
    # ok
    if "财务风险" in text:
        return "严重"
    # ok
    if "信用风险" in text and "考验" in text:
        return "严重"
    # ok
    if "债务" in text and "到期" in text:
        return "严重"
    # ok
    if "退股" in text or "暂停上市" in text:
        return "严重"
    # 待考虑
    # if "债券" in text and "观察" in text:
    #     return "严重"
    # ok
    if "信托贷款" in text and "逾期" in text:
        return "严重"
    # ok
    if "公司负债" in text and "超过" in text:
        return "严重"
    # ok
    if "暂停运营" in text or "负债规模" in text:
        return "严重"
    # ok
    if ("董事长" in text or "总裁" in text) and ("严重违规" in text or "受贿" in text):
        return "严重"
    # ok
    if "市场不看好" in text or "流动性危机" in text:
        return "严重"
    # ok
    if "财务指标" in text and "不利变化" in text and "符合交易所" not in text:
        return "严重"
    # ok
    if ("法律风险" in text and "上升" in text) or ("财务风险" in text and "上升" in text):
        return "严重"
    # ok
    if "利息" in text and "延迟" in text and "兑付" in text:
        return "严重"
    # ok
    if "债券" in text and "支付" in text and "未能" in text:
        return "严重"
    # ok
    if "兑付日期" in text and "延迟" in text:
        return "严重"

    # 正则数字
    if "累计亏损" in text:
        string = re.search("\\d+(\\.\\d+)?亿", text)
        if string:
            num = float(string.group().split("亿")[0])
            if num > 20:
                return "严重"

    if "内部管理" in text and "乱做一团" in text:
        return "严重"

    return ""


if __name__ == "__main__":
    # text = "联合 资信 将 公司 主体 评级 从 AA + 下调 至 AA ， 展望 负面 。 原因 是 其 单一最大股东 东旭 光电 出现实质性违约事件 。"
    # print(negative_news_rule(text))
    raw_data_dir = "/home/cym/Desktop/workspace/sentiment/负面消息/total.xlsx"
    df = pd.read_excel(raw_data_dir)
    df = df[["负面消息内容", "影响程度"]]
    df = df.drop_duplicates()
    # df["label"] = df["负面消息内容"].apply(lambda x:negative_news_rule(x))
    # df.to_csv("test.csv")
