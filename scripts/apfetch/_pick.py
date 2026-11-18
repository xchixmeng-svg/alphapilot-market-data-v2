"""共用小工具：從候選欄位名稱裡找出真正存在的那一個，而不是要求整組欄位名稱完全對死。
官方 API 常常中英文欄名混用（例如同一份資料裡 SecuritiesCompanyCode 配中文的「發言日期」），
用固定組合去比對很容易全部落空；改成「依序嘗試每個候選名字」更能適應這種不一致。"""


def pick(df, *names, optional=False):
    for n in names:
        if n in df.columns:
            return df[n]
    if optional:
        return None
    raise KeyError(f"以下候選欄位都不存在：{list(names)}；實際欄位：{list(df.columns)}")
