import pypinyin

from pypinyin import pinyin, lazy_pinyin, Style
import pandas as pd

df = pd.read_excel(r"D:\SWOTRiverDischarge\SWOTRiverDischarge\test\中文名.xlsx")
pins = []

for index, row in df.iterrows():
    pins.append(lazy_pinyin(row["中文名"]))

nameList = []
for pin in pins:
    givenName = pin[0].capitalize()
    surName = pin[1].capitalize() + "".join(pin[2:])
    nameList.append(surName + " " + givenName)

df["英文名"] = nameList

df.to_excel(r"D:\SWOTRiverDischarge\SWOTRiverDischarge\test\中文名2.xlsx")