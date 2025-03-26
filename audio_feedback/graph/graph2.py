import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "Meiryo"

file_path = r"C:\Users\oobuh\OneDrive - 佐賀大学(edu)\研究\卒論実験.xlsx"
sheet_name = "結果まとめ5"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# カスタムカラー設定
categories = ["断続音", "連続音", "スズの音"]
# custom_palette = {
#     "断続音": "#1f77b4",  # 青色
#     "連続音": "#d62728",  # オレンジ色
#     "スズの音": "#2ca02c",  # 緑色
# }

# 棒グラフの作成
sns.barplot(
    x="Category",
    y="Values",
    order=categories,
    data=df,
    # palette=custom_palette,
    palette="muted",
    width=0.5,
    ci="sd",
    capsize=0.1,  # エラーバーの端にキャップを追加
)


# y軸の範囲の設定
# plt.ylim(0, 100)
plt.grid(True, axis="y", linestyle="--", linewidth=0.7, color="gray")
# グラフの設定
# plt.xlabel("音の種類", fontsize=60)
# plt.ylabel("SUSスコアの平均(点)", fontsize=40)
plt.ylabel("ボールに当てた回数(回)", fontsize=40)

# 目盛りの文字サイズの設定
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

plt.show()
