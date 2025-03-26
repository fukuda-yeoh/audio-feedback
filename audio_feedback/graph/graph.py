import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# plt.rcParams["font.family"] = "Meiryo"

# 日本語用フォント (MS PGothic)
jp_font = fm.FontProperties(fname=r"C:\Windows\Fonts\msgothic.ttc")

# 英数字用フォント (Times New Roman)
en_font = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf")

fig, ax = plt.subplots()

# 日本語フォントを全体に適用
plt.rcParams["font.family"] = jp_font.get_name()

file_path = r"C:\Users\oobuh\OneDrive - 佐賀大学(edu)\研究\卒論実験.xlsx"
sheet_name = "距離平均の結果"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# カスタムカラー設定
categories = ["断続音", "連続音", "スズの音"]
# custom_palette = {
#     "断続音": "#1f77b4",  # 青色
#     "連続音": "#d62728",  # オレンジ色
#     "スズの音": "#2ca02c",  # 緑色
# }

# 箱ひげ図の作成
sns.boxplot(
    x="Category",
    y="Values",
    hue="Category",
    legend=False,
    showmeans=True,
    meanprops={
        "marker": "x",  # マーカーの形状
        "markersize": 12,  # マーカーのサイズ
        # "markerfacecolor": "white", #マーカーの表面の色
        "markeredgecolor": "black",  # マーカーの枠の色
        "markeredgewidth": 2,  # マーカーの枠の太さ
    },
    order=categories,
    data=df,
    palette="muted",
    width=0.5,
    ax=ax,
)
# sns.swarmplot(
#             # data=sub_df,
#             x="condition",
#             y="score",
#             color="k",
#             dodge=True,
#             size=5,
#             alpha=0.5,
#             ax=ax,
#             order=["Condition 1", "Condition 2", "Condition 3" , "Condition 4"]
# )

# グラフを表示
# plt.title("Boxplot Example")
# plt.xlabel("音の種類", fontsize=35)
plt.ylabel("ラケットとボール間の平均距離(cm)", fontsize=40,fontproperties=jp_font,labelpad=35)
# plt.ylabel("SUSスコアの平均(点)", fontsize=40, fontproperties=jp_font, labelpad=35)
# plt.ylim(0, 100)
# plt.ylabel(
#     "ボールに当てた平均回数(回)", fontsize=40, fontproperties=jp_font, labelpad=35
# )
# plt.ylim(0, 10)
# 補助線の追加
plt.grid(True, axis="y", linestyle="--", linewidth=0.7, color="gray")


# 目盛りの文字サイズの設定
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=jp_font, fontsize=40)  # 日本語
ax.set_yticklabels(ax.get_yticks(), fontproperties=en_font, fontsize=30)

plt.show()
