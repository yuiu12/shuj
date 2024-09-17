from renumics import spotlight
from sliceguard import SliceGuard
from sliceguard.data import from_huggingface
from sklearn.metrics import accuracy_score
#https://medium.com/@daniel-klitzke/finding-problematic-data-slices-in-unstructured-data-aeec0a3b9a2a
# df = from_huggingface("Matthijs/snacks",trust_remote_code=True) 

# sg = SliceGuard() 
# issues = sg.find_issues(df,features=["image"],y="label",metric=accuracy_score) 
# report_df,spotlight_data_issues,spotlight_dtypes,spotlight_layout = sg.report(
#     no_browser=True
# )
# spotlight.show(
#     report_df,
#     dtype=spotlight_dtypes,
#     issues = spotlight_data_issues,
#     layout = spotlight_layout,
# )
'''
Sliceguard 将首先计算图像列的嵌入向量，以生成有意义的表示来比较您的图像。
然后，它将在这些嵌入和提供的标签上训练模型，随后为整个数据集生成预测。
然后，Sliceguard 将在嵌入上运行分层聚类算法，以查找具有相似特征的图像组，例如，所有图像不仅包含食物，还包含一个人，或者所有看起来相对较暗的图像。
之后，它将计算为所有找到的集群提供的指标 （准确性），将这些集群标记为明显低于整体准确性的潜在问题。

'''
#音频示例
df = from_huggingface("renumics/emodb")
sg = SliceGuard()
issues = sg.find_issues(df, features=["audio"], y="emotion", metric=accuracy_score)
report_df, spotlight_data_issues, spotlight_dtypes, spotlight_layout = sg.report(
    no_browser=True
)

# # Visualize Detected Issues in Spotlight:
spotlight.show(
     report_df,
     dtype=spotlight_dtypes,
     issues=spotlight_data_issues,
     layout=spotlight_layout,
)

#文本示例
#df = from_huggingface("dair-ai/emotion")
# sg = SliceGuard()
# issues = sg.find_issues(df, features=["text"], y="label", metric=accuracy_score)
# report_df, spotlight_data_issues, spotlight_dtypes, spotlight_layout = sg.report(
#     no_browser=True
# )

# # Visualize Detected Issues in Spotlight:
# spotlight.show(
#     report_df,
#     dtype=spotlight_dtypes,
#     issues=spotlight_data_issues,
#     layout=spotlight_layout,
# )