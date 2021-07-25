# -*- coding: utf-8 -*-


from surprise import KNNBaseline
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
trainset = data.build_full_trainset()# 把加载的数据全部作为训练数据生成训练数据集

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNBaseline(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE、MAE
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)

#预测uid、iid的预测结果
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)
print(pred)