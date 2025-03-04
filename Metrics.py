import pandas as pd
import torch
import numpy as np
import math
from sklearn import metrics as sk_metrics

# from reader.data_reader import DataReader
# from loader.data_loader import DataLoader


# from reader.data_reader import DataReader
# from loader.data_loader import DataLoader


class Metrics(object):
    def __init__(self, configs):
        super(Metrics, self).__init__()
        self.configs = configs

    def get_hit_ratio(self, test_data: pd.DataFrame):  # for implicit feedback
        top_k = self.configs['top_k']
        hrs = {}
        if test_data.empty:
            for current_top_k in range(1, top_k + 1):
                hrs[current_top_k] = 0.0
            return hrs

        assert 'pred' in test_data.columns, "没有预测值"
        test_data['rank'] = test_data['pred'].rank(method='first', ascending=False)
        test_data_rank = int(test_data.head(1)['rank'].iloc[0])  

        for current_top_k in range(1, top_k + 1):
            if test_data_rank <= current_top_k:
                hrs[current_top_k] = 1.0
            else:
                hrs[current_top_k] = 0.0
        return hrs

    def get_ndcg(self, test_data: pd.DataFrame):  # for implicit feedback
        top_k = self.configs['top_k']
        ndcgs = {}
        if test_data.empty:
            for current_top_k in range(1, top_k + 1):
                ndcgs[current_top_k] = 0.0
            return ndcgs

        assert 'pred' in test_data.columns, "没有预测值"
        test_data['rank'] = test_data['pred'].rank(method='first', ascending=False)
        test_data_rank = int(test_data.head(1)['rank'].iloc[0])  
        for current_top_k in range(1, top_k + 1):
            if test_data_rank <= current_top_k:
                ndcgs[current_top_k] = math.log(2) * 1.0 / math.log(1 + test_data_rank)
            else:
                ndcgs[current_top_k] = 0.0
        return ndcgs

    def get_hit_ratio_explicit_client(self, test_data: pd.DataFrame):  # for explicit feedback
        top_k = self.configs['top_k']
        hrs = {}
        if test_data.empty:
            for current_top_k in range(1, top_k + 1):
                hrs[current_top_k] = 0.0
            return hrs

        assert 'pred' in test_data.columns, "没有预测值"

        data = test_data[['pred', 'ratings']].to_numpy()

        real_value_list = sorted(data, key=lambda x: x[1], reverse=True)
        predict_value_list = sorted(data, key=lambda x: x[0], reverse=True)

        test_data['rank'] = test_data['pred'].rank(method='first', ascending=False)
        test_data_rank = int(test_data.head(1)['rank'].iloc[0])  

        for current_top_k in range(1, top_k + 1):
            if test_data_rank <= current_top_k:
                hrs[current_top_k] = 1.0
            else:
                hrs[current_top_k] = 0.0
        return hrs

    def get_ndcg_explicit_client(self, test_data: pd.DataFrame):  # for explicit feedback
        top_k = self.configs['top_k']
        ndcgs = {}
        if test_data.empty:
            for current_top_k in range(1, top_k + 1):
                ndcgs[current_top_k] = 0.0
            return ndcgs
        assert 'pred' in test_data.columns, "没有预测值"

        data = test_data[['pred', 'ratings']].to_numpy()

        real_value_list = sorted(data, key=lambda x: x[1], reverse=True)
        predict_value_list = sorted(data, key=lambda x: x[0], reverse=True)

        for current_top_k in range(1, top_k + 1):
            if len(real_value_list) >= current_top_k:
                idcg, dcg = 0.0, 0.0
                for i in range(current_top_k):
                    idcg += (pow(2, real_value_list[i][1]) - 1) / (math.log(i + 2, 2))
                    dcg += (pow(2, predict_value_list[i][1]) - 1) / (math.log(i + 2, 2))
                if idcg != 0:
                    ndcgs[current_top_k] = float(dcg / idcg)
                else:
                    ndcgs[current_top_k] = 0.0
            else:
                ndcgs[current_top_k] = 0.0
        return ndcgs

    def get_auc(self, test_data: pd.DataFrame):
        pass

    def get_mrr(self, test_data: pd.DataFrame):
        pass

    def get_rmse(self, test_data: pd.DataFrame):
        assert 'pred' in test_data.columns, "没有预测值"
        y = test_data['ratings']
        y_hat = test_data['pred']
        value = sk_metrics.mean_squared_error(y, y_hat) ** 0.5
        return value

    def get_mae(self, test_data: pd.DataFrame):
        assert 'pred' in test_data.columns, "没有预测值"
        y = test_data['ratings']
        y_hat = test_data['pred']
        value = sk_metrics.mean_absolute_error(y, y_hat)
        return value

    def get_rmse_client(self, test_data: pd.DataFrame):
        assert 'pred' in test_data.columns, "没有预测值"
        y = test_data['ratings']
        y_hat = test_data['pred']
        l = len(y)
        value = abs(y - y_hat) ** 2
        value = value.sum()
        result = math.sqrt(value / l)
        return result

    def get_mae_client(self, test_data: pd.DataFrame):
        assert 'pred' in test_data.columns, "没有预测值"
        y = test_data['ratings']
        y_hat = test_data['pred']
        l = len(y)
        value = abs(y - y_hat)
        value = value.sum()
        result = value / l
        return result

    def calDCG_k(self, dictdata, k):
        nDCG = []
        for key in dictdata.keys():
            listdata = dictdata[key]
            real_value_list = sorted(listdata, key=lambda x: x[1], reverse=True)
            idcg = 0.0
            predict_value_list = sorted(listdata, key=lambda x: x[0], reverse=True)
            dcg = 0.0
            if len(listdata) >= k:
                for i in range(k):
                    idcg += (pow(2, real_value_list[i][1]) - 1) / (log(i + 2, 2))
                    dcg += (pow(2, predict_value_list[i][1]) - 1) / (log(i + 2, 2))
                if (idcg != 0):
                    nDCG.append(float(dcg / idcg))
            else:
                continue
        ave_ndcg = np.mean(nDCG)
        # print(nDCG)
        return ave_ndcg