# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import time
from multiprocessing import Pool
from Base import Base
from Client import Client
from Configs import Configs
from DataLoader import DataLoaderCenter


class LightFR(Base):

    def __init__(self):
        super(LightFR, self).__init__()
        self.configs = Configs()
        # 新增：设置并行进程数（根据CPU核心数调整）
        # self.num_processes = 4 
        pass

    def get_random_client_list(self):
        size = int(len(self.user) * self.configs.client_ratio)
        random_client_list = np.random.choice(list(self.user.values()), size)
        return random_client_list

    def get_client_data(self, client_id):
        client = Client(self.configs)
        client.bu = self.B[client_id, :]
        client.D = self.D
        client.data_u = self.rating_matrix[client_id, :]
        client.data_bin_u = self.rating_matrix_bin[client_id, :]
        client.data_len_u = len(self.u_i_r[self.id2user[client_id]])
        return client

    # 新增：包装客户端更新函数供多进程调用
    def _client_update_wrapper(self,client_id):
        client=self.get_client_data(client_id)
        bu, _=client.client_update(client,master_flag=0)
        return client_id,bu

    def train_model(self):
        print("Training started...")
        current_round = 0
        last_loss = 0.0
        start_time = time.time()
        while (current_round < self.configs.global_rounds-40):
            master_flag = 0
            current_round += 1
            sampled_clients = self.get_random_client_list()
            #runing on clients, could be implemented in parallel(origin)
            for u in sampled_clients:
                client = self.get_client_data(u)
                bu, master_flag = client.client_update(client, master_flag)
            #并行化修改部分
            # with Pool(processes=self.num_processes) as pool:
            #     # 并行执行客户端更新
            #     results = pool.map(self._client_update_wrapper, sampled_clients)
                
            #     # 聚合客户端更新结果
            #     for client_id, bu in results:
            #         self.B[client_id, :] = bu  # 更新全局用户矩阵

            # running on the server(keep)
            for i in range(len(self.item)):
                while True:
                    flag = 0
                    di = self.D[i, :]
                    for k in range(self.configs.code_len):
                        # The following can be uploaded by the client side, and we upload the intermediate gradients, i.e., grads_a and grads_b, instead of the raw rating or the user codes. We can use the client-style computation as descriped in the paper, such as B[u,k], rating_matrix[u,i] and rating_matrix_bin[u,i], but it runs slowly.
                        # For efficient training, we use the batch-style computation to calculate the gradients.
                        # The intermediate gradients can be divided into multiple clients, that is loss_total=(self.rating_matrix[:, i] - np.dot(self.B, di.T)) can be reformulated into loss_user=(self.rating_matrix[u, i] - np.dot(self.B[u,:], di.T)), so the loss_total can be regarded as the aggregation from the multiple local loss_user.
                        bk = self.B[sampled_clients, k]
                        grads_a = (self.rating_matrix[sampled_clients, i] - np.dot(self.B[sampled_clients], di.T)) * bk * self.rating_matrix_bin[sampled_clients, i]
                        grads_b = len(self.rating_matrix_bin[sampled_clients, i])
                        # the following performs the simulated aggregation process
                        dik_hat = np.sum(grads_a) + grads_b * di[k]
                        dik_new = np.sign(self.K(dik_hat, di[k]))
                        if (di[k] != dik_new):
                            flag = 1
                            di[k] = dik_new
                    if (flag == 0):
                        break
                    self.D[i, :] = di
                    master_flag = 1

            # calculating the loss for all the clients and upload its loss into the server and then aggregate them
            self.loss = 0.0
            for u in range(len(self.user)):
                client = self.get_client_data(u)
                local_loss = client.calculate_loss()
                self.loss += local_loss

            federated_valid_hr_10, federated_valid_ndcg_10 = self.federated_valid_test_model(
                self.federated_valid_data_path)
            delta_loss = self.loss - last_loss
            print('current_round %d: current_loss = %.5f, delta_loss = %.5f valid_HR@10=%.5f valid_NDCG@10=%.5f' %
                  (current_round, self.loss, delta_loss, federated_valid_hr_10, federated_valid_ndcg_10))
            if (master_flag == 0):
                break
            if (abs(delta_loss) < self.configs.threshold or abs(delta_loss) == abs(self.last_delta_loss)):
                break
            self.last_delta_loss = delta_loss
            last_loss = self.loss
        federated_valid_hr_10, federated_valid_ndcg_10 = self.federated_valid_test_model(self.federated_test_data_path)
        print('test HR@10 = %.5f, NGCD@10 = %.5f' % (federated_valid_hr_10, federated_valid_ndcg_10))
        #计算时间
        end_time=time.time()
        training_time = end_time - start_time
        print(f'Training time: {training_time:.2f} seconds') 

    def federated_valid_test_model(self, path):
        val_data = self.read_federated_valid_dataset(path)
        configs = {'top_k': 10, 'num_negative_test': 49, }
        dl = DataLoaderCenter(configs, val_data)
        val_dataloader = dl.get_val_dataloader()
        hr_10, ndcg_10 = 0.0, 0.0
        len = 0

        # one batch represents a client since there is the same user in a batch
        for batch_id, batch in enumerate(val_dataloader):
            len += 1
            assert isinstance(batch[0], torch.LongTensor)
            users, items, ratings = batch[0], batch[1], batch[2]
            val_data = pd.DataFrame(zip(users.tolist(), items.tolist(), ratings.tolist()),
                                    columns=['user_id', 'item_id', 'ratings'])
            items = [self.item[item] for item in items.tolist()]
            user_id = self.user[int(users[0])]
            client = self.get_client_data(user_id)
            hr, ndcg = client.evaluate_local(items, val_data)

            hr_10 += hr[10]
            ndcg_10 += ndcg[10]

        hr_10 /= len
        ndcg_10 /= len

        return hr_10, ndcg_10

    def main(self):
        self.init_model()
        self.train_model()


if __name__ == '__main__':
    model = LightFR()
    model.main()
