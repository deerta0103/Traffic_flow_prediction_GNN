# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_processing import LoadData
from model import *
import numpy as np
import math
import matplotlib.pyplot as plt
"""
     使用图卷积神经网络实现基于交通流量数据的预测
     Dataset description：
     PeMS04 ，加利福尼亚高速数据，"data.npz"，原始数据shape=(10195,307,3)——间隔5分钟预测1小时(307,3,36)->(307,3,12)
     其中，"3"代表交通流量3种特征(flow，speed，occupancy)。

"""
def MAE(y_true,y_pre):
    y_true=(y_true).detach().cpu().numpy().copy().reshape((-1,1))
    y_pre=(y_pre).detach().cpu().numpy().copy().reshape((-1,1))
    re = np.abs(y_true-y_pre).mean()
    return re

def RMSE(y_true,y_pre):
    y_true=(y_true).detach().cpu().numpy().copy().reshape((-1,1))
    y_pre=(y_pre).detach().cpu().numpy().copy().reshape((-1,1))
    re = math.sqrt(((y_true-y_pre)**2).mean())
    return re

def MAPE(y_true,y_pre):
    y_true=(y_true).detach().cpu().numpy().copy().reshape((-1,1))
    y_pre=(y_pre).detach().cpu().numpy().copy().reshape((-1,1))
    e = (y_true+y_pre)/2+1e-2
    re = (np.abs(y_true-y_pre)/(np.abs(y_true)+e)).mean()
    return re

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Loading Dataset
    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    test_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)




    # Loading Model

    #model = GCN(in_c=6 , hid_c=6 ,out_c=1)
    #model = ChebNet(in_c=6, hid_c=32, out_c=1, K=2)      # 2阶切比雪夫模型
    model = GATNet(in_c=6 , hid_c=6 ,out_c=1, n_heads=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters())

    # Train model
    Epoch = 10
    loss_train_plt = []


    model.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        epoch_mae = 0.0
        epoch_rmse = 0.0
        epoch_mape = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            model.zero_grad()
            predict_value = model(data, device).to(torch.device("cuda"))  # [0, 1] -> recover
            loss = criterion(predict_value, data["flow_y"].to(device))
            epoch_mae += MAE(data["flow_y"].to(device), predict_value)
            epoch_rmse += RMSE(data["flow_y"].to(device), predict_value)
            epoch_mape += MAPE(data["flow_y"].to(device), predict_value)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        end_time = time.time()
        loss_train_plt.append(10 * epoch_loss / (len(train_data) / 64))
        print("Epoch: {:04d}, Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, Time: {:02.2f} mins".format(
                epoch + 1, 10 * epoch_loss / (len(train_data) / 64),
                epoch_mae, epoch_rmse, epoch_mape, (end_time - start_time) / 60))


    # Test Model
    model.eval()
    loss_test_plt = []
    with torch.no_grad():
        total_loss = 0.0

        num = 0
        all_predict_value = 0
        all_y_true = 0

        for data in test_loader:
            predict_value = model(data, device).to(torch.device("cuda"))  # [B, N, 1, D]
            if num == 0:
                all_predict_value = predict_value
                all_y_true = data["flow_y"]
            else:
                all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
                all_y_true = torch.cat([all_y_true, data["flow_y"]], dim=0)

            loss = criterion(predict_value, data["flow_y"].to(device))
            total_loss += loss.item()
            num += 1

        epoch_mae = MAE(all_y_true, all_predict_value)
        epoch_rmse = RMSE(all_y_true, all_predict_value)
        epoch_mape = MAPE(all_y_true, all_predict_value)
        print("Test Loss: {:02.4f}, mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(
            10 * total_loss / (len(test_data) / 64), epoch_mae, epoch_rmse, epoch_mape))

    node_id = 120

    plt.title("Real data")
    plt.xlabel("time/5min")
    plt.ylabel("traffic flow")
    plt.plot(
        test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true.cpu().numpy())[:24 * 12,
        node_id, 0, 0],
        label='Ground Truth')
    plt.legend()
    plt.savefig("Ground truth", dpi=400)
    plt.show()

    plt.title(" visualization for 1 day")
    plt.xlabel("time/5min")
    plt.ylabel("traffic flow")
    plt.plot(
        test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true.cpu().numpy())[:24 * 12, node_id, 0, 0],
        label='Ground Truth')
    plt.plot(
        test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value.cpu().numpy())[:24 * 12, node_id, 0,
        0], label='Prediction')
    plt.legend()
    plt.savefig( " visualization for 1 day.png", dpi=400)
    plt.show()

    plt.title("Training Loss")
    plt.xlabel("time/5min")
    plt.ylabel("traffic flow")
    plt.plot(loss_train_plt, label='loss_train')
    #plt.plot(loss_test_plt, label='loss_test')
    plt.legend()
    plt.savefig("Training loss.png", dpi=400)
    plt.show()

    # plt.title( " visualization (2 weeks)")
    # plt.xlabel("time/5min")
    # plt.ylabel("traffic flow")
    # plt.plot(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true.cpu().numpy())[:, node_id, 0, 0],
    #          label='Ground Truth')
    # plt.plot(
    #     test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value.cpu().numpy())[:, node_id, 0, 0],
    #     label='Prediction')
    # plt.legend()
    # plt.savefig( " visualization for 2 weeks.png", dpi=len(test_data))
    # plt.show()

    mae = MAE(torch.from_numpy(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true.cpu().numpy())),
              torch.from_numpy(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value.cpu().numpy())))
    rmse = RMSE(torch.from_numpy(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true.cpu().numpy())),
                torch.from_numpy(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value.cpu().numpy())))
    mape = MAPE(torch.from_numpy(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_y_true.cpu().numpy())),
                torch.from_numpy(test_data.recover_data(test_data.flow_norm[0], test_data.flow_norm[1], all_predict_value.cpu().numpy())))
    print(
        "Accuracy Indicators Based on Original Values  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse,
                                                                                                             mape))


if __name__ == '__main__':
    main()
    # loss_train_plt,loss_test_plt = main()
    # loss = [i[0] for i in loss_train_plt]
    # for epoch in [i[0] for i in loss_train_plt]:
    #     plt.figure()
    #     plt.plot(epoch, [i[1] for i in loss_train_plt], label="loss")
    #     plt.draw()
    #     plt.show()


