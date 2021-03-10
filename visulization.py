import numpy as np
import matplotlib.pyplot as plt


def get_flow(file_name):  # 将读取文件写成一个函数

    flow_data = np.load(file_name)  # 载入交通流量数据
    print([key for key in flow_data.keys()])  # 打印看看key是什么

    print(flow_data["data"].shape)  # (16992, 307, 3)，16992是时间(59*24*12)，307是节点数，3表示每一维特征的维度（类似于二维的列）
    flow_data = flow_data['data']  # [T, N, D]，T为时间，N为节点数，D为节点特征

    return flow_data


if __name__ == "__main__":
    traffic_data = get_flow("PeMS_04/PeMS04.npz")
    node_id = 10
    print(traffic_data.shape)

    plt.plot(traffic_data[:24 * 12, node_id, 0])  # 0维特征
    plt.savefig("node_{:3d}_1.png".format(node_id))

    plt.plot(traffic_data[:24 * 12, node_id, 1])  # 1维特征
    plt.savefig("node_{:3d}_2.png".format(node_id))

    plt.plot(traffic_data[:24 * 12, node_id, 2])  # 2维特征
    plt.savefig("node_{:3d}_3.png".format(node_id))
