import os
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
from pprint import pprint


class MolNet_pre(InMemoryDataset):
    def __init__(self, root='dataset', dataset=None, xd=None, y=None, transform=None, pre_transform=None, smile_graph=None,
                 fp1_list=None, fp2_list=None, fp3_list=None, fp4_list=None):
        # root is required for save raw data and preprocessed data
        super(MolNet_pre, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        # pprint(self.dataset)
        #
        # if xd is None or y is None:
        #     raise ValueError("Both 'xd' (SMILES strings) and 'y' (labels) must be provided.")
        # if not isinstance(xd, (list, tuple, np.ndarray)) or not isinstance(y, (list, tuple, np.ndarray)):
        #     raise TypeError("'xd' and 'y' must be iterable.")
        # if len(xd) != len(y):
        #     raise ValueError("'xd' and 'y' must have the same length.")
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, smile_graph,fp1_list, fp2_list, fp3_list, fp4_list)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        #pass
        return ['raw_file']

    @property
    def processed_file_names(self):
        return [self.dataset + '_pyg.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, smile_graph,fp1_list, fp2_list, fp3_list, fp4_list):
        # assert (len(xd) == len(y)), "smiles and labels must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):

            smiles = xd[i]
            if smiles is not None:
                # labels = np.asarray([y[i]])
                graphdata = smile_graph[smiles]
                (leng, features, edge_index, edge_attr, adj_order_matrix, dis_order_matrix
                 ,fp1, fp2, fp3, fp4, qm9_feat, electronic_desc) = graphdata
                fp1 = fp1_list[i].squeeze()
                fp2 = fp2_list[i].squeeze()
                fp3 = fp3_list[i].squeeze()
                fp4 = fp4_list[i].squeeze()

                fp1_tensor = torch.tensor(fp1, dtype=torch.float32).reshape(1, -1) # [1, 881]
                fp2_tensor = torch.tensor(fp2, dtype=torch.float32).reshape(1, -1) # [1, 1024]
                fp3_tensor = torch.tensor(fp3, dtype=torch.float32).reshape(1, -1) # [1, 780]
                fp4_tensor = torch.tensor(fp4, dtype=torch.float32).reshape(1, -1) # [1, 1024]

                assert fp1_tensor.shape == (1, 881), f"fp1维度错误: {fp1_tensor.shape}"
                assert fp2_tensor.shape == (1, 1024), f"fp2维度错误: {fp2_tensor.shape}"
                assert fp3_tensor.shape == (1, 780), f"fp3维度错误: {fp3_tensor.shape}"
                assert fp4_tensor.shape == (1, 1024), f"fp4维度错误: {fp4_tensor.shape}"

                electronic_desc = graphdata[-1]

                electronic_desc_tensor = torch.FloatTensor(electronic_desc).view(1, -1)  # shape [1,3]
                qm9_tensor = torch.FloatTensor(qm9_feat).view(1, -1)  # [1,12] 新增行

                electronic_desc = np.nan_to_num(electronic_desc, nan=0.0, posinf=0.0, neginf=0.0)
                qm9_feat = np.nan_to_num(qm9_feat, nan=0.0, posinf=0.0, neginf=0.0)

                electronic_desc_tensor = torch.FloatTensor(electronic_desc).view(1, -1)
                electronic_desc_tensor = torch.nan_to_num(electronic_desc_tensor, nan=0.0)

                qm9_tensor = torch.FloatTensor(qm9_feat).view(1, -1)
                qm9_tensor = torch.nan_to_num(qm9_tensor, nan=0.0)


                if len(edge_index) > 0:
                    GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0).contiguous(),
                                        edge_attr=torch.Tensor(edge_attr),
                                        fp1=fp1_tensor,
                                        fp2=fp2_tensor,
                                        fp3=fp3_tensor,
                                        fp4=fp4_tensor,
                                        qm9_features=qm9_tensor,
                                        electronic_desc=electronic_desc_tensor

                                        )
                    print(len(electronic_desc))
                    GCNData.adj = adj_order_matrix 
                    GCNData.dis = dis_order_matrix  
                    GCNData.leng = [leng] 
                    GCNData.smi = smiles
                    data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])