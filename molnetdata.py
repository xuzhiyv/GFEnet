import networkx as nx
import numpy as np
import pandas as pd
import os
import argparse
import torch
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.preprocessing import StandardScaler

from Dataset import MolNet
from utils import load_fp
from rdkit.Chem import Descriptors
remover = SaltRemover()
smile_graph = {}
meta = ['W', 'U', 'Zr', 'He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']


def load_qm9_mapping(qm9_path):
    """ 加载QM9特征并与SMILES匹配 """
    qm9_df = pd.read_excel(qm9_path)
    required_columns = [
        'mu (D)',
        'alpha (Bohr^3)',
        'homo (eV)',
        'lumo (eV)',
        'gap (eV)',
        'r2 (Bohr^2)',
        'zpve (eV)',
        'u0 (eV)',
        'u298 (eV)',
        'h298 (eV)',
        'g298 (eV)',
        'cv (cal/(mol*K))'
    ]

    missing = [col for col in required_columns if col not in qm9_df.columns]
    assert not missing, f"QM9文件缺少以下必要列: {missing}"

    scaler = StandardScaler()
    features = scaler.fit_transform(qm9_df[required_columns].values)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    qm9_features = {}
    for idx, row in qm9_df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            canon_smi = Chem.MolToSmiles(mol)
            qm9_features[canon_smi] = features[idx]

    return qm9_features


def order_gnn_features(bond):

    bond_type = bond.GetBondType()
    return [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.GetStereo()
    ]


def atom_features(atom):
    res = one_of_k_encoding_unk(atom.GetSymbol(),['C','N','O','F','P','S','Cl','Br','I','B','Si','Unknown']) + \
                    one_of_k_encoding(atom.GetDegree(), [1, 2, 3, 4, 5, 6]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + [atom.GetIsAromatic()] + one_of_k_encoding_unk(atom.GetFormalCharge(), [-1,0,1,3]) + \
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])
    try:
        res = res + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        bonds = atom.GetBonds()
        for bond in bonds:
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and str(bond.GetStereo()) in ["STEREOZ", "STEREOE"]:
                res = res + one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREOZ", "STEREOE"]) + [atom.HasProp('_ChiralityPossible')]
        if len(res) == 33:
            res = res + [False, False] + [atom.HasProp('_ChiralityPossible')]

    electronic_feat = [
            atom.GetMass() / 100.0,
            atom.GetNumImplicitHs(),
            1 if atom.IsInRing() else 0,
            atom.GetFormalCharge() / 4.0
        ]
    return np.array(res + electronic_feat)


def order_gnn_features(bond):
    weight = [1, 2, 3, 1.5]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]

    for i, m in enumerate(bond_feats):
        if m == True and i != 0:
            b = weight[i]
        elif m == True and i == 0:
            if bond.GetIsConjugated() == True:
                b = 1.4
            else:
                b = 1
        else: pass
    return b           


def order_tf_features(bond):
    weight = [1, 2, 3, 1.5]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
    for i, m in enumerate(bond_feats):
        if m == True:
            b = weight[i]
    return b        


def one_of_k_encoding(x, allowable_set):
    # if x not in allowable_set:
    #     raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smiletopyg(smi, fp1, fp2, fp3, fp4, qm9_feature):
    g = nx.Graph()
    mol = Chem.MolFromSmiles(smi)
    # 处理无效SMILES的情况（关键修复点）
    if mol is None:
        print(f"警告：无法解析SMILES字符串 '{smi}'，可能是不合法的分子表示")
        return [smi, None]
    c_size = mol.GetNumAtoms()

    if not mol:
        return [smi, None]

        # 标准化SMILES用于匹配QM9
    canon_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

    # 获取QM9特征，缺失值用0填充
    qm9_feat = qm9_feature.get(canon_smi, np.zeros(12)).astype(np.float32)

    features = []
    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i)
        feature = atom_features(atom)
        features.append((feature / sum(feature)).tolist()) 

    c = []
    adj_order_matrix = np.eye(c_size)
    dis_order_matrix = np.zeros((c_size,c_size))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bfeat = order_gnn_features(bond)
        g.add_edge(a1, a2, weight=bfeat)
        tfft = order_tf_features(bond)
        adj_order_matrix[a1, a2] = tfft
        adj_order_matrix[a2, a1] = tfft
        if bond.GetIsConjugated():
            c = list(set(c).union(set([a1, a2])))

    g = g.to_directed()
    edge_index = np.array(g.edges).tolist()
    edge_attr = []
    for w in list(g.edges.data('weight')):
        edge_attr.append(w[2])

    for i in range(c_size):
        for j in range(i,c_size):
            if adj_order_matrix[i, j] == 0 and i != j:
                conj = False
                paths = None
                try:
                    paths = list(nx.node_disjoint_paths(g, i, j))
                    if len(paths) > 1:
                        paths = sorted(paths, key=lambda i:len(i),reverse=False)
                    for path in paths:
                        if set(path) < set(c):
                            conj = True
                            break
                    if conj:
                        adj_order_matrix[i, j] = 1.2
                        adj_order_matrix[j, i] = 1.2
                except nx.NetworkXNoPath:
                    try:
                        path = nx.shortest_path(g, i, j)
                        dis_order_matrix[i, j] = len(path) - 1
                        dis_order_matrix[j, i] = len(path) - 1
                    except nx.NetworkXNoPath:
                        dis_order_matrix[i, j] = float('1')  # 或者其他合适的默认值
                        dis_order_matrix[j, i] = float('1')
                if paths is not None and paths:
                    path = paths[0]
                    dis_order_matrix[i, j] = len(path) - 1
                    dis_order_matrix[j, i] = len(path) - 1

        # 计算分子级电子特征
    electronic_descriptors = np.array([
        Chem.Descriptors.NumValenceElectrons(mol),
        Chem.Descriptors.TPSA(mol),
        Chem.Descriptors.MolLogP(mol)
    ], dtype=np.float32)
    g = [c_size, features, edge_index, edge_attr, adj_order_matrix, dis_order_matrix, fp1, fp2, fp3, fp4, qm9_feat, electronic_descriptors]
    return [smi, g]


def write(res):
    smi, g = res
    smile_graph[smi] = g


if __name__ == '__main__':
    moldata = 'shyj'
    task = 'clas'
    ncpu = 4
    PubchemFP881_path = r'C:\Users\Administrator\Desktop\dataset\dataset\工作2\数据总\fingerprint\PUBCHEM.csv'
    GraphFP1024_path = r'C:\Users\Administrator\Desktop\dataset\dataset\工作2\数据总\fingerprint\GRAPH.csv'
    APC2D780_path = r'C:\Users\Administrator\Desktop\dataset\dataset\工作2\数据总\fingerprint\APC2D.csv'
    FP1024_path = r'C:\Users\Administrator\Desktop\dataset\dataset\工作2\数据总\fingerprint\FP1024.csv'

    fp1 = load_fp(PubchemFP881_path)
    fp2 = load_fp(GraphFP1024_path)
    fp3 = load_fp(APC2D780_path)
    fp4 = load_fp(FP1024_path)

    fp1 = torch.FloatTensor(fp1)
    fp2 = torch.FloatTensor(fp2)
    fp3 = torch.FloatTensor(fp3)
    fp4 = torch.FloatTensor(fp4)

    labell = ['labels']
    numtasks = 1

    qm9_feat_map = load_qm9_mapping('res/qm9_max6.xlsx')

    processed_data_file = 'dataset/processed/' + moldata+task + '_pyg.pt'
    if not os.path.isfile(processed_data_file):
        try:
            df = pd.read_csv('./dataset/raw/'+moldata+'.csv')
        except:
            print('Raw data ```not found! Put the right raw csvfile in **/dataset/raw/')
        compound_iso_smiles = np.array(df['smiles'])
        ic50s = np.array(df[labell])
        #ic50s = -np.log10(np.array(ic50s))
        pool = Pool(ncpu)
        smis = []
        y = []
        result = []

        for idx,(smi, label) in enumerate(zip(compound_iso_smiles, ic50s)):
            smis.append(smi) 
            y.append(label)
            current_fp1 = fp1[idx]
            current_fp2 = fp2[idx]
            current_fp3 = fp3[idx]
            current_fp4 = fp4[idx]

            result.append(pool.apply_async( smiletopyg, (smi, current_fp1, current_fp2, current_fp3, current_fp4, qm9_feat_map)))
        pool.close()
        pool.join()

        for res in result:
            smi, g = res.get()
            smile_graph[smi] = g

        MolNet(root='./dataset', dataset=moldata+task, xd=smis, y=y, smile_graph=smile_graph,
               fp1_list=fp1, fp2_list=fp2, fp3_list=fp3, fp4_list=fp4)
    
