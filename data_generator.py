import pickle
import dgl
import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

# Feature_Path = "/pubssd/PPIs/Feature/"
Feature_Path = "./Feature/"
# Seed
SEED = 2024
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
BASE_MODEL_TYPE = 'AGAT'  # agat/gcn
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no
USE_EFEATS = True  # True/False
if BASE_MODEL_TYPE == 'GCN':
    USE_EFEATS = False
MAP_CUTOFF = 14
DIST_NORM = 15

INPUT_DIM = 61  #61
HIDDEN_DIM = 256  # hidden size of node features
LAYER = 8  # the number of AGAT layers
DROPOUT = 0.3
ALPHA = 0.7
LAMBDA = 1.5

LEARNING_RATE = 1E-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 50

FLITER_DIM = 512
OUTPUT_SIZE = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_node_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')

    if res_atom_feature.shape[0] != dssp_feature.shape[0]:
        if res_atom_feature.shape[0] > dssp_feature.shape[0]:
            res_atom_feature = res_atom_feature[:dssp_feature.shape[0]]
        else:
            padding = np.zeros((dssp_feature.shape[0] - res_atom_feature.shape[0], res_atom_feature.shape[1]))
            res_atom_feature = np.vstack((res_atom_feature, padding))

    node_features = np.concatenate(
        [dssp_feature, pssm_feature, hmm_feature, res_atom_feature], axis=1)
    return node_features # (seq_len, FEATURE_DIM=61)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list

def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix, xyz_feats, edges, edge_att, edge_feat, plm_features, rsa_features, bfactor_features = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    xyz_feats = torch.cat(xyz_feats)
    edges = edges[0]
    edges = torch.tensor(edges)
    edge_att = torch.cat(edge_att)
    edge_feat = [torch.tensor(ef) if isinstance(ef, np.ndarray) else ef for ef in edge_feat]  # 转换为 Tensor
    edge_feat = torch.cat(edge_feat)
    plm_features = torch.cat(plm_features)
    rsa_features = torch.cat(rsa_features)
    bfactor_features = torch.cat(bfactor_features)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix, xyz_feats, edges, edge_att, edge_feat, plm_features, rsa_features, bfactor_features


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl', fusion_mode='none'):
    # def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='/pubssd/PPIs/Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist
        self.fusion_mode = fusion_mode


    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)
        xyz_feats = pos

        # if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding':
        #     node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)
        node_features = get_node_features(sequence_name)
        node_features = torch.from_numpy(node_features)

        if self.fusion_mode != 'none':
            plm_feature_path = os.path.join(Feature_Path, "esm2", f"{sequence_name}.npy")
            plm_features = np.load(plm_feature_path).astype(np.float32)
            
            # Sanity check on shape/length
            seq_len = len(sequence)
            if plm_features.shape[0] != seq_len:
                if plm_features.shape[0] > seq_len:
                    plm_features = plm_features[:seq_len]
                else:
                    padding = np.zeros((seq_len - plm_features.shape[0], plm_features.shape[1]), dtype=np.float32)
                    plm_features = np.vstack((plm_features, padding))
            plm_features = torch.from_numpy(plm_features)
        else:
            plm_features = torch.zeros((len(sequence), 1280), dtype=torch.float32)

        # Load continuous RSA features
        rsa_path = os.path.join(Feature_Path, "rsa", f"{sequence_name}.npy")
        if os.path.exists(rsa_path):
            rsa_features = np.load(rsa_path).astype(np.float32)
            seq_len = len(sequence)
            if len(rsa_features) != seq_len:
                if len(rsa_features) > seq_len:
                    rsa_features = rsa_features[:seq_len]
                else:
                    rsa_features = np.concatenate([rsa_features, np.full((seq_len - len(rsa_features),), 0.5, dtype=np.float32)])
        else:
            rsa_features = np.full((len(sequence),), 0.5, dtype=np.float32)
        rsa_features = torch.from_numpy(rsa_features)

        # Load per-residue normalised B-factor features
        bfactor_path = os.path.join(Feature_Path, "bfactor", f"{sequence_name}.npy")
        if os.path.exists(bfactor_path):
            bfactor_features = np.load(bfactor_path).astype(np.float32)
            seq_len = len(sequence)
            if len(bfactor_features) != seq_len:
                if len(bfactor_features) > seq_len:
                    bfactor_features = bfactor_features[:seq_len]
                else:
                    bfactor_features = np.concatenate([bfactor_features, np.full((seq_len - len(bfactor_features),), 0.5, dtype=np.float32)])
        else:
            bfactor_features = np.full((len(sequence),), 0.5, dtype=np.float32)
        bfactor_features = torch.from_numpy(bfactor_features)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edges = [radius_index_list[0], radius_index_list[1]]
        edge_feat, edge_att = self.cal_edge_attr(radius_index_list, pos)

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)

        self.add_edges_custom(G, radius_index_list, edge_feat)

        adj_matrix = load_graph(sequence_name)

        return sequence_name, sequence, label, node_features, G, adj_matrix, xyz_feats, edges, edge_att, edge_feat, plm_features, rsa_features, bfactor_features

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        edge_att = torch.FloatTensor(distance)
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list, edge_att

    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)