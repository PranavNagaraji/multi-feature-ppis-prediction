import os
import argparse
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data_generator import *
from EGNN_model import *
from final_model import *
from GraphTransformer_Block import *

parser = argparse.ArgumentParser()
parser.add_argument('--fusion_mode', type=str, default='none', choices=['none', 'concat', 'gated'])
parser.add_argument('--d_proj', type=int, default=128)
parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the model checkpoints")
parser.add_argument('--smoke_test', action='store_true')
args = parser.parse_args()

FUSION_MODE = args.fusion_mode
D_PROJ = args.d_proj
Model_Path = args.model_dir
if not Model_Path.endswith('/'):
    Model_Path += '/'

Dataset_Path = "./Dataset/"


def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}
    gate_records = []

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, G_batch, adj_matrix, xyz_feats, edges, edge_att, edge_feat, plm_features, rsa_features, bfactor_features = data

            if torch.cuda.is_available():
                node_features_dev = Variable(node_features.cuda().float())
                plm_features_dev = Variable(plm_features.cuda().float())
                adj_matrix_dev = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                xyz_feats_dev = Variable(xyz_feats.cuda().float())
                edges_dev = Variable(edges.cuda())
                edge_att_dev = Variable(edge_att.cuda().float())
                edge_feat_dev = Variable(edge_feat.cuda().float())
                y_true_dev = Variable(labels.cuda())

            else:
                node_features_dev = Variable(node_features.float())
                plm_features_dev = Variable(plm_features.float())
                adj_matrix_dev = Variable(adj_matrix)
                xyz_feats_dev = Variable(xyz_feats.float())
                edges_dev = Variable(edges)
                edge_att_dev = Variable(edge_att.float())
                edge_feat_dev = Variable(edge_feat.float())
                y_true_dev = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix_dev = torch.squeeze(adj_matrix_dev)
            y_true_dev = torch.squeeze(y_true_dev)
            y_true_dev = y_true_dev.long()

            y_pred = model(node_features_dev, xyz_feats_dev, edges_dev, edge_att_dev, edge_feat_dev, adj_matrix_dev, plm_features=plm_features_dev)

            # Collect gate values if in gated mode
            if getattr(model, 'fusion_mode', 'none') == 'gated' and model.last_gate_val is not None:
                gates = model.last_gate_val.cpu().numpy().flatten()
                lbls = labels.numpy().flatten()
                # Load actual RSA values from Feature/rsa/ (mirrors data_generator.py lines 155-166)
                rsa_path = os.path.join(Feature_Path, "rsa", f"{sequence_names[0]}.npy")
                seq_len = len(lbls)
                if os.path.exists(rsa_path):
                    rsas = np.load(rsa_path).astype(np.float32)
                    if len(rsas) != seq_len:
                        if len(rsas) > seq_len:
                            rsas = rsas[:seq_len]
                        else:
                            rsas = np.concatenate([rsas, np.full((seq_len - len(rsas),), 0.5, dtype=np.float32)])
                else:
                    rsas = np.full(seq_len, 0.5, dtype=np.float32)
                for g, l, r in zip(gates, lbls, rsas):
                    gate_records.append((g, l, r))

            loss = model.criterion(y_pred, y_true_dev)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true_dev = y_true_dev.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true_dev)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            epoch_loss += loss.item()
            n += 1
    epoch_loss_avg = epoch_loss / n

    return epoch_loss_avg, valid_true, valid_pred, pred_dict, gate_records


def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0.5  # fallback if no threshold improves over this
        # Start from 1 (threshold=0.01) to exclude the degenerate all-positive case
        # (threshold=0.0 gives Recall=1.0, MCC=0 for any model and is not meaningful).
        for threshold in range(1, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def get_validation_dataframes():
    with open(Dataset_Path + "Train_335.pkl", "rb") as f:
        Train_335 = pickle.load(f)
        Train_335.pop('2j3rA', None)

    IDs, sequences, labels = [], [], []
    for ID in Train_335:
        IDs.append(ID)
        item = Train_335[ID]
        sequences.append(item[0])
        labels.append(item[1])

    train_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    all_dataframe = pd.DataFrame(train_dic)

    np.random.seed(SEED)
    kfold = KFold(n_splits=5, shuffle=True)
    val_dataframes = {}
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_dataframe['ID'].values, all_dataframe['label'].values), 1):
        val_dataframes[fold] = all_dataframe.iloc[val_idx, :]
    return val_dataframes


def test(test_dataframe, psepos_path):
    if args.smoke_test:
        test_dataframe = test_dataframe.iloc[:2]
    all_metrics = {
        'binary_acc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'AUC': [],
        'AUPRC': [],
        'mcc': [],
        'threshold': []
    }
        
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe, psepos_path=psepos_path, fusion_mode=FUSION_MODE), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=graph_collate)

    val_dataframes = get_validation_dataframes()
    fold_locked_thresholds = {}

    for model_name in sorted(os.listdir(Model_Path)):
        if not model_name.endswith('.pkl'):
            continue
        print(model_name)
        model = FinalModel(INPUT_DIM, HIDDEN_DIM, FLITER_DIM, OUTPUT_SIZE, DROPOUT, LAYER, fusion_mode=FUSION_MODE, d_proj=D_PROJ)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0', weights_only=True))

        # Compute and lock threshold from validation set (no test label leakage)
        locked_threshold = None
        if model_name.startswith('Fold') and '_best_model.pkl' in model_name:
            try:
                fold_num = int(model_name.split('Fold')[1].split('_')[0])
                val_df = val_dataframes[fold_num]
                val_loader = DataLoader(dataset=ProDataset(dataframe=val_df, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl', fusion_mode=FUSION_MODE), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=graph_collate)
                _, val_true, val_pred, _, _ = evaluate(model, val_loader)
                val_result = analysis(val_true, val_pred, best_threshold=None)
                locked_threshold = val_result['threshold']
                fold_locked_thresholds[fold_num] = locked_threshold
                print(f"Validation threshold for {model_name} (Fold {fold_num}): {locked_threshold:.2f} (val F1: {val_result['f1']:.4f})")
            except Exception as e:
                print(f"Warning: could not compute validation threshold for {model_name}: {e}")
        elif len(fold_locked_thresholds) > 0:
            locked_threshold = float(np.mean(list(fold_locked_thresholds.values())))
            print(f"Using average fold validation threshold for {model_name}: {locked_threshold:.2f}")

        epoch_loss_test_avg, test_true, test_pred, pred_dict, gate_records = evaluate(model, test_loader)

        # Save gate records if in gated mode
        if len(gate_records) > 0:
            import csv
            csv_path = Model_Path + f"{model_name}_gate_records.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['gate_value', 'label', 'rsa'])
                writer.writerows(gate_records)
            print(f"Saved {len(gate_records)} gate records to {csv_path}")

        result_test = analysis(test_true, test_pred, best_threshold=locked_threshold)

        for key in all_metrics:
            all_metrics[key].append(result_test[key])

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUROC: ", result_test['AUC'])
        print("Test MCC: ", result_test['mcc'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Threshold: ", result_test['threshold'])
        print()

        if args.smoke_test:
            break

    return all_metrics

def test_one_dataset(dataset, psepos_path):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    all_metrics = test(test_dataframe, psepos_path)

    average_metrics = {key: np.mean(values[:5]) for key, values in all_metrics.items()}
    print("========== Cross-Validation Results ==========")
    print("Average binary acc: ", average_metrics['binary_acc'])
    print("Average precision: ", average_metrics['precision'])
    print("Average recall: ", average_metrics['recall'])
    print("Average f1: ", average_metrics['f1'])
    print("Average AUROC: ", average_metrics['AUC'])
    print("Average MCC: ", average_metrics['mcc'])
    print("Average AUPRC: ", average_metrics['AUPRC'])
    print("Average threshold: ", average_metrics['threshold'])
    print()


def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_315-28.pkl", "rb") as f:
        Test_315_28 = pickle.load(f)

    with open(Dataset_Path + "UBtest_31-6.pkl", "rb") as f:
        UBtest_31_6 = pickle.load(f)

    Test60_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    Test315_28_psepos_Path = './Feature/psepos/Test315-28_psepos_SC.pkl'
    UBtest31_28_psepos_Path = './Feature/psepos/UBtest31-6_psepos_SC.pkl'

    print("==============================================")
    print("Evaluate on Test_60")
    print("==============================================")
    test_one_dataset(Test_60, Test60_psepos_Path)

    print("==============================================")
    print("Evaluate on Test_315-28")
    print("==============================================")
    test_one_dataset(Test_315_28, Test315_28_psepos_Path)

    print("==============================================")
    print("Evaluate on UBtest_31-6")
    print("==============================================")
    test_one_dataset(UBtest_31_6, UBtest31_28_psepos_Path)


if __name__ == "__main__":
    main()
