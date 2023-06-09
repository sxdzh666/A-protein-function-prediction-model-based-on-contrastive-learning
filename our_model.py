import click as ck
import pandas as pd
from utils import Ontology, to_onehot, MAXLEN
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc
import math
from torch_utils import FastTensorDataLoader
import csv
from torch.optim.lr_scheduler import MultiStepLR

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data_cafa3',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='bp',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=20,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=20,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
def main(data_root, ont, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/deepgocl.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions_deepgocl.pkl'
    go = Ontology(go_file, with_rels=True)
    terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    net = ourModel(n_terms).to(device)
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data

    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[1, 3, ], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        log_file = open(f'{data_root}/train_logs.tsv', 'w')
        logger = csv.writer(log_file, delimiter='\t')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits, conloss = net(batch_features, batch_labels)
                    loss = 0.7 * F.binary_cross_entropy(logits, batch_labels) + 0.3 * conloss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
            train_loss /= train_steps

            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits, conloss = net(batch_features, batch_labels)
                        batch_loss = 0.7 * F.binary_cross_entropy(logits, batch_labels) + 0.3 * conloss
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
                logger.writerow([epoch, train_loss, valid_loss, roc_auc])
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)
            scheduler.step()
        log_file.close()

    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits, conloss = net(batch_features, batch_labels)
                batch_loss = 0.7 * F.binary_cross_entropy(logits, batch_labels) + 0.3 * conloss
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps

        preds = preds.reshape(-1, n_terms)
        preds = propagate_preds(np.array(preds), terms_dict, go)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Ontology - {ont}, Test Loss - {test_loss}, AUC - {roc_auc}')

    test_df['preds'] = list(preds)
    test_df.to_pickle(out_file)


def propagate_preds(preds, terms_dict, go):

    go_terms = list(terms_dict.keys())
    go_ancestors = {}

    for go_term in go_terms:
        ancestors = go.get_anchestors(go_term)
        ancestors = [i for i in ancestors if i in go_terms]
        go_ancestors[go_term] = ancestors

    for i in range(len(preds)):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            ancestors = go_ancestors[go_id]
            ancestor_scores = [preds[i][terms_dict[a]] for a in ancestors]
            avg_ancestor_score = np.mean(ancestor_scores)
            score = preds[i][j]
            if 2 * avg_ancestor_score >= score >= avg_ancestor_score:
                for sup_go in go.get_anchestors(go_id):
                    if sup_go in prop_annots:
                        prop_annots[sup_go] = max(prop_annots[sup_go], score)
                    else:
                        prop_annots[sup_go] = score

        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                preds[i][terms_dict[go_id]] = score

    return preds




def compute_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


class ourModel(nn.Module):

    def __init__(self, n_terms):
        super().__init__()
        kernels = [i + 1 for i in range(1, 7)]
        convs = []
        for kernel in kernels:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=21, out_channels=512, kernel_size=kernel),
                    nn.MaxPool1d(6),
                    nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=kernel),
                    nn.MaxPool1d((1000 - kernel + 1) // 6 - kernel + 1)
                ))
        self.convs = nn.ModuleList(convs)
        self.fc = nn.Linear(in_features=1024 * 6, out_features=n_terms)

    def forward(self, x, y):
        output = []
        for conv in self.convs:
            output.append(conv(x))
        x = th.cat(output, dim=1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        out = th.sigmoid(x)

        sim_label = th.matmul(y, y.transpose(0, 1))
        mask_label = th.logical_not(th.eye(sim_label.shape[0], dtype=th.bool)).to('cuda:0')
        sim_label = th.masked_select(sim_label, mask_label).reshape(sim_label.shape[0], sim_label.shape[1] - 1)
        sim_label = th.exp(th.nn.functional.normalize(sim_label, p=1, dim=1))

        norm_sample = th.nn.functional.normalize(out, p=2, dim=1)
        sim_sample = th.matmul(norm_sample, norm_sample.transpose(0, 1))
        sim_sample = th.exp(sim_sample / 1.0)

        mask_sample = th.logical_not(th.eye(sim_sample.shape[0], dtype=th.bool)).to('cuda:0')
        sim_sample = th.masked_select(sim_sample, mask_sample).reshape(sim_sample.shape[0], sim_sample.shape[1] - 1)
        matrix_log = th.log(th.nn.functional.normalize(sim_sample, p=1, dim=1))
        matrix_dis = th.mul(matrix_log, -sim_label)
        loss = th.mean(matrix_dis)

        return out, loss


def load_data(data_root, ont, terms_file):
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['sup_annotations'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')
    train_data = get_data(train_df, terms_dict)
    valid_data = get_data(valid_df, terms_dict)
    test_data = get_data(test_df, terms_dict)

    return terms_dict, train_data, valid_data, test_data, test_df


def get_data(df, terms_dict):
    data = th.zeros((len(df), 21, MAXLEN), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        seq = row.sequences
        seq = th.FloatTensor(to_onehot(seq))
        data[i, :, :] = seq
        for go_id in row.annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
