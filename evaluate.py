import pandas as pd
import click as ck
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import math
from utils import FUNC_DICT, Ontology, NAMESPACES
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data_cafa3',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model', '-m', default='deepgocl',
    help='Prediction model')
@ck.option(
    '--combine', '-c', is_flag=True,
    help='Prediction model')
def main(data_root, ont, model, combine):
    train_data_file = f'{data_root}/{ont}/train_data.pkl'
    valid_data_file = f'{data_root}/{ont}/valid_data.pkl'
    test_data_file = f'{data_root}/{ont}/predictions_{model}.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['sup_annotations'].values.flatten()

    train_df = pd.read_pickle(train_data_file)
    valid_df = pd.read_pickle(valid_data_file)
    train_df = pd.concat([train_df, valid_df])
    test_df = pd.read_pickle(test_data_file)

    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    eval_preds = []
    for i, row in enumerate(test_df.itertuples()):
        preds = row.preds
        eval_preds.append(preds)

    print('Computing Fmax')
    fmax = 0.0
    prec_value = 0.0
    rec_value = 0.0
    tmax = 0.0

    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = test_df['annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))

    for t in range(0, 101):
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots = set()
            for j, go_id in enumerate(terms):
                if eval_preds[i][j] >= threshold:
                    annots.add(go_id)
            if t == 0:
                preds.append(annots)
                continue
            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)

        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_annotations(go_rels, labels, preds)

        print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}, WFmax:{wf}')
        if fmax < fscore:
            fmax = fscore
            prec_value = prec
            rec_value = rec
            tmax = threshold

    print(model, ont)
    print(f'Fmax: {fmax:0.3f}, Prec: {prec_value:0.3f}, Rec: {rec_value:0.3f}, threshold: {tmax}')


def compute_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(labels, preds):
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total = 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        tpic = 0.0
        for go_id in tp:
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            wp += tpic / (tpic + fpic)

    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf

if __name__ == '__main__':
    main()
