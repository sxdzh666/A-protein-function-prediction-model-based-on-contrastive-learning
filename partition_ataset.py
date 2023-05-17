import click as ck
import numpy as np
import pandas as pd
from collections import Counter
from utils import Ontology, NAMESPACES
import logging
logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--go_file', '-gf', default='data_cafa3/go.obo',
    help='Gene ontology file in OBO format')
@ck.option(
    '--train_data_file', '-df', default='data_cafa3/train_data.pkl',
    help='A result file containing a list of proteins, sequences, and annotations')
@ck.option(
    '--test_data_file', '-df', default='data_cafa3/test_data.pkl',
    help='A result file containing a list of proteins, sequences, and annotations')


def main(go_file, train_data_file, test_data_file):
    go = Ontology(go_file, with_rels=True)
    df_train = pd.read_pickle(train_data_file)
    df_test = pd.read_pickle(test_data_file)

    logging.info(f'The number of protein sequences: {len(df_train)+len(df_test)}')
    logging.info(f'The number of train protein sequences: {len(df_train)}')
    logging.info(f'The number of test protein sequences: {len(df_test)}')
    df = pd.concat([df_train, df_test], axis=0)

    for ont in ['mf', 'bp', 'cc']:
        cnt = Counter()
        index = []
        for i, row in enumerate(df.itertuples()):
            ok = False
            for term in row.annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt[term] += 1
                    ok = True
            if ok:
                index.append(i)

        tdf = df.iloc[index]
        tdf = tdf.reset_index()
        terms = list(cnt.keys())

        print(f'Number of {ont} terms: {len(terms)}')
        print(f'Number of {ont} proteins: {len(tdf)}')

        terms_df = pd.DataFrame({'sup_annotations': terms})
        terms_df.to_pickle(f'data_cafa3/{ont}/terms.pkl')
        cnt_train = Counter()
        index_train = []
        for i, row in enumerate(df_train.itertuples()):
            ok = False
            for term in row.annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt_train[term] += 1
                    ok = True
            if ok:
                index_train.append(i)

        tdf_train = df.iloc[index_train]
        tdf_train = tdf_train.reset_index()


        n = len(tdf_train)
        index = np.arange(n)
        train_n = int(n * 0.9)
        np.random.seed(seed=0)
        np.random.shuffle(index)
        train_df_one = df_train.iloc[index[:train_n]]
        valid_df_one = df_train.iloc[index[train_n:]]


        cnt_test = Counter()
        index_test = []
        for i, row in enumerate(df_test.itertuples()):
            ok = False
            for term in row.annotations:
                if go.get_namespace(term) == NAMESPACES[ont]:
                    cnt_test[term] += 1
                    ok = True
            if ok:
                index_test.append(i)

        tdf_test = df.iloc[index_test]
        tdf_test = tdf_test.reset_index()

        train_df_one.to_pickle(f'data_cafa3/{ont}/train_data.pkl')
        valid_df_one.to_pickle(f'data_cafa3/{ont}/valid_data.pkl')
        tdf_test.to_pickle(f'data_cafa3/{ont}/test_data.pkl')

        print(f'Train/Valid/Test proteins for {ont}: {train_n} / {(n - train_n)} / {len(tdf_test)}')


if __name__ == '__main__':
    main()
