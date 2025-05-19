import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.remove_edges import remove_edges
from src.downstream.downstream_utils import find_all_runs, read_config_info
import os

from src.utils.config_parser import ConfigParser
import torch

from src.postprocessing.reductions.reductions import plot_reductions


def run_downstream_for_run(run_path, checkpoint, complete_data=True, plot_tsne=False):
    version = "duplicates"
    type_feature = "class"

    RUN = {}
    RUN['run'] = run_path
    RUN['checkpoint'] = checkpoint

    # Leggi le informazioni di configurazione
    config_info = read_config_info(run_path)

    parser = ConfigParser()
    edge_index_dict = torch.load(f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}_subsets.pt')
    interval = torch.load(f'src/data/dataset/ogbl_biokg/load_data/interval.pt')
    edge_attr_dict = torch.load(f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}_subsets_attr.pt')
    data = parser._load_dataset(parser, dataset="BioKG", processed=True)
    _, edge_index_dict_train, edge_attr_dict_train = remove_edges(data, interval, p=1)
    z, _ = parser.load_run(run=RUN['run'], checkpoint=RUN['checkpoint'], n_samples=None, shuffle_samples=False)

    negative_edges, negative_edges_attr = parser.generate_negative_samples(data, edge_index_dict, interval,
                                                                           edge_attr_dict)
    if complete_data:
        # Genera negativi anche per gli archi nel grafo di training
        extra_neg_edges, extra_neg_attrs = parser.generate_negative_samples(
            data, edge_index_dict_train, interval, edge_attr_dict_train
        )

        for key in extra_neg_edges:
            if key not in negative_edges:
                negative_edges[key] = []
                negative_edges_attr[key] = []

            # Set per controllo duplicati
            existing_neg_set = set(negative_edges[key])
            true_pos_set = set(map(tuple, edge_index_dict[key].T.tolist()))  # Positivi veri: quelli tolti dal grafo

            for idx, (u, v) in enumerate(extra_neg_edges[key]):
                if (u, v) not in existing_neg_set and (u, v) not in true_pos_set:
                    negative_edges[key].append((u, v))
                    negative_edges_attr[key].append(extra_neg_attrs[key][idx])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    # z = encoder(x, edge_index, edge_attr).cpu()

    results = {}
    splits = {}
    for key in edge_index_dict.keys():
        splits[key] = {}

        total_neg = torch.tensor(negative_edges[key], dtype=torch.long)
        total_neg = total_neg[torch.randperm(total_neg.shape[0])]

        if complete_data:
            train_edges = edge_index_dict_train[key]
            u_train, v_train = train_edges[0, :], train_edges[1, :]
            features_pos_train = torch.cat((z[u_train], z[v_train]), dim=1)
            num_pos_train = features_pos_train.shape[0]

            u_test, v_test = edge_index_dict[key][0, :], edge_index_dict[key][1, :]
            features_pos_test = torch.cat((z[u_test], z[v_test]), dim=1)
            num_pos_test = features_pos_test.shape[0]

            neg_train = total_neg[:num_pos_train]
            neg_test = total_neg[num_pos_train:]

            neg_u_train, neg_v_train = neg_train[:, 0], neg_train[:, 1]
            features_neg_train = torch.cat((z[neg_u_train], z[neg_v_train]), dim=1)

            neg_u_test, neg_v_test = neg_test[:, 0], neg_test[:, 1]
            features_neg_test = torch.cat((z[neg_u_test], z[neg_v_test]), dim=1)

            # ----------------- CONCATENA
            X_train = torch.cat([features_pos_train, features_neg_train], dim=0)
            y_train = torch.cat([torch.ones(len(features_pos_train)), torch.zeros(len(features_neg_train))])

            X_test = torch.cat([features_pos_test, features_neg_test], dim=0)
            y_test = torch.cat([torch.ones(len(features_pos_test)), torch.zeros(len(features_neg_test))])

            splits[key]['X_train'] = X_train.cpu().numpy()
            splits[key]['y_train'] = y_train.cpu().numpy()
            splits[key]['X_test'] = X_test.cpu().numpy()
            splits[key]['y_test'] = y_test.cpu().numpy()
        else:
            train_edges = edge_index_dict[key]
            u, v = train_edges[0, :], train_edges[1, :]
            features_pos = torch.cat((z[u], z[v]), dim=1)
            u, v = torch.tensor(negative_edges[key], dtype=torch.long)[:, 0], torch.tensor(negative_edges[key],
                                                                                           dtype=torch.long)[:, 1]
            features_neg = torch.cat((z[u], z[v]), dim=1)
            X = torch.cat([features_pos, features_neg], dim=0)
            y = torch.cat([torch.ones(len(features_pos)), torch.zeros(len(features_neg))])
            X = X.cpu().numpy()
            y = y.cpu().numpy()
            splits[key]['X_train'], splits[key]['X_test'], splits[key]['y_train'], splits[key][
                'y_test'] = train_test_split(X, y, test_size=0.2, random_state=42)

    num_splits = len(splits)
    if plot_tsne:
        class_names = {0: "Negative samples", 1: "Positive samples"}  # Mappa dei nomi delle classi
        fig, axes = plt.subplots(1, num_splits, figsize=(num_splits * 8, 8))
    for i, (key, value) in enumerate(splits.items()):
        # params = {
        #     'objective': 'binary:logistic',  # binary:logistic       multi:softmax
        #     'max_depth': 6,                  # Maximum depth of a tree
        #     'eta': 0.1,                      # Learning rate
        #     'subsample': 0.6,                # Subsample ratio of the training instance
        #     'eval_metric': 'logloss',        # Logarithmic loss for evaluation
        # }
        # XGBmodel = XGBClassifier(**params)
        XGBmodel = XGBClassifier()
        XGBmodel.fit(value['X_train'], value['y_train'])

        y_pred = XGBmodel.predict(value['X_test'])
        y_pred_proba = XGBmodel.predict_proba(value['X_test'])[:, 1]

        accuracy = accuracy_score(value['y_test'], y_pred)
        auc = roc_auc_score(value['y_test'], y_pred_proba)

        print(f"{key} prediction for {run_path}/{checkpoint}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")

        results[key] = {
            'accuracy': accuracy,
            'auc': auc
        }
        if plot_tsne:
            plot_reductions(value['X_test'], value['y_test'], axes=axes, iteration=i, downstream=True)
    return results


def run_downstream_all(base_dir="runs", output_file="downstream_results.xlsx", complete_data=True):
    """Esegue i test downstream per tutte le run e salva i risultati in Excel."""
    all_runs = find_all_runs(base_dir)

    if not all_runs:
        print(f"Nessuna run trovata nella directory {base_dir}")
        return

    # Prepara un dizionario per salvare tutti i risultati
    all_results = []

    # Esegui il test per ogni run
    for run_folder in all_runs.keys():
        #run_path = os.path.join("runs", run_folder)
        run_path = run_folder
        config_info = read_config_info(run_path)

        checkpoint = all_runs[run_folder]
        print(f"Elaborazione di {run_folder}/{checkpoint}...")
        results = run_downstream_for_run(run_path, checkpoint, complete_data)
        if results:
            # Salva i risultati nella lista con le informazioni sulla run e checkpoint
            for key, metrics in results.items():
                row = {
                    'task': key,
                    'accuracy': metrics['accuracy'],
                    'auc': metrics['auc'],
                    'model': config_info.get('model', 'N/A'),
                    'layer_type': config_info.get('layer_type', 'N/A'),
                    'dim_layers': config_info.get('dim_layer', 'N/A'),
                    'n_layers': config_info.get('num_layers', 'N/A'),
                }
                all_results.append(row)

    # Crea un DataFrame pandas dai risultati
    df_new = pd.DataFrame(all_results)

    if os.path.isfile(output_file):
        df_existing = pd.read_excel(output_file)
        # Unisci esistenti e nuovi
        df = pd.concat([df_existing, df_new], ignore_index=True)
        # Opzionale: rimuovi duplicati (basato su run + checkpoint + edge_type)
    else:
        df = df_new

    # Ordina i risultati per run, checkpoint e edge_type
    df = df.sort_values(by=['task','model', 'layer_type', 'dim_layers'])

    # Salva i risultati in un file Excel
    df.to_excel(output_file, index=False)
    print(f"Risultati salvati in {output_file}")

    # Crea anche un pivot con i risultati per una visualizzazione migliore
    # Prepara una tabella pivot per l'accuratezza
    pivot_acc = pd.pivot_table(df, values='accuracy',
                               index=['model', 'layer_type', 'dim_layers'],
                               columns=['task'],
                               aggfunc='first')

    # Prepara una tabella pivot per l'AUC
    pivot_auc = pd.pivot_table(df, values='auc',
                               index=['model', 'layer_type', 'dim_layers'],
                               columns=['task'],
                               aggfunc='first')

    # Crea anche un foglio con i risultati aggregati per modello/layer_type/dim_latent
    model_stats = df.groupby(['model', 'layer_type', 'dim_layers', 'task']).agg({
        'accuracy': ['mean', 'std', 'max'],
        'auc': ['mean', 'std', 'max']
    }).reset_index()

    # Salva i pivot in fogli separati dello stesso file Excel
    with pd.ExcelWriter(output_file.replace('.xlsx', '_pivot.xlsx')) as writer:
        pivot_acc.to_excel(writer, sheet_name='Accuracy')
        pivot_auc.to_excel(writer, sheet_name='AUC')
        model_stats.to_excel(writer, sheet_name='Model_Stats')

    print(f"Tabelle pivot salvate in {output_file.replace('.xlsx', '_pivot.xlsx')}")