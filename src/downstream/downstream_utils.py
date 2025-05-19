import os
import yaml


def find_all_runs(base_dir):
    """Trova tutte le run e i checkpoint nella directory specificata."""
    all_runs = {}
    checkpoint = ""
    # Cerca tutte le cartelle di run
    for curr_base_directory in base_dir:
        for run_folder in os.listdir(curr_base_directory):
            run_path = os.path.join(curr_base_directory, run_folder)
            if os.path.isdir(run_path):
                # Cerca file di checkpoint in questa run
                for file in os.listdir(run_path):
                    if file.startswith("checkpoint_3000") and file.endswith(".pt"):
                        checkpoint_name = file.split(".")[0]  # rimuovi estensione .pt
                        # Se è specificato un checkpoint specifico, considera solo quello
                        checkpoint = checkpoint_name

                if checkpoint:
                    key = f"{os.path.basename(curr_base_directory)}/{run_folder}"
                    all_runs[key] = checkpoint

    return all_runs


def read_config_info(run_path):
    """Legge il file config.yaml della run e restituisce le informazioni importanti."""
    config_path = os.path.join(run_path, "config.yaml")
    config_info = {}

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Estrai le informazioni rilevanti
            if config:
                # Modello utilizzato
                config_info['model'] = config['model']

                # Tipo di layer
                config_info['layer_type'] = config['encoder']['layer']

                # Dimensioni latenti
                config_info['dim_layer'] = config['encoder']['dim']

                config_info['num_layers'] = config['encoder']['n_layers']

                # Aggiungi altre informazioni rilevanti qui se necessario

    except Exception as e:
        print(f"Errore nella lettura del file config per {run_path}: {e}")

    return config_info
