# On the Alignment of Self-Supervised GML Representations

## 📁 Preparazione del progetto

Prima di eseguire il codice, assicurati di seguire questi passaggi:

1. **Struttura cartelle**
   - All’interno di `src`, crea la seguente struttura:
     ```
     src/data/dataset/ogbl_biokg
     ```
   - Scarica dal Drive il file `load_data.zip` e decomprimilo nella cartella `ogbl_biokg`.

2. **Cartella dei risultati**
   - All’interno di `stunning-enigma`, crea una cartella `runs`.
   - Inserisci in `runs` le seguenti cartelle scaricate dal Drive:
     - `vgae-infomax`
     - `dino`
     - `supervised`
     - 
3. **Creazione ambiente conda**
   - Crea l'ambiente conda dal file `environment.yaml`

## ⚙️ Training di un modello

Configura il file `config_yaml` con i seguenti parametri:

- `model`: scegli tra `DINO`, `DeepGraphInfomax`, `Supervised`
- `layer`: scegli tra `GIN`, `Transformer`
- `n_layers`: imposta il numero di layer (modifica il loader di conseguenza)
- `norm_layer`: scegli tra `BatchNorm1d`, `LayerNorm`
- `act_layer`: scegli tra `GELU`, `ReLU`, `ELU`, `SiLU`
- `loader.batch_size`: imposta la dimensione del batch
- `loader.n_neighbors`: definisci il numero di vicini caricati a ogni layer
- `drug_disease`, `drug_side_effect`, `disease_protein`, `function_function`: in caso di modello Supervised, impostare a True solo il task target del training.
- altri hyperparametri di training

Dopo aver configurato il file YAML, esegui il training con:

```bash
python training.py
```

## 📈 Misurare l'allineamento

Per valutare l’allineamento delle rappresentazioni tra modelli:

1. Modifica il file `config_alignment.yaml`:

   - Nella sezione `models`, specifica i modelli per cui vuoi misurare:
     - **Allineamento intra-bucket** (tra modelli supervised e tra modelli unsupervised)
     - **Allineamento inter-bucket** (all vs all)
   
   - Nella sezione `align_to_one`:
     - `left_models`: i modelli di riferimento (quelli a cui allineare)
     - `right_models`: i modelli da confrontare (quelli che verranno allineati)

2. Esegui il seguente comando:

```bash
python alignment_plots.py config_alignment.yaml
```





