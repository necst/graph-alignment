from src.downstream.run_biokg_downstream import run_downstream_all


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Esegue test downstream per tutti i modelli')
    parser.add_argument('--run_paths', type=str, nargs='+',
                        help='Lista delle directory delle run da testare')
    parser.add_argument('--output', type=str, default='downstream_results.xlsx',
                        help='Nome del file Excel di output')
    parser.add_argument('--complete_data', action='store_true', default=True,
                        help='Usa il dataset completo')
    parser.add_argument('--filter_model', type=str, default=None,
                        help='Filtra solo run con un certo tipo di modello')
    parser.add_argument('--filter_layer', type=str, default=None,
                        help='Filtra solo run con un certo tipo di layer')


    args = parser.parse_args()

    run_downstream_all(
        base_dir=args.run_paths,
        output_file=args.output,
        complete_data=args.complete_data,
    )
        
if __name__ == '__main__':
    main()


