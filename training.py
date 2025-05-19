import yaml

from src.utils.config_parser import ConfigParser
from src.training.trainer import Trainer


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    parser = ConfigParser()
    model, optimizer, criterion, loader, device, config = parser.get_training_setup(config)
 
    trainer = Trainer(model, loader, criterion, optimizer, device, config)
    trainer.train()
    if config['model'] == 'Supervised':
        loss, auc, acc = trainer.test_supervised(model, parser.train_data, parser.test_data, device)


if __name__ == '__main__':
    main()
    