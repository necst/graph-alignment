import pandas as pd
import torch
import torch.nn as nn

import logging
from collections import deque
import yaml
import os
from pathlib import Path
from datetime import datetime, date
import time
import pickle


class Logger():
    def __init__(self, config:dict, level=logging.INFO, log_name:str = 'training.log', log_every:int = 10, average_window:int = 100) -> None:
        self.config = config
        self.log_dir = self._get_output_path() 
        self.log_every = log_every
        self.average_window = average_window
        self.loss_history = deque(maxlen=average_window)
        self.log = pd.DataFrame(columns=['step', 'loss', 'lr', 'wd'])
        
        # configure logger
        self.logger = logging.getLogger("TrainingLogger")
        self.logger.setLevel(level)
        self.logger.propagate = False  # Avoid duplicate logs

        # formatter 
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # file handler (logs to a file)
        file_handler = logging.FileHandler(self.log_dir.joinpath(log_name), mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format) 
        self.logger.addHandler(file_handler)

        # console handler 
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)
  
  
    def start(self, device, num_params):
        self.start_time = time.time()
        self.logger.debug(f'\nStart training on {device} a model w/ {num_params} params')
        self.device = device
        self.num_params = num_params


    def monitor(self, loss:float, step:int):
        # update loss queue (to average loss estimation)
        self.loss_history.append(loss)

        if step == 0 or step % self.log_every == 0:
            loss_smooth = sum(self.loss_history) / max(len(self.loss_history),self.average_window)
            self.logger.info(f'\nStep: {step}, Loss: {loss_smooth:.4f}')


    def save_checkpoint(self, model, step):
        checkpoint = model.state_dict()
        torch.save(checkpoint, self.log_dir.joinpath(f"checkpoint_{step}.pt"))

    @staticmethod 
    def _get_output_path() -> Path:
        root = Path(os.getcwd())
        today = date.today()
        day = today.day
        month = today.month
        year = today.year

        runs_dir = root.joinpath(
            'runs',
        )
    
        training_day = runs_dir.joinpath(
            f'{month}_{day}_{year}',
        )

        # create runs dir if not present
        # count iter of the day, if day dir not found it is the first run of the 
        # day so create the dir, otherwise count the dirs already present and 
        # create dir for the new iteration
        if not os.path.isdir(runs_dir):
            os.mkdir(runs_dir)
        if os.path.isdir(training_day):
            num_iter = len(os.listdir(training_day))+1
        else:
            os.mkdir(training_day)
            num_iter = 1
        # make dir for the current run
        trainig_run = training_day.joinpath(f'run_{num_iter}')
        os.mkdir(trainig_run)

        return trainig_run


    def end(self, config:dict):
            # stop timers
            self.end_time = time.time()
            self.total_time = self.end_time - self.start_time
            self.logger.info(f'\nTraining ended in {self.total_time:.2f} s')
            
            # save backup .yaml config 
            with open(self.log_dir.joinpath('config.yaml'), 'w') as f:
                yaml.dump(config, f)
