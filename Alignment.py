import torch
import torch.nn.functional as F
import numpy as np

from src.postprocessing.alignment.metrics import AlignmentMetrics
from src.utils.config_parser import ConfigParser

# TODO check if pymp in metrics is working
torch.backends.cudnn.deterministic = True


def main():
    interval = torch.load("src\data\dataset\ogbl_biokg\load_data\interval.pt")
    print(interval)

    N_SAMPLES = 5000

    RUN1               = {}
    RUN1['run']        = '3_3_2025/run_3'
    RUN1['checkpoint'] = 'checkpoint_3000'

    RUN2               = {}
    RUN2['run']        = '4_8_2025/run_9'
    RUN2['checkpoint'] = 'checkpoint_6000'
    
    parser = ConfigParser()
    z1, _ = parser.load_run(run=RUN1['run'], checkpoint=RUN1['checkpoint'], n_samples=None)
    z2, _ = parser.load_run(run=RUN2['run'], checkpoint=RUN2['checkpoint'], n_samples=None)
    mask = torch.randperm(z1.shape[0])[:N_SAMPLES]
    calc_metrics(z1[mask, :],z2[mask:16000, :])
    # calc_metrics(z1[mask, :],z2[mask, :])
    

def calc_metrics(z1,z2):
    # subset_indices = torch.randperm(z1.size(dim = 0))[:num_samples]
    # sample_1 = z1[subset_indices]
    # sample_2 = z2[subset_indices]
    
    sample_1 = F.normalize(z1, dim=-1)
    sample_2 = F.normalize(z2, dim=-1)

    import time
    trials = 10

    t0 = time.time()
    for metric in AlignmentMetrics.SUPPORTED_METRICS:

        scores, times = [], []
        for t in range(trials):
            t_st = time.time()

            kwargs = {}
            if 'nn' in metric:
                kwargs['topk'] = 10
            if 'cca' in metric:
                kwargs['cca_dim'] = 10
            if 'kernel' in metric:
                kwargs['dist'] = 'sample'

            score = AlignmentMetrics.measure(metric, sample_1, sample_2, **kwargs)
            scores.append(score)
            times.append(time.time() - t_st)
        print(f"{metric.rjust(20)}: {np.mean(scores):1.3f} [elapsed: {np.mean(times):.2f}s]")

    print(f'Total time: {time.time() - t0:.2f}s')
        
if __name__ == '__main__':
    main()