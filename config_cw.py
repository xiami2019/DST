import argparse
from email.policy import default
from tkinter.messagebox import NO

from utils.io_utils import load_json, save_json, get_or_create_logger

CONFIGURATION_FILE_NAME = "run_config.json"

logger = get_or_create_logger(__name__)

def add_config(parser):
    '''
    define arguments
    '''
    group = parser.add_argument_group("Construction")
    group.add_argument('-backbone', type=str, default='fnlp/cpt-base',
                        choices=['fnlp/cpt-base', 'fnlp/cpt-large', 
                                'fnlp/bart-base-chinese', 'fnlp/bart-large-chinese',
                                'mymusise/CPM-GPT2-FP16', 'mymusise/gpt2-medium-chinese'])
    group.add_argument('-context_size', type=int, default=-1)
    group.add_argument('-ururu', action='store_true')
    group.add_argument("-task", type=str, default="dst",
                       choices=["dst"])

    group = parser.add_argument_group("Training")
    group.add_argument("-max_grad_norm", type=float, default=1.0)
    group.add_argument("-batch_size", type=int, default=8)
    group.add_argument('-ckpt', type=str, default=None)
    group.add_argument("-epochs", type=int, default=10)
    group.add_argument('-train_from', type=str, default=None)
    group.add_argument("-warmup_steps", type=int, default=-1)
    group.add_argument("-warmup_ratio", type=float, default=0.2)
    group.add_argument("-grad_accum_steps", type=int, default=1)
    group.add_argument("-learning_rate", type=float, default=1e-4)
    group.add_argument("-num_train_dialogs", type=int, default=-1)
    group.add_argument("-no_learning_rate_decay", action="store_true")

    group = parser.add_argument_group("Prediction")
    group.add_argument("-pred_data_type", type=str, default="test",
                       choices=["val", "test"])
    group.add_argument("-beam_size", type=int, default=1)
    group.add_argument("-top_k", type=int, default=0)
    group.add_argument("-top_p", type=float, default=0.7)
    group.add_argument("-temperature", type=float, default=1.0)
    group.add_argument("-output", type=str, default=None)
    group.add_argument("-use_true_bs", action="store_true")

    group = parser.add_argument_group("Misc")
    group.add_argument("-run_type", type=str, required=True,
                       choices=["train", "predict"])
    group.add_argument("-seed", type=int, default=42)
    group.add_argument("-max_to_keep_ckpt", type=int, default=10)
    group.add_argument("-special_domain", type=str, default='all')
    group.add_argument("-log_frequency", type=int, default=100)
    group.add_argument("-model_dir", type=str, default="checkpoints")
    group.add_argument("-num_gpus", type=int, default=1)

def get_config():
    '''
    return argument parser instance
    '''
    parser = argparse.ArgumentParser()
    add_config(parser)
    return parser.parse_args()