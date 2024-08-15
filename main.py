import argparse
import datetime
import glob
import importlib
import os
import sys

import pytorch_lightning as pl
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from utils.helpers import instantiate_from_config
from callbacks import SetupCallback


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-c',
        '--config',
        nargs='*',
        metavar='base_config.yaml',
        default=list(),
    )
    parser.add_argument(
        '-t',
        '--train',
        type=str2bool,
        default=True,
        nargs='?',
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=0,
        help='seed for seed_everything'
    )
    parser.add_argument(
        '-f',
        '--fast_dev_run',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default=""
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit"
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None
    )
    parser.add_argument(
        "--scale_lr",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--no_test",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        # default="last.ckpt"
        default=None
    )
    parser.add_argument(
        "-e",
        "--evaluation",
        type=str,
        default="mse"
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return [k for k in vars(args) if hasattr(opt, k) and getattr(opt, k) != getattr(args, k)]


def load_configs(config_paths):
    configs = [OmegaConf.load(cfg) for cfg in config_paths]
    return OmegaConf.merge(*configs)


if __name__=='__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    # Argument validations
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    # ckpt = None
    if opt.resume or opt.test:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        else:
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", opt.ckpt)
        # opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.config = base_configs + opt.config
        nowname = logdir.split("/")[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
        if opt.ckpt is not None:
            ckpt = opt.ckpt
        else:
            ckpt = None
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # Random seed
    seed_everything(opt.seed)

    # Load and merge configs...
    config = load_configs(opt.config)
    lightning_config = config.pop("lightning", OmegaConf.create())
    
    # Configure trainer
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in nondefault_trainer_args(opt): 
        trainer_config[k] = getattr(opt, k)
    if opt.fast_dev_run:
        trainer_config["fast_dev_run"] = True
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    
    # Instantiate data
    data = instantiate_from_config(config.data)
    data.setup()
    
    # Instantiate model
    model = instantiate_from_config(config.model)

    # update learning rate
    batch_size = config.data.params.batch_size
    base_lr_rate = config.model.params.lr
    
    # TODO: scale_lr set relatively high value and this should need to be fixed
    if opt.scale_lr:
        model.learning_rate = batch_size * base_lr_rate
        print(f"[INFO] Setting learning rate to {model.learning_rate:.2e} = {batch_size} (batchsize) * {base_lr_rate:.2e} (base_lr)")
        
    # Currently available loggers
    default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    # "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "version": nowname,
                    "save_dir": logdir
                }
            }
        }
    
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

    # Callbacks
    if not(opt.fast_dev_run):
        trainer_opt.logger = instantiate_from_config(logger_cfg)
        
        trainer_opt.callbacks = [
            instantiate_from_config(lightning_config.callback[callback]) 
            for callback in lightning_config.callback.keys()
        ]
        
        if opt.evaluation == "bleu":
            trainer_opt.callbacks.append(ModelCheckpoint(
                dirpath=ckptdir, 
                filename="epoch={epoch:05}-step={step:07}-bleu4={val/bleu4:.2f}", 
                monitor=model.monitor, 
                auto_insert_metric_name=False, 
                save_top_k=1, 
                mode="max"
            ))
            trainer_opt.callbacks.append(EarlyStopping(
                monitor=model.monitor, verbose=True, patience=10, mode="max"
            ))
        else:
            trainer_opt.callbacks.append(ModelCheckpoint(dirpath=ckptdir, filename="epoch={epoch:05}-step={step:07}-loss={val/loss:.4f}", 
                                                    monitor=model.monitor, auto_insert_metric_name=False, 
                                                    save_top_k=1, mode="min"))
            trainer_opt.callbacks.append(EarlyStopping(monitor=model.monitor, verbose=True, patience=5, mode="min"))
        
        trainer_opt.callbacks.append(SetupCallback(resume=opt.resume, now=now, logdir=logdir, ckptdir=ckptdir, 
                                        cfgdir=cfgdir, config=config, lightning_config=lightning_config))
        
    # Run training or testing
    trainer = Trainer.from_argparse_args(trainer_opt)
    
    if opt.train and opt.resume is not None:
        trainer.fit(model, data, ckpt_path=ckpt)
    elif opt.train:
        if ckpt is not None:
            model.load_pretrained_weights(ckpt)
            trainer.fit(model, data)
            trainer.test(model, data)
        else:
            trainer.fit(model, data)
            trainer.test(model, data)
    elif opt.test:
        trainer.test(model, data, ckpt_path=ckpt)