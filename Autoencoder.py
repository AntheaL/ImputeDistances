import os
import numpy as np
import argparse
import yaml
from models import Autoencoder


parser = argparse.ArgumentParser(description="Running matrix focatorization.")
parser.add_argument("--config", help="configuration file", required=True, type=str)
parser.add_argument(
    "--tag", help="tag for defining logs directory", required=False, type=str
)
parser.add_argument("--src-path", help="input file", required=True, type=str)
parser.add_argument("--ref-path", help="input file", required=True, type=str)
parser.add_argument(
    "--val-mask-path", help="validation mask path", required=False, type=str
)
parser.add_argument("--add-val-mask", required=False, action="store_true")
parser.add_argument("--dst-path", help="output file", required=False, type=str)
parser.add_argument("--n-epochs", help="number of steps", required=False, type=int)
parser.add_argument(
    "--n-reads", help="number of reads per position", required=False, type=str
)


args = parser.parse_args()

with open(args.config, "r") as ymlfile:
    config = yaml.load(ymlfile)

if args.n_epochs:
    config["train"]["n_epochs"] = args.n_epochs

if args.src_path:
    src_path = args.src_path
else:
    assert "src_path" in config, "input file note provided."
    src_path = config["src_path"]
ext = os.path.splitext(src_path)[1]
Rref = None
if ext == ".npy":
    Rin = np.load(src_path)
    if args.ref_path:
        Rref = np.load(args.ref_path)
else:
    Rin = np.loadtxt(src_path)
    if args.ref_path:
        Rref = np.loadtxt(args.ref_path)

assert (
    Rin.shape == Rref.shape
), f"input and reference arrays should have same shape, found {Rin.shape} and {Rref.shape}"

val_mask = None
if args.val_mask_path:
    val_mask = np.load(args.val_mask_path)
elif args.add_val_mask:
    val_mask = np.multiply(Rin, Rref) < 0

print(f"Validation data has {np.sum(np.isnan(Rref))} nan values.")

config["src_path"] = src_path

n_reads = None
if args.n_reads:
    n_reads = np.load(args.n_reads)
elif "n_reads" in config:
    n_reads = np.load(config["n_reads"])

tag = args.tag
if tag is None:
    name, ext = os.path.splitext(os.path.basename(args.src_path))
    exp_rate_str = f"_exp{config['model']['exp_rate']}" if args.n_reads else ""
    tag_fmt = "_b{}_d{}_e{}"
    tag = name + tag_fmt.format(
        config["train"]["batch_size"],
        config["model"]["dropout_probability"],
        exp_rate_str,
    )

logs_dir = os.path.join(config["logs_dir"], tag)
config["train"]["logs_dir"] = logs_dir
os.makedirs(logs_dir, exist_ok=True)
with open(os.path.join(logs_dir, "config.yml"), "a") as f:
    yaml.dump(config, f)

# model: dropout, thresh, exp_rate...
trainer = Autoencoder(
    Rin, n_reads=n_reads, val_data=Rref, val_mask=val_mask, **config["model"]
)
# train: n_epochs, batch_size, log_every, save_every...
trainer.run(**config["train"])
