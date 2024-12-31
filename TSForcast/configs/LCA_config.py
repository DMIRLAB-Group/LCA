import argparse


def get_model_a_parser():
    parser = argparse.ArgumentParser(description="Configuration for Model TSCA", allow_abbrev=False)
    parser.add_argument("--z_dim", type=int, default=20)
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--z_kl_weight", type=float, default=0.1)
    parser.add_argument("--rec_weight", type=float, default=1)
    parser.add_argument("--sparsity_weight", type=float, default=0.1)
    parser.add_argument("--structure_weight", type=float, default=0.1)
    parser.add_argument("--is_ln", action='store_false')
    parser.add_argument("--layer_nums", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--is_norm", action='store_true')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--is_no_prior", action='store_true')
    parser.add_argument("--type", type=str, default="None")

    parser.add_argument('--emb_dim', type=int, default=6, help='dimension of model')

    parser.add_argument('--patch_size', type=int, default=6, help='size of patches')

    # Add other Model A specific arguments here
    return parser

