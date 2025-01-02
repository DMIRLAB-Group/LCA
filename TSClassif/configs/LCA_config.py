import argparse


def get_model_a_parser():
    parser = argparse.ArgumentParser()


    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--start_psuedo_step', default=30, type=int,
                        metavar='W', help='step to start to use pesudo label')
    parser.add_argument('--tar_psuedo_thre', default=0.99, type=float,
                        metavar='W', help='threshold to select pesudo label')

    parser.add_argument("--type", type=str, default="type1")

    parser.add_argument("--z_dim", type=int, default=9)
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--z_kl_weight", type=float, default=0.001)
    parser.add_argument("--rec_weight", type=float, default=0.1)
    parser.add_argument("--sparsity_weight", type=float, default=0.001)
    parser.add_argument("--structure_weight", type=float, default=0.1)
    parser.add_argument("--class_weight", type=float, default=1)
    parser.add_argument("--is_ln", action='store_true')
    parser.add_argument("--layer_nums", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)


    parser.add_argument("--No_prior", action='store_true', help='whether to use norm')

    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='dropout ratio for frame-level feature (default: 0.5)')
    parser.add_argument('--activation', type=str, default='leakyReLU', help='activation')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--emb_dim', type=int, default=32, help='dimension of model')

    parser.add_argument('--patch_size', type=int, default=32, help='size of patches')


    return parser
