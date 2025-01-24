import argparse


def get_model_a_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int,
                        help='manual seed')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='-batch size')
    parser.add_argument('--src', default='hmdb51',
                        help='source domain')
    parser.add_argument('--tar', default='ucf101',
                        help='target domain')
    parser.add_argument('--gpu', default='0', type=str,
                        help='index of GPU to use')
    parser.add_argument('--data_threads', type=int, default=10,
                        help='number of data loading threads')
    parser.add_argument('--num_segments', type=int, default=9,
                        help='the number of frame segment')
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--start_psuedo_step', default=0, type=int,
                        metavar='W', help='step to start to use pesudo label')
    parser.add_argument('--tar_psuedo_thre', default=0.96, type=float,
                        metavar='W', help='threshold to select pesudo label')

    parser.add_argument("--type", type=str, default="None")

    parser.add_argument("--z_dim", type=int, default=20)
    parser.add_argument("--feature_dim", type=int, default=2048)
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--z_kl_weight", type=float, default=0.01)
    parser.add_argument("--rec_weight", type=float, default=0.01)
    parser.add_argument("--sparsity_weight", type=float, default=1)
    parser.add_argument("--structure_weight", type=float, default=100)
    parser.add_argument("--class_weight", type=float, default=1)
    parser.add_argument("--is_ln", action='store_true')
    parser.add_argument("--layer_nums", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--optimizer', type=str,
                        default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N', help='number of total epochs to run')

    parser.add_argument("--No_prior", action='store_true', help='whether to use norm')

    parser.add_argument('--dropout_rate', default=0, type=float,
                        help='dropout ratio for frame-level feature (default: 0.5)')
    parser.add_argument('--activation', type=str, default='leakyReLU', help='activation')
    parser.add_argument("--No_encoder", action='store_true', help='whether to use norm')

    # transformer
    parser.add_argument('--factor', type=int, default=1, help='attn factor')

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    return parser
