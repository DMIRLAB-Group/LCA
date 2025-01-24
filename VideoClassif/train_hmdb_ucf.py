import copy

import torch
import torch.nn as nn
import argparse
import os
import json
import random

import numpy as np

import time

from torch.nn.utils import clip_grad_norm_

from exp import utils
from exp.i_Net import Net
from exp.dataset_n import TSNDataSet
from exp.LCA_config import get_model_a_parser

parser = argparse.ArgumentParser(allow_abbrev=False)
# ========================= Dataset Configs ==========================
parser.add_argument('--dataset', default='hmdb_ucf',
                    help='datasets')
parser.add_argument('--data_root', default='./dataset',
                    help='root directory for data')

parser.add_argument('--input_type', default='feature',
                    choices=['feature', 'image'], help='the type of input')

# ========================= Runtime Configs ==========================

parser.add_argument('--exp_dir', default='experiments',
                    help='base directory of experiments')

parser.add_argument('--result_dir', default='./test_result',
                    help='root directory for data')
parser.add_argument('--log_indicator', default=1, type=int,
                    help='base directory to save logs')
parser.add_argument('--model_dir', default='',
                    help='base directory to save models')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_hp', default=False, action="store_true",
                    help='whether to use the saved hyper-parameters')

parser.add_argument('--save_model', default=0, type=int,
                    help='whether to save models')
parser.add_argument('--parallel_train', default=False,
                    help='whether to use multi-gpus for training')
parser.add_argument('--eval_freq', default=1, type=int,
                    help='evaluation frequency (default: 5)')
parser.add_argument('--weighted_class_loss', type=str,
                    default='Y', choices=['Y', 'N'])
parser.add_argument('--weighted_class_loss_DA', type=str,
                    default='Y', choices=['Y', 'N'])

# ========================= Model Configs ==========================

parser.add_argument('--backbone', type=str, default="I3Dpretrain",
                    choices=['dcgan', 'resnet101', 'I3Dpretrain', 'I3Dfinetune'], help='backbone')
parser.add_argument('--val_segments', type=int, default=-1,
                    help='')
parser.add_argument('--channels', default=3, type=int,
                    help='input channels for image inputs')
parser.add_argument('--add_fc', default=1, type=int, metavar='M',
                    help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2)')
parser.add_argument('--fc_dim', type=int, default=1024,
                    help='dimension of added fc')
parser.add_argument('--frame_aggregation', type=str, default='trn',
                    choices=['rnn', 'trn'], help='aggregation of frame features (none if baseline_type is not video)')

parser.add_argument('--f_dim', type=int, default=256,
                    help='dim of f')
parser.add_argument('--triplet_type', type=str, default='mean',
                    choices=['mean', 'post'], help='type of data to calculate triplet loss')
parser.add_argument('--prior_sample', type=str, default='random',
                    choices=['random', 'post'], help='how to sample prior')

# ========================= Learning Configs ==========================

parser.add_argument('--lr_decay', default=10, type=float,
                    metavar='LRDecay', help='decay factor for learning rate')
parser.add_argument('--lr_adaptive', type=str, default='dann',
                    choices=['none', 'loss', 'dann'])
parser.add_argument('--lr_steps', default=[500, 1000], type=float,
                    nargs="+", metavar='LRSteps', help='epochs to decay learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip_gradient', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')

# ========================= DA Configs ==========================
parser.add_argument('--use_attn', type=str, default='TransAttn',
                    choices=['none', 'TransAttn', 'general'], help='attention-mechanism')
parser.add_argument('--add_loss_DA', type=str, default='none',
                    choices=['none', 'attentive_entropy'], help='add more loss functions for DA')
parser.add_argument('--pretrain_VAE', type=str, default='N',
                    choices=['N', 'Y'], help='whether to pretrain VAE or not')
parser.add_argument('--train', type=str, default='Y',
                    choices=['N', 'Y'], help='whether to pretrain VAE or not')
parser.add_argument('--use_psuedo', type=str, default='N',
                    choices=['N', 'Y'], help='whether to use target psuedo label')

# ========================= Loss Configs ==========================
# Loss_vae + MI(z_f,z_t)
parser.add_argument('--weight_f', type=float, default=1,
                    help='weighting on KL to prior, content vector')
parser.add_argument('--weight_z', type=float, default=1,
                    help='weighting on KL to prior, motion vector')
parser.add_argument('--weight_MI', type=float, default=0,
                    help='weighting on Mutual infomation of f and z')
# loss on z_t: (1) adv_loss (2) cls_loss (3) attendtive entropy
parser.add_argument('--weight_cls', type=float, default=0,
                    help='weighting on video classification loss')
parser.add_argument('--beta', default=[0.75, 0.75, 0.5], type=float, nargs="+", metavar='M',
                    help='weighting for the adversarial loss (use scheduler if < 0; [relation-beta, video-beta, frame-beta])')
parser.add_argument('--weight_entropy', default=0, type=float,
                    help='weighting for the entropy loss')
# loss on z_f: (1) domain_loss (2) triplet_loss
parser.add_argument('--weight_domain', type=float, default=0,
                    help='weighting on domain classification loss')
parser.add_argument('--weight_triplet', type=float,
                    default=0, help='weighting on triplet loss')
parser.add_argument('--weight_adv', type=float, default=0,
                    help='weighting on the adversarial loss')
parser.add_argument('--weight_VAE', type=float, default=1,
                    help='weighting on the VAE loss')

args, remaining_args = parser.parse_known_args()

model_args = get_model_a_parser().parse_args(remaining_args)
opt = utils.merge_args(model_args, args)

if not opt.parallel_train:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

best_tgt_prec = 0
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
CE_loss = nn.CrossEntropyLoss().cuda()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(source_loader, target_loader, model, optimizer, train_file, epoch, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_class = AverageMeter()
    losses_rec = AverageMeter()
    losses_kl = AverageMeter()
    losses_sparse = AverageMeter()
    losses_alignment = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    iter_src = iter(source_loader)
    iter_tar = iter(target_loader)

    epoch_size = len(source_loader)
    len_target_loader = len(target_loader)
    start_steps = epoch * epoch_size
    total_steps = 1000 * epoch_size

    for i in range(epoch_size):
        # if epoch > 100 and random.random() < 0.3:
        #     print("continue")
        #     continue
        # src_data = iter_src.next()
        # tar_data = iter_tar.next()
        src_data = next(iter_src)
        tar_data = next(iter_tar)

        if i % len_target_loader == 0:
            iter_tar = iter(target_loader)

        p = float(i + start_steps) / total_steps
        beta_dann = 2. / (1. + np.exp(-10 * p)) - 1
        beta = [beta_dann if opt.beta[i] < 0 else opt.beta[i]
                for i in range(len(opt.beta))]

        source_data = src_data[0].cuda()
        source_label = src_data[1]
        target_data = tar_data[0].cuda()
        target_label = tar_data[1]

        data_time.update(time.time() - end)

        source_label = source_label.cuda(non_blocking=True)
        target_label = target_label.cuda(non_blocking=True)

        loss = model(source_data, target_data, source_label, epoch)

        losses.update(loss["total_loss"].item())
        losses_class.update(loss["c_loss"].item())
        losses_rec.update(loss["rec_loss"].item())
        losses_sparse.update(loss["sparsity_loss"].item())
        losses_alignment.update(loss["structure_loss"].item())
        losses_kl.update(loss["kld_loss"].item())
        optimizer.zero_grad()

        loss["total_loss"].backward()

        if opt.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), opt.clip_gradient)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if opt.lr_adaptive == 'dann':
            utils.adjust_learning_rate_dann(optimizer, p, opt)
    utils.print_log(
        f"total_loss:{losses.avg}; class_loss:{losses_class.avg}; rec_loss:{losses_rec.avg}"
        f" ;structure_loss:{losses_alignment.avg}; kl_loss:{losses_kl.avg}; sparse_loss:{losses_sparse.avg}",
        train_file)
    return losses.avg, losses_class.avg


def validate(val_loader, model, opt):
    top1 = AverageMeter()
    losses_class = AverageMeter()

    model.eval()

    iter_val = iter(val_loader)
    val_size = len(iter_val)

    for i in range(val_size):
        # val_dataloader = iter_val.next()
        val_dataloader = next(iter_val)
        val_data = val_dataloader[0].cuda()
        val_label = val_dataloader[1]

        val_size_ori = val_data.size()
        batch_val_ori = val_size_ori[0]

        val_label = val_label.cuda(non_blocking=True)

        with torch.no_grad():
            pred = model.inference(val_data)
            loss = opt.criterion_src(pred, val_label)
            losses_class.update(loss.item())

            prec1, prec5 = utils.accuracy(pred.data, val_label, topk=(1, 5))

            top1.update(prec1.item(), val_label.size(0))

    return top1.avg, losses_class.avg


def main(opt):
    global best_tgt_prec

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class_file = '%s/classInd_%s.txt' % (opt.data_root, opt.dataset)
    class_names = [line.strip().split(' ', 1)[1] for line in open(class_file)]
    opt.num_class = len(class_names)

    localtime = time.asctime(time.localtime(time.time()))
    localtime2 = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # path_exp = opt.exp_dir + '/' + model_config + '/' + \
    #            learning_config + '/' + weight_config + '/'

    path_exp = f"{opt.exp_dir}/{opt.src}/{opt.tar}/"

    if not os.path.isdir(path_exp):
        os.makedirs(path_exp, exist_ok=True)

    pretrain_exp = path_exp + 'pretrained_model/'
    if not os.path.isdir(pretrain_exp):
        os.makedirs(pretrain_exp, exist_ok=True)

    if opt.log_indicator == 1:
        train_file = path_exp + 'train_log_{}.txt'.format(localtime2)
    else:
        train_file = None
    utils.print_log("Run time: {}".format(localtime), train_file)
    utils.print_log("Random Seed: {}".format(opt.seed), train_file)
    utils.print_log('Running parameters:', train_file)
    utils.print_log(json.dumps(vars(opt), indent=4,
                               separators=(',', ':')), train_file)

    utils.print_log('loading data...', train_file)
    train_source_list = '%s/%s/list/%s/list_train_%s-%s_%s.txt' % (
        opt.data_root, opt.src, opt.backbone, opt.src, opt.tar, opt.backbone)
    train_target_list = '%s/%s/list/%s/list_train_%s-%s_%s.txt' % (
        opt.data_root, opt.tar, opt.backbone, opt.src, opt.tar, opt.backbone)
    tgt_val_list = '%s/%s/list/%s/list_val_%s-%s_%s.txt' % (
        opt.data_root, opt.tar, opt.backbone, opt.src, opt.tar, opt.backbone)
    src_val_list = '%s/%s/list/%s/list_val_%s-%s_%s.txt' % (
        opt.data_root, opt.src, opt.backbone, opt.tar, opt.src, opt.backbone)

    # tgt_val_list = '%s/%s/list/%s/list_train_%s-%s_%s.txt' % (
    #     opt.data_root, opt.src, opt.backbone, opt.src, opt.tar, opt.backbone)
    # src_val_list = '%s/%s/list/%s/list_train_%s-%s_%s.txt' % (
    #     opt.data_root, opt.tar, opt.backbone, opt.src, opt.tar, opt.backbone)
    # train_source_list = '%s/%s/list/%s/list_val_%s-%s_%s.txt' % (
    #     opt.data_root, opt.tar, opt.backbone, opt.src, opt.tar, opt.backbone)
    # train_target_list = '%s/%s/list/%s/list_val_%s-%s_%s.txt' % (
    #     opt.data_root, opt.src, opt.backbone, opt.tar, opt.src, opt.backbone)
    num_source = sum(1 for i in open(train_source_list))
    num_target = sum(1 for i in open(train_target_list))
    opt.dataset_size = num_source + num_target
    num_val = sum(1 for i in open(tgt_val_list))
    src_aug_num = opt.batch_size - num_source % opt.batch_size
    tar_aug_num = opt.batch_size - num_target % opt.batch_size

    class_id_list = [int(line.strip().split(' ')[2])
                     for line in open(train_source_list)]
    class_id, class_data_counts = np.unique(
        np.array(class_id_list), return_counts=True)
    class_freq = (class_data_counts / class_data_counts.sum()).tolist()
    opt.class_freq = class_freq
    weight_source_class = torch.ones(opt.num_class).cuda()
    weight_domain_loss = torch.Tensor([1, 1]).cuda()

    if opt.weighted_class_loss == 'Y':
        weight_source_class = 1 / torch.Tensor(class_freq).cuda()

    if opt.weighted_class_loss_DA == 'Y':
        weight_domain_loss = torch.Tensor([1 / num_source, 1 / num_target]).cuda()

    opt.criterion_src = torch.nn.CrossEntropyLoss(
        weight=weight_source_class).cuda()
    opt.criterion_domain = torch.nn.CrossEntropyLoss(
        weight=weight_domain_loss).cuda()

    model = Net(opt)
    if not opt.parallel_train:
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    if opt.optimizer == 'SGD':
        utils.print_log('using SGD', train_file)
        optimizer = torch.optim.SGD(model.parameters(
        ), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optimizer == 'Adam':
        utils.print_log('using Adam', train_file)
        optimizer = torch.optim.Adam(
            model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    else:
        utils.print_log('optimizer not support or specified!!!', train_file)
        exit()

    start_epoch = 1
    print('checking the checkpoint......')
    if opt.resume:
        if os.path.isfile(path_exp + opt.resume):
            checkpoint = torch.load(path_exp + opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_tgt_prec = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            utils.print_log(("=> loaded checkpoint '{}' (epoch {})".format(
                opt.resume, checkpoint['epoch'])), train_file)
            if opt.resume_hp:
                utils.print_log(
                    "=> loaded checkpoint hyper-parameters", train_file)
                optimizer.load_state_dict(checkpoint['optimizer'])
        elif os.path.isfile('pretrained_model/' + opt.resume):
            checkpoint = torch.load('pretrained_model/' + opt.resume)
            model.load_state_dict(checkpoint['state_dict'])
            utils.print_log(
                ("=> loaded pretrained VAE '{}'".format(opt.resume)), train_file)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:

        # model.apply(utils.init_weights)
        pass

    utils.print_log(model, train_file)
    utils.print_log('========== start: ' + str(start_epoch),
                    train_file)

    val_segments = opt.val_segments if opt.val_segments > 0 else opt.num_segments
    tar_val_set = TSNDataSet("", tgt_val_list, num_dataload=num_val, num_segments=val_segments,
                             new_length=1, modality='RGB',
                             image_tmpl="img_{:05d}.t7",
                             random_shift=False,
                             test_mode=True,
                             )
    tar_val_loader = torch.utils.data.DataLoader(tar_val_set, batch_size=num_val, shuffle=False,
                                                 num_workers=opt.data_threads, pin_memory=True)

    src_val_set = TSNDataSet("", src_val_list, num_dataload=num_val, num_segments=val_segments,
                             new_length=1, modality='RGB',
                             image_tmpl="img_{:05d}.t7",
                             random_shift=False,
                             test_mode=True
                             )
    src_val_loader = torch.utils.data.DataLoader(src_val_set, batch_size=num_val, shuffle=False,
                                                 num_workers=opt.data_threads, pin_memory=True)

    source_set = TSNDataSet("", train_source_list, num_dataload=(num_source + src_aug_num),
                            num_segments=opt.num_segments,
                            new_length=1, modality='RGB',
                            image_tmpl="img_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            triple=opt.weight_triplet
                            )
    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=opt.batch_size, shuffle=False,
                                                sampler=source_sampler, num_workers=opt.data_threads, pin_memory=True)

    target_set = TSNDataSet("", train_target_list, num_dataload=(num_target + tar_aug_num),
                            num_segments=opt.num_segments,
                            new_length=1, modality='RGB',
                            image_tmpl="img_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            triple=opt.weight_triplet
                            )
    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=opt.batch_size, shuffle=False,
                                                sampler=target_sampler, num_workers=opt.data_threads, pin_memory=True)

    loss_c_current = 999  # random large number
    loss_c_previous = 999  # random large number

    if opt.pretrain_VAE == 'Y':
        is_pretrain = True
        utils.print_log('Pretraining VAE part......', train_file)
        for epoch in range(start_epoch, start_epoch + opt.epochs + 1):
            if opt.lr_adaptive == 'loss':
                utils.adjust_learning_rate_loss(
                    optimizer, opt.lr_decay, loss_c_current, loss_c_previous, '>')
            elif opt.lr_adaptive == 'none' and epoch in opt.lr_steps:
                utils.adjust_learning_rate(optimizer, opt.lr_decay)

            loss, loss_c = train(source_loader, target_loader, model, optimizer,
                                 train_file, epoch, opt)

            loss_c_previous = loss_c_current
            loss_c_current = loss_c

            if epoch % opt.eval_freq == 0 or epoch == opt.epochs:
                prec1 = validate(tar_val_loader, model, opt)
                is_best = False
                if opt.save_model:
                    utils.save_checkpoint({
                        'epoch': 0,
                        'backbone': opt.backbone,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_tgt_prec,
                        'prec1': 0,
                    }, is_best, is_pretrain, pretrain_exp)

    if opt.train == 'Y':
        utils.print_log('start training ......', train_file)
        is_pretrain = False
        best_model = None
        for epoch in range(start_epoch, start_epoch + opt.epochs):
            utils.print_log(f'------------epoch:{epoch}----------', train_file)
            if opt.lr_adaptive == 'loss':
                utils.adjust_learning_rate_loss(
                    optimizer, opt.lr_decay, loss_c_current, loss_c_previous, '>')
            elif opt.lr_adaptive == 'none' and epoch in opt.lr_steps:
                utils.adjust_learning_rate(optimizer, opt.lr_decay)

            loss, loss_c = train(source_loader, target_loader, model, optimizer,
                                 train_file, epoch, opt)

            loss_c_previous = loss_c_current
            loss_c_current = loss_c

            if epoch % opt.eval_freq == 0 or epoch == opt.epochs:
                src_train_prec, src_train_loss = validate(source_loader, model, opt)
                validate(target_loader, model, opt)
                validate(src_val_loader, model, opt)
                tar_val_prec, tar_val_loss = validate(tar_val_loader, model, opt)

                utils.print_log(f"src_train: loss:{src_train_loss};score:{src_train_prec}", train_file)

                utils.print_log(f"tar_val: loss:{tar_val_loss};score:{tar_val_prec}", train_file)
                is_best = tar_val_prec > best_tgt_prec
                if is_best:
                    best_model = copy.deepcopy(model)
                best_tgt_prec = max(tar_val_prec, best_tgt_prec)
                line_update = ' ==> updating the best accuracy\n' if is_best else ''
                line_best = "Best tar_val score {} \n".format(
                    best_tgt_prec) + line_update
                utils.print_log(line_best, train_file)

        os.makedirs(opt.result_dir, exist_ok=True)
        with open(f"./{opt.result_dir}/{opt.src}_{opt.tar}.txt", "a") as f:
            f.write(f"{model_args}\n")
            f.write(f"best:{best_tgt_prec}\n")
        if opt.save_model:
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save(best_model, f"{args.model_dir}/{best_tgt_prec:4f}.pth")


if __name__ == '__main__':
    main(opt)
