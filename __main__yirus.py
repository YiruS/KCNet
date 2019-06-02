from __future__ import absolute_import, print_function

import argparse
import os
import sys

import numpy as np

import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASE_DIR)

from utils.dataset import MNDataset, fastprint
from utils.misc import AverageMeter
from kcnet.kc_module import KCNet
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train(model, train_dataset, criterion, optimizer, epoch, device, args):
    BATCH_SIZE = args.batch_size
    ITER_SIZE = args.iter_size
    TOTAL_TRAIN_DATA = train_dataset.len
    NUM_PTS = args.num_pts
    NUM_BATCH = int(np.ceil((TOTAL_TRAIN_DATA / (BATCH_SIZE * ITER_SIZE))))

    data_idx = 0
    model = model.train()
    losses = AverageMeter()

    tot_loss = []
    fastprint("Training... ")
    for batch_idx in range(NUM_BATCH):
        loss_sum = 0
        optimizer.zero_grad()
        for _iter in range(ITER_SIZE):
            data = train_dataset.getitem(data_idx)
            points, label, indptr, indices = data['data'], \
                    data['label'], \
                    data['indptr'], \
                    data['indices']
            points, label, indptr, indices = torch.from_numpy(points), \
                                            torch.from_numpy(label.reshape(-1)), \
                                            torch.from_numpy(indptr), \
                                            torch.from_numpy(indices)
            points, label, indptr, indices = points.view(NUM_PTS, -1), \
                                            label.view(-1), \
                                            indptr.view(-1), \
                                            indices.view(-1)
            points, label, indptr, indices = Variable(points).float(), \
                                            Variable(label).type(torch.LongTensor), \
                                            indptr, indices
            points, label, indptr, indices = points.to(device), \
                    label.to(device), \
                    indptr.to(device), \
                    indices.to(device)

            pred = model(points, indptr, indices)
            loss = criterion(pred, label) / ITER_SIZE
            loss.backward()

            loss_sum += loss.item()
            data_idx += 1
            losses.update(loss.item(), label.size(0))

        optimizer.step()

        tot_loss.append(loss_sum)
        fastprint('[%d: %d/%d] train loss: %f' %
                  (epoch, batch_idx, NUM_BATCH, loss_sum))

    torch.save(model.state_dict(), '%s/cls_model_%d.pth' % (args.outf, epoch))
    np.savez(os.path.join(args.outf, 'TrainLoss_epoch_{}.npz'.format(epoch)), loss=tot_loss)

def test(model, test_dataset, criterion, epoch, device, args):
    fastprint('Evaluation ... ')
    TOTAL_TEST_DATA = test_dataset.len
    NUM_PTS = args.num_pts

    test_loss = 0.0
    correct = 0.0
    losses = AverageMeter()

    model = model.eval()

    with torch.no_grad():
        for idx in range(TOTAL_TEST_DATA):
            data = test_dataset.getitem(idx)
            points, label, indptr, indices = data['data'], \
                    data['label'], \
                    data['indptr'], \
                    data['indices']
            points, label, indptr, indices = torch.from_numpy(points), \
                    torch.from_numpy(label.reshape(-1)), \
                    torch.from_numpy(indptr), \
                    torch.from_numpy(indices)
            points, label, indptr, indices = points.view(
                NUM_PTS, -1), label.view(-1), indptr.view(-1), indices.view(-1)
            points, label, indptr, indices = Variable(points).float(), Variable(
                label).type(torch.LongTensor), indptr, indices
            points, label, indptr, indices = points.to(device), label.to(
                device), indptr.to(device), indices.to(device)

            pred = model(points, indptr, indices)
            loss = criterion(pred, label)
            # get the index of the max log-probability
            pred = pred.argmax(dim=1, keepdim=True)

            test_loss += loss.item()
            losses.update(loss.item(), label.size(0))
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= float(TOTAL_TEST_DATA)
    acc = 100. * correct / float(TOTAL_TEST_DATA)
    fastprint('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, TOTAL_TEST_DATA, acc))

    return acc


def main(args):
    TRAIN_DATA_PATH = args.train_path
    TEST_DATA_PATH = args.test_path
    NUM_KERNELS = args.num_kernels
    NUM_KPTS = args.num_kpts
    INPUT_DIM = args.input_dim
    SIGMA = args.sigma
    INIT_BOUND = args.init_bound
    CLASS_DIM = args.class_dim
    NUM_EPOCHS = args.nepoch
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    train_dataset = MNDataset(path=TRAIN_DATA_PATH)
    test_dataset = MNDataset(path=TEST_DATA_PATH)
    TOTAL_TRAIN_DATA, TOTAL_TEST_DATA = train_dataset.len, test_dataset.len
    fastprint('    train data size: %d, test data size: %d' %
              (TOTAL_TRAIN_DATA, TOTAL_TEST_DATA))

    assert(BATCH_SIZE == 1)  # make sure batch size is 1

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = KCNet(num_k = NUM_KERNELS,
                  num_kpts = NUM_KPTS,
                  input_dim = INPUT_DIM,
                  sigma = SIGMA,
                  init_bound = INIT_BOUND,
                  class_dim = CLASS_DIM)
    MODEL_PATH = args.model
    if MODEL_PATH != '':
        model.load_state_dict(torch.load(MODEL_PATH))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(device)
    fastprint('    Total params: %.2fM' % (sum(p.numel()
                                               for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    best_accu = 0.0
    eval_accu = np.zeros(NUM_EPOCHS, dtype=np.float32)
    for epoch in range(NUM_EPOCHS):
        train(model, train_dataset, criterion, optimizer, epoch, device, args)
        eval_accu[epoch] = test(model, test_dataset,
                                criterion, epoch, device, args)
        best_accu = max(eval_accu[epoch], best_accu)
        fastprint('** Validation: %f (best) VS. %f (current)' %
                  (best_accu, eval_accu[epoch]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--train_path', type=str,
                        default='data/modelnet/', help='input batch size')
    parser.add_argument('--test_path', type=str,
                        default='data/modelnet/', help='input batch size')
    parser.add_argument('--num_pts', type=int, default=1024,
                        help='#points in each instance')
    parser.add_argument('--num_kernels', type=int, default=16, help='#kernels')
    parser.add_argument('--num_kpts', type=int, default=16,
                        help='#points in one kernel')
    parser.add_argument('--input_dim', type=int, default=3,
                        help='dimension of input point cloud')
    parser.add_argument('--sigma', type=float, default=0.005,
                        help='Sigma in kernel correlation')
    parser.add_argument('--init_bound', type=float, default=0.2,
                        help='range of initialized kernel points')
    parser.add_argument('--class_dim', type=int,
                        default=40, help='#classes in output')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='input batch size')
    parser.add_argument('--iter_size', type=int, default=16,
                        help='iter size similar in Caffe prototxt')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--nepoch', type=int, default=10,
                        help='number of epochs to train for')
    parser.add_argument('--outf', type=str,
                        default='results', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))
    print(args)

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    main(args)
