import argparse
import pickle
from model import *
from train import *
from data_loader import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
    parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
    parser.add_argument('--nodes', type=int, help='number of nodes to use', default=37484)
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--resume', type=str, default=None, help='path of model to resume')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    
    train_data = pickle.load(open('../datasets/yoochoose1_4/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/yoochoose1_4/test.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    
    model, loss_function, optimizer, scheduler = build(args, resume=args.resume)

    model, metrics = train_model(args.epoch, 
                                 model, 
                                 train_data, 
                                 optimizer, 
                                 scheduler, 
                                 loss_function, 
                                 args.device)
    
    print("-"*100)
    print(f'Best Recall@20:\t{metrics["best_result"][0]:.4f}\tBest MMR@20:\t{metrics["best_result"][1]:.4f}')
    
    test_hit, test_mrr = test_model(model, 
                                    test_data)
    print("-"*100)
    print(f'Test\tRecall@20:\t{test_hit:.4f}\tMMR@20:\t{test_mrr:.4f}')

if __name__ == '__main__':
    main()
