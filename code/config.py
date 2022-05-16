import argparse


def get_config():
    parser = argparse.ArgumentParser(description='HPML Final Project. ')
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located. train.csv & train-jpg")
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 250)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--opt', type=str, default='adam', metavar='O', choices=[
                        'sgd', 'sgd_nes', 'adagrad', 'adadelta', 'adam'], help='Optimizer to use (default: sgd)')
    parser.add_argument('--gpu', default=False, action='store_true',
                        help='Switch to use gpu or not')
    parser.add_argument('--num-workers', type=int, default=0,
                        metavar='n', help='number of workers for data loader')
    parser.add_argument('--log-interval', type=int, default=1,
                        metavar='l', help='number of minibatch to lag')
    parser.add_argument('--log', default=False, action='store_true',
                        help='Switch to log time output')
    parser.add_argument('--n-step', type=int, default=2, metavar='s',
                        help='Number of steps to update from parameter server')
    args = parser.parse_args()

    return args
