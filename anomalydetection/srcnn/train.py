import argparse
import csv
import os

import torch.distributed as dist
import torch.utils.data
from torch import nn, optim
import logging

from anomalydetection.utils.utils import *

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

'''
https://github.com/kubeflow/katib/tree/master/examples/v1beta1/trial-images/pytorch-mnist
'''


class SRCNN(nn.Module):
    def __init__(self, window=WINDOW_SIZE):
        self.window = window
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv1d(window, window, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv1d(window, 2 * window, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(2 * window, 4 * window)
        self.fc2 = nn.Linear(4 * window, window)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), self.window, 1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.float().to(device), target.float().to(device)

        output = model(data)
        loss = loss_function(output, target, model, args.weight_decay) / len(data)
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if batch_idx % args.log_interval == 0:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item())
            logging.info(msg)


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logging.info("{{metricName: accuracy, metricValue: {:.4f}}};{{metricName: loss, metricValue: {:.4f}}}\n".format(
        float(correct) / len(test_loader.dataset), test_loss))


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch SRCNN Training")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=0.5, metavar="WD",
                        help="Weight decay (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--no-dist", action="store_true", default=False,
                        help="disables distributed computing")
    parser.add_argument("--seed", type=int, default=54321, metavar="S",
                        help="random seed (default: 54321)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")
    parser.add_argument("--data-path", type=str, default="",
                        help="The path to the training+test data location")
    parser.add_argument("--num-of-anomalous-points", type=int, default=10,
                        help="Number of anomalous points to insert into training data (default: 10)")
    parser.add_argument("--window-steps", type=int, default=128,
                        help="Number of steps to jump in data generator (window steps) (default: 128)")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, metavar="WS",
                        help=f"Window size (default: {WINDOW_SIZE})")

    if dist.is_available():
        parser.add_argument("--backend", type=str, help="Distributed backend",
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()

    number_of_anomalous_points_to_add = int(args.num_of_anomalous_points)
    data_generator_steps = int(args.window_steps)
    window_size = args.window_size

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    if args.log_path == "":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=args.log_path)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print("Using distributed PyTorch with {} backend".format(args.backend))
        dist.init_process_group(backend=args.backend)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    kpis = load_kpi(args.data_path + '/train.csv')
    training_data = []
    generator = DataGenerator(window_size, data_generator_steps, number_of_anomalous_points_to_add)
    for kpi in kpis.values():
        in_value = kpi[1]
        train_data = generator.generate_train_data(in_value)
        training_data += train_data

    '''
    test_hdf = pd.read_hdf(args.data_path + '/test.hdf')
    test_hdf.groupby(test_hdf["KPI ID"])
    for name, kpi in test_hdf:
        in_timestamp = kpi['timestamp'].tolist()
        in_value = kpi['value'].tolist()
        in_label = kpi['label'].tolist()
        train_data = generator.generate_train_data(in_value, insert_anomaly=False)
        training_data += train_data
    '''

    model = SRCNN().to(device)

    if not args.no_dist and is_distributed():
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)

    bp_parameters = filter(lambda p: p.requires_grad, model.parameters())  # back propagation parameters

    optimizer = optim.SGD(bp_parameters, lr=args.lr, momentum=args.momentum)

    training_dataset = SrDataset(args.window_size, training_data)
    train_loader = torch.utils.data.DataLoader(dataset=training_dataset, shuffle=True,
                                               batch_size=args.batch_size, **kwargs)
    #test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False,
    #                                          batch_size=args.test_batch_size, **kwargs)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(args, model, device, test_loader, epoch)
        #adjust_lr(optimizer, epoch, args.lr)

    if args.save_model:
        torch.save(model.state_dict(), "srcnn.pt")


if __name__ == "__main__":
    main()
