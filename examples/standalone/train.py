import torch
from torch.utils.data.dataset import T_co
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time

# from utils import top1accuracy, top5accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_decay = 0.0005
sequence_length = 28
input_size = 28
hidden_size = 128
nlayers = 2
nclasses = 10
batch_size = 64
nepochs = 50
lr = 0.00003

data_dir = "data/"

train_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=True,
    transform=T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]),
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=False,
    transform=T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]),
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def train(num_epochs, opt_state, net_type="RNN"):
    """Implements a learning loop over epochs."""
    # Initialize placeholder for loggin
    log_acc_train, log_acc_test, train_loss = [], [], []

    # Get the initial set of parameters
    params = get_params(opt_state)

    # Loop over the training epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            if net_type == "RNN":
                x = np.array(data).reshape(data.size(0), 28 * 28)

            elif net_type == "HiPPO":
                raise NotImplementedError

            elif net_type == "S4":
                raise NotImplementedError

            else:
                raise ValueError("Unknown network type")

            y = one_hot(np.array(target), num_classes)
            params, opt_state, loss = update(params, x, y, opt_state)
            train_loss.append(loss)

        epoch_time = time.time() - start_time

        print(
            "Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f}".format(
                epoch + 1, epoch_time, train_acc, test_acc
            )
        )

    return train_loss, log_acc_train, log_acc_test


def test(test_loader, model, loss_f):
    """
    Input: test loader (torch loader), model (torch model), loss function
          (torch custom yolov1 loss).
    Output: test loss (torch float).
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            # x = x.reshape(-1, x.shape[1], x.shape[1])
            # x_expanded = x[:, None, ...].expand(x.shape[0], x.shape[1], x.shape[1])
            # x_unsqueezed = torch.unsqueeze(x, dim=-1)
            # x_unsqueezed =
            # out = model(x_expanded)
            # x.reshape(-1, sequence_length, input_size).to(device)
            x = x.unsqueeze(-1)
            vals = (torch.ones(x.shape[0], sequence_length, input_size - 1)).to(device)
            x = torch.cat([x, vals], dim=-1)
            out = model(x)
            del x
            # del x_expanded
            out = F.softmax(out, dim=1)
            # top1_acc = top1accuracy(out, y, batch_size)
            # top5_acc = top5accuracy(out, y, batch_size)
            pred_class = torch.argmax(out, dim=1)
            test_loss_val = loss_f(pred_class.float(), y.float())
            del y
            del out
            del pred_class
        return float(test_loss_val.item())


def main():
    model = SimpleRNN(
        input_size=28,
        hidden_size=128,
        num_layers=2,
        bias=True,
        output_size=10,
        activation="relu",
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_f = nn.CrossEntropyLoss()

    train_loss_lst = []
    test_loss_lst = []
    train_top1acc_lst = []
    test_top1acc_lst = []
    train_top5acc_lst = []
    test_top5acc_lst = []
    last_epoch = 0

    for epoch in range(nepochs - last_epoch):
        # train_loss_value, train_top1acc_value, train_top5acc_value = train(train_loader, model, optimizer, loss_f)
        train_loss_value = train(train_loader, model, optimizer, loss_f)
        train_loss_lst.append(train_loss_value)
        # train_top1acc_lst.append(train_top1acc_value)
        # train_top5acc_lst.append(train_top5acc_value)

        # test_loss_value, test_top1acc_value, test_top5acc_value  = test(test_loader, model, loss_f)
        test_loss_value = test(test_loader, model, loss_f)
        test_loss_lst.append(test_loss_value)
        # test_top1acc_lst.append(test_top1acc_value)
        # test_top5acc_lst.append(test_top5acc_value)

        # print(f"Epoch:{epoch + last_epoch + 1 }  Train[Loss:{train_loss_value}  Top5 Acc:{train_top5acc_value}  Top1 Acc:{train_top1acc_value}]")
        # print(f"Epoch:{epoch + last_epoch + 1 }  Test[Loss:{test_loss_value}  Top5 Acc:{test_top5acc_value}  Top1 Acc:{test_top1acc_value}]")
        # print(f"Epoch:{epoch + last_epoch + 1 }  Train[Loss:{train_loss_value}  Test[Loss:{test_loss_value}]")


if __name__ == "__main__":
    main()
