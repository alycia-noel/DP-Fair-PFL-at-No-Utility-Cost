import argparse
import logging
import random
import warnings
from collections import OrderedDict, defaultdict
from functools import reduce
import numpy as np
import torch
import torch.utils.data
from tqdm import trange
from models import CNNHyper, CNNTarget
from node import BaseNodes
from utils import seed_everything, set_logger

warnings.filterwarnings("ignore")

@torch.no_grad()
def evaluate(nodes, num_nodes, hnet, model, device):

    hnet.eval()

    results = defaultdict(lambda: defaultdict(int))

    preds = []
    true = []
    accuracies = []

    for node_id in range(num_nodes):
        pred_client = []
        true_client = []

        cmodel = model[node_id]
        cmodel.eval()
        cmodel.to(device)

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes.test_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple(t.to(device) for t in batch)

            true_client.extend(y.cpu().numpy())

            weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            cmodel.load_state_dict(weights)

            pred = cmodel(x)
            predictions = pred.argmax(1).eq(y).sum().item()
            running_correct += predictions
            pred_client.append(predictions)
            running_samples += len(y)

        accuracy = running_correct / running_samples

        accuracies.append(accuracy)

        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

        preds.append(pred_client)
        true.append(true_client)

    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in results.values()]

    return results, avg_acc, all_acc

def train(device, data_name, classes_per_node, num_nodes, steps, inner_steps, lr, inner_lr, wd, inner_wd, hyper_hid, n_hidden, bs, n_kernels):

    avg_acc = [[] for _ in range(num_nodes + 1)]

    client_models = []
    client_opt = []
    client_losses = [0 for _ in range(num_nodes)]

    for i in range(1):
        seed_everything(0)

        nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)

        total_num_points = nodes.total_num_points
        num_points_p_c = nodes.num_points_p_c

        # Probability distribution for sampling clients
        p_k = [x / total_num_points for x in num_points_p_c]

        # r_k is the ordering of clients based on loss. it is initially a random ordering
        r_k = [r for r in range(num_nodes)]
        random.shuffle(r_k)

        # Lambda / p_k for regularization (lambda controls trade off between accuracy and client parity)
        lambda_pk = [.5 / p for p in p_k]

        embed_dim = int(1 + num_nodes / 4)

        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid, n_hidden=n_hidden, n_kernels=n_kernels)

        for n in range(num_nodes):
            client_models.append(CNNTarget(n_kernels=n_kernels))
            client_opt.append(torch.optim.SGD(client_models[n].parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd))


        hnet.to(device)

        opt = torch.optim.Adam(params=hnet.parameters(), lr=lr, weight_decay=wd)
        loss = torch.nn.CrossEntropyLoss()
        step_iter = trange(steps)

        for _ in step_iter:
            hnet.train()

            # Sample 10 nodes according to the probability distribution p_k
            node_ids = [np.random.choice(np.arange(0, num_nodes), p=p_k) for _ in range(10)]

            # Update r_k
            for k in range(num_nodes):
                running_r_k = 0

                for j in range(num_nodes):
                    if k == j:
                        continue

                    running_r_k += np.sign(client_losses[k] - client_losses[j])

                r_k[k] = running_r_k

            print(r_k)

            models = [client_models[node_id] for node_id in node_ids]
            inner_opts = [client_opt[node_id] for node_id in node_ids]

            weights = [hnet(torch.tensor([node_id], dtype=torch.long).to(device)) for node_id in node_ids]

            for h in range(10):
                models[h].load_state_dict(weights[h])
                models[h].to(device)
                models[h].train()

            inner_states = [OrderedDict({k: tensor.data for k, tensor in weights[q].items()}) for q in range(10)]
            delta_thetas = [[] for _ in range(10)]

            for x in range(10):
                node_num = node_ids[x]
                model = models[x]
                inner_opt = inner_opts[x]
                inner_state = inner_states[x]
                weight = weights[x]

                for j in range(inner_steps):
                    inner_opt.zero_grad()
                    opt.zero_grad()

                    batch = next(iter(nodes.train_loaders[node_num]))
                    x, y = tuple(t.to(device) for t in batch)

                    pred = model(x)

                    err = loss(pred, y)

                    err.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

                    for group in inner_opt.param_groups:
                        for p in group['params']:
                            p.grad = (1 + lambda_pk[node_num]*r_k[node_num]) * p.grad

                    inner_opt.step()

                client_losses[node_num] = err.item()

                opt.zero_grad()
                final_state = model.state_dict()
                dt = list(OrderedDict({k: inner_state[k] - final_state[k] for k in weight.keys()}).values())

                for t in range(len(dt)):
                    delta_thetas[t].append(dt[t])

            # Need to perform aggregation here
            summed_delta_theta = [[] for _ in range(10)]

            for b in range(10):
                summed_delta_theta[b] = reduce(torch.add, delta_thetas[b])
                summed_delta_theta[b] = (1/10) * summed_delta_theta[b]

            hnet_grads = torch.autograd.grad(list(weight.values()), hnet.parameters(), grad_outputs=summed_delta_theta)

            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)

            opt.step()

        step_results, mean_acc, acc_p_client = evaluate(nodes=nodes, num_nodes=num_nodes, hnet=hnet, model=client_models, device=device)

        logging.info(f"\n\nFinal Results | AVG Acc: {mean_acc:.4f}")
        avg_acc[0].append(mean_acc)
        for n in range(num_nodes):
            avg_acc[n + 1].append(acc_p_client[n])


    print(f"\n\nFinal Results | AVG Acc: {np.mean(avg_acc[0]):.4f}")
    for i in range(num_nodes):
        print("\nClient", i+1, f"Acc: {np.mean(avg_acc[i+1]):.4f}")


def main():

    parser = argparse.ArgumentParser(description="Robust and Fair Personalized Federated Learning at No Utility Cost")

    parser.add_argument("--data_name", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="choice of dataset")
    parser.add_argument("--num_nodes", type=int, default=50, help="number of simulated clients")
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inner_steps", type=int, default=50, help="number of inner steps")
    parser.add_argument("--n_hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner_lr", type=float, default=.005, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner_wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--seed", type=int, default=0, help="seed value")
    parser.add_argument("--device", type=str, default="cuda:2", help="which GPU to use")
    parser.add_argument("--classes_per_node", type=int, default=2, help="how many classes per client")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    args = parser.parse_args()
    set_logger()

    train(
        device=args.device,
        data_name=args.data_name,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        lr=args.lr,
        inner_lr=args.inner_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        bs=args.batch_size,
        n_kernels=args.nkernels
    )


if __name__ == "__main__":
    main()
