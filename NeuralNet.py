import numpy as np
import torch
import torch.nn as nn
from torch.utils.data        import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class WeightsNet(nn.Module):
    def __init__(self, n_features, hidden_layers, n_outputs=2):
        super().__init__()
        layers = []
        in_size = n_features
        for h in hidden_layers:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, n_outputs))
        layers.append(nn.Softmax())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# class WeightsNet(nn.Module):
#     def __init__(self, n_features, hidden_layers, n_outputs=2):
#         super().__init__()
#         assert len(hidden_layers) >= 4, \
#             "hidden_layers deve avere almeno 4 elementi per la skip 2°↔penultimo"

#         # ── Primi due layer ────────────────────────────────────────────────
#         self.pre = nn.Sequential(
#             nn.Linear(n_features,       hidden_layers[0]), nn.ReLU(),
#             nn.Linear(hidden_layers[0], hidden_layers[1]), nn.ReLU(),
#         )

#         # ── Layer centrali (tra i due endpoint della skip) ─────────────────
#         mid_layers, in_size = [], hidden_layers[1]
#         for h in hidden_layers[2:-2]:
#             mid_layers += [nn.Linear(in_size, h), nn.ReLU()]
#             in_size = h
#         self.mid = nn.Sequential(*mid_layers)   # Sequential vuoto = identity

#         # ── Proiezione skip: 2° layer → ingresso del penultimo ─────────────
#         self.skip_proj = (nn.Linear(hidden_layers[1], in_size, bias=False)
#                           if hidden_layers[1] != in_size else nn.Identity())

#         # ── Penultimo + ultimo layer nascosto, poi output ──────────────────
#         self.post = nn.Sequential(
#             nn.Linear(in_size,           hidden_layers[-2]), nn.ReLU(),
#             nn.Linear(hidden_layers[-2], hidden_layers[-1]), nn.ReLU(),
#         )
#         self.out = nn.Linear(hidden_layers[-1], n_outputs)

#     def forward(self, x):
#         h_skip = self.pre(x)                                     # output del 2° layer
#         h      = self.mid(h_skip) + self.skip_proj(h_skip)      # skip arriva prima del penultimo
#         return self.out(self.post(h))
        


def train_neural_network(x_train, x_test, y_train, y_test, hyperparameters):
    """
    Train a 4-hidden-layer neural network and return predictions.

    Parameters
    ----------
    x_train, x_test : np.ndarray   (N, n_features) / (M, n_features)
    y_train, y_test : np.ndarray   (N, 2) / (M, 2)
    hyperparameters : dict
        lr            : float  (default 1e-3)
        epochs        : int    (default 200)
        batch_size    : int    (default 256)
        weight_decay  : float  (default 0)
        hidden_layers : list[int]  (default [100, 80, 50, 10])
        n_outputs     : int    (default 3)

    Returns
    -------
    y_pred_train : np.ndarray (N, 3)
    y_pred_test  : np.ndarray (M, 3)
    model        : WeightsNet
    """

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # multi-GPU

    lr            = hyperparameters.get("lr", 1e-3)
    epochs        = hyperparameters.get("epochs", 1000)
    batch_size    = hyperparameters.get("batch_size", 256)
    weight_decay  = hyperparameters.get("weight_decay", 0.0)
    hidden_layers = hyperparameters.get("hidden_layers", [100, 100, 100, 100])
    lambda_l1     = hyperparameters.get("lambda_l1", 1e-4)
    n_outputs     = hyperparameters.get("n_outputs", 3)
    seed          = hyperparameters.get("seed", 1)

    writer = SummaryWriter(log_dir=f"runs/{seed}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    X_tr = torch.tensor(x_train, dtype=torch.float32, device=device)
    Y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(x_test,  dtype=torch.float32, device=device)
    Y_te = torch.tensor(y_test,  dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_tr, Y_tr),
                        batch_size=batch_size, shuffle=True)

    model = WeightsNet(x_train.shape[1], hidden_layers, n_outputs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"number of params: {num_params}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion      = nn.MSELoss() 
    criterion_l1   = nn.L1Loss()  # MSE + L1 reg
    criterion_comp = nn.MSELoss(reduction='none')   # per-sample, per-output

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb) + lambda_l1 * criterion_l1(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        # ── log once per epoch (after all batches) ──────────────────────────
        model.eval()
        with torch.no_grad():
            train_pred  = model(X_tr)
            val_pred    = model(X_te)
            # mean over samples, shape → (n_outputs,)
            comp_loss_tr  = criterion_comp(train_pred, Y_tr).mean(dim=0)
            comp_loss_val = criterion_comp(val_pred,   Y_te).mean(dim=0)

        writer.add_scalar("Loss/train_total", epoch_loss / len(X_tr), epoch)
        writer.add_scalar("Loss/val_total",   criterion(val_pred, Y_te).item(), epoch)
        for i in range(n_outputs):
            writer.add_scalar(f"Loss_component/train_output{i+1}", comp_loss_tr[i].item(),  epoch)
            writer.add_scalar(f"Loss_component/val_output{i+1}",   comp_loss_val[i].item(), epoch)
        model.train()
        if epoch % 100 == 0 or epoch == 1:
            # print(f"  epoch {epoch:4d}/{epochs}  loss = {epoch_loss / len(X_tr):.6f}")
            for name, param in model.named_parameters():
                writer.add_histogram(f"params/{name}", param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name}", param.grad, epoch)

    model.eval()
    
    writer.add_graph(model, X_tr)
    writer.flush()
    with torch.no_grad():
        y_pred_train = model(X_tr).cpu().numpy()
        y_pred_test  = model(X_te).cpu().numpy()

    # print(y_pred_train)
    # print(y_pred_test.shape)

    return y_pred_train, y_pred_test, model
