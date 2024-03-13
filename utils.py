import numpy as np
import torch

def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = []
    for batch_inputs, batch_labels, batch_weights in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels, batch_weights)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

    return np.nanmean(running_loss)


def validation_step(model, test_loader, criterion, device):
    model.eval()
    running_loss = []
    for batch_inputs, batch_labels, batch_weights in test_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels, batch_weights)

        running_loss.append(loss.item())

    return np.nanmean(running_loss)


def reweighting_fn(model, X_val):
    model = model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        preds = model(torch.Tensor(X_val)).detach().numpy().ravel()
        weights = preds / (1.0 - preds)
    return weights