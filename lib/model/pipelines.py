from tqdm import tqdm
import torch.nn.functional as F
import torch

def train(model, device, optimizer, loader, feature_encoder, target_type):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        x_dict = feature_encoder(batch.x_dict)               # <-- encode the BATCH
        out_dict = model(x_dict, batch.edge_index_dict)      # <-- run on the BATCH
        out = out_dict[target_type][:batch[target_type].batch_size]
        y = batch[target_type].y[:batch[target_type].batch_size].view(-1)
        loss = F.cross_entropy(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / len(loader)                          # <-- use loader

@torch.no_grad()
def test(model, device, loader, feature_encoder, target_type):
    model.eval()
    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        x_dict = feature_encoder(batch.x_dict)               # <-- encode the BATCH
        out_dict = model(x_dict, batch.edge_index_dict)      # <-- run on the BATCH
        out = out_dict[target_type][:batch[target_type].batch_size]
        y = batch[target_type].y[:batch[target_type].batch_size].view(-1)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == y).sum())
        total_examples += y.size(0)
    return total_correct / total_examples
