import torch 
from tqdm import tqdm


def train_loop(dl, model, optimizer, scheduler, criterion, device):

    model.train()  # Put model in train mode.
    losses = []
    predictions = {
        'company': [],
        'address': [],
        'date': [],
        'total': []
    }

    for batch in tqdm(dl, total=len(dl), position=0, leave=True):
        for k,v in batch.items():
            batch[k] = v.to(device)
        optimizer.zero_grad()
        x = batch.pop('images')
        outputs = model(x,batch)
        loss = outputs['losses']
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        for k,v in batch.items():
            batch[k] = v.to('cpu')

        for i,k in enumerate(predictions.keys()):
            predictions[k].append(outputs['preds'][i].detach().cpu().numpy())
    return losses, predictions


def eval_loop(dl, model, criterion, device):
    """
    Loop for evaluting the model.
    Args:
        dl: dataloader for evaluation.
        model: model for training. The model should be on
                the required device.
        criterion: Loss
        device: device for training. Expected 'cpu' or 'cuda'.
    Returns:
        losses: list of mean loss for each batch.
        outputs: list of model activation for each batch.
        dog_or_human: list of targets for dog or human for each batch
        breed_targets: list of targets for dog breeds for each batch
    """
    model.eval()  # Put model in train mode.
    losses = []
    predictions = {
        'company': [],
        'address': [],
        'date': [],
        'total': []
    }
    for batch in tqdm(dl, total=len(dl), position=0, leave=True):
        for k,v in batch.items():
            batch[k] = v.to(device)
        x = batch.pop('images')
        preds = model(x,batch)
        losses.append(loss.item())
        for k,v in batch.items():
            batch[k] = v.to('cpu')

        for i,k in enumerate(predictions.keys()):
            predictions[k].append(outputs['preds'][i].detach().cpu().numpy())
    return losses, outputs,

