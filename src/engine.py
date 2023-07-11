# engine


from tqdm import tqdm

def train_fn(data_loader, model, optimizer):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)



def eval_fn(data_loader, model):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        _, loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)