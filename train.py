from encoder import Encoder
from model import BigramLanguageModel
import torch
import os
import wandb
from alive_progress import alive_bar
import logging
from torch.optim.lr_scheduler import MultiStepLR

def get_batch(dataset: list[int], batch_size: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:

    rand_indices = torch.randint(len(dataset) - context_length, (batch_size,))

    x = torch.stack([torch.tensor(dataset[i:i + context_length]) for i in rand_indices])
    y = torch.stack([torch.tensor(dataset[i + 1:i + context_length + 1]) for i in rand_indices])

    return x, y

def main():

    wandb.init(entity="aydingokce")

    ##### HYPERPARAMS #####
    context_length = 256
    batch_size = 256
    train_test_split = 0.9
    num_epochs = 1
    n_heads = 6
    dropout = 0.2
    embedding_size = 364
    load_path = "model.pth"

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    logging.info(f"using device {device}")

    ######## DATA #########
    data_file = os.path.join(os.path.dirname(__file__), 'data.txt')
    enc = Encoder(data_file)

    train_raw = enc.data[:int(len(enc.data) * train_test_split)]
    test_raw = enc.data[int(len(enc.data) * train_test_split):]

    train_enc = enc.encode(train_raw)
    test_enc = enc.encode(test_raw)

    ######## TRAIN ########


    if load_path is None:
        model = BigramLanguageModel(len(enc.vocabulary), embedding_size, context_length, device, n_heads, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[5000], gamma = 0.1)

        with alive_bar(num_epochs) as bar:
            for epoch_no in range(num_epochs):

                optimizer.zero_grad()
                X, y = get_batch(train_enc, batch_size, context_length)
                X, y = X.to(device), y.to(device)
                _, loss = model(X, y)
                loss.backward()
                item = loss.item()
                optimizer.step()
                bar.text = f"loss: {item}, lr: {scheduler.get_last_lr()}"
                wandb.log({'training loss': item})
                bar()

                scheduler.step()
    
    else:
        model.load_state_dict(torch.load(load_path))

    print(loss.item())
    
    torch.save(model.state_dict(), "model.pth")

    ######## TEST #########
    sample_x, sample_y = get_batch(test_enc, batch_size, context_length)
    input("training complete. press any key to see the output")
    model.eval()
    gen = model.generate(sample_x.to(device), enc.decode)
    #print(enc.decode(gen.cpu().tolist()))



if __name__ == "__main__":
    main()