def fit():
    pass

def train(dataloader, model, loss_fn, optimizer, device):
    print('Training...')
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['source_input_ids'].to(device)
        tgt = batch['target_input_ids'].to(device)
        src_mask = batch['source_mask'].to(device)
        tgt_mask = batch['target_mask'].to(device)

        # Transpose masks to match the expected shape [seq_length, batch_size]
        src_mask = src_mask.t()
        tgt_mask = tgt_mask.t()

        optimizer.zero_grad()
        
        output = model(src, tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        print(output)
        output = output.reshape(-1, output.shape[-1])  # Flatten output for loss calculation
        tgt = tgt.reshape(-1)  # Flatten target for loss calculation

        loss = loss_fn(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print (total_loss)
