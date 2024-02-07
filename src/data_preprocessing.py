import pandas as pd
import torch


def yield_tokens(data_iter: pd.DataFrame, tokenizer, identifier):
    for _, row in data_iter.iterrows():
        yield tokenizer(row[identifier])


def collate_batch(
    batch, device, label_pipeline, text_pipeline, label_identifier, text_identifier
):
    label_list, text_list, offsets = [], [], [0]
    for _, _row in batch:
        label_list.append(label_pipeline(_row[text_identifier]))
        processed_text = torch.tensor(
            text_pipeline(_row[label_identifier]), dtype=torch.int64
        )
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)
