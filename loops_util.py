import numpy as np
def prepare_input(token_ids, tokenizer):
    tokens = tokenizer._ids_to_tokens(token_ids)
    program_tokens = [t for t in tokens if t.startswith("Program_")]
    program_tokens = np.unique(np.random.permutation(program_tokens)).tolist()
    seq = program_tokens + tokens
    seq = ["BOS_None"] + seq + ["EOS_None"]
    ids = tokenizer._tokens_to_ids(seq)
    return ids