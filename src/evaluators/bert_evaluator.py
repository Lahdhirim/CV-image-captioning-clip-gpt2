from bert_score import score


def semantic_similarity(preds, refs):
    _, _, F1 = score(preds, refs, lang="en", verbose=False)
    return F1.mean().item()
