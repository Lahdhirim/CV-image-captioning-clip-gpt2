from bert_score import score


def semantic_similarity(preds, refs):
    P, R, F1 = score(preds, refs, lang="en", verbose=False)
    return P.mean().item(), R.mean().item(), F1.mean().item()
