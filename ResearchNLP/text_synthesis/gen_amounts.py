def pool_size(old_tns, base_sents):
    if old_tns is None:
        return int(len(base_sents) * 1.5)
    else:
        return old_tns


def triangle_total_new_sents(base_sents):
    return int(len(base_sents) * 1.5)
