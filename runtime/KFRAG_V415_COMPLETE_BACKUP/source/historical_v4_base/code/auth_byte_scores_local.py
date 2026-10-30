@torch.no_grad()
def auth_byte_scores_local(final_logits20):

    if tuple(final_logits20.shape) != (16, 20):
        raise RuntimeError(
            f"Expected final_logits20=(16,20), got {tuple(final_logits20.shape)}"
        )

    # source_byte_scores_local() expects the source field at [4:12].
    # Copy auth logits [12:20] into those positions.
    proxy = final_logits20.clone()

    proxy[:, 4:12] = final_logits20[:, 12:20]

    scores = source_byte_scores_local(proxy)

    if tuple(scores.shape) != (16, 256):
        raise RuntimeError(
            f"Expected AUTH byte scores=(16,256), got {tuple(scores.shape)}"
        )

    return scores
