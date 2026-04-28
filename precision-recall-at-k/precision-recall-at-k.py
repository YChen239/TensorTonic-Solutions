def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    if k <= 0:
        return 0.0, 0.0

    # Ensure relevant is a set for fast lookup
    relevant_set = set(relevant)

    # Top-k recommendations
    top_k = recommended[:k]

    # Number of relevant items in top-k
    hits = sum(1 for item in top_k if item in relevant_set)

    precision_at_k = hits / len(top_k) if top_k else 0.0
    recall_at_k = hits / len(relevant_set) if relevant_set else 0.0

    return [precision_at_k, recall_at_k]