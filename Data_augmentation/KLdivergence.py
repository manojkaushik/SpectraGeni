import numpy as np
import pandas as pd
from scipy.stats import entropy

def kl_divergence(real, synthetic, bins=50):
    kl_dict = {}
    for col in real.columns:
        # Categorical
        if real[col].dtype == 'object' or real[col].dtype.name == 'category':
            p = real[col].value_counts(normalize=True)
            q = synthetic[col].value_counts(normalize=True)
            # align on same index
            all_idx = p.index.union(q.index)
            p = p.reindex(all_idx, fill_value=0)
            q = q.reindex(all_idx, fill_value=0)
            kl = entropy(p, q)
        else:
            # Continuous
            min_val = min(real[col].min(), synthetic[col].min())
            max_val = max(real[col].max(), synthetic[col].max())
            hist_range = (min_val, max_val)
            p_hist, _ = np.histogram(real[col], bins=bins, range=hist_range, density=True)
            q_hist, _ = np.histogram(synthetic[col], bins=bins, range=hist_range, density=True)
            # To avoid division by zero or log(0)
            p_hist += 1e-8
            q_hist += 1e-8
            kl = entropy(p_hist, q_hist)
        kl_dict[col] = kl
    
    kl_scores = pd.Series(kl_dict, name="KL Divergence")
    return kl_scores.mean(), kl_scores