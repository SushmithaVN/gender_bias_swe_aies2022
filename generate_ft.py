import numpy as np
import pandas as pd
from scipy.stats import norm
from os import path
import csv

# ==== CONFIG ====

# For Kannada
# EMB_DIR = r"D:\fast_text"
# EMB_FILE = "cc.kn.300.vec"
# OUTPUT_FILE = "word_level_bias_scores_Kannada1.csv"   # this is what your other script expects
# PERMUTATIONS = 1000               # keep small for testing, can increase later
# TOP_N = 100000                    # keep 1000 for quick run, change to 100000 later


# For Tamil
# EMB_DIR = r"D:\fast_text"
# EMB_FILE = "cc.ta.300.vec"
# OUTPUT_FILE = "word_level_bias_scores_Tamil.csv"   # this is what your other script expects
# PERMUTATIONS = 1000               # keep small for testing, can increase later
# TOP_N = 100000                    # keep 1000 for quick run, change to 100000 later


# For Malayalam
# EMB_DIR = r"D:\fast_text"
# EMB_FILE = "cc.ml.300.vec"
# OUTPUT_FILE = "word_level_bias_scores_Malayalam.csv"
# PERMUTATIONS = 1000               # keep small for testing, can increase later
# TOP_N = 100000                    # keep 1000 for quick run, change to


# For Telugu
EMB_DIR = r"D:\fast_text"
EMB_FILE = "cc.te.300.vec"    
OUTPUT_FILE = "word_level_bias_scores_Telugu.csv"
PERMUTATIONS = 1000               # keep small for testing, can increase later
TOP_N = 100000                    # keep 1000 for quick run, change to


# For hindi
# EMB_DIR = r"D:\fast_text"
# EMB_FILE = "cc.hi.300.vec"
# OUTPUT_FILE = "word_level_bias_scores_Hindi1.csv"
# PERMUTATIONS = 1000               # keep small for testing, can increase later
# TOP_N = 100000                    # keep 1000 for quick run, change to


# For bengali
# EMB_DIR = r"D:\fast_text"
# EMB_FILE = "cc.bn.300.vec"
# OUTPUT_FILE = "word_level_bias_scores_Bengali.csv"
# PERMUTATIONS = 1000               # keep small for testing, can increase later
# TOP_N = 100000                    # keep 1000 for quick run, change to


# ==== SC-WEAT function ====
def SC_WEAT(w, A, B, permutations=PERMUTATIONS):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B_normed = B / np.linalg.norm(B, axis=-1, keepdims=True)

    A_assoc = w_normed @ A_normed.T
    B_assoc = w_normed @ B_normed.T
    joint = np.concatenate((A_assoc, B_assoc), axis=-1)

    test_stat = np.mean(A_assoc) - np.mean(B_assoc)
    effect_size = test_stat / np.std(joint, ddof=1)

    midpoint = len(A)
    sample_distribution = np.array([np.random.permutation(joint) for _ in range(permutations)])
    sample_assoc = np.mean(sample_distribution[:, :midpoint], axis=1) - np.mean(sample_distribution[:, midpoint:], axis=1)
    p_value = 1 - norm.cdf(test_stat, np.mean(sample_assoc), np.std(sample_assoc, ddof=1))

    return effect_size, p_value

# ==== Load embeddings (skip 1st row for FastText) ====
print("Loading embeddings...")
embedding_df = pd.read_csv(
    path.join(EMB_DIR, EMB_FILE),
    sep=" ",
    header=None,
    index_col=0,
    na_values=None,
    keep_default_na=False,
    skiprows=1,
    quoting=csv.QUOTE_NONE
)
print("Embeddings loaded.")


# ==== Gender attribute words ====

# Kannada
# female_stimuli = ["ಹೆಣ್ಣು", "ಮಹಿಳೆ", "ಹುಡುಗಿ", "ತಂಗಿ", "ಅವಳು", "ಅವಳಿಗೆ", "ಅವಳದು", "ಮಗಳು"]
# male_stimuli   = ["ಗಂಡು", "ಪುರುಷ", "ಹುಡುಗ", "ಅಣ್ಣ", "ಅವನು", "ಅವನಿಗೆ", "ಅವನದು", "ಮಗ"]

# Tamil
# female_stimuli = ["பெண்", "பெண்மணி", "பெண்ணு", "சகோதரி", "அவள்", "அவளை", "அவளுடைய", "மகள்"]
# male_stimuli   = ["ஆண்", "ஆண்மணி", "ஆண்பிள்ளை", "சகோதரன்", "அவன்", "அவனை", "அவனுடைய", "மகன்"]

# Malayalam
# female_stimuli = ["സ്ത്രീ", "അവൾ", "അവളുടെ", "അവളുടേത്", "പെൺകുട്ടി", "മകൾ", "സഹോദരി", "അമ്മ"]
# male_stimuli   = ["പുരുഷൻ", "അവൻ", "അവന്റെ", "അവന്റേത്", "ആൺകുട്ടി", "മകൻ", "സഹോദരൻ", "അച്ഛൻ"]

# Telugu
female_stimuli = ["స్త్రీ", "ఆమె", "ఆమెది", "ఆమెకు", "పాప", "కుమార్తె", "చెల్లి", "అక్క"]
male_stimuli   = ["పురుషుడు", "అతను", "అతని", "అతనికి", "బాలుడు", "కుమారుడు", "తమ్ముడు", "అన్న"]

# Hindi
# female_stimuli = ["महिला", "वह", "उसकी", "उसका", "औरत", "लड़की", "बेटी", "बहन"]
# male_stimuli   = ["पुरुष", "वह", "उसका", "उसकी", "आदमी", "लड़का", "बेटा", "भाई"]

# Bengali
# female_stimuli = ["মহিলা", "সে", "তার", "তাঁর", "নারী", "মেয়ে", "কন্যা", "বোন"]
# male_stimuli   = ["পুরুষ", "সে", "তার", "তাঁর", "পুরুষ", "ছেলে", "পুত্র", "ভাই"]


female_embs = embedding_df.loc[female_stimuli].to_numpy()
male_embs   = embedding_df.loc[male_stimuli].to_numpy()

# ==== Take top N words ====
targets = embedding_df.index.tolist()[:TOP_N]

print(f"Running SC-WEAT on {TOP_N} words...")
bias_array = np.array([
    SC_WEAT(embedding_df.loc[word].to_numpy(), female_embs, male_embs, PERMUTATIONS)
    for word in targets
])

bias_df = pd.DataFrame(bias_array, index=targets, columns=["female_effect_size", "female_p_value"])
bias_df.to_csv(path.join(EMB_DIR, OUTPUT_FILE))

print(f"✅ Done! Saved {OUTPUT_FILE} in {EMB_DIR}")
