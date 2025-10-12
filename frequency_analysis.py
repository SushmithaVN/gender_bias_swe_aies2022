import numpy as np
import pandas as pd
from os import path


# For kannada
# SOURCE_DIR = f'D:\\fast_text'
# SOURCE_FILE = f'word_level_bias_scores_Kannada1.csv'
# WRITE_FILE = f'bias_frequency_ratios_Kannada1.csv'

# For Tamil
# SOURCE_DIR = f'D:\\fast_text'
# SOURCE_FILE = f'word_level_bias_scores_Tamil.csv'
# WRITE_FILE = f'bias_frequency_ratios_Tamil.csv'

# For Malayalam
# SOURCE_DIR = f'D:\\fast_text'
# SOURCE_FILE = f'word_level_bias_scores_Malayalam.csv'
# WRITE_FILE = f'bias_frequency_ratios_Malayalam.csv'

# For Telugu
SOURCE_DIR = f'D:\\fast_text'
SOURCE_FILE = f'word_level_bias_scores_Telugu.csv'
WRITE_FILE = f'bias_frequency_ratios_Telugu.csv'

# For Hindi
# SOURCE_DIR = f'D:\\fast_text'
# SOURCE_FILE = f'word_level_bias_scores_Hindi1.csv'
# WRITE_FILE = f'bias_frequency_ratios_Hindi1.csv'

# For Bengali
# SOURCE_DIR = f'D:\\fast_text'
# SOURCE_FILE = f'word_level_bias_scores_Bengali.csv'
# WRITE_FILE = f'bias_frequency_ratios_Bengali.csv'


#Read in file of associations for top 100k words
source_df = pd.read_csv(path.join(SOURCE_DIR,SOURCE_FILE), na_values=None, keep_default_na=False)

#Female vs. Male ratio by frequency range, effect size range

frequency_ceilings = [100,1000,10000,100000]
effect_size_floors = [0,.2,.5,.8]

es_list = []

for ceiling in frequency_ceilings:
    head_df = source_df.head(ceiling)
    ceiling_counts = [ceiling]

    for es in effect_size_floors:
        es_df = head_df.loc[head_df['female_effect_size'] >= es]
        es_quantity = len(es_df.index.tolist())
        ceiling_counts.append(es_quantity)

    for es in effect_size_floors:
        es_df = head_df.loc[head_df['female_effect_size'] <= -es]
        es_quantity = len(es_df.index.tolist())
        ceiling_counts.append(es_quantity)

    es_list.append(ceiling_counts)

es_arr = np.array(es_list)
cols = ['num_words']+[f'female_{str(i)}' for i in effect_size_floors]+[f'male_{str(i)}' for i in effect_size_floors]
es_df = pd.DataFrame(es_arr,columns=cols)
es_df.to_csv(path.join(SOURCE_DIR,WRITE_FILE))