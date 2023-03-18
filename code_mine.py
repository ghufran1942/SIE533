# Import Library
import pandas as pd
import numpy as np

# Data Initiatization
# df = pd.read_csv("data/content.csv")


# Conditioning

# condition = (df["language"]=='en')
# filtered_df = df[condition]

# pd.set_option('display.max_rows', None)  # Set the maximum number of rows to display to None (i.e., unlimited)
# pd.set_option('display.max_columns', None)  # Set the maximum number of columns to display to None (i.e., unlimited)
# pd.set_option('display.expand_frame_repr', False)  # Disable line wrapping for the DataFrame representation

# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
# pd.reset_option('display.expand_frame_repr')

# filtered_df.to_csv('data/flitered_df.csv', index=False)

# print(filtered_df["title"])


# Importing the Conditioned data as df.

df = pd.read_csv("data/filtered_df.csv")
print(df)