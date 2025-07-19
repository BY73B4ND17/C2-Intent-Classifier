import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('c2_intent_dataset.csv')

train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['intent'], random_state=42)

test1_df, test2_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['intent'], random_state=42)

test3_df, _ = train_test_split(train_df, test_size=0.9, stratify=train_df['intent'], random_state=42)

test1_df.to_csv('test_c2_dataset_1.csv', index=False)
test2_df.to_csv('test_c2_dataset_2.csv', index=False)
test3_df.to_csv('test_c2_dataset_3.csv', index=False)

print("Created three test datasets: test_c2_dataset_1.csv, test_c2_dataset_2.csv, test_c2_dataset_3.csv")
