import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

tqdm.pandas()  # Enable progress_apply

# === DISPLAY SETTINGS ===
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
sns.set_style("whitegrid")
sns.set_palette("husl")

# === STEP 0: Load Data ===
print("Loading data...")
try:
    train = pd.read_parquet("train_data.parquet")
    test = pd.read_parquet("test_data.parquet")
    events = pd.read_parquet("add_event.parquet")
    transactions = pd.read_parquet("add_trans.parquet")
    offers = pd.read_parquet("offer_metadata.parquet")
    data_dict = pd.read_csv("data_dictionary.csv")
    print("Files loaded successfully.")
except Exception as e:
    print("Failed to load some files:", e)
    exit()

# === STEP 1: Auto-detect relevant columns ===
column_candidates = [col for col in data_dict.columns if 'column' in col.lower()]
type_candidates = [col for col in data_dict.columns if 'type' in col.lower()]
if column_candidates and type_candidates:
    column_name_col = column_candidates[0]
    type_col = type_candidates[0]
    print(f"Auto-selected: column name = '{column_name_col}', type = '{type_col}'")
else:
    raise ValueError("Could not detect suitable columns in data_dictionary.csv")

# === STEP 2: Coerce Types ===
def coerce_column_types(df, dictionary, col_col, type_col):
    for _, row in tqdm(dictionary.iterrows(), total=len(dictionary), desc="Coercing column types"):
        col = row[col_col]
        if col not in df.columns:
            continue
        dtype = str(row[type_col]).strip().lower()
        if dtype == 'numerical':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype in ['categorical', 'one hot encoded']:
            df[col] = df[col].astype('category')
        elif dtype == 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

train = coerce_column_types(train, data_dict, column_name_col, type_col)
test = coerce_column_types(test, data_dict, column_name_col, type_col)
events = coerce_column_types(events, data_dict, column_name_col, type_col)
transactions = coerce_column_types(transactions, data_dict, column_name_col, type_col)
offers = coerce_column_types(offers, data_dict, column_name_col, type_col)

# === STEP 3: Summarize Columns ===
def summarize_from_dictionary(df, dictionary, name, col_col, type_col):
    print(f"\nFull Column Summary — {name}")
    summary_rows = []
    for _, row in tqdm(dictionary.iterrows(), total=len(dictionary), desc=f"Summarizing {name}"):
        col = row[col_col]
        expected_type = row[type_col]
        if col not in df.columns:
            summary_rows.append({
                'Column': col, 'Exists': False, 'Expected Type': expected_type,
                'Actual Dtype': 'N/A', '% Missing': 'N/A',
                '# Unique': 'N/A', 'Sample Values': 'N/A'
            })
            continue
        col_data = df[col]
        summary_rows.append({
            'Column': col, 'Exists': True, 'Expected Type': expected_type,
            'Actual Dtype': col_data.dtype.name,
            '% Missing': f"{col_data.isnull().mean()*100:.2f}%",
            '# Unique': col_data.nunique(dropna=True),
            'Sample Values': col_data.dropna().unique()[:3].tolist()
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df[['Column', 'Exists', 'Expected Type', 'Actual Dtype', '% Missing', '# Unique', 'Sample Values']].to_string(index=False))
    return summary_df

for name, df in tqdm(
    zip(['Train', 'Test', 'Events', 'Transactions', 'Offers'], [train, test, events, transactions, offers]),
    total=5, desc="Summarizing all datasets"
):
    summarize_from_dictionary(df, data_dict, name, column_name_col, type_col)

# === STEP 4: Drop useless columns ===
def drop_useless_columns(df, name, threshold=0.995):
    nulls = df.isnull().mean()
    constant = df.nunique(dropna=False) <= 1
    to_drop = nulls[nulls > threshold].index.tolist() + constant[constant].index.tolist()
    df = df.drop(columns=to_drop)
    if to_drop:
        print(f"{name} — Dropped {len(to_drop)} columns with >{int(threshold*100)}% missing or constant.")
    return df

train = drop_useless_columns(train, "Train")
test = drop_useless_columns(test, "Test")

# === Prepare datasets dictionary for easy iteration ===
datasets = {
    'Train': train,
    'Test': test,
    'Events': events,
    'Transactions': transactions,
    'Offers': offers
}

# === STEP 5: Core EDA Function ===
def clean_eda(df, name):
    print(f"\n{name} Summary:")
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print("Top Missing Columns:")
        print((nulls / len(df) * 100).sort_values(ascending=False).head(10).round(2))
    else:
        print("No missing values.")
    print("Data Types:")
    print(df.dtypes.value_counts())
    dupes = df.duplicated().sum()
    print(f"Duplicates: {dupes} ({dupes / len(df):.2%})")

# === STEP 6: EDA on All Datasets ===
print("\n" + "="*60)
print("DATASET QUALITY ASSESSMENT")
print("="*60)

for name, df in tqdm(datasets.items(), desc="Running clean EDA"):
    clean_eda(df, name)

# === STEP 7: Detect Target-Like Columns ===
print("\nTarget Column Candidates:")
target_candidates = [col for col in train.columns if pd.api.types.is_numeric_dtype(train[col]) and train[col].nunique() <= 10]
if target_candidates:
    for col in target_candidates:
        print(f"\n{col} distribution:")
        dist = train[col].value_counts(normalize=True).sort_index()
        for val, frac in dist.items():
            print(f"  {val}: {frac:.2%}")
        print(f"  Mean: {train[col].mean():.4f}")
else:
    print("No binary/target-like columns found.")

# === STEP 8: Detect ID-Like Columns ===
print("\nID Column Candidates:")
id_columns = [col for col in train.columns if train[col].nunique() > 0.9 * len(train) and train[col].dtype in ['object', 'category']]
if id_columns:
    for col in id_columns:
        print(f"{col}: {train[col].nunique():,} unique values")
else:
    print("No ID-like columns detected.")

# === DONE ===
print("\nEDA complete. All outputs are clean and assumption-free.")