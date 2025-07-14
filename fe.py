import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import warnings
# import gc

# warnings.filterwarnings('ignore')
# tqdm.pandas()

# print("="*60)
# print("AMEX CAMPUS CHALLENGE 2025 - MEMORY-OPTIMIZED FEATURE ENGINEERING")
# print("="*60)

# # Essential columns that should never be dropped
# ESSENTIAL_COLS = ['id1', 'id2', 'id3', 'id4', 'id5', 'y']

# def load_datasets():
#     """Load datasets with optimized data types from the start"""
#     file_list = [
#         ('train', 'train_data.parquet'),
#         ('test', 'test_data.parquet'),
#         ('events', 'add_event.parquet'),
#         ('transactions', 'add_trans.parquet'),
#         ('offers', 'offer_metadata.parquet'),
#         ('data_dict', 'data_dictionary.csv')
#     ]
    
#     data = {}
#     for name, path in tqdm(file_list, desc="📂 Loading datasets"):
#         if path.endswith('.parquet'):
#             data[name] = pd.read_parquet(path)
#         else:
#             data[name] = pd.read_csv(path)
#         print(f"✓ {name}: {data[name].shape} - Memory: {data[name].memory_usage(deep=True).sum() / 1024**2:.1f} MB")
#     return data

# def optimize_dtypes(df, data_dict, dataset_name):
#     """Optimize data types for memory efficiency"""
#     print(f"\n🔄 Optimizing data types for {dataset_name}...")
    
#     initial_memory = df.memory_usage(deep=True).sum() / 1024**2
#     type_mapping = dict(zip(data_dict['masked_column'], data_dict['Type']))
    
#     for col in tqdm(df.columns, desc=f"Optimizing {dataset_name} types"):
#         if col in type_mapping:
#             col_type = type_mapping[col]
#             try:
#                 if col_type == 'Numerical':
#                     # Use downcast for optimal memory usage
#                     df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
#                 elif col_type == 'Categorical':
#                     # Convert to category for memory efficiency
#                     df[col] = df[col].astype('category')
#                 elif col_type == 'One hot encoded':
#                     # Use smallest integer type for binary data
#                     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int8')
#                 elif col_type == 'Key':
#                     # Keep as string for IDs
#                     df[col] = df[col].astype(str)
#             except Exception as e:
#                 print(f"⚠️ Warning: Could not convert {col}: {e}")
    
#     final_memory = df.memory_usage(deep=True).sum() / 1024**2
#     print(f"✅ Memory optimized: {initial_memory:.1f} MB → {final_memory:.1f} MB ({(initial_memory-final_memory)/initial_memory*100:.1f}% reduction)")
#     gc.collect()
#     return df

# def safe_drop_columns(df, cols_to_drop):
#     """Drop columns one by one to avoid memory spikes"""
#     print(f"\n🧹 Dropping {len(cols_to_drop)} columns safely...")
    
#     for col in tqdm(cols_to_drop, desc="Dropping columns"):
#         if col in df.columns:
#             try:
#                 df.drop(columns=[col], inplace=True)  # In-place operation for memory efficiency
#                 gc.collect()  # Force garbage collection after each drop
#             except Exception as e:
#                 print(f"Could not drop {col}: {e}")
    
#     print(f"✅ Columns after dropping: {len(df.columns)}")
#     return df

# def clean_columns(df, dataset_name):
#     """Clean columns while protecting essential IDs"""
#     print(f"\n🧹 Cleaning {dataset_name} columns...")
#     print(f"📊 Columns before cleaning: {len(df.columns)}")
    
#     # Calculate missing percentages
#     missing_pct = df.isnull().mean()
    
#     # Identify columns to drop (excluding essential columns)
#     high_missing = [col for col in missing_pct[missing_pct > 0.99].index 
#                    if col not in ESSENTIAL_COLS]
    
#     constant_cols = [col for col in tqdm(df.columns, desc="Checking constants") 
#                     if df[col].nunique() <= 1 and col not in ESSENTIAL_COLS]
    
#     cols_to_drop = list(set(high_missing + constant_cols))
    
#     print(f"🔍 High missing columns to drop: {len(high_missing)}")
#     print(f"🔍 Constant columns to drop: {len(constant_cols)}")
#     print(f"🛡️ Protected essential columns: {[col for col in ESSENTIAL_COLS if col in df.columns]}")
    
#     df = safe_drop_columns(df, cols_to_drop)
#     return df

# def create_transaction_features(transactions, train, test):
#     """Create transaction features with memory optimization"""
#     print("\n💳 Creating transaction features...")
    
#     required_cols = {'f370', 'f367', 'id2'}
#     if not required_cols.issubset(transactions.columns):
#         print("⚠️ Required transaction columns missing.")
#         return train, test
    
#     # Optimize transaction data types
#     transactions['f370'] = pd.to_datetime(transactions['f370'], errors='coerce')
#     transactions['f367'] = pd.to_numeric(transactions['f367'], errors='coerce', downcast='float')
#     transactions['id2'] = transactions['id2'].astype(str)
#     train['id2'] = train['id2'].astype(str)
#     test['id2'] = test['id2'].astype(str)
    
#     # Create time-based features
#     max_date = transactions['f370'].max()
#     transactions['days_since_transaction'] = (max_date - transactions['f370']).dt.days.astype('int16')
    
#     # Aggregation functions
#     agg_functions = {
#         'f367': ['sum', 'mean', 'std', 'count', 'max', 'min'],
#         'f368': ['nunique'],
#         'f372': ['nunique']
#     }
    
#     # Create aggregated features
#     all_time_agg = transactions.groupby('id2').agg(agg_functions)
#     all_time_agg.columns = ['_'.join(col).strip() for col in all_time_agg.columns]
#     all_time_agg = all_time_agg.add_prefix('trans_all_')
#     all_time_agg.index = all_time_agg.index.astype(str)
    
#     # Add recency feature
#     last_transaction = transactions.groupby('id2')['days_since_transaction'].min()
#     all_time_agg['trans_days_since_last'] = last_transaction
    
#     # Optimize aggregated features data types
#     for col in all_time_agg.select_dtypes(include=['float64']).columns:
#         all_time_agg[col] = all_time_agg[col].astype('float32')
#     for col in all_time_agg.select_dtypes(include=['int64']).columns:
#         all_time_agg[col] = all_time_agg[col].astype('int32')
    
#     print("🔗 Merging transaction features...")
#     train = train.merge(all_time_agg, left_on='id2', right_index=True, how='left')
#     test = test.merge(all_time_agg, left_on='id2', right_index=True, how='left')
    
#     print(f"✅ Created {len(all_time_agg.columns)} transaction features")
#     del all_time_agg, last_transaction
#     gc.collect()
#     return train, test

# def create_event_features(events, train, test, batch_size=5000):
#     """Create event features with batch processing for memory efficiency"""
#     print("\n📊 Creating customer-level event features (batch-wise merge)...")
    
#     required_cols = {'id4', 'id2', 'id3'}
#     if not required_cols.issubset(events.columns):
#         print("⚠️ Required event columns missing.")
#         return train, test
    
#     # Optimize event data types
#     events['id4'] = pd.to_datetime(events['id4'], errors='coerce')
#     events['id2'] = events['id2'].astype(str)
#     train['id2'] = train['id2'].astype(str)
#     test['id2'] = test['id2'].astype(str)
    
#     # Create event indicators
#     events['impression'] = np.int8(1)
#     events['click'] = np.int8(0)  # Placeholder
    
#     # Time-based features
#     max_date = events['id4'].max()
#     events['days_since_event'] = (max_date - events['id4']).dt.days.astype('int16')
    
#     # Customer-level aggregations
#     customer_events = events.groupby('id2').agg({
#         'impression': 'sum',
#         'click': 'sum',
#         'id3': 'nunique',
#         'days_since_event': 'min'
#     })
    
#     customer_events.columns = ['events_total_impressions', 'events_total_clicks', 
#                               'events_unique_offers', 'events_days_since_last']
#     customer_events['events_overall_ctr'] = customer_events['events_total_clicks'] / (customer_events['events_total_impressions'] + 1)
    
#     # Optimize data types
#     for col in customer_events.select_dtypes(include=['float64']).columns:
#         customer_events[col] = customer_events[col].astype('float32')
#     for col in customer_events.select_dtypes(include=['int64']).columns:
#         customer_events[col] = customer_events[col].astype('int32')
    
#     print("🔗 Merging customer-level event features in batches...")
    
#     # Batch-wise merge for train
#     batches = []
#     for start in tqdm(range(0, len(train), batch_size), desc="Merging train event features"):
#         end = min(start + batch_size, len(train))
#         batch = train.iloc[start:end].copy()
#         batch = batch.merge(customer_events, left_on='id2', right_index=True, how='left')
#         batches.append(batch)
#         del batch
#         gc.collect()
    
#     train = pd.concat(batches, ignore_index=True)
#     del batches
#     gc.collect()
    
#     # Batch-wise merge for test
#     batches = []
#     for start in tqdm(range(0, len(test), batch_size), desc="Merging test event features"):
#         end = min(start + batch_size, len(test))
#         batch = test.iloc[start:end].copy()
#         batch = batch.merge(customer_events, left_on='id2', right_index=True, how='left')
#         batches.append(batch)
#         del batch
#         gc.collect()
    
#     test = pd.concat(batches, ignore_index=True)
#     del batches, customer_events
#     gc.collect()
    
#     print(f"✅ Created customer-level event features")
#     return train, test

# def create_offer_features(offers, train, test, batch_size=5000):
#     """Create offer features with batch processing"""
#     print("\n🎯 Creating offer features...")
    
#     # Optimize data types
#     offers['id3'] = offers['id3'].astype(str)
#     train['id3'] = train['id3'].astype(str)
#     test['id3'] = test['id3'].astype(str)
    
#     # Date processing
#     offers['id12'] = pd.to_datetime(offers['id12'], errors='coerce')
#     offers['id13'] = pd.to_datetime(offers['id13'], errors='coerce')
#     offers['offer_duration_days'] = (offers['id13'] - offers['id12']).dt.days.astype('float32')
#     offers['f376'] = pd.to_numeric(offers['f376'], errors='coerce', downcast='float')
    
#     # Select features
#     offer_features = ['id3', 'f376', 'offer_duration_days', 'f375', 'id9']
#     available_features = [col for col in offer_features if col in offers.columns]
    
#     print("🔗 Merging offer features in batches...")
    
#     # Prepare offers for merging
#     offers = offers[available_features].drop_duplicates(subset='id3').set_index('id3')
    
#     # Batch merge for train
#     batches = []
#     for start in tqdm(range(0, len(train), batch_size), desc="Merging train offer features"):
#         end = min(start + batch_size, len(train))
#         batch = train.iloc[start:end].copy()
#         batch = batch.join(offers, on='id3', how='left')
#         batches.append(batch)
#         del batch
#         gc.collect()
    
#     train = pd.concat(batches, ignore_index=True)
#     del batches
#     gc.collect()
    
#     # Batch merge for test
#     batches = []
#     for start in tqdm(range(0, len(test), batch_size), desc="Merging test offer features"):
#         end = min(start + batch_size, len(test))
#         batch = test.iloc[start:end].copy()
#         batch = batch.join(offers, on='id3', how='left')
#         batches.append(batch)
#         del batch
#         gc.collect()
    
#     test = pd.concat(batches, ignore_index=True)
#     del batches, offers
#     gc.collect()
    
#     print(f"✅ Added {len(available_features) - 1} offer features")
#     return train, test

# def create_temporal_features(df):
#     """Create temporal features with safe NaN handling"""
#     print("\n⏰ Creating temporal features...")
    
#     # Safe datetime parsing
#     for col in ['id4', 'id5']:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
    
#     if 'id4' in df.columns:
#         # Extract and safely cast – NaT rows receive -1
#         df['hour'] = df['id4'].dt.hour.fillna(-1).astype('int8')
#         df['day_of_week'] = df['id4'].dt.dayofweek.fillna(-1).astype('int8')
#         df['month'] = df['id4'].dt.month.fillna(-1).astype('int8')
        
#         # Boolean features (NaN becomes False)
#         df['is_weekend'] = (df['day_of_week'] >= 5).fillna(False).astype('int8')
#         df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).fillna(False).astype('int8')
        
#         # Timestamp numeric (keep as int64 to avoid overflow)
#         df['timestamp_numeric'] = df['id4'].astype('int64', errors='ignore') // 10**9
#         df['timestamp_numeric'] = df['timestamp_numeric'].fillna(-1).astype('int64')
    
#     gc.collect()
#     print("✅ Created temporal features (safe casting)")
#     return df

# def create_interaction_features(df):
#     """Create interaction features with memory optimization"""
#     print("\n🔗 Creating interaction features...")
    
#     if 'f53' in df.columns and 'f376' in df.columns:
#         df['customer_value_x_discount'] = (pd.to_numeric(df['f53'], errors='coerce') * 
#                                           pd.to_numeric(df['f376'], errors='coerce')).astype('float32')
    
#     if 'trans_all_f367_mean' in df.columns and 'events_overall_ctr' in df.columns:
#         df['avg_spend_x_ctr'] = (df['trans_all_f367_mean'] * df['events_overall_ctr']).astype('float32')
    
#     gc.collect()
#     print("✅ Created interaction features")
#     return df

# def handle_missing_values(df, strategy='median'):
#     """Handle missing values efficiently"""
#     print("\n🔧 Handling missing values...")
    
#     initial_missing = df.isnull().sum().sum()
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
#     # Handle numeric columns
#     if strategy == 'median':
#         for col in tqdm(numeric_cols, desc="Numeric imputation"):
#             df[col].fillna(df[col].median(), inplace=True)  # In-place for memory efficiency
#     elif strategy == 'mean':
#         for col in tqdm(numeric_cols, desc="Numeric imputation"):
#             df[col].fillna(df[col].mean(), inplace=True)
#     else:
#         df[numeric_cols] = df[numeric_cols].fillna(0)
    
#     # Handle categorical columns
#     for col in tqdm(categorical_cols, desc="Categorical imputation"):
#         if pd.api.types.is_categorical_dtype(df[col]):
#             if 'Unknown' not in df[col].cat.categories:
#                 df[col] = df[col].cat.add_categories('Unknown')
#         df[col].fillna('Unknown', inplace=True)
    
#     final_missing = df.isnull().sum().sum()
#     print(f"✅ Reduced missing values: {initial_missing:,} → {final_missing:,}")
#     gc.collect()
#     return df

# def prepare_final_features(train, test):
#     """Prepare final features ensuring id5 is preserved"""
#     print("\n🎯 Preparing final features...")
    
#     # Debug: Check available columns
#     print(f"📊 Available columns in train: {len(train.columns)}")
#     print(f"📊 Available columns in test: {len(test.columns)}")
    
#     # Define ID columns and target
#     id_cols = ['id1', 'id2', 'id3', 'id5']  # Note: id5 is explicitly included
#     target_col = 'y'
    
#     # Check which ID columns exist
#     existing_train_ids = [col for col in id_cols if col in train.columns]
#     existing_test_ids = [col for col in id_cols if col in test.columns]
    
#     print(f"🔍 ID columns found in train: {existing_train_ids}")
#     print(f"🔍 ID columns found in test: {existing_test_ids}")
    
#     # Warn about missing ID columns
#     missing_train_ids = list(set(id_cols) - set(existing_train_ids))
#     missing_test_ids = list(set(id_cols) - set(existing_test_ids))
    
#     if missing_train_ids:
#         print(f"⚠️ Warning: Missing ID columns in train: {missing_train_ids}")
#     if missing_test_ids:
#         print(f"⚠️ Warning: Missing ID columns in test: {missing_test_ids}")
    
#     # Prepare feature columns
#     feature_cols = [col for col in train.columns if col not in id_cols and col != target_col]
#     common_features = [col for col in feature_cols if col in test.columns]
    
#     # Create final datasets
#     X_train = train[common_features]
#     y_train = train[target_col] if target_col in train.columns else None
#     X_test = test[common_features]
    
#     # Create ID datasets (use all available ID columns)
#     train_ids = train[existing_train_ids]
#     test_ids = test[existing_test_ids]
    
#     print(f"✅ Final feature matrix: {X_train.shape}")
#     print(f"✅ Features: {len(common_features)}")
#     print(f"✅ Train IDs shape: {train_ids.shape}")
#     print(f"✅ Test IDs shape: {test_ids.shape}")
    
#     gc.collect()
#     return X_train, y_train, X_test, train_ids, test_ids

# def main_feature_engineering():
#     """Main feature engineering pipeline with memory optimization"""
#     print("🚀 Starting memory-optimized feature engineering...")
    
#     # Load datasets
#     data = load_datasets()
#     train = data['train']
#     test = data['test']
#     events = data['events']
#     transactions = data['transactions']
#     offers = data['offers']
#     data_dict = data['data_dict']
    
#     print(f"\n📊 Initial data shapes:")
#     print(f"Train: {train.shape}, Test: {test.shape}")
#     print(f"Events: {events.shape}, Transactions: {transactions.shape}, Offers: {offers.shape}")
    
#     # Optimize data types
#     train = optimize_dtypes(train, data_dict, 'train')
#     test = optimize_dtypes(test, data_dict, 'test')
    
#     # Clean columns (protecting essential IDs)
#     train = clean_columns(train, 'train')
#     test = clean_columns(test, 'test')
    
#     # Create features
#     train, test = create_transaction_features(transactions, train, test)
#     train, test = create_event_features(events, train, test, batch_size=5000)
#     train, test = create_offer_features(offers, train, test, batch_size=5000)
    
#     # Create temporal and interaction features
#     train = create_temporal_features(train)
#     test = create_temporal_features(test)
#     train = create_interaction_features(train)
#     test = create_interaction_features(test)
    
#     # Handle missing values
#     train = handle_missing_values(train)
#     test = handle_missing_values(test)
    
#     # Prepare final features
#     X_train, y_train, X_test, train_ids, test_ids = prepare_final_features(train, test)
    
#     # Save processed data
#     print("💾 Saving processed data...")
#     X_train.to_parquet('X_train_features.parquet', compression='snappy')
#     X_test.to_parquet('X_test_features.parquet', compression='snappy')
#     train_ids.to_parquet('train_ids.parquet', compression='snappy')
#     test_ids.to_parquet('test_ids.parquet', compression='snappy')
    
#     if y_train is not None:
#         pd.DataFrame({'y': y_train}).to_parquet('y_train.parquet', compression='snappy')
    
#     print("\n✅ Feature engineering complete!")
#     print(f"📊 Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
#     print(f"🎯 Target distribution: {y_train.value_counts().to_dict() if y_train is not None else 'N/A'}")
#     print(f"🆔 Train IDs: {train_ids.shape}, Test IDs: {test_ids.shape}")
    
#     # Final memory cleanup
#     del train, test, events, transactions, offers
#     gc.collect()
    
#     return X_train, y_train, X_test, train_ids, test_ids

# if __name__ == "__main__":
#     print("🚀 Starting Amex Campus Challenge 2025 Memory-Optimized Feature Engineering...")
#     X_train, y_train, X_test, train_ids, test_ids = main_feature_engineering()
    
#     print("\n" + "="*60)
#     print("🎉 MEMORY-OPTIMIZED FEATURE ENGINEERING COMPLETED!")
#     print("="*60)
#     print("\n📁 Files created:")
#     print("- X_train_features.parquet")
#     print("- X_test_features.parquet") 
#     print("- y_train.parquet")
#     print("- train_ids.parquet")
#     print("- test_ids.parquet")
#     print("\n🚀 Ready for modeling with id5 preserved!")
import pandas as pd

# Load the test data
df = pd.read_parquet('test_data.parquet')

# Check if 'id5' exists
if 'id5' in df.columns:
    print("✅ 'id5' column found in test_data.parquet")
    
    # Print dtype and missing percentage
    print(f"🕒 id5 dtype: {df['id5'].dtype}")
    print(f"❓ Missing values in id5: {df['id5'].isnull().sum()} / {len(df)} ({df['id5'].isnull().mean()*100:.2f}%)")

    # Show value counts or top unique entries
    print("\n🔍 Sample id5 values:")
    print(df['id5'].dropna().unique()[:5])
else:
    print("❌ 'id5' column NOT FOUND in test_data.parquet")
import pandas as pd

# Load original test data
test_data = pd.read_parquet('test_data.parquet')

# Extract ID columns (ensure id5 is included)
id_cols = ['id1', 'id2', 'id3', 'id5']
id_df = test_data[id_cols].copy()

# Save the fixed ID file
id_df.to_parquet('test_ids.parquet', index=False)
print("✅ test_ids.parquet regenerated with id5 included.")
