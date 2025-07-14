# enhanced_ammodel.py — Improved ML Model Training + Submission Script
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import logging
from typing import Dict, Tuple, Any
import json
from tqdm import tqdm

# Custom transformer for boolean columns
class BooleanToIntTransformer(BaseEstimator, TransformerMixin):
    """Convert boolean columns to integer (0, 1) for proper imputation"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.astype(int)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

# === CONFIG ===
class Config:
    TRAIN_PATH = "train_features.parquet"
    TEST_PATH = "test_features.parquet"
    RAW_TRAIN_PATH = "train_data.parquet"
    TEST_IDS_PATH = "test_data.parquet"
    TARGET_COL = "y"
    ID_COLS = ["id1", "id2", "id3", "id5"]
    SUBMISSION_PATH = "submission.csv"
    MODEL_PATH = "trained_model.pkl"
    RESULTS_PATH = "model_results.json"
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Model configurations
    MODELS = {
        'hist_gradient_boost': {
            'model': HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=RANDOM_STATE
            ),
            'name': 'Histogram Gradient Boosting'
        },
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'name': 'Random Forest'
        },
        'logistic_regression': {
            'model': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),
            'name': 'Logistic Regression'
        }
    }

class DataLoader:
    """Handles data loading and validation"""
    
    @staticmethod
    def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate training and test data"""
        try:
            logger.info("📥 Loading processed features...")
            train = pd.read_parquet(config.TRAIN_PATH)
            test = pd.read_parquet(config.TEST_PATH)
            
            # Validate data shapes
            if train.empty or test.empty:
                raise ValueError("❌ Loaded data is empty")
                
            logger.info(f"✅ Loaded: Train {train.shape}, Test {test.shape}")
            return train, test
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise

    @staticmethod
    def restore_target(train: pd.DataFrame, config: Config) -> pd.DataFrame:
        """Restore target column if missing"""
        if config.TARGET_COL not in train.columns:
            logger.info(f"🔁 Target column '{config.TARGET_COL}' missing. Attempting to restore...")
            try:
                raw_train = pd.read_parquet(config.RAW_TRAIN_PATH)
                if config.TARGET_COL not in raw_train.columns:
                    raise ValueError(f"❌ '{config.TARGET_COL}' column not found in raw data either.")
                
                train[config.TARGET_COL] = pd.to_numeric(raw_train[config.TARGET_COL], errors='coerce')
                logger.info(f"✅ Restored '{config.TARGET_COL}' from raw data")
                
            except Exception as e:
                logger.error(f"❌ Failed to restore target: {e}")
                raise
                
        return train

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def _init_(self, config: Config):
        self.config = config
        self.results = {}
        self.best_model = None
        self.best_score = 0.0
        
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create preprocessing pipeline"""
        # Separate different data types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        boolean_features = X.select_dtypes(include=['bool']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove boolean features from numeric features if they were included
        numeric_features = [col for col in numeric_features if col not in boolean_features]
        
        transformers = []
        
        # Numeric transformer
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        # Boolean transformer - convert to int and then impute
        if boolean_features:
            boolean_transformer = Pipeline(steps=[
                ('bool_to_int', BooleanToIntTransformer()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('bool', boolean_transformer, boolean_features))
        
        # Categorical transformer
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        if not transformers:
            # If no transformers, create a passthrough
            logger.warning("⚠ No transformers created, using passthrough")
            from sklearn.preprocessing import FunctionTransformer
            return Pipeline([('passthrough', FunctionTransformer())])
        
        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor
    
    def train_and_evaluate_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_val: pd.DataFrame, y_val: pd.Series, 
                                model_name: str, model_config: Dict) -> Dict[str, Any]:
        """Train and evaluate a single model"""
        logger.info(f"⚙ Training {model_config['name']}...")
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X_train)
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_config['model'])
        ])
        
        start_time = time.time()
        
        # Train model with progress bar
        with tqdm(total=100, desc=f"Training {model_config['name']}", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # Simulate training progress for display
            pbar.update(10)  # Preprocessing
            
            # Train model
            pipeline.fit(X_train, y_train)
            pbar.update(60)  # Training
            
            # Validation predictions
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            pbar.update(15)  # Validation
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_val, y_proba)
            pbar.update(10)  # Metrics
            
            # Cross-validation with progress
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE),
                scoring='roc_auc'
            )
            pbar.update(5)  # CV complete
        
        training_time = time.time() - start_time
        
        results = {
            'model_name': model_config['name'],
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'pipeline': pipeline
        }
        
        logger.info(f"✅ {model_config['name']} - ROC AUC: {roc_auc:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Update best model
        if roc_auc > self.best_score:
            self.best_score = roc_auc
            self.best_model = pipeline
            
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train and compare all models"""
        logger.info("🚀 Starting model training and comparison...")
        
        all_results = {}
        
        # Create progress bar for overall training
        model_progress = tqdm(self.config.MODELS.items(), 
                            desc="Training Models", 
                            unit="model",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} models [{elapsed}<{remaining}]')
        
        for model_name, model_config in model_progress:
            model_progress.set_postfix(current=model_config['name'])
            
            try:
                results = self.train_and_evaluate_model(
                    X_train, y_train, X_val, y_val, model_name, model_config
                )
                all_results[model_name] = results
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_config['name']}: {e}")
                continue
        
        # Find best model - handle case where no models were trained successfully
        if not all_results:
            logger.error("❌ No models were trained successfully!")
            raise ValueError("No models were trained successfully")
            
        best_model_name = max(all_results.keys(), key=lambda x: all_results[x]['roc_auc'])
        logger.info(f"🏆 Best model: {all_results[best_model_name]['model_name']} "
                   f"with ROC AUC: {all_results[best_model_name]['roc_auc']:.4f}")
        
        return all_results
    
    def detailed_evaluation(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Perform detailed evaluation of best model"""
        if self.best_model is None:
            logger.error("❌ No best model found for detailed evaluation")
            return
            
        logger.info("📊 Detailed Validation Metrics:")
        
        y_pred = self.best_model.predict(X_val)
        y_proba = self.best_model.predict_proba(X_val)[:, 1]
        
        # Classification report
        print(classification_report(y_val, y_pred, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # ROC AUC
        roc_auc = roc_auc_score(y_val, y_proba)
        logger.info(f"🎯 ROC AUC Score: {roc_auc:.4f}")

class PredictionGenerator:
    """Handles test predictions and submission generation"""
    
    @staticmethod
    def align_test_features(test: pd.DataFrame, train_columns: list) -> pd.DataFrame:
        """Align test features with training features"""
        logger.info("🔄 Aligning test features with train...")
        
        missing_cols = set(train_columns) - set(test.columns)
        extra_cols = set(test.columns) - set(train_columns)
        
        if missing_cols:
            logger.warning(f"⚠ Missing columns in test: {missing_cols}")
            for col in missing_cols:
                test[col] = 0
        
        if extra_cols:
            logger.warning(f"⚠ Extra columns in test: {extra_cols}")
            
        # Ensure same order
        test = test[train_columns]
        return test
    
    @staticmethod
    def generate_submission(model: Pipeline, test: pd.DataFrame, config: Config) -> None:
        """Generate final submission file"""
        logger.info("🔍 Predicting on test set...")
        
        try:
            # Predict with progress bar
            with tqdm(total=100, desc="Generating Predictions", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                pbar.update(20)  # Start prediction
                test_proba = model.predict_proba(test)[:, 1]
                pbar.update(60)  # Prediction complete
                
                # Load original test IDs
                logger.info("📦 Constructing final submission...")
                test_ids = pd.read_parquet(config.TEST_IDS_PATH)[config.ID_COLS]
                pbar.update(10)  # IDs loaded
                
                if len(test_ids) != len(test_proba):
                    raise ValueError("❌ Length mismatch between test IDs and predictions.")
                
                submission = test_ids.copy()
                submission["pred"] = test_proba
                pbar.update(5)  # Submission prepared
                
                # Save submission
                submission.to_csv(config.SUBMISSION_PATH, index=False)
                pbar.update(5)  # File saved
            
            logger.info(f"✅ Saved submission to {config.SUBMISSION_PATH}")
            
            # Basic statistics
            logger.info(f"📈 Prediction statistics:")
            logger.info(f"   Mean: {test_proba.mean():.4f}")
            logger.info(f"   Std: {test_proba.std():.4f}")
            logger.info(f"   Min: {test_proba.min():.4f}")
            logger.info(f"   Max: {test_proba.max():.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate submission: {e}")
            raise

def save_results(results: Dict, model: Pipeline, config: Config) -> None:
    """Save model and results"""
    try:
        # Save model
        joblib.dump(model, config.MODEL_PATH)
        logger.info(f"💾 Saved model to {config.MODEL_PATH}")
        
        # Save results (excluding pipeline objects)
        results_to_save = {}
        for model_name, model_results in results.items():
            results_to_save[model_name] = {
                'model_name': model_results['model_name'],
                'roc_auc': model_results['roc_auc'],
                'cv_mean': model_results['cv_mean'],
                'cv_std': model_results['cv_std'],
                'training_time': model_results['training_time']
            }
        
        with open(config.RESULTS_PATH, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        logger.info(f"📊 Saved results to {config.RESULTS_PATH}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")

def main():
    """Main execution function"""
    try:
        # Initialize configuration
        config = Config()
        
        # Load data
        data_loader = DataLoader()
        train, test = data_loader.load_data(config)
        train = data_loader.restore_target(train, config)
        
        # Prepare features and target
        with tqdm(total=100, desc="Data Preparation", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            X = train.drop(columns=[config.TARGET_COL])
            y = train[config.TARGET_COL]
            pbar.update(40)
            
            # Handle missing values in target
            if y.isnull().any():
                logger.warning("⚠ Found missing values in target, dropping rows...")
                mask = ~y.isnull()
                X = X[mask]
                y = y[mask]
            pbar.update(30)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
            )
            pbar.update(30)
        
        logger.info(f"✅ Split: Train ({X_train.shape[0]}, {X_train.shape[1]}), "
                   f"Val ({X_val.shape[0]}, {X_val.shape[1]})")
        
        # Train models
        trainer = ModelTrainer(config)
        results = trainer.train_all_models(X_train, y_train, X_val, y_val)
        
        # Check if any models were trained successfully
        if not results:
            logger.error("❌ No models were trained successfully. Falling back to simple model.")
            # Fallback to a simple model without preprocessing
            from sklearn.ensemble import HistGradientBoostingClassifier
            
            # Convert boolean columns to int manually
            X_train_clean = X_train.copy()
            X_val_clean = X_val.copy()
            test_clean = test.copy()
            
            # Convert boolean columns to int
            bool_cols = X_train_clean.select_dtypes(include=['bool']).columns
            for col in bool_cols:
                X_train_clean[col] = X_train_clean[col].astype(int)
                X_val_clean[col] = X_val_clean[col].astype(int)
                test_clean[col] = test_clean[col].astype(int)
            
            # Simple model training
            logger.info("⚙ Training fallback model...")
            
            with tqdm(total=100, desc="Fallback Training", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                fallback_model = HistGradientBoostingClassifier(
                    max_iter=100, 
                    learning_rate=0.1, 
                    random_state=config.RANDOM_STATE
                )
                pbar.update(10)
                
                fallback_model.fit(X_train_clean, y_train)
                pbar.update(60)
                
                # Evaluate fallback model
                y_pred = fallback_model.predict(X_val_clean)
                y_proba = fallback_model.predict_proba(X_val_clean)[:, 1]
                roc_auc = roc_auc_score(y_val, y_proba)
                pbar.update(20)
                
                # Generate submission with fallback model
                test_aligned = PredictionGenerator.align_test_features(test_clean, X_train_clean.columns.tolist())
                pbar.update(10)
            
            logger.info(f"✅ Fallback model ROC AUC: {roc_auc:.4f}")
            print(classification_report(y_val, y_pred, digits=4))
            
            logger.info("🔍 Predicting on test set with fallback model...")
            test_proba = fallback_model.predict_proba(test_aligned)[:, 1]
            
            # Create submission
            test_ids = pd.read_parquet(config.TEST_IDS_PATH)[config.ID_COLS]
            submission = test_ids.copy()
            submission["pred"] = test_proba
            submission.to_csv(config.SUBMISSION_PATH, index=False)
            logger.info(f"✅ Saved fallback submission to {config.SUBMISSION_PATH}")
            
        else:
            # Normal flow with successful model training
            # Detailed evaluation
            trainer.detailed_evaluation(X_val, y_val)
            
            # Align test features
            test_aligned = PredictionGenerator.align_test_features(test, X_train.columns.tolist())
            
            # Generate submission
            Pr