#!/usr/bin/env python3
"""
Machine Learning Model Training and Testing Script

This script:
1. Loads and combines all original datasets
2. Removes highly correlated features
3. Trains multiple ML models
4. Tests the models on adversarial datasets
5. Compares performance between original and adversarial data
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


class NetworkTrafficMLPipeline:
    """Complete ML pipeline for network traffic classification"""
    
    def __init__(self, original_data_dir: str, adversarial_data_dir: str, 
                 results_dir: str = "ml_results"):
        self.original_data_dir = Path(original_data_dir)
        self.adversarial_data_dir = Path(adversarial_data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # ML components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.feature_names = []
        self.removed_features = []
        
        # Results storage
        self.results = {
            'original_performance': {},
            'adversarial_performance': {},
            'feature_analysis': {},
            'correlation_analysis': {}
        }
    
    def load_and_combine_datasets(self, data_dir: Path, dataset_type: str = "original") -> pd.DataFrame:
        """Load and combine all CSV files from a directory"""
        print(f"\nLoading {dataset_type} datasets from {data_dir}")
        
        all_data = []
        csv_files = list(data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
        for csv_file in csv_files:
            print(f"  Loading: {csv_file.name}")
            df = pd.read_csv(csv_file)
            
            # Add source file info for tracking
            df['source_file'] = csv_file.stem
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  Combined dataset shape: {combined_df.shape}")
        print(f"  Label distribution:\n{combined_df['Label'].value_counts()}")
        
        return combined_df
    
    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for ML training/testing"""
        print(f"\nPreprocessing data (training={is_training})")
        
        # Remove non-feature columns
        non_feature_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'source_file']
        feature_cols = [col for col in df.columns if col not in non_feature_cols + ['Label']]
        
        # Extract features and labels
        X = df[feature_cols].copy()
        y = df['Label'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Convert to numeric (handle any string columns)
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(X[col].median())
                except:
                    print(f"Warning: Could not convert {col} to numeric")
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        if is_training:
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Use existing encoders/scalers
            # Ensure same features as training
            missing_features = set(self.feature_names) - set(X.columns)
            extra_features = set(X.columns) - set(self.feature_names)
            
            if missing_features:
                print(f"Warning: Missing features in test data: {missing_features}")
                for feat in missing_features:
                    X[feat] = 0
            
            if extra_features:
                print(f"Warning: Extra features in test data (removing): {extra_features}")
                X = X.drop(columns=list(extra_features))
            
            # Reorder columns to match training
            X = X[self.feature_names]
            
            y_encoded = self.label_encoder.transform(y)
            X_scaled = self.scaler.transform(X)
        
        print(f"  Features shape: {X_scaled.shape}")
        print(f"  Labels shape: {y_encoded.shape}")
        print(f"  Unique labels: {np.unique(y_encoded)}")
        
        return X_scaled, y_encoded
    
    def remove_correlated_features(self, X: np.ndarray, correlation_threshold: float = 0.95) -> np.ndarray:
        """Remove highly correlated features"""
        print(f"\nRemoving features with correlation > {correlation_threshold}")
        
        # Create DataFrame for correlation analysis
        df_features = pd.DataFrame(X, columns=self.feature_names)
        
        # Calculate correlation matrix
        corr_matrix = df_features.corr().abs()
        
        # Find highly correlated feature pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to remove
        features_to_remove = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > correlation_threshold)
        ]
        
        self.removed_features = features_to_remove
        remaining_features = [f for f in self.feature_names if f not in features_to_remove]
        
        print(f"  Removed {len(features_to_remove)} highly correlated features")
        print(f"  Remaining features: {len(remaining_features)}")
        
        # Store correlation analysis
        self.results['correlation_analysis'] = {
            'threshold': correlation_threshold,
            'removed_features': features_to_remove,
            'remaining_features': remaining_features,
            'correlation_matrix_shape': corr_matrix.shape
        }
        
        # Update feature names and return filtered data
        self.feature_names = remaining_features
        feature_indices = [i for i, f in enumerate(df_features.columns) if f in remaining_features]
        
        return X[:, feature_indices]
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train multiple ML models"""
        print("\nTraining ML models...")
        
        # Define models
        model_configs = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=10
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                multi_class='ovr'
            )
        }
        
        trained_models = {}
        performance_results = {}
        
        for model_name, model in model_configs.items():
            print(f"  Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                trained_models[model_name] = model
                performance_results[model_name] = metrics
                
                print(f"    Accuracy: {metrics['accuracy']:.4f} (+/- {metrics['cv_std']*2:.4f})")
                
            except Exception as e:
                print(f"    Error training {model_name}: {str(e)}")
                continue
        
        self.models = trained_models
        return performance_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add AUC if probabilities available and multiclass
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def test_on_adversarial(self, X_adv: np.ndarray, y_adv: np.ndarray) -> Dict[str, Any]:
        """Test trained models on adversarial data"""
        print("\nTesting models on adversarial data...")
        
        adversarial_results = {}
        
        for model_name, model in self.models.items():
            print(f"  Testing {model_name}...")
            
            try:
                # Predict on adversarial data
                y_pred = model.predict(X_adv)
                y_pred_proba = model.predict_proba(X_adv) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_adv, y_pred, y_pred_proba)
                adversarial_results[model_name] = metrics
                
                print(f"    Adversarial Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"    Error testing {model_name}: {str(e)}")
                continue
        
        return adversarial_results
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance from trained models"""
        print("\nAnalyzing feature importance...")
        
        feature_importance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_imp_dict = dict(zip(self.feature_names, importances))
                
                # Sort by importance
                sorted_features = sorted(feature_imp_dict.items(), key=lambda x: x[1], reverse=True)
                feature_importance[model_name] = {
                    'top_10': sorted_features[:10],
                    'all_features': sorted_features
                }
                
                print(f"  {model_name} - Top 5 features:")
                for feat, imp in sorted_features[:5]:
                    print(f"    {feat}: {imp:.4f}")
        
        return feature_importance
    
    def generate_reports(self):
        """Generate comprehensive reports and visualizations"""
        print("\nGenerating reports...")
        
        # Save detailed results
        results_file = self.results_dir / "ml_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate performance comparison
        self._plot_performance_comparison()
        
        # Generate feature importance plots
        self._plot_feature_importance()
        
        # Generate confusion matrices
        self._plot_confusion_matrices()
        
        print(f"  Results saved to {self.results_dir}")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison between original and adversarial data"""
        if not self.results['original_performance'] or not self.results['adversarial_performance']:
            return
        
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        models = list(self.results['original_performance'].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            original_scores = [self.results['original_performance'][model][metric] for model in models]
            adversarial_scores = [self.results['adversarial_performance'][model][metric] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[i].bar(x - width/2, original_scores, width, label='Original', alpha=0.8)
            axes[i].bar(x + width/2, adversarial_scores, width, label='Adversarial', alpha=0.8)
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(models, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance for models that support it"""
        if 'feature_importance' not in self.results:
            return
        
        for model_name, importance_data in self.results['feature_importance'].items():
            if 'top_10' not in importance_data:
                continue
            
            features, importances = zip(*importance_data['top_10'])
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Feature Importance - {model_name}')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f"feature_importance_{model_name.lower()}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        # This would require storing predictions, which we can add if needed
        pass
    
    def run_complete_pipeline(self, correlation_threshold: float = 0.95, test_size: float = 0.2):
        """Run the complete ML pipeline"""
        print("=" * 80)
        print("NETWORK TRAFFIC ML PIPELINE")
        print("=" * 80)
        
        # Step 1: Load original data
        original_data = self.load_and_combine_datasets(self.original_data_dir, "original")
        
        # Step 2: Preprocess original data
        X_orig, y_orig = self.preprocess_data(original_data, is_training=True)
        
        # Step 3: Remove correlated features
        X_orig_filtered = self.remove_correlated_features(X_orig, correlation_threshold)
        
        # Step 4: Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_orig_filtered, y_orig, test_size=test_size, random_state=42, stratify=y_orig
        )
        
        print(f"\nData split:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        
        # Step 5: Train models
        original_performance = self.train_models(X_train, y_train, X_val, y_val)
        self.results['original_performance'] = original_performance
        
        # Step 6: Feature importance analysis
        feature_importance = self.analyze_feature_importance()
        self.results['feature_importance'] = feature_importance
        
        # Step 7: Load and test on adversarial data
        adversarial_data = self.load_and_combine_datasets(self.adversarial_data_dir, "adversarial")
        
        # For adversarial data, we need to handle the feature mismatch more carefully
        # First get the feature columns from adversarial data (excluding label and non-feature cols)
        non_feature_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'source_file']
        adv_feature_cols = [col for col in adversarial_data.columns if col not in non_feature_cols + ['Label']]
        
        # Find common features between training and adversarial data
        common_features = list(set(self.feature_names) & set(adv_feature_cols))
        missing_in_adv = set(self.feature_names) - set(adv_feature_cols)
        
        print(f"\nFeature alignment:")
        print(f"  Training features: {len(self.feature_names)}")
        print(f"  Adversarial features: {len(adv_feature_cols)}")
        print(f"  Common features: {len(common_features)}")
        print(f"  Missing in adversarial: {len(missing_in_adv)}")
        
        if missing_in_adv:
            print(f"  Missing features: {list(missing_in_adv)[:5]}...")
        
        # Create adversarial feature matrix with only common features
        X_adv_common = adversarial_data[common_features].copy()
        y_adv = adversarial_data['Label'].copy()
        
        # Handle missing values and convert to numeric
        X_adv_common = X_adv_common.fillna(X_adv_common.median())
        for col in X_adv_common.columns:
            if X_adv_common[col].dtype == 'object':
                try:
                    X_adv_common[col] = pd.to_numeric(X_adv_common[col], errors='coerce')
                    X_adv_common[col] = X_adv_common[col].fillna(X_adv_common[col].median())
                except:
                    print(f"Warning: Could not convert {col} to numeric")
        
        # Remove infinite values
        X_adv_common = X_adv_common.replace([np.inf, -np.inf], np.nan)
        X_adv_common = X_adv_common.fillna(X_adv_common.median())
        
        # Encode labels and scale features for common features only
        y_adv_encoded = self.label_encoder.transform(y_adv)
        
        # We need to retrain models with only common features
        print(f"\nRetraining models with {len(common_features)} common features...")
        
        # Filter original training data to common features
        common_feature_indices = [i for i, f in enumerate(self.feature_names) if f in common_features]
        X_train_common = X_train[:, common_feature_indices]
        X_val_common = X_val[:, common_feature_indices]
        
        # Update feature names to common features
        self.feature_names = common_features
        
        # Retrain models with common features
        original_performance = self.train_models(X_train_common, y_train, X_val_common, y_val)
        self.results['original_performance'] = original_performance
        
        # Feature importance analysis
        feature_importance = self.analyze_feature_importance()
        self.results['feature_importance'] = feature_importance
        
        # Scale adversarial data and test
        X_adv_scaled = self.scaler.fit_transform(X_adv_common)
        
        adversarial_performance = self.test_on_adversarial(X_adv_scaled, y_adv_encoded)
        self.results['adversarial_performance'] = adversarial_performance
        
        # Step 8: Generate reports
        self.generate_reports()
        
        # Step 9: Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print a summary of results"""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"\nFeatures:")
        print(f"  Original features: {len(self.feature_names) + len(self.removed_features)}")
        print(f"  Removed (correlation): {len(self.removed_features)}")
        print(f"  Used for training: {len(self.feature_names)}")
        
        print(f"\nModel Performance (Original Data):")
        for model, metrics in self.results['original_performance'].items():
            print(f"  {model}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1-Score (macro): {metrics['f1_macro']:.4f}")
            print(f"    CV Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
        
        print(f"\nModel Performance (Adversarial Data):")
        for model, metrics in self.results['adversarial_performance'].items():
            print(f"  {model}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1-Score (macro): {metrics['f1_macro']:.4f}")
        
        print(f"\nPerformance Degradation:")
        for model in self.results['original_performance'].keys():
            orig_acc = self.results['original_performance'][model]['accuracy']
            adv_acc = self.results['adversarial_performance'][model]['accuracy']
            degradation = ((orig_acc - adv_acc) / orig_acc) * 100
            print(f"  {model}: {degradation:.2f}% accuracy drop")


def main():
    """Main function to run the ML pipeline"""
    # Configuration
    original_data_dir = "original_dataset"
    adversarial_data_dir = "adversarial_dataset"
    results_dir = "ml_results"
    correlation_threshold = 0.95  # Remove features with correlation > 95%
    
    # Initialize and run pipeline
    pipeline = NetworkTrafficMLPipeline(
        original_data_dir=original_data_dir,
        adversarial_data_dir=adversarial_data_dir,
        results_dir=results_dir
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        correlation_threshold=correlation_threshold,
        test_size=0.2
    )
    
    print(f"\nPipeline completed successfully!")
    print(f"Results and plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
