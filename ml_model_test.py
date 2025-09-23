#!/usr/bin/env python3
"""
Machine Learning Model Training and Testing Script

This script:
1. Loads original datasets and augments training with DDoS FGSM data
2. Removes highly correlated features
3. Trains multiple ML models
4. Tests the models on original datasets only
5. Compares performance between training and test data
"""

import glob
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
            'original_labelwise_performance': {},
            'adversarial_labelwise_performance': {},
            'feature_analysis': {},
            'correlation_analysis': {}
        }
        
        # Store predictions for confusion matrices
        self.validation_predictions = {}  # Predictions on validation set (common features)
        self.test_predictions = {}        # Predictions on test set
        self.validation_true_labels = None
        self.test_true_labels = None
        
        # Keep original names for backward compatibility
        self.original_predictions = {}
        self.adversarial_predictions = {}
        self.original_true_labels = None
        self.adversarial_true_labels = None
    
    def load_and_combine_datasets(self, dataset_type: str = "training") -> pd.DataFrame:
        """Load and combine all CSV files from a directory"""
        print(f"\nLoading {dataset_type} datasets...")
        
        ## Define the folders which contains the datasets
        original_data_folder = "original_dataset"
        adversarial_data_folder = "adversarial_dataset"
        
        if dataset_type == "training":
            ## 1. Import all csv files in the original dataset folder
            data_df_1 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(original_data_folder, "*.csv"))])

            ## 2. Import ddos_attack_fgsm.csv in the adversarial dataset folder
            data_df_2 = pd.read_csv(os.path.join(adversarial_data_folder, "ddos_attack_fgsm.csv"))

            # ## Change the Label of data_df_2 to "DDOS_FGSM"
            # data_df_2['Label'] = "DDOS_FGSM"

            ## 3. Combine the two dataframes
            combined_df = pd.concat([data_df_1, data_df_2], ignore_index=True)
            # combined_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(original_data_folder, "*.csv"))])
        else:
            combined_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(original_data_folder, "*.csv"))])
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
                
                # Calculate label-wise metrics
                labelwise_metrics = self._calculate_labelwise_metrics(y_val, y_pred)
                
                # Store predictions for confusion matrix (validation set with common features)
                self.validation_predictions[model_name] = y_pred
                if self.validation_true_labels is None:
                    self.validation_true_labels = y_val
                
                # Also store in original_predictions for backward compatibility
                self.original_predictions[model_name] = y_pred
                if self.original_true_labels is None:
                    self.original_true_labels = y_val
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                trained_models[model_name] = model
                performance_results[model_name] = metrics
                
                # Store label-wise performance separately
                if not hasattr(self, 'original_labelwise_results'):
                    self.original_labelwise_results = {}
                self.original_labelwise_results[model_name] = labelwise_metrics
                
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
    
    def _calculate_labelwise_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate per-class/label performance metrics"""
        # Get unique classes in both true and predicted
        unique_classes_true = np.unique(y_true)
        unique_classes_pred = np.unique(y_pred)
        
        # Get class names from label encoder for all possible classes
        all_classes = np.unique(np.concatenate([unique_classes_true, unique_classes_pred]))
        class_names = []
        
        for class_idx in all_classes:
            try:
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
                class_names.append((class_idx, class_name))
            except:
                print(f"    Warning: Could not decode class {class_idx}")
        
        # Calculate per-class metrics with labels parameter to handle missing classes
        labels = [idx for idx, _ in class_names]
        target_names = [name for _, name in class_names]
        
        precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        # Create labelwise results dictionary
        labelwise_metrics = {}
        
        for i, (class_idx, class_name) in enumerate(class_names):
            if i < len(precision_per_class):  # Safety check
                labelwise_metrics[class_name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(np.sum(y_true == class_idx)),
                    'predicted_count': int(np.sum(y_pred == class_idx))
                }
        
        return labelwise_metrics
    
    def test_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Test trained models on the test data"""
        print("\nTesting models on test data...")
        
        test_results = {}
        test_labelwise_results = {}
        
        for model_name, model in self.models.items():
            print(f"  Testing {model_name}...")
            
            try:
                # Predict on test data
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                test_results[model_name] = metrics
                
                # Calculate label-wise metrics
                labelwise_metrics = self._calculate_labelwise_metrics(y_test, y_pred)
                test_labelwise_results[model_name] = labelwise_metrics
                
                # Store predictions for confusion matrix (test set)
                self.test_predictions[model_name] = y_pred
                if self.test_true_labels is None:
                    self.test_true_labels = y_test
                
                # Also store in adversarial_predictions for backward compatibility
                self.adversarial_predictions[model_name] = y_pred
                if self.adversarial_true_labels is None:
                    self.adversarial_true_labels = y_test
                
                print(f"    Test Accuracy: {metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"    Error testing {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store test label-wise results
        self.adversarial_labelwise_results = test_labelwise_results
        
        return test_results
    
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
        
        # Generate label-wise performance comparison
        self._plot_labelwise_performance_comparison()
        
        # Generate feature importance plots
        self._plot_feature_importance()
        
        # Generate confusion matrices
        self._plot_confusion_matrices()
        
        # Generate validation vs test comparison
        self._plot_validation_vs_test_comparison()
        
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
    
    def _plot_labelwise_performance_comparison(self):
        """Plot label-wise performance comparison between original and adversarial data"""
        if not hasattr(self, 'original_labelwise_results') or not hasattr(self, 'adversarial_labelwise_results'):
            return
        
        # Get all unique labels
        all_labels = set()
        for model_results in self.original_labelwise_results.values():
            all_labels.update(model_results.keys())
        all_labels = sorted(list(all_labels))
        
        models = list(self.original_labelwise_results.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        # Create subplots for each metric
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            # Prepare data for plotting
            x = np.arange(len(all_labels))
            width = 0.15
            
            for model_idx, model_name in enumerate(models):
                # Original performance
                orig_values = []
                adv_values = []
                
                for label in all_labels:
                    orig_val = self.original_labelwise_results[model_name].get(label, {}).get(metric, 0)
                    adv_val = self.adversarial_labelwise_results[model_name].get(label, {}).get(metric, 0)
                    orig_values.append(orig_val)
                    adv_values.append(adv_val)
                
                # Plot bars
                offset = (model_idx - len(models)/2 + 0.5) * width
                ax.bar(x + offset - width/2, orig_values, width/2, 
                      label=f'{model_name} (Original)', alpha=0.8)
                ax.bar(x + offset + width/2, adv_values, width/2, 
                      label=f'{model_name} (Adversarial)', alpha=0.6)
            
            ax.set_title(f'{metric.replace("_", " ").title()} by Attack Type')
            ax.set_xlabel('Attack Types')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xticks(x)
            ax.set_xticklabels(all_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "labelwise_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a detailed heatmap for F1-scores
        self._plot_labelwise_heatmap()
    
    def _plot_labelwise_heatmap(self):
        """Create heatmap showing F1-score degradation per class"""
        if not hasattr(self, 'original_labelwise_results') or not hasattr(self, 'adversarial_labelwise_results'):
            return
        
        # Get all unique labels and models
        all_labels = set()
        for model_results in self.original_labelwise_results.values():
            all_labels.update(model_results.keys())
        all_labels = sorted(list(all_labels))
        models = list(self.original_labelwise_results.keys())
        
        # Create data matrices for original and adversarial F1-scores
        orig_f1_matrix = np.zeros((len(models), len(all_labels)))
        adv_f1_matrix = np.zeros((len(models), len(all_labels)))
        degradation_matrix = np.zeros((len(models), len(all_labels)))
        
        for i, model_name in enumerate(models):
            for j, label in enumerate(all_labels):
                orig_f1 = self.original_labelwise_results[model_name].get(label, {}).get('f1_score', 0)
                adv_f1 = self.adversarial_labelwise_results[model_name].get(label, {}).get('f1_score', 0)
                
                orig_f1_matrix[i, j] = orig_f1
                adv_f1_matrix[i, j] = adv_f1
                
                # Calculate degradation percentage
                if orig_f1 > 0:
                    degradation_matrix[i, j] = ((orig_f1 - adv_f1) / orig_f1) * 100
                else:
                    degradation_matrix[i, j] = 0
        
        # Create subplots for the three heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(25, 8))
        
        # Original F1-scores
        im1 = axes[0].imshow(orig_f1_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
        axes[0].set_title('Original F1-Scores')
        axes[0].set_xticks(range(len(all_labels)))
        axes[0].set_xticklabels(all_labels, rotation=45, ha='right')
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(all_labels)):
                axes[0].text(j, i, f'{orig_f1_matrix[i, j]:.3f}', 
                           ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im1, ax=axes[0])
        
        # Adversarial F1-scores
        im2 = axes[1].imshow(adv_f1_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
        axes[1].set_title('Adversarial F1-Scores')
        axes[1].set_xticks(range(len(all_labels)))
        axes[1].set_xticklabels(all_labels, rotation=45, ha='right')
        axes[1].set_yticks(range(len(models)))
        axes[1].set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(all_labels)):
                axes[1].text(j, i, f'{adv_f1_matrix[i, j]:.3f}', 
                           ha="center", va="center", color="white", fontsize=10)
        
        plt.colorbar(im2, ax=axes[1])
        
        # Performance degradation
        im3 = axes[2].imshow(degradation_matrix, cmap='OrRd', aspect='auto', vmin=0, vmax=100)
        axes[2].set_title('F1-Score Degradation (%)')
        axes[2].set_xticks(range(len(all_labels)))
        axes[2].set_xticklabels(all_labels, rotation=45, ha='right')
        axes[2].set_yticks(range(len(models)))
        axes[2].set_yticklabels(models)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(all_labels)):
                color = "white" if degradation_matrix[i, j] > 50 else "black"
                axes[2].text(j, i, f'{degradation_matrix[i, j]:.1f}%', 
                           ha="center", va="center", color=color, fontsize=10)
        
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "labelwise_f1_heatmap.png", dpi=300, bbox_inches='tight')
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
        """Plot confusion matrices for all models on both original and adversarial data"""
        if not self.original_predictions or not self.adversarial_predictions:
            print("    No predictions available for confusion matrices")
            return
        
        # Get class names
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.label_encoder.classes_))]
        
        for model_name in self.models.keys():
            if model_name not in self.original_predictions or model_name not in self.adversarial_predictions:
                continue
                
            # Create figure with subplots for original and adversarial
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Original data confusion matrix
            cm_orig = confusion_matrix(self.original_true_labels, self.original_predictions[model_name])
            
            # Normalize confusion matrix to show percentages
            cm_orig_norm = cm_orig.astype('float') / cm_orig.sum(axis=1)[:, np.newaxis]
            
            # Plot original confusion matrix
            im1 = axes[0].imshow(cm_orig_norm, interpolation='nearest', cmap='Blues')
            axes[0].set_title(f'{model_name} - Training Data (Original + DDoS FGSM)\nConfusion Matrix (Normalized)', fontsize=14)
            axes[0].set_xlabel('Predicted Label', fontsize=12)
            axes[0].set_ylabel('True Label', fontsize=12)
            
            # Add class names to ticks
            tick_marks = np.arange(len(class_names))
            axes[0].set_xticks(tick_marks)
            axes[0].set_yticks(tick_marks)
            axes[0].set_xticklabels(class_names, rotation=45, ha='right')
            axes[0].set_yticklabels(class_names)
            
            # Add text annotations
            thresh = cm_orig_norm.max() / 2.
            for i, j in np.ndindex(cm_orig_norm.shape):
                axes[0].text(j, i, f'{cm_orig_norm[i, j]:.2f}\n({cm_orig[i, j]})',
                           ha="center", va="center",
                           color="white" if cm_orig_norm[i, j] > thresh else "black",
                           fontsize=10)
            
            plt.colorbar(im1, ax=axes[0])
            
            # Test data confusion matrix
            cm_adv = confusion_matrix(self.adversarial_true_labels, self.adversarial_predictions[model_name])
            
            # Normalize confusion matrix to show percentages
            cm_adv_norm = cm_adv.astype('float') / cm_adv.sum(axis=1)[:, np.newaxis]
            
            # Plot test data confusion matrix
            im2 = axes[1].imshow(cm_adv_norm, interpolation='nearest', cmap='Greens')
            axes[1].set_title(f'{model_name} - Test Data (Original Only)\nConfusion Matrix (Normalized)', fontsize=14)
            axes[1].set_xlabel('Predicted Label', fontsize=12)
            axes[1].set_ylabel('True Label', fontsize=12)
            
            # Add class names to ticks
            axes[1].set_xticks(tick_marks)
            axes[1].set_yticks(tick_marks)
            axes[1].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1].set_yticklabels(class_names)
            
            # Add text annotations
            thresh = cm_adv_norm.max() / 2.
            for i, j in np.ndindex(cm_adv_norm.shape):
                axes[1].text(j, i, f'{cm_adv_norm[i, j]:.2f}\n({cm_adv[i, j]})',
                           ha="center", va="center",
                           color="white" if cm_adv_norm[i, j] > thresh else "black",
                           fontsize=10)
            
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f"confusion_matrix_{model_name.lower()}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a comparison plot showing the difference in confusion matrices
        self._plot_confusion_matrix_comparison()
    
    def _plot_confusion_matrix_comparison(self):
        """Create a comparison plot showing the confusion matrix differences"""
        if not self.original_predictions or not self.adversarial_predictions:
            return
        
        # Get class names
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.label_encoder.classes_))]
        
        for model_name in self.models.keys():
            if model_name not in self.original_predictions or model_name not in self.adversarial_predictions:
                continue
            
            # Calculate confusion matrices
            cm_orig = confusion_matrix(self.original_true_labels, self.original_predictions[model_name])
            cm_adv = confusion_matrix(self.adversarial_true_labels, self.adversarial_predictions[model_name])
            
            # Normalize both matrices
            cm_orig_norm = cm_orig.astype('float') / cm_orig.sum(axis=1)[:, np.newaxis]
            cm_adv_norm = cm_adv.astype('float') / cm_adv.sum(axis=1)[:, np.newaxis]
            
            # Calculate difference (degradation)
            cm_diff = cm_orig_norm - cm_adv_norm
            
            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(30, 8))
            
            # Original confusion matrix
            im1 = axes[0].imshow(cm_orig_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            axes[0].set_title(f'{model_name} - Training Data (Original + DDoS FGSM)\n(Normalized)', fontsize=14)
            axes[0].set_xlabel('Predicted Label', fontsize=12)
            axes[0].set_ylabel('True Label', fontsize=12)
            
            tick_marks = np.arange(len(class_names))
            axes[0].set_xticks(tick_marks)
            axes[0].set_yticks(tick_marks)
            axes[0].set_xticklabels(class_names, rotation=45, ha='right')
            axes[0].set_yticklabels(class_names)
            
            # Add text annotations
            for i, j in np.ndindex(cm_orig_norm.shape):
                axes[0].text(j, i, f'{cm_orig_norm[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if cm_orig_norm[i, j] > 0.5 else "black",
                           fontsize=10)
            
            plt.colorbar(im1, ax=axes[0])
            
            # Test data confusion matrix
            im2 = axes[1].imshow(cm_adv_norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
            axes[1].set_title(f'{model_name} - Test Data (Original Only)\n(Normalized)', fontsize=14)
            axes[1].set_xlabel('Predicted Label', fontsize=12)
            axes[1].set_ylabel('True Label', fontsize=12)
            
            axes[1].set_xticks(tick_marks)
            axes[1].set_yticks(tick_marks)
            axes[1].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1].set_yticklabels(class_names)
            
            # Add text annotations
            for i, j in np.ndindex(cm_adv_norm.shape):
                axes[1].text(j, i, f'{cm_adv_norm[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if cm_adv_norm[i, j] > 0.5 else "black",
                           fontsize=10)
            
            plt.colorbar(im2, ax=axes[1])
            
            # Difference matrix (degradation)
            im3 = axes[2].imshow(cm_diff, interpolation='nearest', cmap='RdYlBu_r', 
                               vmin=-1, vmax=1)
            axes[2].set_title(f'{model_name} - Performance Difference\n(Training - Test)', fontsize=14)
            axes[2].set_xlabel('Predicted Label', fontsize=12)
            axes[2].set_ylabel('True Label', fontsize=12)
            
            axes[2].set_xticks(tick_marks)
            axes[2].set_yticks(tick_marks)
            axes[2].set_xticklabels(class_names, rotation=45, ha='right')
            axes[2].set_yticklabels(class_names)
            
            # Add text annotations
            for i, j in np.ndindex(cm_diff.shape):
                color = "white" if abs(cm_diff[i, j]) > 0.5 else "black"
                axes[2].text(j, i, f'{cm_diff[i, j]:.2f}',
                           ha="center", va="center", color=color, fontsize=10)
            
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f"confusion_matrix_comparison_{model_name.lower()}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_validation_vs_test_comparison(self):
        """Create comparison plots between validation set (training data) and test set (original only)"""
        if not self.validation_predictions or not self.test_predictions:
            print("    No validation or test predictions available for comparison")
            return
        
        print("    Creating validation vs test comparison plots...")
        
        # Get class names
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.label_encoder.classes_))]
        
        for model_name in self.models.keys():
            if model_name not in self.validation_predictions or model_name not in self.test_predictions:
                continue
            
            # Calculate confusion matrices
            cm_val = confusion_matrix(self.validation_true_labels, self.validation_predictions[model_name])
            cm_test = confusion_matrix(self.test_true_labels, self.test_predictions[model_name])
            
            # Normalize both matrices
            cm_val_norm = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
            cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
            
            # Calculate difference (performance change from validation to test)
            cm_diff = cm_val_norm - cm_test_norm
            
            # Create figure with four subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # Validation confusion matrix (top-left)
            im1 = axes[0,0].imshow(cm_val_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
            axes[0,0].set_title(f'{model_name} - Validation Set\n(Training: Original + DDoS FGSM)', fontsize=14)
            axes[0,0].set_xlabel('Predicted Label', fontsize=12)
            axes[0,0].set_ylabel('True Label', fontsize=12)
            
            tick_marks = np.arange(len(class_names))
            axes[0,0].set_xticks(tick_marks)
            axes[0,0].set_yticks(tick_marks)
            axes[0,0].set_xticklabels(class_names, rotation=45, ha='right')
            axes[0,0].set_yticklabels(class_names)
            
            # Add text annotations
            for i, j in np.ndindex(cm_val_norm.shape):
                axes[0,0].text(j, i, f'{cm_val_norm[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if cm_val_norm[i, j] > 0.5 else "black",
                             fontsize=9)
            
            plt.colorbar(im1, ax=axes[0,0])
            
            # Test confusion matrix (top-right)
            im2 = axes[0,1].imshow(cm_test_norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
            axes[0,1].set_title(f'{model_name} - Test Set\n(Original Data Only)', fontsize=14)
            axes[0,1].set_xlabel('Predicted Label', fontsize=12)
            axes[0,1].set_ylabel('True Label', fontsize=12)
            
            axes[0,1].set_xticks(tick_marks)
            axes[0,1].set_yticks(tick_marks)
            axes[0,1].set_xticklabels(class_names, rotation=45, ha='right')
            axes[0,1].set_yticklabels(class_names)
            
            # Add text annotations
            for i, j in np.ndindex(cm_test_norm.shape):
                axes[0,1].text(j, i, f'{cm_test_norm[i, j]:.2f}',
                             ha="center", va="center",
                             color="white" if cm_test_norm[i, j] > 0.5 else "black",
                             fontsize=9)
            
            plt.colorbar(im2, ax=axes[0,1])
            
            # Performance difference matrix (bottom-left)
            im3 = axes[1,0].imshow(cm_diff, interpolation='nearest', cmap='RdYlBu_r', 
                                 vmin=-1, vmax=1)
            axes[1,0].set_title(f'{model_name} - Performance Difference\n(Validation - Test)', fontsize=14)
            axes[1,0].set_xlabel('Predicted Label', fontsize=12)
            axes[1,0].set_ylabel('True Label', fontsize=12)
            
            axes[1,0].set_xticks(tick_marks)
            axes[1,0].set_yticks(tick_marks)
            axes[1,0].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1,0].set_yticklabels(class_names)
            
            # Add text annotations
            for i, j in np.ndindex(cm_diff.shape):
                color = "white" if abs(cm_diff[i, j]) > 0.5 else "black"
                axes[1,0].text(j, i, f'{cm_diff[i, j]:.2f}',
                             ha="center", va="center", color=color, fontsize=9)
            
            plt.colorbar(im3, ax=axes[1,0])
            
            # Side-by-side bar chart comparison (bottom-right)
            # Calculate per-class accuracy for both validation and test
            val_class_acc = np.diag(cm_val_norm)
            test_class_acc = np.diag(cm_test_norm)
            
            x = np.arange(len(class_names))
            width = 0.35
            
            bars1 = axes[1,1].bar(x - width/2, val_class_acc, width, label='Validation (Original+DDoS_FGSM)', alpha=0.8, color='blue')
            bars2 = axes[1,1].bar(x + width/2, test_class_acc, width, label='Test (Original Only)', alpha=0.8, color='green')
            
            axes[1,1].set_title(f'{model_name} - Per-Class Accuracy Comparison', fontsize=14)
            axes[1,1].set_xlabel('Attack Types', fontsize=12)
            axes[1,1].set_ylabel('Accuracy', fontsize=12)
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(class_names, rotation=45, ha='right')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f"validation_vs_test_comparison_{model_name.lower()}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a summary comparison across all models
        self._plot_validation_vs_test_summary()
    
    def _plot_validation_vs_test_summary(self):
        """Create a summary plot comparing validation vs test performance across all models"""
        if not self.validation_predictions or not self.test_predictions:
            return
        
        print("    Creating validation vs test summary plot...")
        
        # Get class names and models
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.label_encoder.classes_))]
        models = list(self.models.keys())
        
        # Calculate accuracy matrices for validation and test
        val_acc_matrix = np.zeros((len(models), len(class_names)))
        test_acc_matrix = np.zeros((len(models), len(class_names)))
        difference_matrix = np.zeros((len(models), len(class_names)))
        
        for i, model_name in enumerate(models):
            if model_name in self.validation_predictions and model_name in self.test_predictions:
                # Calculate confusion matrices
                cm_val = confusion_matrix(self.validation_true_labels, self.validation_predictions[model_name])
                cm_test = confusion_matrix(self.test_true_labels, self.test_predictions[model_name])
                
                # Normalize confusion matrices
                cm_val_norm = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
                cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
                
                # Extract per-class accuracies (diagonal elements)
                val_class_acc = np.diag(cm_val_norm)
                test_class_acc = np.diag(cm_test_norm)
                
                for j in range(len(class_names)):
                    if j < len(val_class_acc):  # Safety check
                        val_acc_matrix[i, j] = val_class_acc[j]
                        test_acc_matrix[i, j] = test_class_acc[j]
                        
                        # Calculate difference (can be positive or negative)
                        difference_matrix[i, j] = test_class_acc[j] - val_class_acc[j]
        
        # Create figure with three heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(25, 8))
        
        # Validation accuracy heatmap
        im1 = axes[0].imshow(val_acc_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        axes[0].set_title('Validation Set Accuracy\n(Training: Original + DDoS FGSM)', fontsize=14)
        axes[0].set_xticks(range(len(class_names)))
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].set_yticks(range(len(models)))
        axes[0].set_yticklabels(models)
        axes[0].set_xlabel('Attack Types', fontsize=12)
        axes[0].set_ylabel('Models', fontsize=12)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(class_names)):
                axes[0].text(j, i, f'{val_acc_matrix[i, j]:.3f}', 
                           ha="center", va="center", color="white", fontsize=10)
        
        plt.colorbar(im1, ax=axes[0])
        
        # Test accuracy heatmap
        im2 = axes[1].imshow(test_acc_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
        axes[1].set_title('Test Set Accuracy\n(Original Data Only)', fontsize=14)
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].set_yticks(range(len(models)))
        axes[1].set_yticklabels(models)
        axes[1].set_xlabel('Attack Types', fontsize=12)
        axes[1].set_ylabel('Models', fontsize=12)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(class_names)):
                axes[1].text(j, i, f'{test_acc_matrix[i, j]:.3f}', 
                           ha="center", va="center", color="white", fontsize=10)
        
        plt.colorbar(im2, ax=axes[1])
        
        # Performance difference heatmap
        im3 = axes[2].imshow(difference_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[2].set_title('Accuracy Difference\n(Test - Validation)', fontsize=14)
        axes[2].set_xticks(range(len(class_names)))
        axes[2].set_xticklabels(class_names, rotation=45, ha='right')
        axes[2].set_yticks(range(len(models)))
        axes[2].set_yticklabels(models)
        axes[2].set_xlabel('Attack Types', fontsize=12)
        axes[2].set_ylabel('Models', fontsize=12)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(class_names)):
                color = "white" if abs(difference_matrix[i, j]) > 0.5 else "black"
                axes[2].text(j, i, f'{difference_matrix[i, j]:.3f}', 
                           ha="center", va="center", color=color, fontsize=10)
        
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "validation_vs_test_summary_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_pipeline(self, correlation_threshold: float = 0.95, test_size: float = 0.2):
        """Run the complete ML pipeline with corrected scaling and feature selection logic."""
        print("=" * 80)
        print("NETWORK TRAFFIC ML PIPELINE (Corrected Logic)")
        print("=" * 80)

        # Step 1: Load and preprocess training data (without scaling)
        training_data = self.load_and_combine_datasets("training")
        
        # Separate features and labels
        non_feature_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'source_file', 'Label']
        feature_cols = [col for col in training_data.columns if col not in non_feature_cols]
        X = training_data[feature_cols].copy()
        y_series = training_data['Label'].copy()

        # Clean data: handle missing, non-numeric, and infinite values
        X = X.fillna(X.median())
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        # Encode labels
        y = self.label_encoder.fit_transform(y_series)
        self.feature_names = list(X.columns)
        print(f"\nInitial data loaded and cleaned. Features: {X.shape[1]}")

        # Step 2: Remove correlated features from the DataFrame
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
        
        X_filtered = X.drop(columns=to_drop)
        self.removed_features = to_drop
        self.feature_names = list(X_filtered.columns) # Final feature set
        print(f"Removed {len(to_drop)} correlated features. Remaining features: {len(self.feature_names)}")

        # Step 3: Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_filtered, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"\nData split: Training ({X_train.shape}), Validation ({X_val.shape})")

        # Step 4: Fit the scaler on the training data ONLY
        print("Fitting scaler on the training data...")
        self.scaler.fit(X_train)

        # Step 5: Scale the training and validation data
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        print("Training and validation data scaled.")

        # Step 6: Train models
        self.results['original_performance'] = self.train_models(X_train_scaled, y_train, X_val_scaled, y_val)
        self.results['original_labelwise_performance'] = self.original_labelwise_results

        # Step 7: Feature importance analysis
        self.results['feature_importance'] = self.analyze_feature_importance()

        # Step 8: Load and prepare test data
        print("\nLoading and preparing test data...")
        test_data = self.load_and_combine_datasets("test")
        X_test = test_data[self.feature_names].copy() # Use the final feature list
        y_test_series = test_data['Label'].copy()

        # Clean test data
        X_test = X_test.fillna(X_test.median())
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())
        
        y_test = self.label_encoder.transform(y_test_series)

        # Step 9: Scale the test data using the already-fitted scaler
        print("Scaling test data...")
        X_test_scaled = self.scaler.transform(X_test)
        print(f"  Test data scaled shape: {X_test_scaled.shape}")

        # Step 10: Test models on the scaled test data
        self.test_results = self.test_models(X_test_scaled, y_test)
        self.results['adversarial_performance'] = self.test_results
        self.results['adversarial_labelwise_performance'] = self.adversarial_labelwise_results
        
        # Step 11: Generate reports and print summary
        self.generate_reports()
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
        
        print(f"\nModel Performance (Test Data - Original Only):")
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
        
        print(f"\n⚠️  TRAINING vs TEST ANALYSIS:")
        print(f"  Training Data: Original data + DDoS FGSM (data augmentation)")
        print(f"  Test Data: Original data only (clean test set)")
        print(f"  This comparison shows how well the model generalizes from")
        print(f"  augmented training data (including adversarial examples) to clean test data.")
        
        # Print validation vs test accuracy comparison
        if hasattr(self, 'validation_predictions') and hasattr(self, 'test_predictions'):
            print(f"\nValidation vs Test Accuracy Comparison:")
            for model_name in self.models.keys():
                if model_name in self.validation_predictions and model_name in self.test_predictions:
                    from sklearn.metrics import accuracy_score
                    val_acc = accuracy_score(self.validation_true_labels, self.validation_predictions[model_name])
                    test_acc = accuracy_score(self.test_true_labels, self.test_predictions[model_name])
                    difference = test_acc - val_acc  # Can be positive (improvement) or negative (degradation)
                    
                    print(f"  {model_name}:")
                    print(f"    Validation Accuracy: {val_acc:.4f}")
                    print(f"    Test Accuracy: {test_acc:.4f}")
                    if difference >= 0:
                        print(f"    Improvement: +{difference:.4f} ({(difference/val_acc)*100:.2f}%)")
                    else:
                        print(f"    Degradation: {difference:.4f} ({(difference/val_acc)*100:.2f}%)")
        
        # Print label-wise performance summary
        if hasattr(self, 'original_labelwise_results') and hasattr(self, 'adversarial_labelwise_results'):
            print(f"\nLabel-wise Performance Summary:")
            
            # Get all unique labels
            all_labels = set()
            for model_results in self.original_labelwise_results.values():
                all_labels.update(model_results.keys())
            all_labels = sorted(list(all_labels))
            
            for model_name in self.results['original_performance'].keys():
                print(f"\n  {model_name} - F1-Score by Attack Type:")
                print(f"    {'Attack Type':<20} {'Original':<10} {'Adversarial':<12} {'Degradation':<12}")
                print(f"    {'-'*54}")
                
                for label in all_labels:
                    orig_f1 = self.original_labelwise_results[model_name].get(label, {}).get('f1_score', 0)
                    adv_f1 = self.adversarial_labelwise_results[model_name].get(label, {}).get('f1_score', 0)
                    
                    if orig_f1 > 0:
                        degradation = ((orig_f1 - adv_f1) / orig_f1) * 100
                    else:
                        degradation = 0
                    
                    print(f"    {label:<20} {orig_f1:<10.3f} {adv_f1:<12.3f} {degradation:<12.1}%")


def main():
    """Main function to run the ML pipeline"""
    # Configuration
    original_data_dir = "original_dataset"
    adversarial_data_dir = "adversarial_dataset"
    results_dir = "ml_test_results"
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
