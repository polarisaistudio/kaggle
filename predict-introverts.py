"""
Kaggle Playground Series S5E7: Introvert vs Extrovert Prediction
Complete ML Pipeline for Personality Classification

Author: Polaris AI Studio
Competition: https://www.kaggle.com/competitions/playground-series-s5e7
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class DataExplorer:
    """Comprehensive data exploration and analysis"""
    
    def __init__(self):
        self.feature_importance = {}
        self.correlation_matrix = None
        
    def load_and_explore_data(self, train_path, test_path):
        """Load data and perform initial exploration"""
        print("="*50)
        print("LOADING AND EXPLORING DATA")
        print("="*50)
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Basic info
        print("\nTraining data info:")
        print(train_df.info())
        
        print("\nFirst few rows:")
        print(train_df.head())
        
        # Check for missing values
        print("\nMissing values:")
        print(train_df.isnull().sum().sort_values(ascending=False))
        
        # Target distribution
        if 'target' in train_df.columns:
            print("\nTarget distribution:")
            print(train_df['target'].value_counts())
            print(f"Target balance: {train_df['target'].mean():.3f}")
            
        return train_df, test_df
    
    def analyze_features(self, df, target_col='target'):
        """Analyze feature distributions and correlations"""
        print("\n" + "="*50)
        print("FEATURE ANALYSIS")
        print("="*50)
        
        # Separate features and target
        if target_col in df.columns:
            features = df.drop(columns=[target_col, 'id'] if 'id' in df.columns else [target_col])
            target = df[target_col]
            
            # Feature correlations with target
            correlations = features.corrwith(target).abs().sort_values(ascending=False)
            print("\nTop 10 features correlated with target:")
            print(correlations.head(10))
            
            # Feature correlation matrix
            self.correlation_matrix = features.corr()
            
            # Identify highly correlated features
            high_corr_pairs = []
            for i in range(len(self.correlation_matrix.columns)):
                for j in range(i+1, len(self.correlation_matrix.columns)):
                    if abs(self.correlation_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((
                            self.correlation_matrix.columns[i],
                            self.correlation_matrix.columns[j],
                            self.correlation_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                print(f"\nHighly correlated feature pairs (>0.8):")
                for pair in high_corr_pairs:
                    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        return features, target if target_col in df.columns else None
    
    def visualize_data(self, df, target_col='target'):
        """Create visualization plots"""
        if target_col not in df.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        df[target_col].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Target Distribution')
        axes[0,0].set_xlabel('Class')
        axes[0,0].set_ylabel('Count')
        
        # Feature correlation heatmap (top features)
        features = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
        top_features = features.corrwith(df[target_col]).abs().sort_values(ascending=False)[:10]
        corr_matrix = df[top_features.index.tolist() + [target_col]].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
        axes[0,1].set_title('Top Features Correlation Matrix')
        
        # Feature importance from Random Forest
        X = features
        y = df[target_col]
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.head(10).plot(x='feature', y='importance', kind='barh', ax=axes[1,0])
        axes[1,0].set_title('Top 10 Feature Importance (Random Forest)')
        
        # Class distribution by top feature
        top_feature = importance_df.iloc[0]['feature']
        df.boxplot(column=top_feature, by=target_col, ax=axes[1,1])
        axes[1,1].set_title(f'{top_feature} Distribution by Class')
        
        plt.tight_layout()
        plt.show()

class PersonalityFeatureEngineer:
    """Psychology-informed feature engineering for personality prediction"""
    
    def __init__(self):
        self.feature_interactions = []
        self.psychological_scales = []
        self.scaler = None
        self.poly_features = None
    
    def create_interaction_features(self, df):
        """Create meaningful feature interactions based on psychology"""
        print("\nCreating interaction features...")
        
        # Identify potential social and energy-related features
        # These are example feature names - adjust based on actual dataset
        social_features = [col for col in df.columns if any(term in col.lower() 
                          for term in ['social', 'group', 'party', 'friend', 'people'])]
        
        energy_features = [col for col in df.columns if any(term in col.lower() 
                          for term in ['energy', 'tired', 'recharge', 'active'])]
        
        communication_features = [col for col in df.columns if any(term in col.lower() 
                                 for term in ['talk', 'speak', 'communicate', 'express'])]
        
        if len(social_features) >= 2:
            # Social energy interactions
            df['social_energy_index'] = df[social_features].mean(axis=1)
            
        if len(communication_features) >= 2:
            # Communication complexity
            df['communication_complexity'] = df[communication_features].mean(axis=1)
        
        # Create interaction between numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            # High-value interactions
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to avoid explosion
                for col2 in numeric_cols[i+1:6]:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        return df
    
    def create_psychological_scales(self, df):
        """Build composite psychological measures"""
        print("Creating psychological scales...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 3:
            # Create scales based on feature groups
            n_features = len(numeric_cols)
            
            # Extraversion-like scale (first third of features)
            extraversion_features = numeric_cols[:n_features//3]
            df['extraversion_score'] = df[extraversion_features].mean(axis=1)
            
            # Introversion-like scale (middle third)
            introversion_features = numeric_cols[n_features//3:2*n_features//3]
            df['introversion_score'] = df[introversion_features].mean(axis=1)
            
            # Energy source preference
            df['energy_source_ratio'] = (
                df['extraversion_score'] / (df['introversion_score'] + 0.1)
            )
            
            # Behavioral consistency
            df['behavioral_consistency'] = df[numeric_cols].std(axis=1)
            
            # Extreme response tendency
            df['extreme_response_tendency'] = (
                (df[numeric_cols] > df[numeric_cols].quantile(0.8)).sum(axis=1) +
                (df[numeric_cols] < df[numeric_cols].quantile(0.2)).sum(axis=1)
            )
        
        return df
    
    def create_polynomial_features(self, df, degree=2):
        """Create polynomial features for key variables"""
        print("Creating polynomial features...")
        
        # Select most important features for polynomial expansion
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Use top correlated features with target if available
            key_features = numeric_cols[:4]  # Limit to prevent explosion
            
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            poly_data = poly.fit_transform(df[key_features])
            
            # Get feature names
            poly_feature_names = poly.get_feature_names_out(key_features)
            
            # Add polynomial features
            for i, name in enumerate(poly_feature_names):
                if name not in key_features:  # Don't duplicate original features
                    df[f'poly_{name}'] = poly_data[:, i]
            
            self.poly_features = poly
        
        return df
    
    def engineer_features(self, df):
        """Complete feature engineering pipeline"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        # Store original feature count
        original_features = len(df.columns)
        
        # Apply all feature engineering steps
        df = self.create_interaction_features(df)
        df = self.create_psychological_scales(df)
        df = self.create_polynomial_features(df)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        
        new_features = len(df.columns)
        print(f"Feature engineering complete: {original_features} → {new_features} features")
        
        return df

class PersonalityModelBuilder:
    """Advanced model building and ensemble methods"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_selector = None
        self.scaler = StandardScaler()
    
    def build_baseline_model(self, X, y):
        """Simple but effective baseline approach"""
        print("\n" + "="*50)
        print("BUILDING BASELINE MODEL")
        print("="*50)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Simple Random Forest baseline
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Validation predictions
        val_pred = rf_model.predict(X_val_scaled)
        val_prob = rf_model.predict_proba(X_val_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_val, val_pred)
        auc_score = roc_auc_score(y_val, val_prob)
        
        print(f"Baseline Accuracy: {accuracy:.4f}")
        print(f"Baseline AUC: {auc_score:.4f}")
        
        self.models['baseline_rf'] = rf_model
        
        return rf_model, accuracy, auc_score
    
    def build_advanced_models(self, X, y):
        """Build multiple advanced models"""
        print("\n" + "="*50)
        print("BUILDING ADVANCED MODELS")
        print("="*50)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        models_config = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                feature_fraction=0.8,
                random_state=42,
                verbose=-1
            ),
            'logistic': LogisticRegression(
                C=0.1,
                penalty='l1',
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            val_pred = model.predict(X_val_scaled)
            val_prob = model.predict_proba(X_val_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_val, val_pred)
            auc = roc_auc_score(y_val, val_prob)
            
            model_scores[name] = {'accuracy': accuracy, 'auc': auc}
            self.models[name] = model
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return model_scores
    
    def build_ensemble(self, X, y):
        """Create ensemble of best models"""
        print("\n" + "="*50)
        print("BUILDING ENSEMBLE MODEL")
        print("="*50)
        
        # Use pre-trained models or train new ones
        if not self.models:
            self.build_advanced_models(X, y)
        
        # Select best performing models for ensemble
        best_models = []
        for name, model in self.models.items():
            if name != 'baseline_rf':  # Exclude baseline
                best_models.append((name, model))
        
        # Create voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=best_models,
            voting='soft'
        )
        
        # Split data for ensemble training
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train ensemble
        voting_ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate ensemble
        val_pred = voting_ensemble.predict(X_val_scaled)
        val_prob = voting_ensemble.predict_proba(X_val_scaled)[:, 1]
        
        accuracy = accuracy_score(y_val, val_pred)
        auc = roc_auc_score(y_val, val_prob)
        
        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print(f"Ensemble AUC: {auc:.4f}")
        
        self.models['ensemble'] = voting_ensemble
        self.best_model = voting_ensemble
        
        return voting_ensemble, accuracy, auc

class AdvancedPersonalityClassifier:
    """Advanced techniques for personality classification"""
    
    def __init__(self):
        self.feature_selector = None
        self.calibrator = None
        self.best_params = None
    
    def feature_selection_optimization(self, X, y, model=None):
        """Optimize feature selection for personality prediction"""
        print("\n" + "="*50)
        print("OPTIMIZING FEATURE SELECTION")
        print("="*50)
        
        if model is None:
            model = xgb.XGBClassifier(random_state=42)
        
        # Recursive feature elimination with cross-validation
        selector = RFECV(
            estimator=model,
            step=1,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print("Running recursive feature elimination...")
        selector.fit(X, y)
        self.feature_selector = selector
        
        print(f"Optimal number of features: {selector.n_features_}")
        print(f"CV Score with optimal features: {selector.grid_scores_.max():.4f}")
        
        # Get selected features
        selected_features = X.columns[selector.support_]
        print(f"Selected features: {list(selected_features)}")
        
        return selector.transform(X), selected_features
    
    def hyperparameter_optimization(self, X, y):
        """Optimize hyperparameters with grid search"""
        print("\n" + "="*50)
        print("OPTIMIZING HYPERPARAMETERS")
        print("="*50)
        
        # Hyperparameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        print("Running grid search...")
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def probability_calibration(self, model, X, y):
        """Calibrate prediction probabilities"""
        print("\nCalibrating prediction probabilities...")
        
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='isotonic',
            cv=3
        )
        
        calibrated_model.fit(X, y)
        self.calibrator = calibrated_model
        
        return calibrated_model

class PersonalityPipeline:
    """Complete pipeline for personality prediction"""
    
    def __init__(self):
        self.data_explorer = DataExplorer()
        self.feature_engineer = PersonalityFeatureEngineer()
        self.model_builder = PersonalityModelBuilder()
        self.advanced_classifier = AdvancedPersonalityClassifier()
        
        self.train_df = None
        self.test_df = None
        self.final_model = None
    
    def run_complete_pipeline(self, train_path, test_path, target_col='target'):
        """Run the complete ML pipeline"""
        print("🧠 PERSONALITY PREDICTION PIPELINE")
        print("="*60)
        
        # 1. Data Loading and Exploration
        self.train_df, self.test_df = self.data_explorer.load_and_explore_data(train_path, test_path)
        
        if target_col in self.train_df.columns:
            features, target = self.data_explorer.analyze_features(self.train_df, target_col)
            
            # 2. Feature Engineering
            engineered_train = self.feature_engineer.engineer_features(self.train_df.copy())
            engineered_test = self.feature_engineer.engineer_features(self.test_df.copy())
            
            # Prepare features for modeling
            feature_cols = [col for col in engineered_train.columns 
                          if col not in [target_col, 'id']]
            
            X = engineered_train[feature_cols]
            y = engineered_train[target_col]
            X_test = engineered_test[feature_cols]
            
            # 3. Baseline Model
            baseline_model, baseline_acc, baseline_auc = self.model_builder.build_baseline_model(X, y)
            
            # 4. Advanced Models
            model_scores = self.model_builder.build_advanced_models(X, y)
            
            # 5. Ensemble Model
            ensemble_model, ensemble_acc, ensemble_auc = self.model_builder.build_ensemble(X, y)
            
            # 6. Feature Selection Optimization
            X_selected, selected_features = self.advanced_classifier.feature_selection_optimization(X, y)
            
            # 7. Hyperparameter Optimization
            optimized_model = self.advanced_classifier.hyperparameter_optimization(X_selected, y)
            
            # 8. Final Model with Calibration
            final_model = self.advanced_classifier.probability_calibration(optimized_model, X_selected, y)
            
            self.final_model = final_model
            
            # 9. Generate Predictions
            # Scale test features
            X_test_scaled = self.model_builder.scaler.transform(X_test)
            X_test_selected = self.advanced_classifier.feature_selector.transform(X_test_scaled)
            
            # Final predictions
            test_predictions = final_model.predict_proba(X_test_selected)[:, 1]
            
            # Create submission
            submission = pd.DataFrame({
                'id': self.test_df['id'] if 'id' in self.test_df.columns else range(len(test_predictions)),
                'target': test_predictions
            })
            
            submission.to_csv('personality_prediction_submission.csv', index=False)
            print("\n✅ Submission saved to 'personality_prediction_submission.csv'")
            
            # Print final summary
            print("\n" + "="*60)
            print("FINAL RESULTS SUMMARY")
            print("="*60)
            print(f"Baseline Model (Random Forest): {baseline_acc:.4f}")
            print(f"Best Individual Model: {max([scores['accuracy'] for scores in model_scores.values()]):.4f}")
            print(f"Ensemble Model: {ensemble_acc:.4f}")
            print(f"Final Optimized Model: Ready for submission")
            print(f"Selected Features: {len(selected_features)}")
            
            return submission
        else:
            print(f"Error: Target column '{target_col}' not found in training data")
            return None

def main():
    """Main execution function"""
    
    # Configuration
    TRAIN_PATH = "train.csv"  # Update with actual path
    TEST_PATH = "test.csv"    # Update with actual path
    TARGET_COL = "target"     # Update with actual target column name
    
    # Initialize and run pipeline
    pipeline = PersonalityPipeline()
    
    try:
        submission = pipeline.run_complete_pipeline(TRAIN_PATH, TEST_PATH, TARGET_COL)
        
        if submission is not None:
            print("\n🎉 Pipeline completed successfully!")
            print("📊 Check 'personality_prediction_submission.csv' for your predictions")
            print("🏆 Good luck in the competition!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. Please update TRAIN_PATH and TEST_PATH.")
        print(f"   Current paths: {TRAIN_PATH}, {TEST_PATH}")
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        print("   Check your data format and column names")

if __name__ == "__main__":
    main()

# Additional utility functions for analysis
def analyze_predictions(y_true, y_pred, y_prob):
    """Analyze model predictions in detail"""
    print("PREDICTION ANALYSIS")
    print("="*40)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

def feature_importance_analysis(model, feature_names):
    """Analyze feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("TOP 20 MOST IMPORTANT FEATURES:")
        print("="*40)
        print(importance_df.head(20))
        
        return importance_df
    else:
        print("Model does not have feature_importances_ attribute")
        return None
