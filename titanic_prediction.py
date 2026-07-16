"""
Titanic Survival Prediction Project

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
import os

# Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

print("="*60)
print("TITANIC SURVIVAL PREDICTION PROJECT")
print("="*60)


# STEP 1: Load and Explore the Data

print("STEP 1: LOADING AND EXPLORING THE DATA")

# Load the dataset
data_path = os.path.join(DATA_DIR, 'Titanic-Dataset.csv')
df = pd.read_csv(data_path)
print(f"Dataset loaded successfully! Shape: {df.shape}")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Data overview
print("\n" + "-"*40)
print("DATASET INFORMATION:")
print("-"*40)
print(df.info())

print("\n" + "-"*40)
print("STATISTICAL SUMMARY:")
print("-"*40)
print(df.describe())

# Check for missing values
print("\n" + "-"*40)
print("MISSING VALUES:")
print("-"*40)
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing %': missing_percent
})
print(missing_df[missing_df['Missing Count'] > 0])


# STEP 2: Exploratory Data Analysis (EDA) and Visualization

print("STEP 2: EXPLORATORY DATA ANALYSIS")

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# 2.1 Survival Distribution
plt.figure(figsize=(10, 6))
survival_counts = df['Survived'].value_counts()
colors = ['#ff6b6b', '#51cf66']
plt.pie(survival_counts, labels=['Not Survived', 'Survived'], 
        autopct='%1.1f%%', startangle=90, colors=colors,
        explode=(0.05, 0.05), shadow=True)
plt.title('Titanic Passenger Survival Distribution', fontsize=16, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, '1_survival_distribution_pie.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.2 Survival by Gender
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Sex', hue='Survived', palette=['#ff6b6b', '#51cf66'])
plt.title('Survival by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig(os.path.join(OUTPUT_DIR, '2_survival_by_gender.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.3 Survival by Passenger Class
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Pclass', hue='Survived', palette=['#ff6b6b', '#51cf66'])
plt.title('Survival by Passenger Class', fontsize=16, fontweight='bold')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig(os.path.join(OUTPUT_DIR, '3_survival_by_class.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.4 Age Distribution by Survival
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30, palette=['#ff6b6b', '#51cf66'])
plt.title('Age Distribution by Survival', fontsize=16, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig(os.path.join(OUTPUT_DIR, '4_age_distribution_by_survival.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.5 Fare Distribution by Survival
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Fare', hue='Survived', kde=True, bins=30, palette=['#ff6b6b', '#51cf66'])
plt.title('Fare Distribution by Survival', fontsize=16, fontweight='bold')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.xlim(0, 300)
plt.legend(['Not Survived', 'Survived'])
plt.savefig(os.path.join(OUTPUT_DIR, '5_fare_distribution_by_survival.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.6 Correlation Heatmap
plt.figure(figsize=(12, 10))
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, mask=mask)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, '6_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.7 Survival by Embarkation Port
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Embarked', hue='Survived', palette=['#ff6b6b', '#51cf66'])
plt.title('Survival by Embarkation Port', fontsize=16, fontweight='bold')
plt.xlabel('Embarkation Port')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig(os.path.join(OUTPUT_DIR, '7_survival_by_embarkation.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.8 Family Size vs Survival
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='FamilySize', y='Survived', palette='viridis')
plt.title('Survival Rate by Family Size', fontsize=16, fontweight='bold')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.savefig(os.path.join(OUTPUT_DIR, '8_survival_by_family_size.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.9 Age Group vs Survival (for visualization only)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
plt.figure(figsize=(12, 6))
age_survival = df.groupby('AgeGroup')['Survived'].mean()
age_survival.plot(kind='bar', color='steelblue')
plt.title('Survival Rate by Age Group', fontsize=16, fontweight='bold')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.ylim(0, 1)
for i, v in enumerate(age_survival):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, '9_survival_by_age_group.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2.10 Fare Category vs Survival (for visualization only)
df['FareCategory'] = pd.qcut(df['Fare'].fillna(0), q=4, labels=['Low', 'Medium', 'High', 'Very High'])
plt.figure(figsize=(12, 6))
fare_survival = df.groupby('FareCategory')['Survived'].mean()
fare_survival.plot(kind='bar', color='steelblue')
plt.title('Survival Rate by Fare Category', fontsize=16, fontweight='bold')
plt.xlabel('Fare Category')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.ylim(0, 1)
for i, v in enumerate(fare_survival):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, '10_survival_by_fare_category.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All EDA visualizations saved to output directory!")


# STEP 3: Data Preprocessing

print("STEP 3: DATA PREPROCESSING")

# Create a copy of the dataset
df_processed = df.copy()

# Drop columns that are not useful for prediction
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'AgeGroup', 'FareCategory']
df_processed = df_processed.drop(columns=columns_to_drop, errors='ignore')

print(f"Columns after dropping unnecessary features: {df_processed.columns.tolist()}")

# Handle missing values - Age
print("\nHandling missing values...")
age_median = df_processed['Age'].median()
df_processed['Age'] = df_processed['Age'].fillna(age_median)
print(f"Age missing values filled with median: {age_median}")

# Handle missing values - Embarked
embarked_mode = df_processed['Embarked'].mode()[0]
df_processed['Embarked'] = df_processed['Embarked'].fillna(embarked_mode)
print(f"Embarked missing values filled with mode: {embarked_mode}")

# Handle missing values - Fare
fare_median = df_processed['Fare'].median()
df_processed['Fare'] = df_processed['Fare'].fillna(fare_median)
print(f"Fare missing values filled with median: {fare_median}")

# Verify no missing values remain
print(f"\nRemaining missing values after imputation: {df_processed.isnull().sum().sum()}")

# Encode categorical variables
print("\nEncoding categorical variables...")
le_sex = LabelEncoder()
df_processed['Sex'] = le_sex.fit_transform(df_processed['Sex'])  # male=0, female=1

le_embarked = LabelEncoder()
df_processed['Embarked'] = le_embarked.fit_transform(df_processed['Embarked'])

print("Categorical variables encoded ")

# Create additional features
print("\nCreating additional features...")
df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
df_processed['Age*Class'] = df_processed['Age'] * df_processed['Pclass']

print("Additional features created:")
print(df_processed[['FamilySize', 'IsAlone', 'Age*Class']].head())

# Final check for NaN values
print("\n" + "-"*40)
print("FINAL DATA CHECK:")
print("-"*40)
print(f"Total NaN values in processed data: {df_processed.isnull().sum().sum()}")
print(f"Data types:\n{df_processed.dtypes}")


# STEP 4: Split Data into Features and Target

print("STEP 4: SPLITTING DATA FOR MODELING")

# Separate features and target
X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures: {X.columns.tolist()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training set survival distribution:\n{y_train.value_counts()}")
print(f"Test set survival distribution:\n{y_test.value_counts()}")

# Scale numerical features
print("\nScaling numerical features...")
scaler = StandardScaler()
numeric_features = ['Age', 'Fare', 'FamilySize', 'Age*Class']

# Create scaled copies
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Fit and transform on training data, transform on test data
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# Check for NaN after scaling (should be None)
print(f"NaN in X_train_scaled: {X_train_scaled.isnull().sum().sum()}")
print(f"NaN in X_test_scaled: {X_test_scaled.isnull().sum().sum()}")


# STEP 5: Model Training and Evaluation

print("STEP 5: MODEL TRAINING AND EVALUATION")

# Define models to test
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Store results
results = {}
predictions = {}

print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\n{'-'*40}")
    print(f"Model: {name}")
    print(f"{'-'*40}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'ROC AUC': roc_auc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    }
    predictions[name] = y_pred
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Compare models

print("MODEL COMPARISON")

results_df = pd.DataFrame(results).T
print(results_df)


# STEP 6: Model Comparison Visualization


print("\n" + "="*60)
print("STEP 6: MODEL COMPARISON VISUALIZATION")
print("="*60)

# 6.1 Model Performance Comparison
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'ROC AUC', 'CV Mean']
x = np.arange(len(metrics))
width = 0.2
colors_models = ['#ff6b6b', '#51cf66', '#4dabf7', '#fcc419']

for i, (model_name, scores) in enumerate(results.items()):
    values = [scores[metric] for metric in metrics]
    plt.bar(x + i*width, values, width, label=model_name, color=colors_models[i])

plt.xlabel('Metric', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xticks(x + width*1.5, metrics)
plt.legend(loc='lower right')
plt.ylim(0.7, 1.0)
for i, (model_name, scores) in enumerate(results.items()):
    values = [scores[metric] for metric in metrics]
    for j, v in enumerate(values):
        plt.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
plt.savefig(os.path.join(OUTPUT_DIR, '11_model_performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.2 Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    axes[idx].set_title(f'{name}\nAccuracy: {results[name]["Accuracy"]:.3f}', fontsize=12)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.3 ROC Curves
plt.figure(figsize=(10, 8))

for name, model in models.items():
    # Get predictions
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["ROC AUC"]:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for All Models', fontsize=16, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, '13_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6.4 Feature Importance (Random Forest)
plt.figure(figsize=(12, 8))
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
for i, v in enumerate(feature_importance['Importance']):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center')
plt.savefig(os.path.join(OUTPUT_DIR, '14_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All comparison visualizations saved to output directory!")


# STEP 7: Hyperparameter Tuning (Best Model)


print("STEP 7: HYPERPARAMETER TUNING")

# Find the best performing model
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
print(f"Best performing model: {best_model_name}")

if best_model_name == 'Random Forest':
    print("\nTuning Random Forest hyperparameters...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    best_rf = grid_search.best_estimator_
    y_pred_tuned = best_rf.predict(X_test_scaled)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    print(f"Tuned model test accuracy: {tuned_accuracy:.4f}")
    best_model = best_rf
else:
    best_model = models[best_model_name]
    print(f"Using {best_model_name} without tuning")


# STEP 8: Save the Best Model

print("STEP 8: SAVING THE BEST MODEL")


import joblib

# Save the best model
model_path = os.path.join(MODEL_DIR, 'titanic_best_model.pkl')
joblib.dump(best_model, model_path)
print(f"Best model saved to: {model_path}")

# Save the scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# Save encoders
encoders_path = os.path.join(MODEL_DIR, 'encoders.pkl')
joblib.dump({'sex': le_sex, 'embarked': le_embarked}, encoders_path)
print(f"Encoders saved to: {encoders_path}")


# STEP 9: Summary Report


print("PROJECT SUMMARY REPORT")

print(f"""
PROJECT: Titanic Survival Prediction

DATASET INFORMATION:
- Total samples: {len(df)}
- Features: {df.shape[1]}
- Survival Rate: {df['Survived'].mean():.2%}

MISSING VALUES HANDLED:
- Age: {df['Age'].isnull().sum()} missing values (filled with median: {age_median:.2f})
- Embarked: {df['Embarked'].isnull().sum()} missing values (filled with mode: {df['Embarked'].mode()[0]})
- Fare: {df['Fare'].isnull().sum()} missing values (filled with median: {fare_median:.2f})
- Cabin: Dropped (too many missing values)
- Ticket: Dropped (not useful for prediction)
- Name: Dropped (not useful for prediction)

FEATURES USED:
- Pclass (Passenger Class)
- Sex (Encoded)
- Age (Scaled)
- SibSp (Siblings/Spouses Aboard)
- Parch (Parents/Children Aboard)
- Fare (Scaled)
- Embarked (Encoded)
- FamilySize (Engineered)
- IsAlone (Engineered)
- Age*Class (Engineered)

MODEL PERFORMANCE:
""")

for name, scores in results.items():
    print(f"{name}:")
    print(f"  - Accuracy: {scores['Accuracy']:.4f}")
    print(f"  - ROC AUC: {scores['ROC AUC']:.4f}")
    print(f"  - CV Score: {scores['CV Mean']:.4f} (+/- {scores['CV Std']:.4f})")

print(f"\nBEST MODEL: {best_model_name}")
print(f"Best Model Accuracy: {results[best_model_name]['Accuracy']:.4f}")

print(f"\nOUTPUT FILES GENERATED:")
print(f"  - Data directory: {DATA_DIR}")
print(f"  - Output directory: {OUTPUT_DIR} (14 visualizations)")
print(f"  - Model directory: {MODEL_DIR} (3 files)")

