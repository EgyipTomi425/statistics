import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

pd.set_option('display.float_format', lambda x: '%.2f' % x)
np.set_printoptions(suppress=True, precision=2)

data = pd.read_csv('../rice.csv')
y = data['Class']

features = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']
X = data[features]

stats = pd.DataFrame({
    'Mean': X.mean(),
    'Std': X.std(),
    'Skew': X.apply(skew),
    'Kurtosis': X.apply(lambda x: kurtosis(x, fisher=False))
})
plt.figure(figsize=(12,6))
sns.heatmap(stats, annot=True, fmt='.2f', cmap='YlGnBu', cbar=True)
plt.title('Feature Statistics (All Features)')
plt.show()

plt.figure(figsize=(12,6))
for i, col in enumerate(features):
    plt.subplot(2,4,i+1)
    sns.boxplot(x=y, y=X[col])
    plt.title(col)
plt.suptitle('Boxplots Grid')
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

plt.figure(figsize=(12,6))
for i, col in enumerate(features):
    plt.subplot(2,4,i+1)
    sns.histplot(data=X, x=col, hue=y, kde=True, element='step', stat='density')
    plt.title(col)
plt.suptitle('Feature Distributions by Class')
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

corr = X.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=None)
plt.title('Correlation Matrix (Full)')
plt.show()

sns.pairplot(data=pd.concat([X, y], axis=1), hue='Class', corner=False)
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
sigma_vals = np.sqrt(pca.explained_variance_)
comp_weights = pca.components_

plt.figure(figsize=(8,5))
plt.plot(range(1,len(explained_var)+1), explained_var, 'o-', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

fig, ax = plt.subplots(figsize=(8,4))
ax.axis('off')
table_data = [['Comp', 'Sigma', 'Var %', 'Cum %']] + \
             [[i+1, f'{sigma_vals[i]:.5f}', f'{explained_var[i]*100:.4f}', f'{np.cumsum(explained_var)[i]*100:.4f}'] 
              for i in range(len(explained_var))]
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1]*4)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('=== PCA SVD Summary ===', fontweight='bold')
plt.show()

top_vars = {}
for i in range(len(features)):
    sorted_idx = np.argsort(np.abs(pca.components_[i]))[::-1][:3]
    top_vars[f'PC{i+1}'] = [(features[j], pca.components_[i,j]) for j in sorted_idx]

fig, ax = plt.subplots(figsize=(8,4))
ax.axis('off')
table_data = [['Component', 'Top 3 Variables']] + \
             [[comp, ', '.join([f'{v[0]}({v[1]:.6f})' for v in vars_])] for comp, vars_ in top_vars.items()]
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.15,0.7])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
plt.title('--- Top 3 variables per component ---', fontweight='bold')
plt.show()

fig, ax = plt.subplots(figsize=(10,4))
ax.axis('off')
table_data = [features] + [[f'{comp_weights[i,j]:.6f}' for j in range(len(features))] for i in range(len(features))]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('--- Component weights (all features) ---', fontweight='bold')
plt.show()

split_size = 800
sss = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=42)

for train_idx, test_idx in sss.split(X_scaled, y):
    X_train = X_scaled[train_idx]
    X_test = X_scaled[test_idx]

    X_train_pca2 = X_pca[train_idx, :2]
    X_test_pca2 = X_pca[test_idx, :2]

    X_train_pca3 = X_pca[train_idx, :3]
    X_test_pca3 = X_pca[test_idx, :3]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

models = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": GaussianNB(),
    "SVM_linear": SVC(kernel='linear', probability=True),
    "SVM_rbf": SVC(kernel='rbf', probability=True),
    "SVM_poly": SVC(kernel='poly', degree=3, probability=True)
}

def run_models_grid(models, X_tr, X_te, y_tr, y_te, title):
    n = len(models)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols,4*rows))
    axes = axes.flatten()
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_tr, y_tr)
        y_pred_train = model.predict(X_tr)
        y_pred_test = model.predict(X_te)
        acc_train = accuracy_score(y_tr, y_pred_train)
        acc_test = accuracy_score(y_te, y_pred_test)
        ConfusionMatrixDisplay.from_predictions(
            y_te, y_pred_test, ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"{name}\nTrain Acc={acc_train:.3f}, Test Acc={acc_test:.3f}")
    for i in range(len(models), len(axes)):
        fig.delaxes(axes[i])
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

run_models_grid(models, X_train, X_test, y_train_enc, y_test_enc, "Confusion Matrices — All Features")
run_models_grid(models, X_train_pca3, X_test_pca3, y_train_enc, y_test_enc, "Confusion Matrices — PCA (3 Components)")
run_models_grid(models, X_train_pca2, X_test_pca2, y_train_enc, y_test_enc, "Confusion Matrices — PCA (2 Components)")

n_models = len(models)
cols = 3
rows = int(np.ceil(n_models / cols))
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train_pca2, y_train_enc)
    y_pred_train = model.predict(X_train_pca2)
    y_pred_test = model.predict(X_test_pca2)
    acc_train = accuracy_score(y_train_enc, y_pred_train)
    acc_test = accuracy_score(y_test_enc, y_pred_test)
    x_min, x_max = X_test_pca2[:,0].min()-1, X_test_pca2[:,0].max()+1
    y_min, y_max = X_test_pca2[:,1].min()-1, X_test_pca2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3)
    except:
        pass
    sns.scatterplot(x=X_test_pca2[:,0], y=X_test_pca2[:,1],
                    hue=le.inverse_transform(y_test_enc),
                    palette='Set1', ax=ax, edgecolor='k', s=40, legend=False)
    ax.set_title(f"{name}\nTrain Acc={acc_train:.3f}, Test Acc={acc_test:.3f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

for i in range(n_models, len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Decision Surfaces on Test Set (2D PCA)", fontsize=16)
plt.tight_layout()
plt.show()