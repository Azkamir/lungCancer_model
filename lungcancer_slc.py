
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv('/SLC_ds.csv')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
X = df.drop('LUNG_CANCER',axis='columns')
y = df.LUNG_CANCER

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
fs_model = RandomForestClassifier(n_estimators=100, random_state=42)
fs_model.fit(x_train, y_train)
selector = SelectFromModel(fs_model, threshold='median')
selector.fit(x_train, y_train)
x_train_fs = selector.transform(x_train)
x_test_fs = selector.transform(x_test)
selected_features = X.columns[selector.get_support()].tolist()
print("Selected Features:", selected_features)

rf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
svm = SVC(kernel='rbf', C=0.7, gamma='scale', probability=True, random_state=42)
lr = LogisticRegression(C=0.5, solver='liblinear',  random_state=42)

ensemble = VotingClassifier(estimators=[
    ('lr', lr),
    ('rf', rf),
    ('svm', svm)
], voting='soft', weights=[1, 2, 1])

pipeline = ImbPipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(k_neighbors=4, random_state=42)),
    ('model', ensemble)
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
cv_results = cross_validate(pipeline, x_train_fs, y_train, cv=cv, scoring=scoring)

print("\nCross-Validation Results (10-fold):")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.3f} ± {scores.std():.3f}")

pipeline.fit(x_train_fs, y_train)
y_pred = pipeline.predict(x_test_fs)

print("\nClassification Report on Hold-out Test Set:")
print(classification_report(y_test, y_pred))
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.3f}")

import shap
import matplotlib.pyplot as plt
final_model = pipeline.named_steps['model']

explainer = shap.Explainer(final_model.predict_proba, x_train_fs)

shap_values = explainer(x_test_fs)

shap.summary_plot(shap_values, x_test_fs, feature_names=selected_features)

shap.summary_plot(shap_values, x_test_fs, feature_names=selected_features, plot_type="bar")

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log_reg = final_model.named_estimators_['lr']

explainer = shap.Explainer(log_reg, x_train_fs)
shap_values = explainer(x_test_fs)
mean_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance = pd.DataFrame({
    "Feature": selected_features,
    "Importance": mean_shap
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(8,6))
plt.barh(shap_importance["Feature"], shap_importance["Importance"], color="skyblue")
plt.xlabel("Mean(|SHAP Value|)", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Feature Importance (SHAP)", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()