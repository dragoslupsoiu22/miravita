"""Prototip educațional pentru setul Wisconsin de cancer mamar.

Nu este destinat diagnosticării sau îngrijirii pacienților.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_breast_cancer()

    # În setul sklearn, malignant este codificat cu 0, benign cu 1.
    # Transformăm etichetele astfel încât malignant să fie clasa pozitivă (1).
    X = data.data
    y = (data.target == 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    malignant_scores = model.predict_proba(X_test)[:, 1]

    # Ordinea claselor: benign (0), malignant (1)
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print("Setul Wisconsin — evaluare pe date de test")
    print("Clasa pozitivă: malignă")
    print(f"Acuratețe:    {accuracy_score(y_test, predictions):.3f}")
    print(
        "Precizie:     "
        f"{precision_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(
        "Sensibilitate:"
        f" {recall_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(f"Specificitate: {specificity:.3f}")
    print(f"ROC AUC:      {roc_auc_score(y_test, malignant_scores):.3f}")
    print("Matrice de confuzie (rânduri=real, coloane=prezis; benign, malign):")
    print(f"[[{tn:3d} {fp:3d}]")
    print(f" [{fn:3d} {tp:3d}]]")
    print("\nDoar pentru învățare; nu folosiți pentru decizii medicale.")


if __name__ == "__main__":
    main()
