import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Assuming you have:
# z: numpy array of shape [n_samples, embedding_size]
# labels: numpy array of shape [n_samples] with binary values (0, 1)


#in postprocessing notebook you can run the test


def downstream_proteins(data,z, labels):
    def train_xgb_classifier(task, random_state=42):
        # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(
        #     z, labels, test_size=test_size, stratify=labels
        # )

        # Initialize and train the XGBoost classifier
        y_train = labels_train[:, task]
        y_test = labels_test[:, task]

        num_pos = (y_train == 1).sum()
        num_neg = (y_train == 0).sum()
        scale_pos_weight = num_neg / num_pos

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,  # Number of boosting rounds
            learning_rate=0.1,  # Step size shrinkage to prevent overfitting
            max_depth=3,  # Maximum depth of a tree
            # min_child_weight=1,     # Minimum sum of instance weight needed in a child
            # gamma=0,                # Minimum loss reduction required for split
            subsample=0.8,  # Subsample ratio of training instances
            colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',  # Binary classification
            eval_metric='logloss',  # Evaluation metric
            random_state=random_state
        )

        # Train the model
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Make predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        # report = classification_report(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

        return xgb_model, accuracy, auc


    X_train = z[data.train_idx, :]
    # test_idx = torch.cat((data.valid_idx, data.test_idx), dim=0)
    X_test = z[data.test_idx, :]
    labels_train = labels[data.train_idx, :]
    labels_test = labels[data.test_idx, :]
    # Example usage
    avg_accuracy = 0
    avg_auc = 0
    for task in range(z.shape[1]):
        if task == 112: break
        model, accuracy, auc = train_xgb_classifier(task=task)
        avg_accuracy += accuracy
        avg_auc += auc
    avg_accuracy /= z.shape[1]
    avg_auc /= z.shape[1]
    print(f"Average accuracy: {avg_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")