import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 7,
    'min_samples_leaf': 10,
    'min_samples_split': 20,
    'random_state': 42,
    'n_jobs': -1
}
OUTPUT_FILE = 'model_rf.bin'

def load_and_prepare_data():
    red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('data/winequality-white.csv', sep=';')
    
    red_wine['is_red'] = 1
    white_wine['is_red'] = 0
    
    df = pd.concat([red_wine, white_wine], ignore_index=True, sort=False)
    
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    df['label'] = (df['quality'] >= 7).astype(int)
    del df['quality']
    
    return df

def run_training_pipeline(df):
    full_train, test = train_test_split(df, test_size=0.10, random_state=0, stratify=df['label'])
    
    X_full_train = full_train.drop(columns='label')
    y_full_train = full_train['label']
    
    X_test = test.drop(columns='label')
    y_test = test['label']
    
    print("Training RandomForest (full_train)...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_full_train, y_full_train)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    roc_test = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC (Test): {roc_test:.4f}")
    
    return model

def export_model(model):
    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump(model, f_out)
    print(f"Model exported to {OUTPUT_FILE}")

if __name__ == '__main__':
    df_wine = load_and_prepare_data()
    final_model = run_training_pipeline(df_wine)
    export_model(final_model)