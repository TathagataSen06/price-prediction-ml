"""
Complete 75K Predictions - Final Aggressive Model
Generates predictions for ALL 75,000 test samples with exact sample_id matching
"""
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def smape(actual, predicted):
    return np.mean(np.abs(predicted - actual) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100

def extract_ultimate_features(df):
    """Ultimate feature extraction - same as final_aggressive_push.py"""
    f = pd.DataFrame(index=df.index)
    
    # Ultimate IPQ extraction
    def extract_ipq_ultimate(text):
        patterns = [
            r'(?:IPQ|Pack Quantity|Quantity|Pack|Count):\s*(\d+)',
            r'(\d+)\s*(?:pack|units?|pieces?|count|pcs|items?|ct)',
            r'set\s+of\s+(\d+)',
            r'(\d+)\s*(?:-|x|×)\s*pack',
            r'(\d+)\s*(?:piece|pc|unit|each)',
            r'quantity[:\s]*(\d+)',
            r'(\d+)\s*in\s*(?:1|one|pack)',
            r'(\d+)\s*per\s*(?:pack|box|case)',
            r'bulk\s*(\d+)',
            r'(\d+)\s*(?:roll|sheet|bag|box)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                return min(val, 1000)  # Cap at 1000
        return 1
    
    f['ipq'] = df['catalog_content'].apply(extract_ipq_ultimate)
    f['ipq_log'] = np.log1p(f['ipq'])
    f['ipq_sqrt'] = np.sqrt(f['ipq'])
    f['ipq_cbrt'] = np.power(f['ipq'], 1/3)
    
    # Advanced text statistics
    f['len'] = df['catalog_content'].str.len()
    f['len_log'] = np.log1p(f['len'])
    f['len_sqrt'] = np.sqrt(f['len'])
    
    f['words'] = df['catalog_content'].str.split().str.len()
    f['words_log'] = np.log1p(f['words'])
    
    f['sentences'] = df['catalog_content'].str.count(r'[.!?]+') + 1
    f['avg_word_len'] = f['len'] / np.maximum(f['words'], 1)
    f['words_per_sentence'] = f['words'] / np.maximum(f['sentences'], 1)
    
    # Character analysis
    f['digits'] = df['catalog_content'].str.count(r'\d')
    f['capitals'] = df['catalog_content'].str.count(r'[A-Z]')
    f['special_chars'] = df['catalog_content'].str.count(r'[^\w\s]')
    f['digit_ratio'] = f['digits'] / np.maximum(f['len'], 1)
    f['capital_ratio'] = f['capitals'] / np.maximum(f['len'], 1)
    
    # Price-predictive keywords (comprehensive)
    keywords = {
        'premium': ['premium', 'luxury', 'deluxe', 'professional', 'pro', 'elite', 'superior', 'high-end', 'top-tier'],
        'budget': ['budget', 'basic', 'economy', 'cheap', 'value', 'standard', 'affordable', 'discount'],
        'size': ['large', 'xl', 'xxl', 'big', 'jumbo', 'king', 'giant', 'mega', 'super', 'extra'],
        'quality': ['quality', 'durable', 'heavy-duty', 'commercial', 'industrial', 'grade', 'certified'],
        'brand': ['apple', 'samsung', 'sony', 'nike', 'canon', 'hp', 'dell', 'microsoft', 'intel'],
        'material': ['steel', 'aluminum', 'plastic', 'wood', 'metal', 'ceramic', 'glass', 'leather'],
        'tech': ['smart', 'digital', 'electronic', 'wireless', 'bluetooth', 'wifi', 'usb', 'led']
    }
    
    for category, words in keywords.items():
        pattern = '|'.join([f'\\b{w}\\b' for w in words])
        f[f'{category}_count'] = df['catalog_content'].str.count(pattern, flags=re.I)
        f[f'has_{category}'] = (f[f'{category}_count'] > 0).astype(int)
    
    # Advanced number extraction
    def extract_numbers_ultimate(text):
        patterns = [r'\d+\.\d+', r'\d+']
        numbers = []
        for pattern in patterns:
            numbers.extend([float(m) for m in re.findall(pattern, text)])
        return numbers if numbers else [0]
    
    df['numbers'] = df['catalog_content'].apply(extract_numbers_ultimate)
    f['max_num'] = df['numbers'].apply(lambda x: min(max(x), 10000))
    f['max_num_log'] = np.log1p(f['max_num'])
    f['min_num'] = df['numbers'].apply(min)
    f['num_count'] = df['numbers'].apply(len)
    f['num_mean'] = df['numbers'].apply(np.mean)
    f['num_std'] = df['numbers'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    f['num_range'] = f['max_num'] - f['min_num']
    
    # Interaction features
    f['ipq_x_len'] = f['ipq'] * f['len_log']
    f['premium_x_len'] = f['premium_count'] * f['len_log']
    f['quality_x_ipq'] = f['quality_count'] * f['ipq_log']
    f['brand_x_premium'] = f['brand_count'] * f['premium_count']
    
    # Price indicators
    f['has_price'] = df['catalog_content'].str.contains(r'\$\d+|\d+\s*(?:dollar|usd)', flags=re.I).astype(int)
    f['has_dimensions'] = df['catalog_content'].str.contains(r'\d+\s*(?:inch|cm|mm|ft)', flags=re.I).astype(int)
    f['has_weight'] = df['catalog_content'].str.contains(r'\d+\s*(?:lb|kg|oz|gram)', flags=re.I).astype(int)
    
    return f.fillna(0)

def add_tfidf_features(train_df, test_df, max_features=500):
    """Add TF-IDF features"""
    print("  📝 Processing TF-IDF features...")
    
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    
    # Fit on combined text
    all_text = pd.concat([train_df['catalog_content'], test_df['catalog_content']])
    tfidf_matrix = tfidf.fit_transform(all_text)
    
    # Reduce dimensions
    svd = TruncatedSVD(n_components=50, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    
    # Split back
    n_train = len(train_df)
    train_tfidf = tfidf_reduced[:n_train]
    test_tfidf = tfidf_reduced[n_train:]
    
    # Create DataFrames
    cols = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    
    return (pd.DataFrame(train_tfidf, columns=cols, index=train_df.index),
            pd.DataFrame(test_tfidf, columns=cols, index=test_df.index))

def train_ultimate_ensemble(X_train, y_train_t):
    """Train ultimate ensemble"""
    print("  🤖 Training ensemble models...")
    models = []
    
    # Model 1: Random Forest
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_t)
    models.append(('rf', rf))
    print("    ✅ Random Forest trained")
    
    # Model 2: Extra Trees
    et = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    et.fit(X_train, y_train_t)
    models.append(('et', et))
    print("    ✅ Extra Trees trained")
    
    # Model 3: Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    gb.fit(X_train, y_train_t)
    models.append(('gb', gb))
    print("    ✅ Gradient Boosting trained")
    
    # Model 4: Neural Network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    nn = MLPRegressor(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=300,
        random_state=42
    )
    nn.fit(X_train_scaled, y_train_t)
    models.append(('nn', nn, scaler))
    print("    ✅ Neural Network trained")
    
    return models

def ensemble_predict_ultimate(models, X):
    """Ultimate ensemble prediction"""
    predictions = []
    
    for item in models:
        if len(item) == 2:  # Regular model
            name, model = item
            pred = model.predict(X)
        else:  # Neural network with scaler
            name, model, scaler = item
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
        
        predictions.append(pred)
    
    return np.mean(predictions, axis=0)

def main():
    print("🚀 COMPLETE 75K PREDICTIONS - FINAL AGGRESSIVE MODEL")
    print("Generating predictions for ALL 75,000 test samples")
    print("=" * 70)
    
    # Load complete datasets
    print("📂 Loading complete datasets...")
    train_df = pd.read_csv("student_resource/dataset/train.csv")
    test_df = pd.read_csv("student_resource/dataset/test.csv")
    
    print(f"✅ Training data: {len(train_df):,} samples")
    print(f"✅ Test data: {len(test_df):,} samples")
    
    # Verify test data integrity
    print(f"\n🔍 Test data verification:")
    print(f"  Sample ID range: {test_df['sample_id'].min()} - {test_df['sample_id'].max()}")
    print(f"  Unique sample IDs: {test_df['sample_id'].nunique()}")
    print(f"  Duplicates: {test_df['sample_id'].duplicated().sum()}")
    
    if test_df['sample_id'].nunique() != len(test_df):
        print("⚠️  Warning: Duplicate sample IDs found!")
    else:
        print("✅ All sample IDs are unique")
    
    # Use substantial training data for better model
    train_subset = train_df.head(25000)  # Use 25k for training
    print(f"📊 Using {len(train_subset):,} training samples for model")
    
    # Split training data for validation
    train_split, val_split = train_test_split(train_subset, test_size=0.15, random_state=42)
    
    print(f"  Training split: {len(train_split):,}")
    print(f"  Validation split: {len(val_split):,}")
    
    # Feature extraction
    print(f"\n🔧 Feature extraction for all datasets...")
    print("  📊 Extracting ultimate features...")
    
    train_features = extract_ultimate_features(train_split)
    val_features = extract_ultimate_features(val_split)
    test_features = extract_ultimate_features(test_df)
    
    print(f"    ✅ Basic features: {train_features.shape[1]}")
    
    # Add TF-IDF features
    train_tfidf, test_tfidf = add_tfidf_features(train_split, test_df)
    val_tfidf, _ = add_tfidf_features(val_split, test_df.head(100))  # Small subset for val
    
    print(f"    ✅ TF-IDF features: {train_tfidf.shape[1]}")
    
    # Combine features
    X_train = pd.concat([train_features, train_tfidf], axis=1)
    X_val = pd.concat([val_features, val_tfidf], axis=1)
    X_test = pd.concat([test_features, test_tfidf], axis=1)
    
    y_train = train_split['price'].values
    y_val = val_split['price'].values
    
    print(f"📊 Final feature count: {X_train.shape[1]}")
    print(f"📊 Training shape: {X_train.shape}")
    print(f"📊 Test shape: {X_test.shape}")
    
    # Target transformation
    print(f"\n⚡ Target transformation...")
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    y_train_t = pt.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    print(f"  Original target range: ${y_train.min():.2f} - ${y_train.max():.2f}")
    print(f"  Transformed target range: {y_train_t.min():.3f} - {y_train_t.max():.3f}")
    
    # Train ensemble
    print(f"\n🤖 Training ultimate ensemble...")
    models = train_ultimate_ensemble(X_train.values, y_train_t)
    
    # Quick validation
    print(f"\n📈 Quick validation check...")
    val_pred_t = ensemble_predict_ultimate(models, X_val.values)
    val_pred = pt.inverse_transform(val_pred_t.reshape(-1, 1)).ravel()
    val_pred = np.maximum(val_pred, 0.01)
    
    val_smape = smape(y_val, val_pred)
    print(f"  Validation SMAPE: {val_smape:.2f}")
    
    # Generate predictions for ALL 75k test samples
    print(f"\n🎯 Generating predictions for ALL {len(test_df):,} test samples...")
    
    # Process in batches to handle memory
    batch_size = 5000
    all_predictions = []
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_X = X_test.iloc[i:batch_end].values
        
        print(f"  Processing batch {i//batch_size + 1}/{(len(X_test) + batch_size - 1)//batch_size}: samples {i+1}-{batch_end}")
        
        # Predict
        batch_pred_t = ensemble_predict_ultimate(models, batch_X)
        batch_pred = pt.inverse_transform(batch_pred_t.reshape(-1, 1)).ravel()
        batch_pred = np.maximum(batch_pred, 0.01)
        
        all_predictions.extend(batch_pred)
    
    # Apply final calibration (based on validation performance)
    final_predictions = np.array(all_predictions)
    
    # Simple calibration based on validation
    if val_smape < 60:
        # Good performance, light calibration
        calibration_factor = y_val.mean() / val_pred.mean()
        final_predictions = final_predictions * calibration_factor
    else:
        # Need stronger calibration
        final_predictions = final_predictions * 1.5
    
    final_predictions = np.maximum(final_predictions, 0.01)
    
    # Create final output with EXACT sample_id matching
    print(f"\n📋 Creating final output...")
    
    final_output = pd.DataFrame({
        'sample_id': test_df['sample_id'].values,  # Exact match
        'price': final_predictions
    })
    
    # Verify output integrity
    print(f"\n🔍 Output verification:")
    print(f"  Output samples: {len(final_output):,}")
    print(f"  Expected samples: {len(test_df):,}")
    print(f"  Sample ID match: {(final_output['sample_id'] == test_df['sample_id']).all()}")
    print(f"  Price range: ${final_output['price'].min():.2f} - ${final_output['price'].max():.2f}")
    print(f"  Mean price: ${final_output['price'].mean():.2f}")
    print(f"  Zero prices: {(final_output['price'] <= 0).sum()}")
    
    # Save final predictions
    output_filename = "complete_75k_final_predictions.csv"
    final_output.to_csv(output_filename, index=False)
    
    print(f"\n✅ COMPLETE 75K PREDICTIONS GENERATED!")
    print("=" * 70)
    print(f"📁 Output file: {output_filename}")
    print(f"📊 Total predictions: {len(final_output):,}")
    print(f"📈 Validation SMAPE: {val_smape:.2f}")
    print(f"💰 Price statistics:")
    print(f"  Min: ${final_output['price'].min():.2f}")
    print(f"  Max: ${final_output['price'].max():.2f}")
    print(f"  Mean: ${final_output['price'].mean():.2f}")
    print(f"  Median: ${final_output['price'].median():.2f}")
    print(f"  Std: ${final_output['price'].std():.2f}")
    print(f"🔧 Features used: {X_train.shape[1]}")
    print(f"🤖 Models: RF + ET + GB + NN ensemble")
    print(f"🎯 Ready for competition submission!")
    
    return final_output

if __name__ == "__main__":
    main()