import os
import re # ファイル名のサニタイズ用にインポート
import shutil
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import hdbscan

# --- 0. ログ設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 設定 ---
# ユーザーの環境に合わせて変更
INPUT_DIR = './input'
OUTPUT_DIR = './output'

# --- パラメータ調整エリア ---
# MIN_DF = 2 が最も良い結果だったため、それを採用
MIN_DF = 2
MAX_DF = 0.85
NGRAM_RANGE = (1, 2)
N_COMPONENTS = 50
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES = 1
METRIC = 'euclidean'
# フォルダ名にするキーワードの数
NUM_KEYWORDS = 3

# --- 2. ユーティリティ関数 ---
j_tokenizer = Tokenizer()

def japanese_tokenizer(text):
    tokens = []
    for token in j_tokenizer.tokenize(text):
        part_of_speech = token.part_of_speech.split(',')[0]
        if part_of_speech in ['名詞', '動詞', '形容詞', '副詞']:
            tokens.append(token.base_form)
    return tokens

def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return ""

def sanitize_filename(name):
    """ファイル名として使えない文字を_に置換する"""
    return re.sub(r'[\\/*?:"<>|]', '_', name)

# --- ★★★ 新しい関数 ★★★ ---
def get_cluster_keywords(cluster_id, labels, tfidf_matrix, vectorizer, top_n=3):
    """
    指定されたクラスタIDの代表的なキーワードを抽出する関数
    """
    # クラスタに属する文書のインデックスを取得
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        return ""
    
    # クラスタのTF-IDFベクトルを平均化
    cluster_vectors = tfidf_matrix[indices]
    mean_tfidf_vector = np.array(cluster_vectors.mean(axis=0)).flatten()
    
    # 上位N件の単語のインデックスを取得（スコアが高い順）
    top_indices = mean_tfidf_vector.argsort()[-top_n:][::-1]
    
    # インデックスを実際の単語に変換
    feature_names = vectorizer.get_feature_names_out()
    keywords = [feature_names[i] for i in top_indices]
    
    return "_".join(keywords)

# --- 4. メイン処理 ---
def main():
    if os.path.exists(OUTPUT_DIR):
        logging.info(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

    # ... (PDF読み込みとテキスト抽出部分は変更なし) ...
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files: logging.warning(f"No PDF files found in '{INPUT_DIR}' directory."); return
    filepaths = [os.path.join(INPUT_DIR, f) for f in pdf_files]
    logging.info("Step 1: Extracting text from PDFs...")
    documents = [extract_text_from_pdf(path) for path in filepaths]
    valid_indices = [i for i, doc in enumerate(documents) if doc and doc.strip()]
    if not valid_indices: logging.error("Could not extract any text from the PDFs."); return
    documents = [documents[i] for i in valid_indices]
    filepaths = [filepaths[i] for i in valid_indices]
    pdf_files = [os.path.basename(p) for p in filepaths]
    logging.info(f"Successfully extracted text from {len(documents)} PDFs.")

    # ... (ベクトル化、LSA、クラスタリング部分は変更なし) ...
    logging.info("Step 2: Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(tokenizer=japanese_tokenizer, max_df=MAX_DF, min_df=MIN_DF, ngram_range=NGRAM_RANGE)
    tfidf_matrix = vectorizer.fit_transform(documents)
    if tfidf_matrix.shape[1] == 0: logging.error(f"No features were extracted."); return
    logging.info(f"TF-IDF matrix created with shape: {tfidf_matrix.shape}")

    logging.info("Step 3: Applying LSA for dimensionality reduction...")
    if tfidf_matrix.shape[1] <= N_COMPONENTS:
        logging.warning(f"Feature count is less than N_COMPONENTS. Skipping LSA.")
        processed_matrix = tfidf_matrix.toarray()
    else:
        lsa = make_pipeline(TruncatedSVD(n_components=N_COMPONENTS, random_state=42), Normalizer(copy=False))
        processed_matrix = lsa.fit_transform(tfidf_matrix)
        logging.info(f"Matrix shape after LSA: {processed_matrix.shape}")

    logging.info("Step 4: Clustering using HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, metric=METRIC, prediction_data=True)
    labels = clusterer.fit_predict(processed_matrix)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)
    logging.info(f"\n--- Clustering Results ---\nFound {num_clusters} clusters.\nNumber of noise points: {num_noise} / {len(documents)}\n--------------------------\n")

    # --- ★★★ ここからが新しい結果整理のロジック ★★★ ---
    logging.info("Step 5: Generating cluster keywords and organizing files...")

    # クラスタIDとキーワードベースのフォルダ名を対応付ける辞書を作成
    cluster_id_to_foldername = {}
    for cluster_id in sorted(list(set(labels))):
        if cluster_id == -1:
            folder_name = 'cluster_noise'
        else:
            keywords = get_cluster_keywords(cluster_id, labels, tfidf_matrix, vectorizer, top_n=NUM_KEYWORDS)
            # 安全なファイル名に変換し、IDを接頭辞として付与して一意性を保証
            safe_keywords = sanitize_filename(keywords)
            folder_name = f"cluster_{cluster_id}_{safe_keywords}"
        cluster_id_to_foldername[cluster_id] = folder_name

    # 結果をDataFrameにまとめる（キーワードフォルダ名も追加）
    results_df = pd.DataFrame({
        'filename': pdf_files,
        'filepath': filepaths,
        'cluster_id': labels
    })
    results_df['cluster_folder'] = results_df['cluster_id'].map(cluster_id_to_foldername)
    
    csv_path = os.path.join(OUTPUT_DIR, 'clustering_results.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"Results with folder names saved to '{csv_path}'")

    # キーワードフォルダ名を使ってディレクトリを作成し、PDFをコピー
    for folder_name in set(cluster_id_to_foldername.values()):
        cluster_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(cluster_dir, exist_ok=True)

        files_in_cluster = results_df[results_df['cluster_folder'] == folder_name]
        logging.info(f"Copying {len(files_in_cluster)} files to '{cluster_dir}'")
        for _, row in files_in_cluster.iterrows():
            shutil.copy(row['filepath'], cluster_dir)
            
    logging.info(f"\nProcess complete. PDFs have been organized into keyword-based subdirectories inside '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    main()