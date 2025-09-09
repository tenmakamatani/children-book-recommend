import json
import numpy as np
from janome.tokenizer import Tokenizer
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import f1_score, jaccard_score, precision_score

# ==== 1. データ読み込み ====
data = json.loads(Path("extracted_metadata_raw.json").read_text(encoding="utf-8"))

texts = [item["summary"] for item in data]   # summaryをスペース区切りの文章に
labels = [item["subjects"] for item in data]          # subjectsは主題リスト


# ==== 2. ラベルをmulti-hot化 ====
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)


# ==== 3. 学習・テスト分割 ====
X_train, X_test, Y_train, Y_test = train_test_split(
    texts, Y, test_size=0.2, random_state=42
)

t = Tokenizer()
def tokenize_ja(text: str):
    toks = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]  # "名詞","動詞","形容詞",...
        if pos in ("名詞", "動詞", "形容詞"):
            base = token.base_form if token.base_form != "*" else token.surface
            toks.append(base)
    return toks

# ==== 4. 特徴量: TF-IDF ====
word_vec = TfidfVectorizer(
    analyzer="word",
    tokenizer=tokenize_ja,   # ★ 形態素解析
    token_pattern=None,      # ★ 正規表現トークナイズを無効化（必須）
    lowercase=False,
    ngram_range=(1, 2),      # ★ unigram+bigram
    min_df=2,
    max_features=200_000
)
char_vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),      # ★ 3〜5文字
    min_df=2,
    max_features=200_000
)
vectorizer = FeatureUnion([
    ("word", word_vec),
    ("char", char_vec)
])
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==== 5. モデル: One-vs-Rest ロジスティック回帰 ====

clf = OneVsRestClassifier(LogisticRegression(max_iter=300, class_weight="balanced"))
clf.fit(X_train_tfidf, Y_train)

def match_cardinality(P, Y_true):
    """
    予測確率行列 P (N×L)
    正解ラベル行列 Y_true (N×L)
    各サンプルについて、予測の1の数 = 正解の1の数 になるように変換
    """
    N, L = P.shape
    Y_pred = np.zeros_like(P, dtype=int)
    
    for i in range(N):
        # 正解ラベル数
        k = int(Y_true[i].sum())
        if k == 0:
            continue  # 正解ラベルが0なら全部0のまま
        # 確率上位kを1に
        idx = np.argpartition(-P[i], k-1)[:k]
        Y_pred[i, idx] = 1
    return Y_pred

Y_pred = match_cardinality(clf.predict_proba(X_test_tfidf), Y_test)

# ==== 6. テストデータで評価 ====
print(f"micro-F1 :", f1_score(Y_test, Y_pred, average="micro", zero_division=0))
print(f"macro-F1 :", f1_score(Y_test, Y_pred, average="macro", zero_division=0))
print(f"Jaccard (samples):", jaccard_score(Y_test, Y_pred, average="samples", zero_division=0))
