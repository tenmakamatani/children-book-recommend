import json
import numpy as np
from janome.tokenizer import Tokenizer
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import f1_score, jaccard_score, precision_score
from pathlib import Path

# ==== 1. データ読み込み ====
data = json.loads(Path("extracted_metadata_raw.json").read_text(encoding="utf-8"))

titles = [item["title"] for item in data]
texts = [item["summary"] for item in data]   # summaryをスペース区切りの文章に
labels = [item["subjects"] for item in data]          # subjectsは主題リスト


# ==== 2. ラベルをmulti-hot化 ====
mlb = MultiLabelBinarizer()
X = texts
Y = mlb.fit_transform(labels)


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

X_tfidf = vectorizer.fit_transform(X)

# ==== 5. モデル: One-vs-Rest ロジスティック回帰 ====

clf = OneVsRestClassifier(LogisticRegression(max_iter=300, class_weight="balanced"))
clf.fit(X_tfidf, Y)

# 全データの予測確率ベクトルを算出 & 正規化
Y_prob_all = clf.predict_proba(X_tfidf)  # shape (N, L)
Y_prob_all = np.asarray(Y_prob_all, dtype=np.float32)
Y_prob_all_norm = Y_prob_all / (np.linalg.norm(Y_prob_all, axis=1, keepdims=True) + 1e-12)


def recommend_top1_index_labelonly(i):
    """
    本 i をクエリに、ラベル確率ベクトルのコサイン類似のみでトップ1を返す
    """
    q_prob = Y_prob_all_norm[i]                 # (L,)
    scores = Y_prob_all_norm @ q_prob           # (N,) ← コサイン類似
    scores[i] = -1.0                            # 自分自身は除外
    j = int(np.argmax(scores))
    return j, float(scores[j])


def f1_of_label_sets(y_true_i, y_true_j):
    """
    ラベル集合間のF1（Dice係数）
    """
    a = y_true_i.astype(bool)
    b = y_true_j.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0  # 両方空集合なら完全一致とみなす
    return 2.0 * inter / denom


def evaluate_corpus_top1_f1_labelonly():
    """
    全件をクエリにして、ラベル確率ベクトル類似のみで推薦 → Top1 のF値を平均
    """
    N = len(texts)
    f1_list = []
    for i in range(N):
        j, _ = recommend_top1_index_labelonly(i)
        f1 = f1_of_label_sets(Y[i], Y[j])
        f1_list.append(f1)
    return float(np.mean(f1_list)), f1_list

def evaluate_corpus_micro_dice_labelonly():
    """
    全件をクエリにしてTop-1推薦を取得し、
    全サンプルまとめて micro版 Dice (＝F1) を計算する
    """
    N = len(texts)
    y_true_all = []
    y_pred_all = []

    for i in range(N):
        j, _ = recommend_top1_index_labelonly(i)
        y_true_all.append(Y[i])
        y_pred_all.append(Y[j])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # 集合F1（=Dice）は sklearn.metrics.f1_score(micro) と等価
    micro_f1 = f1_score(y_true_all, y_pred_all, average="micro", zero_division=0)
    return micro_f1

def predict_subjects(text: str, top_k: int = 5):
    """
    未知テキストから主題ベクトルを生成し、
    上位 top_k 件の主題名と確率を返す
    """
    # 文章をベクトル化
    Xq = vectorizer.transform([text])

    # 確率ベクトルを予測
    probs = clf.predict_proba(Xq)[0]

    # 上位top_kのインデックス
    top_idx = np.argsort(-probs)[:top_k]

    # 主題名と確率をまとめて返す
    return [(mlb.classes_[i], float(probs[i])) for i in top_idx]

# ==== 動作確認 ====
print("\n[Example] recommend_top1_index_labelonly (item #0 as query)")
j, score = recommend_top1_index_labelonly(0)
print(f"query=0 -> top1={j}, score={score:.4f}")

mean_f1, _ = evaluate_corpus_top1_f1_labelonly()
print(f"[Eval@Top1-F(label-only)] mean F1 = {mean_f1:.4f}")

micro_f1 = evaluate_corpus_micro_dice_labelonly()
print(f"[Eval@Micro-F(label-only)] Dice = {micro_f1:.4f}")

query = "夏休みの午後、兄と妹は川辺で石を集めて遊んでいた。やがて夕立が降り、二人は笑いながら家へ駆け戻り、母の温かい声に迎えられた。"
results = predict_subjects(query, top_k=5)
print("予測された主題トップ5:")
for subj, p in results:
    print(f"{subj}: {p:.3f}")