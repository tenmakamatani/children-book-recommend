import os
import json
from pathlib import Path
from janome.tokenizer import Tokenizer
from collections import defaultdict, Counter

# ./data以下のメタデータのjsonを全て取得する
def load_all_metadata(base_dir='./data'):
    metadata_list = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_metadata.json'):
                json_path = Path(root) / file
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metadata_list.append(data)
                except Exception as e:
                    print(f"Failed to load {json_path}: {e}")
    
    return metadata_list

# jsonからあらすじと主題のリストを抽出する
def extract_summary_and_subjects(all_metadata):
    extracted = []

    for i, metadata in enumerate(all_metadata):
        try:
            summary = metadata['item_7_description_22']['attribute_value_mlt'][0]['subitem_description']
        except (KeyError, IndexError, TypeError):
            summary = None

        try:
            subjects_raw = metadata['item_7_text_24']['attribute_value_mlt']
            subjects = [entry['subitem_text_value'] for entry in subjects_raw if 'subitem_text_value' in entry]
        except (KeyError, TypeError):
            subjects = []

        extracted.append({
            'index': i,
            'summary': summary,
            'subjects': subjects
        })

    return extracted

# 形態素解析
tokenizer = Tokenizer()
def tokenize(text):
    words = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        base = token.base_form

        if pos in {'名詞', '動詞', '形容詞'} and base != '*':
            words.append(base)

    return words

# 特徴語リストの構築
def build_topic_keywords(extracted_entries):
    topic_word_counts = defaultdict(Counter)

    for entry in extracted_entries:
        summary = entry['summary']
        subjects = entry['subjects']

        if not summary or not subjects:
            continue

        words = tokenize(summary)

        for subject in subjects:
            topic_word_counts[subject].update(words)

    return topic_word_counts

# 文章から主題を推定する
def predict_subjects(text, topic_word_counts):
    # 解析して名詞だけ取り出し
    words = tokenize(text)

    word_set = set(words)  # 重複を省いて効率UP
    results = []

    for subject, word_counter in topic_word_counts.items():
        count = sum(1 for word in word_set if word in word_counter)
        results.append({'subject': subject, 'count': count})

    # 一致数の多い順にソート（オプション）
    results.sort(key=lambda x: x['count'], reverse=True)
    
    return results

# train dataとtest dataにわける
def split_data(data, train_ratio=0.8):
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]

# 評価関数
def evaluate_predictions(test_data, topic_word_counts):
    total_correct = 0
    total_expected = 0

    for entry in test_data:
        true_subjects = set(entry['subjects'])
        if not entry['summary'] or not true_subjects:
            continue

        pred_result = predict_subjects(entry['summary'], topic_word_counts)
        predicted_subjects = [p['subject'] for p in pred_result if p['count'] > 0]

        # ランキング上位 N 件に絞る（N = 正解ラベル数）
        n = len(true_subjects)
        top_n_preds = set(predicted_subjects[:n])

        total_correct += len(true_subjects & top_n_preds)
        total_expected += n

    accuracy = (total_correct / total_expected) * 100 if total_expected else 0
    return {
        'Total Correct Matches': total_correct,
        'Total Ground Truth Labels': total_expected,
        'Accuracy (%)': round(accuracy, 2)
    }


def main():
    all_metadata = load_all_metadata()
    all_extracted_metadata = extract_summary_and_subjects(all_metadata)
    [train_data, test_data] = split_data(all_extracted_metadata, 0.8)
    topic_keywords = build_topic_keywords(train_data)
    result = evaluate_predictions(test_data, topic_keywords)
    print(result)

main()