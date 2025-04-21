import os
import json
from pathlib import Path
from janome.tokenizer import Tokenizer
from collections import defaultdict, Counter

stopwords = {
    '', 'こと', 'これ', 'それ', 'あれ', 'もの', 'ため', 'ところ', 'よう', 'さん', 'そう', 'の', 'に',
    'が', 'は', 'を', 'と', 'も', 'で', 'から', 'まで', 'より', 'へ', 'など', 'や', 'し', 'して',
    'また', 'そして', 'しかし', 'でも', 'ある', 'いる', 'なる', 'ない', 'できる', 'する', 'した',
    'ような', 'ように', 'だけ', 'その', 'この', 'あの', 'れる'
}

# ./data以下のメタデータのjsonを全て取得する。ファイルに書き出すために初回のみ実行が必要
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

# jsonからあらすじと主題のリストを抽出する。ファイルに書き出すために初回のみ実行が必要
def extract_summary_and_subjects(all_metadata):
    extracted = []

    for i, metadata in enumerate(all_metadata):
        try:
            summary = metadata['item_7_description_22']['attribute_value_mlt'][0]['subitem_description']
        except (KeyError, IndexError, TypeError):
            summary = None

        try:
            subjects_raw = metadata['item_7_text_24']['attribute_value_mlt']
            subjects = []
            for entry in subjects_raw:
                subject = entry['subitem_text_value']
                if subject == ' 仲間意識を育てる':
                    subject = '仲間意識を育てる'
                if subject == '一人の時間を持つ':
                    print('///////')
                    print(subject)
                    subject = '一人の時間をもつ'
                    print(subject)
                if subject == '価値観を持つ':
                    subject = '価値観をもつ'
                if subject == '手袋・帽子・靴下・傘・マフラー等と':
                    subject = '手袋・帽子・靴下・傘・マフラー等と遊ぶ'
                if subject == '死　':
                    subject = '死'
                if subject == '自尊心を持つ':
                    subject = '自尊心をもつ'
                if subject == '雪と遊ぶ遊ぶ':
                    subject = '雪と遊ぶ'
                if subject == '買い物をする':
                    subject = '買物をする'
                if subject == '心':
                    continue
                subjects.append(subject)
            
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

        if pos in {'名詞', '動詞', '形容詞'} and base not in stopwords and base != '*':
            words.append(base)

    return words

# extracted_metadataを書き出す
def export_extracted_metadata(extracted_metadata, filepath='extracted_metadata.json'):
    simple_data = []

    for entry in extracted_metadata:
        summary = entry['summary']
        subjects = entry['subjects']
        tokenized_summary = tokenize(summary)

        simple_data.append({
            'summary': tokenized_summary,  # 形態素解析済み単語リスト
            'subjects': subjects
        })

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, ensure_ascii=False, indent=2)

# extracted_metadataを読み込む
def load_extracted_metadata(filepath='extracted_metadata.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 特徴語リストの構築
def build_topic_keywords(extracted_entries):
    topic_word_counts = defaultdict(Counter)

    for entry in extracted_entries:
        words = entry['summary']
        subjects = entry['subjects']

        if not words or not subjects:
            continue

        for subject in subjects:
            topic_word_counts[subject].update(words)

    return topic_word_counts

# 文章から主題を推定する
def predict_subjects(summary, topic_word_counts):
    results = []

    for subject, word_counter in topic_word_counts.items():
        count = sum(1 for word in summary if word in word_counter)
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

# TopicKeywordsの上位n個ずつをtxtファイルに出力する
def export_topic_keywords_to_txt(topic_keywords, filepath='topic_keywords.txt', top_n=10):
    with open(filepath, mode='w', encoding='utf-8') as f:
        for subject, counter in topic_keywords.items():
            f.write(f"{subject}\n")
            for word, count in counter.most_common(top_n):
                f.write(f"  - {word}: {count}\n")
            f.write("\n")

def main():
    # 基本的に実行不要。summaryの形態素解析方法を変更した時のみ実行する
    # ファイルの読み込み→あらすじ、主題情報の抽出→整形(形態素解析、表記揺れの修正)→ファイルへの書き出し を行っている

    # all_metadata = load_all_metadata()
    # all_extracted_metadata = extract_summary_and_subjects(all_metadata)
    # export_extracted_metadata(all_extracted_metadata)
    all_extracted_metadata = load_extracted_metadata()
    [train_data, test_data] = split_data(all_extracted_metadata, 0.8)
    topic_keywords = build_topic_keywords(train_data)
    export_topic_keywords_to_txt(topic_keywords)
    result = evaluate_predictions(test_data, topic_keywords)
    print(result)

main()