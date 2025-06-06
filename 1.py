import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def normalize_text(text):
    text = re.sub(r'#.*', '', text)  # Удаление комментариев
    text = re.sub(r'//.*', '', text) 
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)  # Удаление многострочных комментариев
    text = re.sub(r'\".*?\"|\'.*?\'', 'STR', text)  # Замена строк на "STR"
    text = re.sub(r'\d+', 'NUM', text)  # Замена чисел на "NUM"
    text = re.sub(r'\b[_a-zA-Z][_a-zA-Z0-9]*\b', 'VAR', text)  # Замена имён переменных/функций/классов на "VAR"
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    return text.lower().strip()  


def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer() # встроениенная функция чтобы узнать "важность"
    tfidf = vectorizer.fit_transform([text1, text2]) #  анализируем частоту и преобразуем в формат vectorizer(там что то типо матрица на основе  словаря )
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2]) # вычисляем косинусное сходство между двумя текстами
    return similarity[0][0] * 100

def sequence_similarity(text1, text2):
    matcher = SequenceMatcher(None, text1, text2) # сравниваем две последовательности и находит наиболее длинные совпадающие подпоследовательности
    return matcher.ratio() * 100

def extract_structure(text):
    lines = text.split('\n')
    structure = []

    for line in lines:
        stripped = line.strip().lower() # удаляем все лишние пробелы

        if stripped.startswith(('def ', 'function ', 'public ', 'private ', 'void ')):
            structure.append('FUNC')
        elif stripped.startswith(('if ', 'elif ', 'else', 'switch')):
            structure.append('COND')
        elif stripped.startswith(('for ', 'while ')):
            structure.append('LOOP')
        elif stripped.startswith('class '):
            structure.append('CLASS')
        else:
            structure.append('CODE')

    return ' '.join(structure)

def structural_similarity(text1, text2):
    struct1 = extract_structure(text1)
    struct2 = extract_structure(text2)
    return SequenceMatcher(None, struct1, struct2).ratio() * 100

def compare_files(file1, file2):
    textik1 = read_file(file1)
    textik2 = read_file(file2)

    norm1 = normalize_text(textik1)
    norm2 = normalize_text(textik2)

    tfidf_score = tfidf_similarity(norm1, norm2)
    sequence_score = sequence_similarity(norm1, norm2)
    structural_score = structural_similarity(textik1, textik2)

    combined_score = (tfidf_score * 0.33 +
                      sequence_score * 0.33 +
                      structural_score * 0.33)

    return round(combined_score, 2)

if __name__ == "__main__":
    file_path_1 = "file1.txt"
    file_path_2 = "file3.txt"

    similarity = compare_files(file_path_1, file_path_2)

    print(f"Процент схожести: {similarity}%")
