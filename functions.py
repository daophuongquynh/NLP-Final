# Tải thư viện cần thiết
import pandas as pd
import chardet
import pandas as pd
import re
import string
import contractions
import unidecode
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Hàm đọc dữ liệu
def read_process_file(path):
    with open(path, 'rb') as f:
        result = chardet.detect(f.read())
    df = pd.read_csv(path, encoding=result['encoding'])
    df = df.drop_duplicates(keep='first')
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')
    df.columns = ['target', 'message']
    le = LabelEncoder()
    le.fit(df['target'])
    df['target_encoded'] = le.transform(df['target'])
    return df

# Hàm xử lý stopwords
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'ur', "i'm", 'c', 'ill', 'ive', 'dnt', 'dont', 'dun', 'bcoz', 'n']
    stop_words = stop_words + more_stopwords
    abbreviations = []
    new_words = []
    for word in stop_words:
        if "'" in word:
            abbreviations.append(word)
    for word in abbreviations:
        new_word = re.sub("'", "", word)
        new_words.append(new_word)
    for word in new_words:
        if word not in stop_words:
            stop_words.append(word)
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

# Hàm lemmatization
def lemmatize_words(text):
    words = word_tokenize(text)
    lemma = WordNetLemmatizer()
    lemma_words = [lemma.lemmatize(word) for word in words]
    lemma_text = ' '.join(lemma_words)
    return lemma_text

# Hàm làm sạch đoạn văn bản
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\(.*?\)', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>', '', text)
    text = contractions.fix(text)
    text = unidecode.unidecode(text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = remove_stopwords(text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\d+[\.\/-]\d+[\.\/-]\d+', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = lemmatize_words(text)
    return text

# Hàm vector hóa dữ liệu
def vectorize_text(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return X_train_vectorized, X_test_vectorized, vectorizer


# Hàm xây dựng mô hình dự đoán SVM
def train_and_predict_SVM(X_train, X_test, y_train, y_test):
    result = []
    # Tạo mô hình SVM với kernel tuyến tính
    svm = SVC(kernel='linear')
    
    # Huấn luyện mô hình trên dữ liệu huấn luyện
    svm.fit(X_train, y_train)
    
    # Dự đoán nhãn cho dữ liệu kiểm tra
    y_pred = svm.predict(X_test)
    
    # Đánh giá độ chính xác của mô hình
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    result = result + [accuracy,precision,recall,f1]
    return svm,y_pred,result

# Hàm dự đoán
def predict(text, vectorizer, svm):
    #red = [220,50,50]
    #green = [100,220,50]
    
    processed_text = clean_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    
    # Dự đoán bằng mô hình SVM đã huấn luyện từ train_and_predict_SVM
    prediction = svm.predict(text_vectorized)
    if prediction[0] == 1:
        # In ra kết quả 'message is spam'
        return "message is spam"
    else:
        # In ra kết quả 'message is not spam'
        return "message is not spam"

# Chạy thử các functions đã xây dựng    
df =  read_process_file('spam.csv')
df['message_clean'] = df['message'].apply(clean_text)
X_train, X_test, y_train, y_test = train_test_split(df['message_clean'], df['target_encoded'], test_size=.2, random_state=1)
X_train_vectorized, X_test_vectorized, vectorizer = vectorize_text(X_train, X_test)
svm, y_pred, result = train_and_predict_SVM(X_train_vectorized,X_test_vectorized, y_train, y_test)
a = predict('Congratulations! You’ve won a $500 Amazon gift card. Claim it here [Link]',vectorizer,svm)
