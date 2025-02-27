import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, flash, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from io import BytesIO
import base64
import emoji
import string
from nltk.corpus import stopwords
import matplotlib
from youtube_scraper import scrape_video_comments
import spacy  # Added NLP library
from flask_sqlalchemy import SQLAlchemy  # Import SQLAlchemy

matplotlib.use('Agg')
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the root logger level to DEBUG

# Define a logger for the Flask app
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# -------------------
# MySQL Database Setup
# -------------------
# Configure your MySQL database URI here.
# For example: 'mysql+mysqlconnector://username:password@localhost/db_name'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://YOUR_USERNAME:YOUR_PASSWORD@localhost/YOUR_DATABASE'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with the Flask app
db = SQLAlchemy(app)

# Example Model to store comments and predictions
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment_id = db.Column(db.String(255))
    author = db.Column(db.String(255))
    content = db.Column(db.Text)
    prediction = db.Column(db.Integer)  # 0 for non-spam, 1 for spam

    def __init__(self, comment_id, author, content, prediction):
        self.comment_id = comment_id
        self.author = author
        self.content = content
        self.prediction = prediction

# Create tables if not exists (run once)
with app.app_context():
    db.create_all()

# -------------------
# Other Configurations
# -------------------
ensemble = None
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
MYDATA_FOLDER = 'mydata'
RESULT_FOLDER = 'results'
app.config['MYDATA_FOLDER'] = MYDATA_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(MYDATA_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_model_and_changes(ensemble, spam_match):
    # Save trained ensemble model
    joblib.dump(ensemble, 'ensemble_model.pkl')
    
    # Save spam words
    pd.DataFrame(spam_match, columns=['spam_words']).to_csv('spam_words.csv', index=False)


def load_model():
    global ensemble, spam_match
    try:
        ensemble = joblib.load('ensemble_model.pkl')
        spam_match = pd.read_csv('spam_words.csv')['spam_words'].tolist()
        print("Model loaded successfully")
    except FileNotFoundError:
        print("Model file not found")
    except Exception as e:
        print("Error loading model:", e)


def generate_pie_chart(predictions):
    unique, counts = np.unique(predictions, return_counts=True)
    sizes = dict(zip(unique, counts))
    labels = ['Non-Spam', 'Spam']
    sizes = [sizes.get(0, 0), sizes.get(1, 0)]
    colors = ['#99ff99', '#ff9999']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    pie_buf = BytesIO()
    plt.savefig(pie_buf, format='png')
    pie_buf.seek(0)
    pie_image = base64.b64encode(pie_buf.getvalue()).decode('utf-8')
    pie_buf.close()
    return pie_image

def generate_heatmap(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Non-Spam', 'Spam'])
    plt.yticks([0.5, 1.5], ['Non-Spam', 'Spam'])
    plt.tight_layout()
    heatmap_buf = BytesIO()
    plt.savefig(heatmap_buf, format='png')
    heatmap_buf.seek(0)
    heatmap_image = base64.b64encode(heatmap_buf.getvalue()).decode('utf-8')
    heatmap_buf.close()
    return heatmap_image

# Dummy data for initial spam words
spam_match = ["check my video", "Follow me", "watch my videos", "subscribe", "Please share", "Check out", "my channel", "my page", "giftcard", "promos", "sex", "channel", "facebook", "soundcloud", "support", "website"]
sr = set(stopwords.words('english'))

# Load spaCy English model (make sure you have downloaded the 'en_core_web_sm' model)
nlp = spacy.load('en_core_web_sm')

def preprocess_data(data):
    """
    Enhanced NLP preprocessing:
    1. Cleans and lowercases the text.
    2. Removes punctuation.
    3. Applies spaCy lemmatization.
    4. Filters out stop words.
    5. Calculates additional features (word count, spam count, URL presence, etc.).
    6. Performs sentiment analysis using NLTK's VADER.
    """
    import string
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Step 1: Clean text -> lowercase and remove punctuation.
    data['CONTENT_CLEAN'] = data['CONTENT'].str.lower().apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation))
    )

    # Step 2: Use spaCy to perform tokenization and lemmatization.
    data['CONTENT_LEMMA'] = data['CONTENT_CLEAN'].apply(
        lambda x: " ".join([token.lemma_ for token in nlp(x)])
    )

    # Step 3: Remove stop words from the lemmatized text.
    data['CONTENT_FLTR'] = data['CONTENT_LEMMA'].apply(
        lambda x: [word for word in x.split() if word not in sr]
    )

    # Step 4: Calculate text-based features.
    data['WORD_COUNT'] = data['CONTENT_FLTR'].apply(len)
    
    # COMMENT_LEN can be the same as WORD_COUNT here, or you can choose other logic.
    data['COMMENT_LEN'] = data['CONTENT_FLTR'].apply(len)
    
    # Count occurrences of spam words in the original content (case-insensitive)
    data['SPM_CNT'] = data['CONTENT'].str.upper().str.count('|'.join(spam_match))
    
    # Check if content contains URLs
    data['IS_URL'] = data['CONTENT'].str.upper().str.contains('HTTP|HTTPS|WWW|.COM')
    
    # Ratio of spam words to comment length 
    # (replace 0 with 1 to avoid division by zero errors)
    data['SPM_to_COMNT'] = data['SPM_CNT'] / data['COMMENT_LEN'].replace(0, 1)
    
    # Binary flag for long comments (using a threshold of 50 words)
    data['spm_len'] = np.where(data['COMMENT_LEN'] > 50, 1, 0)
    
    # Step 5: Sentiment Analysis using VADER
    sid = SentimentIntensityAnalyzer()
    data['sentiment'] = data['CONTENT'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    return data


def update_spam_match_with_ngrams(spam_match, new_words):
    updated_spam_match = spam_match.copy()
    updated_spam_match.extend(new_words)
    
    if new_words and any(new_words):
        if isinstance(new_words, str):
            new_words = [new_words]
        
        for phrase in new_words:
            if ' ' in phrase:
                max_n = len(phrase.split())
                for n in range(1, max_n + 1):
                    vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word', stop_words=None)
                    ngrams = vectorizer.fit_transform([phrase]).toarray()
                    ngram_features = vectorizer.get_feature_names_out()
                    updated_spam_match.extend(ngram_features)
    
    updated_spam_match = list(set(updated_spam_match))
    return updated_spam_match

# Load the model and spam words list when the application starts up
try:
    ensemble = joblib.load('ensemble_model.pkl')
    spam_match = pd.read_csv('spam_words.csv')['spam_words'].tolist()
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model file not found")
except Exception as e:
    print("Error loading model:", e)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    flash_message = session.pop('flash_message', None)
    return render_template('index.html', spam_match=spam_match, flash_message=flash_message)



from flask import session

@app.route('/update_spam', methods=['POST'])
def update_spam():
    action = request.form.get('action')
    updated_spam_words = request.form.getlist('word')

    for word in updated_spam_words:
        if action == 'add':
            if word not in spam_match:  # Check if word is not already in the list
                spam_match.append(word)
            else:
                session['flash_message'] = 'Word "{}" is already present in the spam list.'.format(word)
        elif action == 'delete':
            if word in spam_match:
                spam_match.remove(word)

    # Redirect to the index page
    return redirect(url_for('index'))



@app.route('/scrape_youtube', methods=['GET', 'POST'])
def scrape_youtube():
    if request.method == 'POST':
        video_url = request.form.get('video_url')
        if not video_url:
            flash('Please provide a YouTube video URL.')
            return redirect(url_for('scrape_youtube'))
        
        csv_filename, comments = scrape_video_comments(video_url)
        if not comments:
            flash('Failed to scrape comments. Please check the YouTube URL.')
            return redirect(url_for('scrape_youtube'))
        
        df = pd.DataFrame(comments)
        first_few_comments = df.head().to_html()
        
        return render_template('scrape_youtube.html', comments=first_few_comments, csv_filename=csv_filename)
    return render_template('scrape_youtube.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        voting_option = request.form.get('voting_option')
        dataset_option = request.form.get('dataset_option')
        
        if voting_option not in ['1', '2', '3'] or dataset_option not in ['1', '2']:
            flash('Invalid option selected.')
            return redirect(url_for('model'))
        
        if dataset_option == '1':
            file_path = r'D:\TE code\New Folder\final\My flask app\uploads\Youtube05-Shakira.csv'
        else:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
        
        ytd_data = pd.read_csv(file_path)

        ytd_data['SPM_CNT'] = ytd_data['CONTENT'].str.upper().str.count(str.upper("|".join(spam_match)))
        ytd_data['IS_URL'] = ytd_data['CONTENT'].str.upper().str.contains(str.upper('http|https|www|.com'))
        ytd_data['CONTENT_FLTR'] = ytd_data['CONTENT'].str.lower().apply(lambda x: [item for item in str.split(x) if item not in sr])
        ytd_data['COMMENT_LEN'] = ytd_data['CONTENT_FLTR'].str.len()
        ytd_data['SPM_to_COMNT'] = ytd_data['SPM_CNT'] / ytd_data['COMMENT_LEN']
        ytd_data['spm_len'] = np.where(ytd_data['COMMENT_LEN'] > 50, 1, 0)
        ytd_data['WORD_COUNT'] = ytd_data['CONTENT_FLTR'].apply(len)
        ytd_data['CHARACTER_COUNT'] = ytd_data['CONTENT'].apply(len)
        ytd_data['CONTAINS_NUMBERS'] = ytd_data['CONTENT'].str.contains(r'\d', regex=True).astype(int)
        ytd_data['CONTAINS_EMOJIS'] = ytd_data['CONTENT'].apply(lambda x: emoji.emoji_count(x))
        ytd_data['CONTAINS_PUNCTUATION'] = ytd_data['CONTENT'].apply(lambda x: any(char in string.punctuation for char in x)).astype(int)
        ytd_data['CONTAINS_ALL_CAPS'] = ytd_data['CONTENT'].apply(lambda x: any(word.isupper() for word in x.split())).astype(int)

        estimators_option1 = [('log', LogisticRegression(max_iter=1000)), ('cart', DecisionTreeClassifier(criterion='gini', random_state=40)), ('svm', SVC(kernel="rbf", C=100, gamma=0.1,probability=True)), ('nb', GaussianNB())]
        estimators_option2 = [('log', LogisticRegression(max_iter=1000)), ('cart', DecisionTreeClassifier(criterion='gini', random_state=40)), ('svm', SVC(kernel="rbf", C=100, gamma=0.1,probability=True)), ('nb', GaussianNB())]
        estimators_option3 = [('log', LogisticRegression(max_iter=1000)), ('nb', GaussianNB())]

        if voting_option == '1':
            estimators = estimators_option1
            voting_type = "Hard Voting"
        elif voting_option == '2':
            estimators = estimators_option2
            voting_type = "Soft Voting (All Models)"
        else:
            estimators = estimators_option3
            voting_type = "Soft Voting (Log & Naive)"

        ensemble = VotingClassifier(estimators, voting='hard' if voting_option == '1' else 'soft')
        X = ytd_data[['SPM_CNT', 'IS_URL', 'COMMENT_LEN', 'SPM_to_COMNT', 'spm_len', 'WORD_COUNT', 'CHARACTER_COUNT', 'CONTAINS_NUMBERS', 'CONTAINS_EMOJIS', 'CONTAINS_PUNCTUATION', 'CONTAINS_ALL_CAPS']].values

        y = ytd_data ['CLASS']
        
        # Train the ensemble model on the dataset
        ensemble.fit(X, y)

        # Make predictions
        Y_pred = ensemble.predict(X)
   
        accuracy = accuracy_score(y, Y_pred)
        conf_matrix = confusion_matrix(y, Y_pred)


        # Obtain accuracy, confusion matrix, and probability estimates
        accuracy = accuracy_score(y, Y_pred)
        conf_matrix = confusion_matrix(y, Y_pred)
        if voting_option == '1':
            Y_pred_prob = Y_pred
            fpr, tpr, _ = roc_curve(y, Y_pred)
        else:
            Y_pred_prob = ensemble.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, Y_pred_prob)
            
        auc = roc_auc_score(y, Y_pred_prob)
        heatmap_image = generate_heatmap(conf_matrix)

        # Plot ROC Curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label="{} (AUC = {:.2f})".format(voting_type, auc))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - {}'.format(voting_type))
        plt.legend()
        roc_buf = BytesIO()
        plt.savefig(roc_buf, format='png')
        roc_buf.seek(0)
        roc_image = base64.b64encode(roc_buf.getvalue()).decode('utf-8')
        roc_buf.close()



        # Generate pie chart
        pie_chart = generate_pie_chart(Y_pred)
        
        


        # Separate spam and non-spam comments with their IDs and authors
        spam_comments_data = ytd_data .loc[Y_pred == 1, ['COMMENT_ID', 'AUTHOR', 'CONTENT']].values.tolist()
        non_spam_comments_data = ytd_data .loc[Y_pred == 0, ['COMMENT_ID', 'AUTHOR', 'CONTENT']].values.tolist()
        
        # Create separate DataFrames for spam and non-spam comments
        spam_df = pd.DataFrame(spam_comments_data, columns=['COMMENT_ID', 'AUTHOR', 'CONTENT'])
        non_spam_df = pd.DataFrame(non_spam_comments_data, columns=['COMMENT_ID', 'AUTHOR', 'CONTENT'])

        # Get the filename from the file path
        filename = os.path.basename(file_path)
        
        # Extract the dataset name without extension
        dataset_name = os.path.splitext(filename)[0]
        
        # Create a folder with the dataset name if it doesn't exist
        dataset_folder = os.path.join(app.config['RESULT_FOLDER'], dataset_name)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        # Save DataFrames as CSV files in the folder
        spam_csv_path = os.path.join(dataset_folder, 'spam_comments.csv')
        non_spam_csv_path = os.path.join(dataset_folder, 'non_spam_comments.csv')
        spam_df.to_csv(spam_csv_path, index=False)
        non_spam_df.to_csv(non_spam_csv_path, index=False)

        save_model_and_changes(ensemble, spam_match)

        return render_template('model_results.html', spam_csv_path=spam_csv_path,non_spam_csv_path=non_spam_csv_path, accuracy=accuracy, spam_match=spam_match,conf_matrix=conf_matrix, roc_image=roc_image, pie_chart=pie_chart, spam_comments=spam_comments_data, non_spam_comments=non_spam_comments_data, heatmap_image=heatmap_image)


    return render_template('model.html')



import os

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    global ensemble
    # Initialize variables
    spam_predictions_df = pd.DataFrame()
    non_spam_predictions_df = pd.DataFrame()
    pie_chart = None
    total_comments = 0

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                unlabeled_data = pd.read_csv(file_path)
            except Exception as e:
                flash('Error reading the uploaded file.')
                return redirect(request.url)

        # Validate the uploaded file
        if not all(col in unlabeled_data.columns for col in ['COMMENT_ID', 'AUTHOR', 'CONTENT']):
            flash('Uploaded file is missing required columns.')
            return redirect(url_for('classify'))

        unclass_data = preprocess_data(unlabeled_data)  # Preprocess the data
        X_unlabeled = unclass_data[['SPM_CNT', 'IS_URL', 'COMMENT_LEN', 'SPM_to_COMNT', 'spm_len', 'WORD_COUNT']].values

        if ensemble is None:
            flash('Model is not trained. Please train the model first.')
            return redirect(url_for('home'))

        predictions = ensemble.predict(X_unlabeled)

        if len(predictions) == 0:
            flash('No predictions available.')
            return redirect(url_for('classify'))

        total_comments = len(unclass_data)

        unclass_data['PREDICTION'] = predictions
        unclass_data['CLASS'] = ['Spam' if pred == 1 else 'Not Spam' for pred in predictions]

        # Save predictions into the MySQL database
        for index, row in unclass_data.iterrows():
            new_comment = Comment(
                comment_id=str(row['COMMENT_ID']),
                author=row['AUTHOR'],
                content=row['CONTENT'],
                prediction=int(row['PREDICTION'])
            )
            db.session.add(new_comment)
        db.session.commit()

        # Separate spam and non-spam comments with their IDs and authors
        spam_predictions_df = unclass_data[unclass_data['PREDICTION'] == 1]
        non_spam_predictions_df = unclass_data[unclass_data['PREDICTION'] == 0]

        # Get the filename from the file path
        filename = os.path.basename(file_path)
        
        # Extract the dataset name without extension
        dataset_name = os.path.splitext(filename)[0]
        
        # Create a folder with the dataset name if it doesn't exist
        dataset_folder = os.path.join(app.config['RESULT_FOLDER'], dataset_name)
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
                
        # Save spam and non-spam comments as separate CSV files
        spam_csv_path = os.path.join(dataset_folder, 'spam_comments.csv')
        non_spam_csv_path = os.path.join(dataset_folder, 'non_spam_comments.csv')
        spam_predictions_df.to_csv(spam_csv_path, index=False)
        non_spam_predictions_df.to_csv(non_spam_csv_path, index=False)

        pie_chart = generate_pie_chart(predictions)
        return render_template('classify.html', 
                               spam_data=spam_predictions_df, 
                               non_spam_data=non_spam_predictions_df, 
                               pie_chart=pie_chart, 
                               total_comments=total_comments,
                               spam_csv_path=spam_csv_path,  # Pass the paths to the template
                               non_spam_csv_path=non_spam_csv_path)

    return render_template('classify.html', 
                           spam_data=spam_predictions_df, 
                           non_spam_data=non_spam_predictions_df, 
                           pie_chart=pie_chart, 
                           total_comments=total_comments)   


if __name__ == '__main__':
    app.run(debug=True)