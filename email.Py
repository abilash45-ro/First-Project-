import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. CREATE MOCK DATASET (For Demonstration)
# ==========================================
# If you have a CSV, replace this section with: df = pd.read_csv('your_file.csv')
data = {
    'email_text': [
        "Dear customer, please verify your bank account details immediately at http://secure-login-bank-update.com",
        "Hey, are we still meeting for lunch at 1 PM today? Let me know.",
        "URGENT: Your Netflix subscription has expired. Click here to renew: http://netflix-free-trial.net",
        "Attached is the weekly project status report. Please review by EOD.",
        "Congratulations! You won a $1000 Amazon gift card. Claim now at http://win-free-prizes.biz",
        "Hi Team, please find the meeting minutes attached from yesterday's sync.",
        "Suspicious login attempt detected on your account. Reset password immediately.",
        "Can you send me the updated spreadsheet whenever you get a chance?"
    ],
    'label': ['Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe']
}

df = pd.DataFrame(data)

# ==========================================
# 2. FEATURE EXTRACTION & SPLITTING
# ==========================================
# Extract text features using TF-IDF (Term Frequency-Inverse Document Frequency)
# This handles both keywords and URL patterns embedded in the text
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['email_text'])
y = df['label']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ==========================================
# 3. MODEL TRAINING
# ==========================================
model = LogisticRegression()
model.fit(X_train, y_train)

# ==========================================
# 4. MODEL EVALUATION
# ==========================================
# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate and display Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("=========================================")
print(f"Model Classification Accuracy: {accuracy * 100:.2f}%")
print("=========================================\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate and Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['Safe', 'Phishing'])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Safe', 'Phishing'], 
            yticklabels=['Safe', 'Phishing'])
plt.title('Phishing Email Detection - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save the confusion matrix plot to submit alongside your code
plt.savefig('confusion_matrix.png')
plt.show()
