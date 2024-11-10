import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def predictD():
    
     df = pd.read_csv('traffic.csv')
     # Display the first few rows of the dataset
     #print(df.head())

     # Convert 'Time' to hour (assuming 'Time' column is in HH:MM:SS format)
     df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour

# Convert 'Date' to day (assuming 'Date' column is in YYYY-MM-DD format)
     df['Date'] = pd.to_datetime(df['Date']).dt.day

# Convert 'Day of the week' to numerical
     label_encoder = LabelEncoder()
     df['Day of the week'] = label_encoder.fit_transform(df['Day of the week'])

# Features and target
     X = df[['Time', 'Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']]
     y = df['Traffic Situation']

# Split the data into training and testing sets
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
     clf = DecisionTreeClassifier()
     clf.fit(X_train, y_train)

# Predict on the test set
     y_pred = clf.predict(X_test)

     
     #accuracy = accuracy_score(y_test, y_pred)
     #print(f"Accuracy: {accuracy * 100:.2f}%")
     #print("Classification Report:\n", classification_report(y_test, y_pred))

     
     '''traffic_summary = df.groupby(['Time', 'Traffic Situation']).size().unstack().fillna(0)
     traffic_summary.plot(kind='bar', stacked=True, figsize=(10, 6))
     plt.title('Traffic Situation by Time of Day')
     plt.xlabel('Time of Day (Hour)')
     plt.ylabel('Count')
     plt.xticks(rotation=45)
     plt.legend(title='Traffic Situation')
     plt.tight_layout()
     # Create directory tree
     plot_dir = os.path.join('static', 'plots')
     os.makedirs(plot_dir, exist_ok=True)
     plot_path = os.path.join(plot_dir, 'traffic_plot.png')
    
     plt.savefig(plot_path)
     plt.close()
     #plt.savefig('static/traffic_plot.png')
     #plt.close()
     #plt.show()'''
     return accuracy_score(y_test, y_pred)*100

     
     


