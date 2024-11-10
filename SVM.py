import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def predictS():
# Load the dataset (replace 'traffic_data.csv' with your actual dataset file name)
    df = pd.read_csv('traffic.csv')

# Display the first few rows of the dataset
#print(df.head())

# Convert 'Time' to hour (assuming 'Time' column is in 12-hour HH:MM:SS AM/PM format)
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

# Initialize and train the Support Vector Classifier
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train, y_train)

# Predict on the test set
    y_pred = clf.predict(X_test)

# Evaluate the model
    return accuracy_score(y_test, y_pred)*100
    #print(f"Accuracy: {accuracy * 100:.2f}%")
#print("Classification Report:\n", classification_report(y_test, y_pred))

# Plotting the data with bar chart
# Summarize traffic situation by 'Time'
#traffic_summary = df.groupby(['Time', 'Traffic Situation']).size().unstack().fillna(0)

# Plotting the bar chart
"""traffic_summary.plot(kind='bar', stacked=True, figsize=(10, 6))*
plt.title('Traffic Situation by Time of Day')
plt.xlabel('Time of Day (Hour)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Traffic Situation')
plt.tight_layout()
plt.show()"""