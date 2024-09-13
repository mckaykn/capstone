import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

cheater_data = np.load('cheaters/cheaters.npy', allow_pickle=True)
legit_data = np.load('legit/legit.npy', allow_pickle=True)

cheater_subset = cheater_data[:2000]
legit_subset = legit_data[:10000]

# Combine the subsets
subset_data = np.vstack((cheater_subset, legit_subset))

# Create labels: 1 for cheaters, 0 for legit
subset_labels = np.hstack((
    np.ones(cheater_subset.shape[0]),
    np.zeros(legit_subset.shape[0])
))

# Repeat labels to match the shape of the flattened subset data
subset_labels_repeated = np.repeat(subset_labels, 30 * 192)

# Reshape the combined data to (total_subset_samples, 5)
subset_data_reshaped = subset_data.reshape(-1, 5)

# Perform stratified split to ensure equal representation in training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    subset_data_reshaped, subset_labels_repeated, test_size=0.2, random_state=42, stratify=subset_labels_repeated)

# Define custom class weights
class_weights = {0: 1.2, 1: 6}  # You may need to adjust these values based on performance

# Initialize the classifier with custom class weighting
classifier = LogisticRegression(max_iter=500, class_weight=class_weights, random_state=42)

# # Train the classifier
# classifier.fit(X_train, y_train)

# # Save the trained classifier
# with open('classifier.pkl', 'wb') as f:
#     pickle.dump(classifier, f)

# Load the pre-made classification algorithm
with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate the classification report with zero_division parameter
classification_report_text = classification_report(y_test, y_pred, zero_division=0)
print(f'Classification Report: \n{classification_report_text}')

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n{conf_matrix}')

# Calculate percentages
conf_matrix_percentages = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix with percentages
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=False, fmt='.2f', cmap='Blues', xticklabels=['Non-Cheater', 'Cheater'],
            yticklabels=['Non-Cheater', 'Cheater'])

# Annotate with raw numbers
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j + 0.5, i + 0.5, f"{conf_matrix[i, j]}\n({conf_matrix_percentages[i, j]:.2f}%)",
                 ha='center', va='center', color='black', fontsize=12)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with Percentages')
plt.show()


def average_fires_per_engagement(data):
    firing_data = data[:, :, :, 4]
    sum_firing_per_engagement = np.sum(firing_data, axis=2)
    avg_firing = np.mean(sum_firing_per_engagement)
    return avg_firing


def calculate_time_to_kill(data):
    time_to_kill_list = []
    one_tap_count = 0

    for player_data in data:
        for engagement in player_data:
            firing_sequence = engagement[:, 4]
            first_firing_index = np.argmax(firing_sequence)

            if first_firing_index < 160:
                time_to_kill_timesteps = 160 - first_firing_index
                time_to_kill_seconds = time_to_kill_timesteps / 32
                time_to_kill_list.append(time_to_kill_seconds)
            if first_firing_index == 160:
                time_to_kill_seconds = 0
                time_to_kill_list.append(time_to_kill_seconds)
                one_tap_count += 1

    if time_to_kill_list:
        average_time_to_kill_seconds = np.mean(time_to_kill_list)
    else:
        average_time_to_kill_seconds = None

    average_one_tap_count = one_tap_count / len(data)  # Average one-tap count per engagement
    return time_to_kill_list, average_time_to_kill_seconds, average_one_tap_count


avg_fires_cheater = average_fires_per_engagement(cheater_data)
avg_fires_legit = average_fires_per_engagement(legit_data)

time_to_kill_cheater, avg_time_to_kill_cheater, cheater_one_tap_count = calculate_time_to_kill(cheater_data)
time_to_kill_legit, avg_time_to_kill_legit, legit_one_tap_count = calculate_time_to_kill(legit_data)

ttk_data = pd.DataFrame({
    'Time to Kill (seconds)': time_to_kill_cheater + time_to_kill_legit,
    'Label': ['Cheater'] * len(time_to_kill_cheater) + ['Legit'] * len(time_to_kill_legit)
})

avg_fires_data = pd.DataFrame({
    'Type': ['Cheater', 'Legit'],
    'Average Fires': [average_fires_per_engagement(cheater_data), average_fires_per_engagement(legit_data)]
})
# Plotting

# Convert to DataFrame for easier plotting
metrics_data = pd.DataFrame({
    'Type': ['Cheater', 'Legit', 'Cheater', 'Legit'],
    'Metric': ['Average Fires', 'Average Fires', 'Average One-Taps', 'Average One-Taps'],
    'Value': [avg_fires_cheater, avg_fires_legit, cheater_one_tap_count, legit_one_tap_count]
})

# Grouped bar plot for average fires and average one-taps
plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Type', y='Value', hue='Metric', data=metrics_data, palette='viridis')

for p in barplot.patches:
    height = p.get_height()
    barplot.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.title('Average Fires and Average One-Taps per Engagement')
plt.xlabel('Player Type')
plt.ylabel('Value')
plt.legend(title='Metric')
plt.show()

# Histogram for time to kill
plt.figure(figsize=(10, 6))
sns.histplot(ttk_data, x='Time to Kill (seconds)', hue='Label', kde=True, element='step', stat='density',
             common_norm=False)
plt.title('Time to Kill Histogram')
plt.xlabel('Time to Kill (seconds)')
plt.ylabel('Density')
plt.show()
