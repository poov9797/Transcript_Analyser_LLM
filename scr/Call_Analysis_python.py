import os
import re
import pandas as pd
import csv
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from openai import OpenAI

# Constants for file paths and URLs
PATH_TO_FILE = "/Users/pravin/Desktop/Src/call_analysis_results.csv"
TRANSCRIPT_FOLDER_PATH = '/Users/pravin/Desktop/transcripts_v3/transcripts_v3'
OLLAMA_URL = "http://localhost:11434/v1/"  # Local URL for the LLM API
OLLAMA_API_KEY = "ollama"  # API key for the LLM service

# Function to load all transcript files from the specified folder
def load_transcripts(folder_path):
    """
    Load all .txt transcript files from the specified folder.

    Args:
        folder_path (str): The path to the folder containing transcript files.

    Returns:
        list: A list of dictionaries with filenames as keys and file content as values.
    """
    transcripts = []
    filenames = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".txt")])

    for filename in filenames:
        with open(os.path.join(folder_path, filename), 'r') as file:
            transcripts.append({filename: file.read()})
    return transcripts

# Function to extract numerical part from filenames for sorting
def extract_number(filename):
    """
    Extract the numerical part of a filename.

    Args:
        filename (str): The filename from which to extract the number.

    Returns:
        int: The extracted number from the filename.
    """
    return int(re.search(r'\d+', filename).group())

# Function to build a prompt for LLM
def build_prompt(transcript):
    """
    Build the LLM prompt for analyzing the conversation.

    Args:
        transcript (str): The conversation transcript to be analyzed.

    Returns:
        str: The formatted prompt for the LLM model.
    """
    prompt_template = f"""
    The following is a conversation between a customer and an agent. Analyze the conversation and do the following:
    
    1. Determine if the customer's issue was resolved or if follow-up action is needed.
    2. Determine the overall sentiment of the customer during the conversation (positive, negative, or neutral).
    
    Transcript: {transcript}

    Based on the findings provide an output in the below format:
    Call Outcome: [Issue Resolved/Follow-up Action Needed]
    Sentiment: [Positive/Negative/Neutral]
    """
    return prompt_template.strip()

# Function to interact with LLM to get call outcomes and sentiment
def llm_analyze(transcript):
    """
    Use LLM to analyze the transcript for call outcome and sentiment.

    Args:
        transcript (str): The conversation transcript to be analyzed.

    Returns:
        str: The LLM response containing call outcome and sentiment analysis.
    """
    prompt = build_prompt(transcript)
    ollama_client = OpenAI(base_url=OLLAMA_URL, api_key=OLLAMA_API_KEY)
    
    response = ollama_client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Function to determine call outcomes for all transcripts
def determine_call_outcomes(transcripts):
    """
    Determine call outcomes and sentiments using LLM for each transcript.

    Args:
        transcripts (list): A list of transcripts to be analyzed.

    Returns:
        list: A list of LLM responses for each transcript.
    """
    outcomes = []
    for transcript in tqdm(transcripts, desc="Processing Transcripts"):
        outcome = llm_analyze(transcript)
        outcomes.append(outcome)
    return outcomes

# Function to categorize the outcome based on the LLM response
def categorize_outcome(call_outcome):
    """
    Categorize the call outcome into 'Issue Resolved' or 'Follow-up Action Needed'.

    Args:
        call_outcome (str): The raw outcome returned by the LLM.

    Returns:
        str: Categorized outcome.
    """
    call_outcome_lower = call_outcome.lower()
    if "follow-up" in call_outcome_lower:
        return "Follow-up Action Needed"
    elif "issue" in call_outcome_lower or "resolved" in call_outcome_lower:
        return "Issue Resolved"
    else:
        return "Unknown"

# Function to categorize sentiment based on the LLM response
def categorize_sentiment(call_outcome):
    """
    Categorize the sentiment based on the LLM response.

    Args:
        call_outcome (str): The raw outcome returned by the LLM.

    Returns:
        str: Categorized sentiment (Positive/Negative/Neutral).
    """
    call_outcome_lower = call_outcome.lower()
    if "positive" in call_outcome_lower:
        return "Positive"
    elif "negative" in call_outcome_lower:
        return "Negative"
    elif "neutral" in call_outcome_lower:
        return "Neutral"
    else:
        return "Unknown"

# Main function to process the transcripts and save results
def main():
    # Load and sort transcripts
    transcripts = load_transcripts(TRANSCRIPT_FOLDER_PATH)
    sorted_transcripts = sorted(transcripts, key=lambda x: extract_number(list(x.keys())[0]))
    
    # Determine call outcomes and sentiments
    call_outcomes = determine_call_outcomes(sorted_transcripts)

    # Read existing analysis results into DataFrame
    df = pd.read_csv(PATH_TO_FILE)
    
    # Add LLM analysis results to DataFrame
    df['Ground_truth_outcome'] = call_outcomes
    df['Actual_Sentiment'] = df['Ground_truth_outcome'].apply(categorize_sentiment)
    df['Actual_Outcome'] = df['Ground_truth_outcome'].apply(categorize_outcome)
    
    # Save the results back to CSV
    df.to_csv('call_analysis_ground_truth_results.csv', index=False)
    print("Results saved to call_analysis_ground_truth_results.csv")

    # Evaluate model performance
    y_true_sentiments = df['Actual_Sentiment']  # Actual sentiments
    y_true_outcomes = df['Actual_Outcome']      # Actual outcomes
    y_pred_sentiments = df["Sentiment"]         # Model predictions
    y_pred_outcomes = df["Output"]              # Model predictions

    # Print classification reports
    print("Sentiment Analysis Report:")
    print(classification_report(y_true_sentiments, y_pred_sentiments))

    print("Call Outcome Classification Report:")
    print(classification_report(y_true_outcomes, y_pred_outcomes))

    # Visualize call outcomes distribution
    plot_pie_chart(df['Output'], 'Call Outcome Distribution')

    # Visualize sentiment distribution
    plot_bar_chart(df['Sentiment'], 'Sentiment Distribution')

# Function to plot a pie chart for call outcomes
def plot_pie_chart(data, title):
    """
    Plot a pie chart for the given data.

    Args:
        data (pd.Series): The data to plot (e.g., call outcomes).
        title (str): The title of the chart.
    """
    outcome_counts = data.value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
    plt.axis('equal')
    plt.title(title)
    plt.show()

# Function to plot a bar chart for sentiment distribution
def plot_bar_chart(data, title):
    """
    Plot a bar chart for the given data.

    Args:
        data (pd.Series): The data to plot (e.g., sentiment distribution).
        title (str): The title of the chart.
    """
    sentiment_counts = data.value_counts()
    sentiment_percentage = (sentiment_counts / sentiment_counts.sum()) * 100
    plt.figure(figsize=(8, 6))
    plt.bar(sentiment_percentage.index, sentiment_percentage, color=['#ff9999', '#66b3ff', '#99ff99'])
    
    for i, value in enumerate(sentiment_percentage):
        plt.text(i, value + 0.5, f'{value:.1f}%', ha='center')

    plt.title(title)
    plt.xlabel('Sentiment')
    plt.ylabel('Percentage')
    plt.show()

if __name__ == "__main__":
    main()
