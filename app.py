import pickle
from sklearn.feature_extraction.text import CountVectorizer
import gradio as gr


# List of possible target categories
target_names = ["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting", "Mortgage/Loan"]

# Loading count vector and tf-idf transformer and logistic regression model which are saved after training
load_vec = CountVectorizer(vocabulary=pickle.load(open("./count_vector.pkl", "rb")))
load_tfdif = pickle.load(open("./tfdif.pkl", "rb"))
load_log_reg = pickle.load(open("./log_reg_model.pkl", "rb"))

def topic_predictor(complaint_text:str):
    """
    This function takes a text input (complaint) and predicts the topic category.
    """

    test_count = load_vec.transform([complaint_text])   # Convert input text to count vector
    test_tfdif = load_tfdif.transform(test_count)  # Apply the TF-IDF transformation

    # Predict the topic
    prediction = load_log_reg.predict(test_tfdif)

    # Return the predicted topic
    return target_names[prediction[0]]


# Create the Gradio interface
interface = gr.Interface(
    fn=topic_predictor,  # Function to be called on user input
    inputs=gr.Textbox(lines=2, placeholder="Enter your complaint here..."),  # User input (textbox)
    outputs="text",  # Output format (text output)
    title="Topic Prediction",  # Title of the app
    description="Enter a complaint to get its predicted category (e.g., 'Bank Account services', 'Credit card or prepaid card', etc.)."  # Description
)
# Launch the interface
interface.launch()
