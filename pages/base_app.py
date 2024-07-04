import streamlit as st
import pickle
import os
from wordcloud import WordCloud, STOPWORDS  # Correct import
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from textblob import TextBlob
import pandas as pd

# Model Options with Absolute Paths
model_options = {
    "Logistic Regression": os.path.join(os.path.dirname(__file__), 'models/logistic_regression_pipeline.pkl'),
    "Support Vector Classifier": os.path.join(os.path.dirname(__file__), 'models/pipeline_svc.pkl'),
    "Naive bayes Classifier": os.path.join(os.path.dirname(__file__), 'models/pipeline_nb.pkl')
}

# Set page config (title, icon) ONLY for the main page
st.set_page_config(
    page_title="Text Prediction App", page_icon=":crystal_ball:"
)

# Manual Sidebar Navigation
st.sidebar.title("Navigation")

if "page" not in st.session_state:
    st.session_state.page = "home"  # Start on the home page

# Page Selection Buttons
if st.sidebar.button("Home"):
    st.session_state.page = "home"
    st.experimental_rerun()
elif st.sidebar.button("Predict"):
    st.session_state.page = "main_page"
    st.experimental_rerun()
elif st.sidebar.button("About Us"):
    st.session_state.page = "about_us"
    st.experimental_rerun()
elif st.sidebar.button("Project Overview"):  # New button
    st.session_state.page = "project_overview"
    st.experimental_rerun()
elif st.sidebar.button("Explore Data"):
    st.session_state.page = "explore_data"
    st.experimental_rerun()

# Home Page
if st.session_state.page == 'home':
    # Create two columns with width ratio 1:4
    col1, col2 = st.columns([1, 4])  

    # Display logo in the first column (adjust width as needed)
    with col1:
        st.image("logo1.jpg", width=100)  # Replace with your actual logo file

    # Display title in the second column
    with col2:
        st.title("Welcome to StoryStream")

    # Display app image (centered)
    st.image("app_image.jpg") 

    # Rest of your home page content
    st.markdown(
        """
        <div style="text-align: center;">
            <p style="font-size: 1.2em;">
                <b>Empowering Editorial Teams to Uncover Hidden Trends in News & Content</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Story Stream app is your AI-powered editorial assistant, designed to help you stay ahead of the curve. Our platform harnesses cutting-edge machine learning to analyze your text data and extract valuable insights that can drive your content strategy.
        """
    )

    with st.expander(":white[**Discover Story Stream's Key Features:**]"):
        st.markdown("- **Predict:** **Uncover the topics and categories hidden within your news articles and social media posts**.")
        st.markdown("- **Explore Data:** **Visualize trending keywords and phrases to identify hot topics and emerging stories**.")
        st.markdown("- **About Us:** **Learn more about the team of data scientists and journalists behind Story Stream**.")

#about us page
elif st.session_state.page == 'about_us':
    st.title("About Story Stream")

    st.markdown(
        """
        :black[Welcome to Story Stream! We are a team of passionate storytellers and data scientists 
        dedicated to helping you understand and categorize your text data.]

        **Our Mission:**
        :black[To empower individuals and organizations to unlock the insights hidden within their text data.]

        **Our Team:**
        """
    )
    # Define a dictionary to store image paths for each team member
    team_member_images = {
        "Neo Modibedi": "NeoModibedi.jpg",  # Replace with actual image filenames
        "Thapelo Robyn Raphala": "Robyn.jpg",
        "Thandekile Sikhakhane": "Thandekile.jpg",
        "Mbalenhle Lenepa": "Mbalenhle.jpg"
    }

    # Create columns for each team member and display image and description
    for member, image_file in team_member_images.items():
        col1, col2 = st.columns([1, 3])  # Adjust column ratio as needed

        # Display image in the first column (adjust width as needed)
        with col1:
            st.image(image_file, width=100)

        # Display member description in the second column
        with col2:
            st.markdown(f"- **{member}:** Experienced software engineer specializing in building scalable web applications.")
    st.markdown(
        """
        **Contact Us:**

        - **Email:** storystream@example.com
        - **Website:** https://www.storystream.com 

        We'd love to hear from you! Try out our text prediction app on the "Predict" page and let us know what you think.
        """
    )
#Project Overview page
if st.session_state.page == 'project_overview':
    st.title("Project Overview")

    st.subheader("The Problem")
    st.write(
        """
        News outlets and content creators face challenges in quickly categorizing and understanding vast amounts of text data.  Manual classification is time-consuming and prone to errors. Additionally, identifying key trends and patterns within different content categories is difficult without specialized tools.
        """
    )

    st.subheader("The Solution: StoryStream")
    st.write(
        """
        StoryStream is an AI-powered application designed to streamline content analysis and classification. It leverages machine learning models to automatically predict the category of news articles and social media posts.  This empowers editorial teams to:

        * **Quickly Classify Content:** Save time and resources by automating the categorization process.
        * **Identify Trends:** Gain insights into popular topics and emerging stories.
        * **Refine Content Strategy:**  Make data-driven decisions to optimize content creation and engagement.
        """
    )

    st.subheader("Key Features")
    st.markdown("""
        * **Text Prediction:**  Utilizes trained machine learning models to accurately predict the category of input text.
        * **Explore Data:**  Provides tools to visualize word clouds, word frequencies, and named entities within text data.
        * **Sentiment Analysis:**  Assesses the overall sentiment (positive, negative, neutral) of text content.
        """)

    st.subheader("Impact and Benefits")
    st.write(
        """
        StoryStream is a valuable asset for newsrooms, content marketers, and anyone working with large volumes of text data. It enhances efficiency, uncovers hidden insights, and ultimately enables users to create more relevant and impactful content.
        """
    )

# Explore data page
elif st.session_state.page == 'explore_data':
    st.title("Explore Your Data")
    st.markdown("Our Explore data feature will take your input text and derive insights.")
    st.markdown("- The page will display a visual that shows the common words in your text,the bolder the text, the more common it is.This is called a **word cloud**")
    st.markdown("- The page will display a barplot that shows the count of the frequently used words")
    st.markdown("- Extract entities from the text and count their frequencies")
    st.markdown("- Additionally, our feature allows you to detect the sentiment in the text,is it a positive tone,negative tone or neutral?")

    st.header("Enter Text:")
    text = st.text_area("Type or paste your text here", height=200)
    sentiment = 0 # Initialize sentiment outside the if statement
    sentiment_label = 'Neutral'
    if text:
        # Word Cloud Generation (same as before)
        stopwords = set(STOPWORDS)
        additional_stopwords = {'said', 'would', 'could', 'also','and','the','that','of'}
        stopwords.update(additional_stopwords)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

        # Display Word Cloud
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)

        # Top 10 Words Bar Chart
        words = text.split()
        word_counts = Counter(words)
        top_words = word_counts.most_common(10)

        # Create and display the bar chart
        top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        st.bar_chart(top_words_df.set_index('Word'))
        
        

        nlp = spacy.load("en_core_web_sm")  # Load a spaCy English model
        doc = nlp(text)

        # Extract entities and count their frequencies
        entity_counts = Counter([ent.label_ for ent in doc.ents])

        # Create and display the bar chart
        entity_df = pd.DataFrame(entity_counts.items(), columns=['Entity', 'Frequency'])
        st.bar_chart(entity_df.set_index('Entity'))  # Display NER bar chart
        
        # Categorize sentiment
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        # Display Sentiment
        # Sentiment Analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        # Categorize sentiment
        if sentiment > 0:
            sentiment_label = "Positive"
        elif sentiment < 0:
            sentiment_label = "Negative"
        

        # Display Sentiment
        st.subheader("Sentiment Analysis:")
        st.write(f"Overall Sentiment: {sentiment_label} ({sentiment:.2f})")

        # Create and display the pie chart
        labels = ["Positive", "Negative", "Neutral"]
        sizes = [
            max(0, sentiment),
            max(0, -sentiment), 
            1 - abs(sentiment)  # Neutral is the remaining portion
        ]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig)


    else:
      st.warning("Please enter some text to generate a word cloud.")
elif st.session_state.page == 'main_page':  # Added condition for main_page
    st.header("Predict Content Category \U0001F44B")  
    st.markdown("**Choose a model and enter your text below:**")
    # Model Selection
    selected_model_name = st.selectbox(
        "Choose a model:", list(model_options.keys())
    )

    # User Input
    st.header('Enter Text:')
    user_text = st.text_area('Type or paste your text here', height=200)

    # Prediction Button
    if st.button('predict'):
        if user_text:
            with st.spinner('Classifying...'):
                try:
                    selected_model_file = model_options[selected_model_name]
                    if not os.path.exists(selected_model_file):
                        raise FileNotFoundError(f"Model file not found: {selected_model_file}")  
                    
                    with open(selected_model_file, 'rb') as file:
                        model = pickle.load(file)

                    prediction = model.predict([user_text])[0]
                    st.success(f"Text classified successfully as '{prediction}!'")

                except FileNotFoundError as fnf_error:
                    st.error(f"Model not found. Please check the model path: {fnf_error}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning('Please enter some text.')

    # Model Descriptions
    with st.expander("About the Models"):
        st.write("**Logistic Regression:** A simple yet effective linear model for classification.")
        st.write("**Naive Bayes Classifier:**  A Naive Bayes Classifier is a simple probabilistic algorithm that makes predictions based on Bayes' theorem, assuming that features are conditionally independent given the class.")
        st.write("**Support Vector Machine:** A powerful model for finding complex patterns in data.")

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
    st.markdown(f"""
         <style>
         .stApp {{
             background: url("https://th.bing.com/th/id/R.d1963c15d5340e33e2efcc6c9252fd93?rik=aT2erCwidDkqSQ&riu=http%3a%2f%2fwonderfulengineering.com%2fwp-content%2fuploads%2f2014%2f07%2fLandscape-wallpapers-5.jpg&ehk=eQXzJTa5%2bRMjLs1CeOGXY4ZD0EWMFDU%2b1N%2bMclWB%2b1s%3d&risl=&pid=ImgRaw&r=0");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()