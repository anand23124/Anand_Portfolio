import streamlit as st
import os
# --- PROJECTS ---
BLOGS = {
    "While Studying Batch Normalization , Did You Wonder: Why Don’t We Call It Batch Standardization?": {
        "link": "https://www.linkedin.com/posts/anandsahu_deeplearning-machinelearning-ai-activity-7268134713477771264-Q6Wi?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/batch.png",  
    },
    "Hume.ai: Emotionally Intelligent AI for a Better Tomorrow": {
        "link": "https://www.linkedin.com/posts/anandsahu_artificialintelligence-emotionalintelligence-activity-7267031848931708928-BIqU?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/hume.png",  
    },
    "Unlocking Efficiency: Quantizing Mistral Models to 4-Bit Precision": {
        "link": "https://www.linkedin.com/posts/anandsahu_github-anand23124quantizationmistral-activity-7266815813431480320-1NRc?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/quantization.png",  
    },
    "📈 Python became the most popular language of 2024": {
        "link": "https://www.linkedin.com/posts/anandsahu_github-python-activity-7259448200158138369-1OTe?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/python_popular.png",  
    },
    "🚀 Diving deep into LangGraph": {
        "link": "https://www.linkedin.com/posts/anandsahu_langgraph-multiagentchatbot-conversationalai-activity-7247168817137635329-L6GM?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/langgraph.png",  
    },
    "📚 Exploring Knowledge Graphs": {
        "link": "https://www.linkedin.com/posts/anandsahu_neo4j-groqapi-knowledgegraph-activity-7236621475623911424-iMCH?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/knowlegegraph.png",  
    },
}


# --- PYTORCH PROJECTS ---
PYTORCH_PROJECTS = {
    "⚡ RAG Flash Reranker": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/5.RAG_Flask_Reranker_Superfast_Reranking",
        "image": "./assets/pytorch/flash.png",  
    },
    "🤝 RAG Fusion": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/4.RAG_Reciprocal_Rank_Fusion",
        "image": "./assets/pytorch/fusion.png",  
    },
    "🔧 Lost in Middle Phenomenon": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/3.RAG_Lost_in_the_middle_problem",
        "image": "./assets/pytorch/lost.png",  
    },
    "🔎 Exploring Hybrid Search": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/1.Hybrid_Search_From_Scratch",
        "image": "./assets/pytorch/hybrid.png",  
    },
    "🔍 Reranking Techniques": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/2.Reranking_From_Scratch",
        "image": "./assets/pytorch/rag_tech.png",  
    },
    "🌐 Sequence-to-Sequence Model for Translation": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/7.Seq2Seq_Model/Seq2Seq_model.ipynb",
        "image": "./assets/pytorch/seq2seq.png",  
    },
    "💬 Sentiment Analysis with LSTM": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/6.Sentiment_Analysis_LSTM/Sentiment_Analysis_LSTM.ipynb",
        "image": "./assets/pytorch/sentiment.png",  
    },
    "📝 Sentiment Analysis with Word Embeddings": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/5.Sentiment_Analysis_CBOW/Sentiment_Analysis_using_CBOW.ipynb",
        "image": "./assets/pytorch/sentiment.png",  
    },
    "🎯 Classification with PyTorch": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/4.Classification_Neural_Network/Classification_using_Neural_Network.ipynb",
        "image": "./assets/pytorch/classification.png",  
    },
    "📊 Logistic Regression in PyTorch": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/3.Logistic_Regression/Logistic_regression_with_pytorch.ipynb",
        "image": "./assets/pytorch/logistic.png",  
    },
    "📈 Linear Regression in PyTorch": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/2.Linear_Regression/Linear_regression_with_pytorch.ipynb",
        "image": "./assets/pytorch/linear.png",  
    },
    "🚀 Exploring PyTorch Basics": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/1.Introduction/Introduction_to_torch.ipynb",
        "image": "./assets/pytorch/pytorch_basics.png",  
    },
}


# --- GEN AI PROJECTS ---
GEN_AI_PROJECTS = {
    "🌟 Intelligent Document Processing and Querying System Using FastAPI": {
        "link": "https://www.linkedin.com/posts/anandsahu_thankyou-ai-nlp-activity-7263482567889207296-Zv52?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/intelligent_document.png",  
    },
    "Automated FAQ Chatbot for Course Providers": {
        "link": "https://www.linkedin.com/posts/anandsahu_nlp-machinelearning-chatbot-activity-7229775942724800512-d9Km?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/autofaq.png",  
    },
    "Gen AI: Restaurant Name & Menu Generator": {
        "link": "https://www.linkedin.com/posts/anandsahu_generativeai-langchain-googlegeminipro-activity-7231974175509233665-Bcl5?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/restaurant.png",  
    },
    "🔍 Transforming How We Analyze News": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7232967376575582208-mYy_?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/newsai.png",  
    },
    "🔍 Solving Real-World Problems: Retail Store Database Manager": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7234054525936930816-LlnT?utm_source=share&utm_medium=member_desktop",
        "image": "assets/genai/retail.png",  
    },
    "🔍 Building an Advanced RAG Pipeline: A Multi-Source Information Retrieval Model": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7234779346081595393-aC3G?utm_source=share&utm_medium=member_desktop",
        "image": "assets/genai/advanceRAG.png",  
    },
    "🔍 Building a Custom ChatGPT Model": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7235504103043280897-M8t1?utm_source=share&utm_medium=member_desktop",
        "image": "assets/genai/customgpt.png",  
    },
    "🔍 Tailoring Cold Emails with AI for Business Development Success 📧": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-businessdevelopment-coldemail-activity-7236311001594269696-WMS-?utm_source=share&utm_medium=member_desktop",
        "image": "assets/genai/coldemail.png",  
    },
    "Invoice Information Extractor": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7236591299343093761-WkhZ?utm_source=share&utm_medium=member_desktop",
        "image": "assets/genai/invoice.png",  
    },
    "Nutritional Content Analyzer": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7237316060276813824-6yhz?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/nutrition.png",  
    },
    "🚀 Efficiently Handling 2000 PDFs with Response Time Under Second! 🕒📄": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-nlp-llm-activity-7240221781666447360-fUyW?utm_source=share&utm_medium=member_desktop",
        "image": "assets/genai/pdf2000.png",  
    },
    "🌟 AI-Powered Personalized Diet Plans! 🌟": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-healthtech-dietplan-activity-7243476653346091009--7dJ?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/aidiet.png",  
    },
    "Excited to introduce StudyBuddy AI—your personal learning companion that makes studying easier! 📘✨": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-learninginpublic-learningmadeeasy-activity-7246381639444951040-sKIU?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/studybuddy.png",  
    },
    "Deply LLM with FastAPI": {
        "link": "https://www.linkedin.com/posts/anandsahu_fastapi-ai-llm-activity-7246768779341258752-scb3?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/llm_api.png",  
    },
    "🚀 Custom NER Model for Medical Insights! 🏥✨": {
        "link": "https://www.linkedin.com/posts/anandsahu_nlp-spacy-customner-activity-7247840762439094272-U5AV?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/ner_medical.png",  
    },
    "🌟 Introducing AskHR AI: Revolutionizing Employee Support 🤖✨": {
        "link": "https://www.linkedin.com/posts/anandsahu_askhr-ai-employeesupport-activity-7248932982034513920-8ldz?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/genai/Askhr.png",  
    },
}


st.set_page_config(page_title="Projects | Anand Sahu", page_icon=":wave:")

# Page Title
st.title("Projects & Blogs")
st.write("---")

# Sidebar Option Menu
menu_option = st.sidebar.radio(
    "Select Category",
    options=[ "Gen AI Projects","PyTorch Projects","Blogs"],
    index=0,  # Default to "All Projects"
)

import os
import streamlit as st

import os
import streamlit as st

# Function to display blogs with a fallback for missing images
def display_projects(blog_dict, image_width=300):
    for blog_name, blog_details in blog_dict.items():
        image_path = blog_details.get("image", "")
        
        # Check if the image is a local file and exists
        if image_path.startswith("./assets") and os.path.exists(image_path):
            st.image(
                image_path,
                caption=blog_name,
                use_container_width=False,
                width=image_width,
            )
        else:
            # Display the placeholder image link
            st.image(
                image_path,  # The placeholder link is stored in the dictionary
                caption=blog_name,
                use_container_width=False,
                width=image_width,
            )
        
        # Add the link to the blog below the image
        st.markdown(f"[Read More]({blog_details['link']})", unsafe_allow_html=True)
        st.write("---")  # Divider between blogs


# Display projects based on selected category
if menu_option == "Gen AI Projects":
    st.subheader("🧠 Generative AI Projects")
    display_projects(GEN_AI_PROJECTS)
elif menu_option == "PyTorch Projects":
    st.subheader("🔥 PyTorch Projects")
    display_projects(PYTORCH_PROJECTS)
elif menu_option == "Blogs":
    st.subheader("✍️ Blogs")
    display_projects(BLOGS)

# Footer
st.write("---")
st.markdown(
    "💡 **Pro Tip**: Click on Read More to explore more!"
)
