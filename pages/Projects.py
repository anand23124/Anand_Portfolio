import streamlit as st

# --- PROJECTS ---
BLOGS = {
    "📈 Python became the most popular language of 2024": {
        "link": "https://www.linkedin.com/posts/anandsahu_github-python-activity-7259448200158138369-1OTe?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Python+Popularity",  # Replace with your image URL
    },
    "🚀 Diving deep into LangGraph": {
        "link": "https://www.linkedin.com/posts/anandsahu_langgraph-multiagentchatbot-conversationalai-activity-7247168817137635329-L6GM?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=LangGraph",  # Replace with your image URL
    },
    "📚 Exploring Knowledge Graphs": {
        "link": "https://www.linkedin.com/posts/anandsahu_neo4j-groqapi-knowledgegraph-activity-7236621475623911424-iMCH?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Knowledge+Graph",  # Replace with your image URL
    },
}


# --- PYTORCH PROJECTS ---
PYTORCH_PROJECTS = {
    "⚡ RAG Flash Reranker": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/5.RAG_Flask_Reranker_Superfast_Reranking",
        "image": "https://via.placeholder.com/300x200?text=RAG+Flash+Reranker",  # Replace with your image URL
    },
    "🤝 RAG Fusion": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/4.RAG_Reciprocal_Rank_Fusion",
        "image": "https://via.placeholder.com/300x200?text=RAG+Fusion",  # Replace with your image URL
    },
    "🔧 Lost in Middle Phenomenon": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/3.RAG_Lost_in_the_middle_problem",
        "image": "https://via.placeholder.com/300x200?text=Lost+in+Middle",  # Replace with your image URL
    },
    "🔎 Exploring Hybrid Topics": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/1.Hybrid_Search_From_Scratch",
        "image": "https://via.placeholder.com/300x200?text=Hybrid+Search",  # Replace with your image URL
    },
    "🔍 Reranking Techniques": {
        "link": "https://github.com/anand23124/RAG_Different_Topics/tree/master/2.Reranking_From_Scratch",
        "image": "https://via.placeholder.com/300x200?text=Reranking+Techniques",  # Replace with your image URL
    },
    "🌐 Sequence-to-Sequence Model for Translation": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/7.Seq2Seq_Model/Seq2Seq_model.ipynb",
        "image": "https://via.placeholder.com/300x200?text=Seq2Seq+Model",  # Replace with your image URL
    },
    "💬 Sentiment Analysis with LSTM": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/6.Sentiment_Analysis_LSTM/Sentiment_Analysis_LSTM.ipynb",
        "image": "https://via.placeholder.com/300x200?text=Sentiment+Analysis+LSTM",  # Replace with your image URL
    },
    "📝 Sentiment Analysis with Word Embeddings": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/5.Sentiment_Analysis_CBOW/Sentiment_Analysis_using_CBOW.ipynb",
        "image": "https://via.placeholder.com/300x200?text=Sentiment+Analysis+CBOW",  # Replace with your image URL
    },
    "🎯 Classification with PyTorch": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/4.Classification_Neural_Network/Classification_using_Neural_Network.ipynb",
        "image": "https://via.placeholder.com/300x200?text=Classification+NN",  # Replace with your image URL
    },
    "📊 Logistic Regression in PyTorch": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/3.Logistic_Regression/Logistic_regression_with_pytorch.ipynb",
        "image": "https://via.placeholder.com/300x200?text=Logistic+Regression",  # Replace with your image URL
    },
    "📈 Linear Regression in PyTorch": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/2.Linear_Regression/Linear_regression_with_pytorch.ipynb",
        "image": "https://via.placeholder.com/300x200?text=Linear+Regression",  # Replace with your image URL
    },
    "🚀 Exploring PyTorch Basics": {
        "link": "https://github.com/anand23124/Pytorch_Learning/blob/master/1.Introduction/Introduction_to_torch.ipynb",
        "image": "https://via.placeholder.com/300x200?text=PyTorch+Basics",  # Replace with your image URL
    },
}


# --- GEN AI PROJECTS ---
GEN_AI_PROJECTS = {
    "🌟 Intelligent Document Processing and Querying System": {
        "link": "https://www.linkedin.com/posts/anandsahu_thankyou-ai-nlp-activity-7263482567889207296-Zv52?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Intelligent+Document+Processing",  # Replace with your image URL
    },
    "Automated FAQ Chatbot for Course Providers": {
        "link": "https://www.linkedin.com/posts/anandsahu_nlp-machinelearning-chatbot-activity-7229775942724800512-d9Km?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=FAQ+Chatbot",  # Replace with your image URL
    },
    "Gen AI: Restaurant Name & Menu Generator": {
        "link": "https://www.linkedin.com/posts/anandsahu_generativeai-langchain-googlegeminipro-activity-7231974175509233665-Bcl5?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Restaurant+Name+Generator",  # Replace with your image URL
    },
    "🔍 Transforming How We Analyze News": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7232967376575582208-mYy_?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=News+Analysis",  # Replace with your image URL
    },
    "🔍 Solving Real-World Problems: Retail Store Database Manager": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7234054525936930816-LlnT?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Retail+Store+Manager",  # Replace with your image URL
    },
    "🔍 Building an Advanced RAG Pipeline: A Multi-Source Information Retrieval Model": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7234779346081595393-aC3G?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=RAG+Pipeline",  # Replace with your image URL
    },
    "🔍 Building a Custom ChatGPT Model": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7235504103043280897-M8t1?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Custom+ChatGPT",  # Replace with your image URL
    },
    "🔍 Tailoring Cold Emails with AI for Business Development Success 📧": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-businessdevelopment-coldemail-activity-7236311001594269696-WMS-?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Cold+Emails+AI",  # Replace with your image URL
    },
    "Invoice Information Extractor": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7236591299343093761-WkhZ?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Invoice+Extractor",  # Replace with your image URL
    },
    "Nutritional Content Analyzer": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-machinelearning-langchain-activity-7237316060276813824-6yhz?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=Nutritional+Analyzer",  # Replace with your image URL
    },
    "🚀 Efficiently Handling 2000 PDFs with Response Time Under Second! 🕒📄": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-nlp-llm-activity-7240221781666447360-fUyW?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=2000+PDFs+Processing",  # Replace with your image URL
    },
    "🌟 AI-Powered Personalized Diet Plans! 🌟": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-healthtech-dietplan-activity-7243476653346091009--7dJ?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=AI+Diet+Plans",  # Replace with your image URL
    },
    "Excited to introduce StudyBuddy AI—your personal learning companion that makes studying easier! 📘✨": {
        "link": "https://www.linkedin.com/posts/anandsahu_ai-learninginpublic-learningmadeeasy-activity-7246381639444951040-sKIU?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=StudyBuddy+AI",  # Replace with your image URL
    },
    "Thrilled to share my recent journey with FastAPI": {
        "link": "https://www.linkedin.com/posts/anandsahu_fastapi-ai-llm-activity-7246768779341258752-scb3?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=FastAPI+Journey",  # Replace with your image URL
    },
    "🚀 Custom NER Model for Medical Insights! 🏥✨": {
        "link": "https://www.linkedin.com/posts/anandsahu_nlp-spacy-customner-activity-7247840762439094272-U5AV?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=NER+Medical+Model",  # Replace with your image URL
    },
    "🌟 Introducing AskHR AI: Revolutionizing Employee Support 🤖✨": {
        "link": "https://www.linkedin.com/posts/anandsahu_askhr-ai-employeesupport-activity-7248932982034513920-8ldz?utm_source=share&utm_medium=member_desktop",
        "image": "https://via.placeholder.com/300x200?text=AskHR+AI",  # Replace with your image URL
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

# Function to display projects with or without images
def display_projects(project_dict, show_images=True):
    for project_name, project_details in project_dict.items():
        if show_images and "image" in project_details:  # If images are enabled and available
            st.markdown(
                f"""
                <a href="{project_details['link']}" target="_blank">
                    <img src="{project_details['image']}" alt="{project_name}" style="width:300px; height:auto; margin-bottom:10px;">
                </a>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"### {project_name}")
        else:  # Text-only display
            st.write(f"- [{project_name}]({project_details['link']})")
        st.write("---")

# Content Container
with st.container():
    if menu_option == "Gen AI Projects":
        st.subheader("Gen AI Projects")
        display_projects(GEN_AI_PROJECTS)
    elif menu_option == "PyTorch Projects":
        st.subheader("PyTorch Projects")
        display_projects(PYTORCH_PROJECTS)  # Disable images for PyTorch projects
    elif menu_option == "Blogs":
        st.subheader("Blogs")
        display_projects(BLOGS)