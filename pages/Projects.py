import streamlit as st
import os
# --- PROJECTS ---
BLOGS = {
    "🚀 what is context length in AI models? Does a Huge Model Always Mean a Huge Context Length? 🤔": {
        "link": "https://www.linkedin.com/posts/anandsahu_what-is-context-length-in-ai-models-activity-7275004375318626304-MIzX?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/context.png",  
    },
    "🚀 Fine-Tuned Models vs. Large Language Models: Which Should You Choose? 🤔": {
        "link": "https://www.linkedin.com/posts/anandsahu_fine-tuned-models-vs-llms-how-to-choose-activity-7274642002753847296-YJQf?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/FINEvsLLM.png",  
    },
    "🚀 Choosing the Right Python Version for Your Projects: Why Not Always the Latest? 🐍": {
        "link": "https://www.linkedin.com/posts/anandsahu_why-not-the-latest-choosing-the-right-python-activity-7273215506130616320-SgXd?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/version.png",  
    },
    "🚀 Discover the Future of LLM Optimization! 🔍": {
        "link": "https://www.linkedin.com/posts/anandsahu_rag-vs-fine-tuning-exploring-top-emerging-activity-7272830051543363584-SKf9?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/RAGvsFine.png",  
    },
    "💡 Building a RAG System Without LangChain! 💡": {
        "link": "https://www.linkedin.com/posts/anandsahu_building-a-rag-retrieval-augmented-generation-activity-7272105309995741184-8hDi?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/pine.png",  
    },
    "🚀 Understanding the Building Blocks of LLMs: Characters, Tokens, and Chunks! 🧩": {
        "link": "https://www.linkedin.com/posts/anandsahu_decoding-the-jargon-characters-tokens-activity-7271018162869915648-3kU2?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/token.png",  
    },
    "🚀 Understanding LLM API Pricing: A Deep Dive into Gemini API 💡": {
        "link": "https://www.linkedin.com/posts/anandsahu_how-do-llm-apis-charge-money-a-simple-guide-activity-7270655708109361152-AVIq?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/api.png",  
    },
    "🚀 Demystifying Quantization in LLMs 🧠": {
        "link": "https://www.linkedin.com/posts/anandsahu_simplifying-quantization-in-llms-gguf-gptq-activity-7270427845716271104-eLtQ?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/quant.png",  
    },
    "🌟 Exploring the Evolution of Generative AI 🚀": {
        "link": "https://www.linkedin.com/posts/anandsahu_the-journey-of-generative-ai-from-deep-learning-activity-7269338515660816385-iu3v?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/blogs/evolution.png",  
    },
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
# --- MLOPS PROJECTS ---

MLOPS_PROJECTS = {
    "🔧 Demonstrating My Skills Through a Production-Ready MLOps Project!": {
        "link": "https://www.linkedin.com/posts/anandsahu_mlops-aws-docker-activity-7276539257219858432-6TYx?utm_source=share&utm_medium=member_desktop",
        "image": "./assets/mlops/mlops.png",  
    }
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
    options=["Blogs","Gen AI Projects","PyTorch Projects","MLOPS Projects"],
    index=0,  # Default to "All Projects"
)
from PIL import Image

def display_projects(blog_dict, image_width=300, image_height=200, columns=3):
    blog_items = list(blog_dict.items())
    
    # Custom CSS for consistent spacing and layout
    st.markdown(
        """
        <style>
        .project-card {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 10px;
            margin: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #f9f9f9;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .project-card img {
            margin-bottom: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    for i in range(0, len(blog_items), columns):
        # Create a row of columns
        cols = st.columns(columns)
        
        # Populate each column with blog details
        for col, (blog_name, blog_details) in zip(cols, blog_items[i : i + columns]):
            with col:
                image_path = blog_details.get("image", "")
                
                # Start the project card
                st.markdown('<div class="project-card">', unsafe_allow_html=True)
                
                # Check if the image is a local file and exists
                if image_path.startswith("./assets") and os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        img = img.resize((image_width, image_height))
                        st.image(
                            img,
                            caption=None,  # We'll add the caption manually for better styling
                            use_container_width=False,
                        )
                    except Exception as e:
                        st.error(f"Error loading image for {blog_name}: {e}")
                else:
                    # Display the placeholder image link
                    st.image(
                        image_path,  # The placeholder link is stored in the dictionary
                        caption=None,  # We'll add the caption manually for better styling
                        use_container_width=False,
                        width=image_width,
                    )
                
                # Add the blog name and link
                st.markdown(f"**{blog_name}**", unsafe_allow_html=True)
                st.markdown(f"[Read More]({blog_details['link']})", unsafe_allow_html=True)
                
                # End the project card
                st.markdown('</div>', unsafe_allow_html=True)


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
elif menu_option == "MLOPS Projects":
    st.subheader("🤖 MLOPS Projects")
    display_projects(MLOPS_PROJECTS)

# Footer
st.write("---")
st.markdown(
    "💡 **Pro Tip**: Click on Read More to explore more!"
)
