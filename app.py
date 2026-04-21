from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

app = Flask(__name__, template_folder='.', static_folder='.')

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

client = OpenAI(api_key=api_key)

# Resume data
RESUME = """
VENKATA KRISHNAN
Contact: +353 892161460 | vishalkrishnan3116@gmail.com | linkedin.com/in/venkatakrishnanM
Visa: Stamp 4 — Eligible to work without restrictions in Ireland. No sponsorship required.

PROFILE:
- Data Scientist with 8+ years of experience in Data Science, ML, and Generative AI.
- End-to-end owner of AI solutions from architecture to production deployment.
- Filed patents in NLP and Data Science methodologies.
- Expert in Python, PySpark, SQL, Azure, AWS, H2O, Keras, Ktrain.
- Built production-grade GenAI apps using Flask, Gradio, OpenAI, Azure OpenAI.
- Specialises in prompt engineering, LangChain, LangGraph, AI agent workflows.
- Deep expertise in NLP, RNNs, LSTMs, and large-scale data pipelines.

EXPERIENCE:

Data Scientist | Ardonagh Group, Ireland | June 2024 – Present
- Deep domain expertise in insurance brokerage workflows and data-driven decision making.
- Applied ML and GenAI to improve operational efficiency and analytical decision-making.
- Built GenAI-powered internal apps streamlining broker workflows.
- Deployed predictive solutions on complex, unstructured insurance documents.
- Automated brokerage processes with AI-driven document extraction tools.
- Led demos and executive presentations to senior leadership and CEOs.

Senior Data Scientist | DraftKings, Ireland | Oct 2023 – March 2024
- Analysed target markets for user betting; improved user engagement.
- Engineered linear quadratic estimation method for player rating models.

Data Scientist | Optum Technology (UHG), Ireland | Oct 2018 – Oct 2023
- Delivered data-driven solutions for complex US healthcare business problems.
- Processed large-scale datasets using PySpark and Python.
- Developed NLP and deep learning models on healthcare data.
- Mentored interns and junior team members.

PROJECTS:

SmartBroker — AI-Powered Insurance Broker Platform
- Cloud-native RAG platform on Azure for insurance document analysis.
- Semantic retrieval, vector search, multi-context conversational interfaces.
- LangChain & LangGraph orchestration; deployed in production at a leading Irish brokerage.
- Tech: GenAI, RAG, NLP, LangChain, LangGraph, Flask, Streamlit, Azure, CI/CD

Medicare Disability — NLP Model
- Predicted Medicare-eligible members to reduce claim audit effort.
- Tech: NLP, FastText, KTrain, Airflow

Claim Cost Savings — LSTM Trend Detection
- LSTM models detecting trend changes in medical expenditure.
- Tech: LSTM, Keras, Isolation Forest, Anomaly Detection

Model Interpretability — Reviewer Productivity
- Explainability layer for ML-based claim rejections.
- Tech: CatBoost, Faiss, Clustering

CAQH Model — XGBoost COB Prediction
- Reduced false positives in Coordination of Benefits identification.
- Tech: H2O, XGBoost, Sparkling Water, PySpark

Provider Segmentation
- Analytical model to segment providers by claim submission patterns.
- Tech: PySpark, Segmentation, Airflow

TECHNICAL SKILLS:
- Programming: Python, PySpark, R
- AI & GenAI: LLMs, NLP, OpenAI, Azure OpenAI, LangChain, LangGraph, Prompt Engineering, AI Agents
- Cloud: Azure (Blob, Synapse, Fabric), Databricks, AWS S3
- Databases: Azure Cosmos DB, SQL Server, MySQL, Oracle, Hive
- Visualisation: Tableau, Power BI, Excel
- Tools: VS Code, Jupyter Notebook, PyCharm, DBeaver
- Methods: Agile / Iterative Development

EDUCATION:
- Master's in Data Analytics, National College of Ireland, Dublin | Jan 2017 – Jan 2018 | Grade: 2:1
- B.E. in Electronics and Communication, MNM Jain College, Chennai | June 2011 – May 2015 | Grade: First Class (1:1)

ACHIEVEMENTS:
- Spotlight Award – Best Newcomer: Awarded for rapid impact and exceptional contributions.
- Spotlight Award – Rising Star: Recognising sustained high performance and growing leadership.
- Filed patents in NLP and Data Science methodologies.

CERTIFICATIONS:
- Attended COMPSTAT 2022 — The 24th International Conference on Computational Statistics.
- NLP Processing with Classification and Vector Spaces.
- Medicare and Retirement Foundation Certificate.

VISA STATUS:
- Stamp 4 visa holder in Ireland — fully eligible to work without restrictions. No sponsorship required.
"""

SYSTEM_PROMPT = f"""You are SmartBot, the personal AI assistant of Venkata Krishnan (Venkat), a senior Data Scientist based in Ireland with 8+ years of experience.

Answer questions about Venkat in a professional, warm, and concise way — as his personal representative. Use the resume below as your sole source of truth.

If asked something not in the resume, say you don't have that detail and suggest contacting Venkat on LinkedIn (linkedin.com/in/venkatakrishnanM) or email (vishalkrishnan3116@gmail.com). Never fabricate information.

Here is Venkat's full resume:

{RESUME}"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Build messages for OpenAI
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        
        # Add conversation history
        for msg in conversation_history:
            messages.append({'role': msg['role'], 'content': msg['content']})
        
        # Add current user message
        messages.append({'role': 'user', 'content': user_message})
        
        # Call OpenAI
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        return jsonify({'reply': assistant_message})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
