# Chat Bot by Data Science

Welcome to the Chat Bot by Data Science application, designed to interact with audio documents using natural language processing capabilities. This app integrates various technologies to facilitate audio transcription, text chunking, embedding generation, and conversational AI.

## Features:
- Audio Processing: Upload audio files (formats: WAV, MP3, M4A) to transcribe and analyze.
- Natural Language Understanding: Utilizes LangChain for text chunking and OpenAIEmbeddings for generating embeddings.
- Conversational AI: Implements a chatbot using LangChain's ChatOpenAI model to answer questions based on audio document content.
- Predefined Questions: Displays predefined questions related to the audio content and retrieves answers.
- Rate Limit Management: Implements a 20-second wait between predefined questions to manage rate limits effectively.

### How to Use:
- Upload Your Audio: Choose an audio file and click on "Process" to transcribe it.
- Ask Questions: After processing, ask questions about the audio content in the text input field provided.
  - Sidebar Options:
    - API Key Input: Enter your OpenAI API key and Hugging Face Hub API token for access.
    - Description: Provides an overview of the app's functionality and components.
    - Predefined Questions: Displays and retrieves answers to predefined questions related to the audio content.

### Installation and Setup:
To run the application locally, ensure you have the necessary libraries installed:

```bash
pip install streamlit langchain whisper
```

### Run the application using:

```
streamlit run app.py
```

### Data Sources and Technologies:
- Whisper: Used for audio transcription.
- LangChain: Handles text chunking, embeddings, and conversational AI.
- OpenAIEmbeddings: Generates embeddings for text chunks.
- Hugging Face Hub: Provides additional NLP models and functionalities.

Explore and interact with your audio documents using the Chat Bot by Data Science app!
