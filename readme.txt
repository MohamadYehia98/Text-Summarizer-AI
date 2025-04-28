- Text Summarization.AI

I implemented an abstractive text summarization system using the T5-small pretrained model from Hugging Face’s
Transformers library. The goal was to generate concise and meaningful summaries from longer input texts.
To achieve this, I fine-tuned the T5-small model on a custom dataset containing 10,000 text-summary pairs, 
with 8,000 examples for training and 2,000 for testing.


How to Run the Project ?
Follow these steps to set up and run the project on your local machine:

1- Clone the github repo to your machine
    command : git clone <repository-url>
  
2- Open the project folder in the VScode or other code editor

3- Open a terminal (PowerShell or your terminal of choice)

4- Create a virtual enviroment:
    command: py -3.10 -m venv venv 
  
5- Activate the virtual enviroment:
    command: .venv\Scripts\activate 
    
6- Install project dependencies: 
    command: pip install -r requirements.txt 
    
7- Run the application using Uvicorn: 
    command: uvicorn main:app --reload 