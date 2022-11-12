from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json



def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()



def message(user_input):
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    return (answer['챗봇'])



# ------------------------------------Flask App -------------------------------------
app = Flask(__name__)
@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/get')
def Chatbot():

    user_msg = request.args.get('msg')

    return message(user_msg)


if __name__=="__main__":
    app.run()
