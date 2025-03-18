import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import streamlit as st

model = joblib.load('svc_baru.pkl')
vectorizer = joblib.load('vec_baru.pkl')
kamus = joblib.load('kamus2.pkl')

def cleaning(komen):
    komen = komen.lower()
    komen = re.sub(r'[^a-zA-Z\s]', '', komen)
    komen = re.sub(r'(.)\1{2,}', r'\1', komen)
    komen = re.sub(r'\b\w\b', '', komen)
    komen = re.sub(r'\s+', ' ', komen)
    return komen

def tokenizing(komen):
    from nltk.tokenize import word_tokenize
    komen = word_tokenize(komen)
    return komen

def normalisasi(komen):
    komen = [kamus[word] if word in kamus else word for word in komen]
    return komen

def stopwords(komen):
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    stopword = factory.get_stop_words()
    kecuali = ['baik', 'buruk', 'bagus', 'jelek', 'yang', 'sejauh', 'ini', 'belum', 'ada', 'cukup', 'untuk', 'saat', 'ini', 'tidak', 'ada', 'belum', 'sangat', 'kurang']
    stopstop = [word for word in stopword if word not in kecuali]
    
    # from nltk.corpus import stopwords
    # stopword_eng = set(stopwords.words("english"))
    
    
    komen = [word for word in komen if word not in stopstop]
    # komen = [word for word in komen if word not in stopword_eng]
    return komen

def stemming(komen):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    komen = [stemmer.stem(word) for word in komen]
    komen = ' '.join(komen)
    
    return komen


def pipeline(komen):
    komen = cleaning(komen)
    # st.write("After cleaning")
    # st.caption(komen)
    komen = tokenizing(komen)
    # st.write("After tokenizing")
    # st.caption(komen)
    komen = normalisasi(komen)
    # st.write("After normalisasi")
    # st.caption(komen)
    komen = stopwords(komen)
    # st.write("After stopwords")
    # st.caption(komen)
    komen = stemming(komen)
    # st.write("After stemming")
    # st.caption(komen)
    
    from wordcloud import WordCloud
    
    wordcloud  = WordCloud(
        width= 800,
        height= 400,
        background_color= "white"
    ).generate(komen)
    st.image(wordcloud.to_image())
    
    return komen

def prediksi(komen):
    result_proces = pipeline(komen)
    tfidf_result = vectorizer.transform([result_proces])
    # st.write(tfidf_result)
    
    # prediksi
    result_prediksi = model.predict(tfidf_result)
    st.write(result_prediksi)
    
    if result_prediksi == "positif":
        st.info("Survey Siswa Tersebut POSITIF")
    else:
        st.info("Survey Siswa Tersebut NEGATIF")
        
    
    
    
    
# =================== WEB APSS ==========================

st.title("CLASSIFICATION SURVEY KEPUASAN SMK")

with st.sidebar:
    st.header("Berikut Pilihan Kategori Kepuasan Siswa")
    selectment = st.multiselect("Pilih Kategori : ",
                options= ["positif", "negatif"]   ,
                default= ["positif", "negatif"]
    )



df = pd.read_csv("Survey Kepuasan SMK_Train.csv", delimiter= ";")
df_main = df[df['Label'].isin(selectment)]

tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
with tab1:
    st.header("Tabel Survey Kepuasan")
    st.write(df_main)
    
    
    
    st.header("Jumlah Kategori Kepuasan Siswa")
    fig, ax = plt.subplots(figsize = (10,7))
    sns.barplot(
    x = df_main['Label'].value_counts().index,
    y = df_main['Label'].value_counts().values,
    ax = ax
    )
    for i in range(len(df_main['Label'].value_counts().index)):
        plt.text(i, df_main['Label'].value_counts().values[i], df_main['Label'].value_counts().values[i], va = "bottom" )
    ax.set_title("Kepuasan Siswa")
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize = (10,7))
    ax.pie(
        df_main['Label'].value_counts(),
        autopct= "%1.1f%%",
        labels= df_main['Label'].value_counts().index
    )
    ax.set_title("Presentase Nilai Sentimen", fontsize = 18)
    st.pyplot(fig)
    
    
    st.header("Kata Kata")
    from wordcloud import WordCloud
    fig, ax = plt.subplots(figsize = (10,7))
    text_gabung = " ".join(df_main['Survey'])
    wordcloud = WordCloud(
        width= 800,
        height= 400,
        background_color= "white"
    ).generate(text_gabung)
    ax.imshow(wordcloud)
    ax.set_title("Word Cloud", fontsize = 18, fontweight = "bold")
    ax.axis("Off")
    st.pyplot(fig)
    
    
    st.header("Persebaran")
    from sklearn.decomposition import PCA
    pca = PCA(n_components= 2)

    X_pca1 = pca.fit_transform(vectorizer.transform(df_main['Survey']))
    
    fig, ax = plt.subplots(figsize = (10,7))
    sns.scatterplot(
    x = X_pca1[:, 0],
    y = X_pca1[:, 1],
    hue = df_main["Label"],
    ax = ax
    )
    ax.set_title("Persebaran Berdasarkan Sentiment", fontsize = 18)
    st.pyplot(fig)
    
    

with tab2:
    user_input = st.text_area("Masukkan Kata")
    button1 = st.button("Submit", key= "1")
    
    if button1:
        if user_input:
            st.write("Kalimat anda : ")
            st.caption(user_input)
            
            result_final = prediksi(user_input)
            st.write(result_final)
    

with tab3:
    df_input = st.file_uploader("Masukkan File yang ingin di Prediksi (csv only): ", type= "csv")
    button2 = st.button("Submit", key= "2")
    
    if button2:
        if df_input:
            df_input = pd.read_csv(df_input, delimiter= ";")
            df_input = df_input.iloc[:, [0]]
            df_input.columns = ["Survey"]
            
            st.header("Data Anda : ")
            st.write(df_input)
            
            
            def preprocess_input_df(komen):
                komen = cleaning(komen)
                komen = tokenizing(komen)
                komen = normalisasi(komen)
                komen = stopwords(komen)
                komen = stemming(komen)
                return komen
                
            # Prosess df masuk
            df_input["Clean_Text"] = df_input["Survey"].apply(preprocess_input_df)
            # st.write(df_input)
            
            from wordcloud import WordCloud
            fig, ax = plt.subplots(figsize = (10,7))
            text_gabung_input = " ".join(df_input["Clean_Text"])
            wordcloud  = WordCloud(
                width= 800,
                height= 400,
                background_color= "white"
            ).generate(text_gabung_input)
            ax.imshow(wordcloud)
            ax.set_title("Word Cloud", fontsize = 18, fontweight = "bold")
            ax.axis("Off")
            st.pyplot(fig)
            
            # Prediksi
            tfidf_df_input = vectorizer.transform(df_input['Clean_Text'])
            prediksi_input = model.predict(tfidf_df_input)
            
            df_input["Label"] = prediksi_input
            st.header("Data Prediksi Sentiment :")
            st.write(df_input)
            
            
            
            # Nilai Unik
            st.header("Jumlah Kategori Kepuasan Siswa")
            fig, ax = plt.subplots(figsize = (10,7))
            sns.barplot(
            x = df_input['Label'].value_counts().index,
            y = df_input['Label'].value_counts().values,
            ax = ax
            )
            for i in range(len(df_input['Label'].value_counts().index)):
                plt.text(i, df_input['Label'].value_counts().values[i], df_input['Label'].value_counts().values[i], va = "bottom" )
            ax.set_title("Kepuasan Siswa")
            st.pyplot(fig)
            
            fig, ax = plt.subplots(figsize = (10,7))
            ax.pie(
                df_input['Label'].value_counts(),
                autopct= "%1.1f%%",
                labels= df_input['Label'].value_counts().index
            )
            ax.set_title("Presentase Nilai Sentimen", fontsize = 18)
            st.pyplot(fig)
            
            # Persebaran Data
            X_pca2 = pca.fit_transform(tfidf_df_input)
    
            fig, ax = plt.subplots(figsize = (10,7))
            sns.scatterplot(
            x = X_pca2[:, 0],
            y = X_pca2[:, 1],
            hue = df_input["Label"],
            ax = ax
            )
            ax.set_title("Persebaran Berdasarkan Sentiment", fontsize = 18)
            st.pyplot(fig)
            
            
            
            
            
                
    
            
            
