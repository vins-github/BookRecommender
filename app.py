import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import os  

# Load Data
artifacts_path = '/Users/arvindewonoto/socs/BookRecommender/artifacts'

# Memuat file pickle
final_rating = pickle.load(open(os.path.join(artifacts_path, 'final_rating.pkl'), 'rb'))
model = pickle.load(open(os.path.join(artifacts_path, 'model.pkl'), 'rb'))
book_names = pickle.load(open(os.path.join(artifacts_path, 'book_names.pkl'), 'rb'))
book_pivot = pickle.load(open(os.path.join(artifacts_path, 'book_pivot.pkl'), 'rb'))

# Pendahuluan
st.title("Selamat Datang di Sistem Rekomendasi Buku 📚")

st.markdown("""
### Mengapa Program Ini Dibuat?
Dalam era informasi digital, banyaknya jumlah buku yang tersedia dapat membuat pembaca kebingungan dalam memilih bacaan yang sesuai. 
Kami membuat sistem rekomendasi buku ini untuk:
1. **Memudahkan pengguna** menemukan buku yang relevan berdasarkan preferensi mereka.
2. **Menghemat waktu** dalam memilih buku dengan rekomendasi otomatis menggunakan Machine Learning.
3. Memberikan **pengalaman personalisasi** bagi setiap pengguna dengan fitur-fitur seperti rekomendasi berbasis genre, pengguna, dan pencarian cepat.

Sistem ini dirancang untuk memudahkan anda mencari buku yang cocok sesuai preferensi, genre dan merekomendasikan buku bagi seseorang yang sedang binggung ingin membaca buku apa.
""")

# Fetch Poster Function
def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url

# Recommend Books Function
def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url

# Streamlit User Interface
st.header('Sistem Rekomendasi Buku Menggunakan Machine Learning')

# Dropdown for Book Selection
selected_books = st.selectbox(
    "Ketik atau pilih buku dari dropdown",
    book_names
)

if st.button('Tampilkan Rekomendasi'):
    recommended_books, poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommended_books[2])
        st.image(poster_url[2])
    with col3:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommended_books[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommended_books[5])
        st.image(poster_url[5])

# Search Function for Books
def search_books(query, book_list):
    closest_matches = process.extract(query, book_list, limit=5)
    return [match[0] for match in closest_matches]

st.subheader("Mencari Buku")
user_input = st.text_input("Masukan Nama Buku")

if user_input:
    suggestions = search_books(user_input, book_names)
    st.write("Apakah maksud kamu:")
    for suggestion in suggestions:
        if st.button(suggestion):
            selected_books = suggestion
            recommended_books, poster_url = recommend_book(selected_books)
            st.subheader("Buku Rekomendasi")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_books[1])
                st.image(poster_url[1])
            with col2:
                st.text(recommended_books[2])
                st.image(poster_url[2])
            with col3:
                st.text(recommended_books[3])
                st.image(poster_url[3])
            with col4:
                st.text(recommended_books[4])
                st.image(poster_url[4])
            with col5:
                st.text(recommended_books[5])
                st.image(poster_url[5])
#genre filter
st.header('Rekomendasi Buku Berdasarkan Genre')
genre_keywords = {
    "Fantasy": ["magic", "dragon", "wizard", "kingdom", "sword"],
    "Romance": ["love", "heart", "kiss", "wedding", "romance"],
    "Horror": ["ghost", "haunted", "dark", "fear", "death"],
    "Mystery": ["mystery", "detective", "crime", "case", "murder"],
    "Science Fiction": ["alien", "space", "robot", "future", "galaxy"]
}
def filter_books_by_genre(selected_genre, book_names):
    keywords = genre_keywords.get(selected_genre, [])
    filtered_books = [book for book in book_names if any(keyword.lower() in book.lower() for keyword in keywords)]
    return filtered_books
# Dropdown untuk memilih genre
selected_genre = st.selectbox("Pilih sebuah Genre:", list(genre_keywords.keys()))

if st.button('Show Books by Genre'):
    filtered_books = filter_books_by_genre(selected_genre, book_names)

    if not filtered_books:
        st.write("Tidak ada buku ditemukan untuk genre ini")
    else:
        st.write(f"Books in {selected_genre} genre:")
        for book in filtered_books[:10]:  # Batasi hasil untuk tampilan
            st.text(book)

st.header('Rekomendasi Buku Random')
# Fitur Suggest a Random Book dengan Expander
if st.button('Suggest a Random Book'):
    random_book = np.random.choice(book_names)
    st.write(f"Bagaimana dengan buku ini **{random_book}**?")

    # Dapatkan rekomendasi berdasarkan buku acak
    recommended_books, poster_url = recommend_book(random_book)

    # Gunakan expander untuk membuat rekomendasi bisa diminimalkan
    with st.expander("Lihat Buku Rekomendasi"):
        st.subheader("Buku Rekomendasi:")
        for book, poster in zip(recommended_books, poster_url):
            st.text(book)
            st.image(poster)