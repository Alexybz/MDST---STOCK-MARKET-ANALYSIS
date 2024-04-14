from PIL import Image
import streamlit as st
import pickle as pk

def main():
    st.title('Stock Analysis Homepage')

    st.header('Home')

    st.markdown('Visit our [GitHub](https://github.com) repository!')


    if st.button('Directory'):
        st.write('Hello, welcome to our app!')

    st.header('Intro ')
    st.markdown("""---""")
    st.write('This is our stock predictor model: ')
    image_path = 'Predictor_model.png'

    image = Image.open(image_path)
    st.image(image, caption='Stock Predictor Model', use_column_width=True)
    
if __name__ == '__main__':
    main()
