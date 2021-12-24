#!/usr/bin/env python
# coding: utf-8

# In[9]:


from fer import FER
from PIL import Image 
import matplotlib.pyplot as plt
import streamlit as st
import imageio
st.write('''
#  Emotion Detector
''')
st.write("A Image Classification Web App That Detects the Emotions Based On An Image")
file = st.file_uploader("Please Upload an image of Person With Face", type=['jpg','png'])
if file is None:
  st.text("Please upload an image file")
else:
  image = imageio.imread(file)
  emo_detector = FER(mtcnn=True)
  captured_emotions = emo_detector.detect_emotions(image)
# Print all captured emotions with the image
  st.write(captured_emotions[0]['emotions'])
  #plt.imshow(image)

  # Use the top Emotion() function to call for the dominant emotion in the image
  dominant_emotion, emotion_score = emo_detector.top_emotion(image)
  st.write(dominant_emotion, emotion_score)
  st.image(image, use_column_width=True)

# In[ ]:




