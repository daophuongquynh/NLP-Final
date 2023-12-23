# Tải thư viện cần thiết
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_dark
import pandas as pd
import numpy as np
import chardet
import pandas as pd
import numpy as np
import re
import string
import contractions
import unidecode
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from functions import clean_text, predict, vectorizer, svm 

# Kiểm tra đoạn tin nhắn là spam hoặc ham
def check_spam():
    input_value = dpg.get_value("Input")
    input_value = clean_text(input_value)
    outcome = predict(input_value,vectorizer,svm)
    dpg.set_value(text_id, outcome) 
    if outcome == "message is spam":
        dpg.bind_item_theme(text_id, red_text_theme)
    else:
        dpg.bind_item_theme(text_id, green_text_theme)

# Điều chỉnh kích thước của window và viewport
screen_width = 1920 // 2
screen_height = 750

dpg.create_context()
dpg.create_viewport(title='Main Window', width=screen_width, height=screen_height, resizable=False)
dpg.setup_dearpygui()

# def show_style_editor(sender, data):
#     dpg.show_style_editor()

# Upload hình ảnh vào giao diện
width, height, channels, data = dpg.load_image("logo_spamFilter.png")
with dpg.texture_registry():
    texture_id = dpg.add_static_texture(width, height, data, tag="image_id")

# Định nghĩa các nút bấm, hướng dẫn nhập thông tin và dòng hiển thị kết quả
with dpg.window(label="SMS Spam Detector", pos=(0,0),width=screen_width,height=screen_height) as window:
    dpg.set_global_font_scale(1.35)
    dpg.add_image(texture_id,pos=(20,30))
    dpg.add_spacer(height=185)
    dpg.add_separator()
    dpg.add_spacer(height=40)
    bt1 = dpg.add_button(label="Please enter an SMS message of your choice to check if it's spam or not",width=700)
    # dpg.add_button(label='Show Style Editor', callback=show_style_editor)
    dpg.add_spacer(height=40)
    input_text_id = dpg.add_input_text(hint="Type a message here!",width=800,tag="Input",label="Input")
    dpg.add_spacer(height=30)
    bt2 = dpg.add_button(label='Check',callback=check_spam,width=100,height=60)
    dpg.add_spacer(height=30)
    dpg.add_separator()
    dpg.add_spacer(height=45)
    text_id = dpg.add_input_text(default_value="The predicted result will be put here",width=600)

# Căn chỉnh định dạng (màu sắc, vị trí) của nút bấm
background_color = (37,37,38)
with dpg.theme() as button_right:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign,0.0, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5.0, category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Button,background_color, category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,background_color, category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,background_color, category=dpg.mvThemeCat_Core)


with dpg.theme(default_theme=True) as theme_id:
     with dpg.theme_component(dpg.mvAll):
         dpg.add_theme_color(dpg.mvThemeCol_WindowBg,background_color, category=dpg.mvThemeCat_Core)

# Set theme chung cho toàn bộ các nút bấm, text input         
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,10.0,category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,40,65,category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg,(233,138,21),category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Button,(233,138,21),category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvNodeCol_TitleBar,(233,138,21),category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,(233,138,21),category=dpg.mvThemeCat_Core)





with dpg.theme() as text_id_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,background_color,category=dpg.mvThemeCat_Core)
dpg.bind_item_theme(text_id,text_id_theme)

with dpg.theme() as red_text_theme:
    with dpg.theme_component(dpg.mvInputText):
        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,background_color,category=dpg.mvThemeCat_Core)


with dpg.theme() as green_text_theme:
    with dpg.theme_component(dpg.mvInputText):
        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg,background_color,category=dpg.mvThemeCat_Core)


dpg.bind_theme(global_theme)
dpg.bind_item_theme(bt1,button_right)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()



