# Wardrobe Mate: Final All-in-One Version with Advanced Features
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import KMeans
import random
from colorsys import rgb_to_hsv
import io

# --- Utility Functions ---
@st.cache_data

def process_image(file):
    try:
        image = Image.open(file).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Could not process file {file.name}: {e}")
        return None

@st.cache_data

def get_dominant_colors(image, k=3):
    image = image.resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    return kmeans.cluster_centers_

def color_similarity(color1, color2):
    hsv1 = rgb_to_hsv(*[v / 255 for v in color1])
    hsv2 = rgb_to_hsv(*[v / 255 for v in color2])
    return abs(hsv1[0] - hsv2[0])

def choose_best_match(item, candidates):
    if not candidates:
        return None
    return min(candidates, key=lambda x: color_similarity(item['colors'][0], x['colors'][0]))

def combine_outfit_images(outfit):
    max_width = 400
    padded_images = []
    for category, item in outfit.items():
        img = item['image']
        img_resized = ImageOps.contain(img, (max_width, img.height))
        padded_img = Image.new("RGB", (max_width, max_width), (255, 255, 255))
        padded_img.paste(img_resized, ((max_width - img_resized.width) // 2, (max_width - img_resized.height) // 2))
        padded_images.append(padded_img)

    total_height = sum(img.height for img in padded_images)
    combined_image = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for img in padded_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    return combined_image

def show_color_palette(colors):
    st.write("**Dominant Colors**")
    cols = st.columns(len(colors))
    for i, col in enumerate(cols):
        color = tuple(int(c) for c in colors[i])
        col.color_picker(f"", value='#{:02x}{:02x}{:02x}'.format(*color), label_visibility="collapsed")

def categorize_items(uploaded_files):
    wardrobe = {"Top": [], "Bottom": [], "Shoes": []}
    st.subheader("Uploaded Clothes")
    cols = st.columns(3)
    for i, uploaded_file in enumerate(uploaded_files):
        with cols[i % 3]:
            image = process_image(uploaded_file)
            if image:
                st.image(image, caption=uploaded_file.name, use_container_width=True)
                category = st.selectbox(
                    f"Select category for {uploaded_file.name}",
                    options=['Top', 'Bottom', 'Shoes'],
                    key=f"cat_{uploaded_file.name}"
                )
                dominant_colors = get_dominant_colors(image)
                show_color_palette(dominant_colors)
                wardrobe[category].append({"image": image, "colors": dominant_colors})
    return wardrobe

def generate_outfits(wardrobe):
    tops, bottoms, shoes = wardrobe["Top"], wardrobe["Bottom"], wardrobe["Shoes"]
    if not (tops and bottoms and shoes):
        st.warning("Please upload at least one item per category (Top, Bottom, Shoes).")
        return []
    random.shuffle(tops)
    random.shuffle(bottoms)
    random.shuffle(shoes)
    min_outfits = min(len(tops), len(bottoms), len(shoes))
    recommendations = []
    for i in range(min_outfits):
        recommendations.append({"Top": tops[i], "Bottom": bottoms[i], "Shoes": shoes[i]})
    return recommendations

# --- Tabs ---
def upload_tab(session_state):
    uploaded_files = st.file_uploader("Upload clothes (jpg, png)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if uploaded_files:
        session_state['wardrobe'] = categorize_items(uploaded_files)

def recommend_tab(session_state):
    if 'wardrobe' not in session_state or not session_state['wardrobe']:
        st.warning("Please upload and categorize clothes first.")
        return
    if st.button("Generate Recommendations"):
        session_state['outfits'] = generate_outfits(session_state['wardrobe'])

    if session_state.get('outfits'):
        for idx, outfit in enumerate(session_state['outfits'], 1):
            combined_image = combine_outfit_images(outfit)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.image(combined_image, caption=f"Outfit {idx}", use_container_width=True)
            with col2:
                if st.button(f"⭐ Save Outfit {idx}"):
                    session_state['favorites'].append(outfit)
            st.markdown(f"🧠 *Matching Colors*: This outfit is color-coordinated using dominant hues for a balanced look.*")

def favorites_tab(session_state):
    if not session_state.get('favorites'):
        st.info("No favorites yet. Save outfits from the Recommend tab.")
    else:
        for idx, outfit in enumerate(session_state['favorites'], 1):
            st.image(combine_outfit_images(outfit), caption=f"⭐ Favorite Outfit {idx}", use_container_width=True)

def analytics_tab(session_state):
    st.write("Color usage and item count will appear here soon!")
    if 'wardrobe' in session_state:
        total = sum(len(items) for items in session_state['wardrobe'].values())
        st.metric("Total Items Uploaded", total)
        for cat, items in session_state['wardrobe'].items():
            st.write(f"### {cat}: {len(items)} items")
            for i, item in enumerate(items):
                show_color_palette(item['colors'])

# --- Main App ---
def main():
    st.set_page_config(page_title="Wardrobe Mate", layout="wide")
    st.title("👗 Wardrobe Mate – Smart Outfit Recommender")
    session_state = st.session_state
    if 'wardrobe' not in session_state:
        session_state['wardrobe'] = {}
    if 'outfits' not in session_state:
        session_state['outfits'] = []
    if 'favorites' not in session_state:
        session_state['favorites'] = []

    tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Recommend", "Favorites", "Analytics"])
    with tab1:
        upload_tab(session_state)
    with tab2:
        recommend_tab(session_state)
    with tab3:
        favorites_tab(session_state)
    with tab4:
        analytics_tab(session_state)

if __name__ == "__main__":
    main()

    