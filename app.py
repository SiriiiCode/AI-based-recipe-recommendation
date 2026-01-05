import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Smart Dish Recommendation",
    layout="centered",
    page_icon="🍳"
)

# ================== CUSTOM STYLES ==================
st.markdown("""
<style>
.big-input-label {
    font-size: 40px;
    font-weight: 600;
    margin-bottom: 5px;
}
.small-label {
    font-size: 16px;
    font-weight: 500;
    margin-top: 10px;
}
.project-title {
    font-size: 60px;
}
.tag {
    allignment: center;
    font-size: 15px
}
</style>
""", unsafe_allow_html=True)

# ================== TITLE ==================
st.markdown('<div class="project-title">🍳 Smart Dish Recommendation System</div>', unsafe_allow_html=True)
st.caption('<div class="tag">AI-based ingredient matching and dish recommendation system</div>', unsafe_allow_html=True)

# ================== LOAD DATA ==================
data = pd.read_csv("recipes.csv")
data["ingredients"] = data["ingredients"].str.lower().str.replace(",", " ").str.strip()

# ================== TF-IDF MODEL ==================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["ingredients"])

# ================== USER INPUT ==================
st.markdown('<div class="big-input-label">🥕 Enter ingredients you have:</div>', unsafe_allow_html=True)
ingredients = st.text_input("", placeholder="egg, onion, oil")

st.markdown('<div class="small-label">🍽 Select Food Type</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    veg = st.checkbox("Veg")
with col2:
    non_veg = st.checkbox("Non-Veg")

# ================== BUTTON ==================
if st.button("🍽 Get Recommendations"):

    if not ingredients:
        st.warning("Please enter ingredients!")
        st.stop()

    user_input = ingredients.lower().replace(",", " ").strip()
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, tfidf_matrix)[0]
    data["similarity"] = similarity

    # Filter by food type
    if veg and not non_veg:
        filtered = data[data["type"].str.lower() == "veg"]
    elif non_veg and not veg:
        filtered = data[data["type"].str.lower() == "non-veg"]
    else:
        filtered = data.copy()

    # Sort and select top matches
    filtered = filtered.sort_values(by="similarity", ascending=False)
    results = filtered.head(3)

    st.markdown("## 🍽 Recommended Dishes")

    if results.empty:
        st.warning("No matching recipes found.")
    else:
        for _, row in results.iterrows():
            recipe_set = set(row["ingredients"].split())
            user_set = set(user_input.split())
            missing = recipe_set - user_set
            score = round(row["similarity"], 2)

            st.markdown(f"### 🍲 {row['dish_name']}")
            st.write(f"⏱ **Time:** {row['time']} minutes")
            st.write(f"🔥 **Calories:** {row['calories']} kcal")

            if score >= 0.6:
                st.success(f"Excellent Match ({score})")
            elif score >= 0.3:
                st.warning(f"Partial Match ({score})")
            else:
                st.info(f"Low Match ({score})")

            st.progress(score)

            st.write(
                f"❗ **Missing Ingredients:** {', '.join(missing) if missing else 'None! You have everything 🎉'}"
            )

            st.divider()
