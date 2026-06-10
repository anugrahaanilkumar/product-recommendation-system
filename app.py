# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# # Set up page configurations
# st.set_page_config(
#     page_title="E-Commerce Cluster Recommender",
#     page_icon="🛍️",
#     layout="wide"
# )

# # 1. Load the Exported Pickle File efficiently using Streamlit caching
# @st.cache_resource
# def load_recommendation_artifacts():
#     with open('final_birch_recommendation_model.pkl', 'rb') as f:
#         artifacts = pickle.load(f)
#     return artifacts

# try:
#     artifacts = load_recommendation_artifacts()
#     df_final = artifacts['df_final']
#     df_filtered=artifacts['df_filtered']
#    # user_cluster_map = artifacts['user_cluster_map']
# except FileNotFoundError:
#     st.error("⚠️ 'final_birch_recommendation_model.pkl' not found. Please run your training script first to export the model.")
#     st.stop()

# # --- SIDEBAR: Project Context & Metrics ---
# st.sidebar.title("📊 Product Recommendation")
# st.sidebar.markdown("""
# **Model Architecture:**
# * **Clustering Engine:** KMEANS
# * **Target Segments:** 11 Clusters
# * **Dataset Scale:** ~7.8 Million Rows
# """)

# # Display basic system statistics
# total_users = df_final['userId'].nunique()
# total_products = df_final['productId'].nunique()
# st.sidebar.metric(label="Total Profiled Users", value=f"{total_users:,}")
# st.sidebar.metric(label="Total Active Products", value=f"{total_products:,}")

# # --- MAIN APP INTERFACE ---
# st.title("🛍️ E-Commerce Product Recommendation System")
# st.write("This application uses **KMeans Clustering** to group users based on their buying behaviors and recommend products favored by their peer group.")

# st.divider()
# st.title("🛍️ User Recommendation")
# # User Selection Row
# col1, col2 = st.columns([2, 1])

# with col1:
#     # Get a sample list of users to display in the dropdown
#     sample_users = df_final['userId'].tolist()
#     selected_user = st.selectbox(
#         "🕵️‍♂️ Select or Type a User ID to generate recommendations:",
#         options=sample_users
#     )

# with col2:
#     num_recommendations = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5)

# # --- RECOMMENDATION LOGIC EXECUTION ---
# if st.button("✨ Generate Real-Time Recommendations", type="primary"):

#     with st.spinner("Analyzing consumer profile and cluster trends..."):

#         # 1. Find the selected user's cluster assignment
#         user_cluster_row = df_final[df_final['userId'] == selected_user]

#         if user_cluster_row.empty:
#             st.warning("User ID not found in the trained database matrix.")
#         else:
#             user_cluster = user_cluster_row['kmeans_cluster'].iloc[0]

#             # 2. Extract products this user has already rated (so we don't re-recommend them)
#             already_rated_products = df_final[df_final['userId'] == selected_user]['productId'].tolist()

#             # 3. Pull behavioral profiles of all peers belonging to the same cluster
#             peer_profiles = df_final[df_final['kmeans_cluster'] == user_cluster]

#             # 4. Filter out already rated products from peer data
#             valid_recommendations = peer_profiles[~peer_profiles['productId'].isin(already_rated_products)]

#             # 5. Aggregate metrics to identify high-quality, popular products within the cluster
#             top_products = (
#                 valid_recommendations.groupby('productId')['Rating']
#                 .agg(Average_Rating='mean', Total_Votes='count')
#                 .reset_index()
#             )

#             # Sort strategically: Priority to higher average rating, backed by high vote volume
#             top_products = top_products.sort_values(
#                 by=['Average_Rating', 'Total_Votes'],
#                 ascending=[False, False]
#             ).head(num_recommendations)

#             # --- DISPLAY RESULTS UI ---
#             st.success(f"Successfully matched user **{selected_user}** to Behavioral Segment Profile: **Cluster {user_cluster}**")

#             st.subheader(f"🎯 Top {num_recommendations} Personalized Products")

#             if top_products.empty:
#                 st.info("No unrated products found within this user segment cluster.")
#             else:
#                 # Format output table nicely for the user
#                 top_products.columns = ['Product ID', 'Predicted Affinity (Avg Rating)', 'Segment Demand Count']
#                 top_products.reset_index(drop=True, inplace=True)

#                 # Style formatting for rating decimals
#                 st.dataframe(
#                     top_products.style.format({'Predicted Affinity (Avg Rating)': '{:.2f} ⭐'}),
#                     use_container_width=True
#                 )

#                 # Context block explaining the recommendation rationale
#                 st.caption(
#                     f"*Rationale: These products are highly rated by users within Cluster {user_cluster} "
#                     f"and have not yet been evaluated or purchased by User {selected_user}.*"
#                 )

# st.divider()
# def get_cluster_recommendations(cluster_id, top_n):
#     # 1. Filter data for only users in this specific cluster
#     cluster_data = df_final[df_final['kmeans_cluster'] == cluster_id]

#     # 2. Calculate the popularity (count) and quality (mean rating) for each product
#     product_stats = cluster_data.groupby('productId')['Rating'].agg(['count', 'mean'])

#     # 3. Filter for products that have a high average rating and reasonable vote count
#     # Sorting by 'mean' rating first, then 'count' of ratings
#     top_products = product_stats.sort_values(by=['mean', 'count'], ascending=False)

#     return top_products.head(top_n)
# # ==========================================================


# # 3. Streamlit User Interface (UI)
# st.title("🛍️ Cluster Recommendation")

# # Create a dropdown menu for selecting the cluster ID
# available_clusters = sorted(df_final['kmeans_cluster'].dropna().unique())
# selected_cluster = st.selectbox("Select Cluster ID:", options=available_clusters)

# # Create a slider for selecting how many products to recommend
# selected_top_n = st.slider("Number of Recommendations:", min_value=1, max_value=20, value=5)

# # 4. Run your exact function when the user changes inputs
# st.subheader(f"🔥 Top Recommendations for Cluster {selected_cluster}")

# if st.button("✨ Generate Real-Time Recommendations for the cluster ", type="primary"):
#     # Calling your function exactly as you designed it:
#     recomm = get_cluster_recommendations(cluster_id=selected_cluster, top_n=selected_top_n)
#     # 5. Display the output DataFrame directly in the Streamlit App window
#     if not recomm.empty:
#         st.dataframe(recomm, use_container_width=True)
#     else:
#         st.warning("No data found for this cluster.")


# st.title("🛍️ User Recommendation")
# # User Selection Row
# col3, col4 = st.columns([4, 3])

# with col3:
#     # Get a sample list of users to display in the dropdown
#     sample_users = df_final['userId'].tolist()
#     selected_user = st.selectbox(
#         "🕵️‍♂️ Select or Type a User ID to generate recommendations:",
#         options=sample_users,
#         key='user'
#     )

# with col4:
#     selected_no_recomm = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5, key='2')

# def get_user_recommendations(target_user, df, top_n_user):

#     # 1. Identify what cluster the user belongs to
#     user_rows = df[df['userId'] == target_user]
#     if user_rows.empty:
#         print(f" User '{target_user}' not found in the dataset.")
#         return None

#     user_cluster = user_rows['kmeans_cluster'].iloc[0]

#     # 2. Extract products this user has already interacted with
#     already_rated = user_rows['productId'].tolist()

#     # 3. Pull historical interactions of everyone in the same cluster
#     cluster_peers = df[df['kmeans_cluster'] == user_cluster]

#     # 4. Filter out items the target user has already seen
#     recommendation_pool = cluster_peers[~cluster_peers['productId'].isin(already_rated)]

#     # 5. Calculate average ratings and popularity metrics for the remaining pool
#     top_products = (
#         recommendation_pool.groupby('productId')['Rating']
#         .agg(Predicted_Rating='mean', Total_Cluster_Reviews='count')
#         .reset_index()
#     )

#     # 6. Sort by highest rating first, using review count to break ties
#     sorted_recommendations = top_products.sort_values(
#         by=['Predicted_Rating', 'Total_Cluster_Reviews'],
#         ascending=[False, False]
#     ).head(top_n_user).reset_index(drop=True)

#     return sorted_recommendations

# if st.button("✨ Generate Real-Time Recommendations for the user ", type="primary",key=3):
#     # Calling your function exactly as you designed it:
#     user_recomm = get_user_recommendations(target_user=selected_user,df=df_final,top_n_user=selected_no_recomm)
#     # 5. Display the output DataFrame directly in the Streamlit App window
#     if not user_recomm.empty:
#         st.dataframe(user_recomm, use_container_width=True)
#     else:
#         st.warning("No data found for this user.")




# st.divider()

# st.title("🛍️ Product Recommendation")
# from scipy.sparse import csr_matrix
# from sklearn.metrics.pairwise import cosine_similarity
# @st.cache_resource
# def build_matrices(_df):
#     # We use a leading underscore (_df) to tell Streamlit not to hash the massive dataframe
#     user_matrix = csr_matrix((_df['Rating'], (_df['user_idx'], _df['product_idx'])))
#     prod_matrix = user_matrix.T

#     # Pre-build mappings once
#     categories = _df['productId'].astype('category').cat.categories
#     idx_to_p = dict(enumerate(categories))
#     p_to_idx = {v: k for k, v in idx_to_p.items()}

#     return prod_matrix, idx_to_p, p_to_idx

# # Fetch everything instantly from memory cache
# product_user_matrix, idx_to_prod, prod_to_idx = build_matrices(df_filtered)


# # 2. Interactive App UI Elements
# st.title("🛍️ Real-Time Product Recommender")

# all_products = sorted(df_filtered['productId'].unique())
# selected_product = st.selectbox("Select a Product:", options=all_products, key="prod_select")
# selected_top_product = st.slider("Number of Recommendations:", min_value=1, max_value=20, value=5, key="top_slider")


# # 3. Recommendation Function
# def get_cosine_product_recommendations(target_product_id, top_n=5):
#     if target_product_id not in prod_to_idx:
#         return f"⚠️ Product '{target_product_id}' not found."

#     target_idx = prod_to_idx[target_product_id]

#     # Quick single-row similarity check against the stored matrix
#     target_vector = product_user_matrix[target_idx]
#     similarity_scores = cosine_similarity(target_vector, product_user_matrix).ravel()

#     similar_indices = np.argsort(similarity_scores)[::-1]

#     prod_recommendations = []
#     for idx in similar_indices:
#         if idx == target_idx:
#             continue

#         prod_id = idx_to_prod[idx]
#         score = similarity_scores[idx]

#         if score <= 0.0:
#             continue

#         prod_recommendations.append({
#             'productId': prod_id,
#             'Cosine_Similarity': round(float(score), 4)
#         })

#         if len(prod_recommendations) >= top_n:
#             break

#     return pd.DataFrame(prod_recommendations)


# st.divider()

# # 4. Trigger UI Processing
# if st.button("✨ Generate Real-Time Product Recommendations", type="primary",key='product'):
#     with st.spinner("Calculating recommendation vectors..."):
#         get_prod_recommendations = get_cosine_product_recommendations(
#             target_product_id=selected_product,
#             top_n=selected_top_product
#         )

#         if isinstance(get_prod_recommendations, pd.DataFrame) and not get_prod_recommendations.empty:
#             st.dataframe(get_prod_recommendations, use_container_width=True, hide_index=True)
#         elif isinstance(get_prod_recommendations, pd.DataFrame) and get_prod_recommendations.empty:
#             st.warning("No overlapping consumer patterns found for this item.")
#         else:
#             st.error(get_prod_recommendations)


import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity



# Set up page configurations
st.set_page_config(
    page_title="E-Commerce Cluster Recommender",
    page_icon="🛍️",
    layout="wide"
)

# ==========================================
# 1. OPTIMIZED DATA CACHING (The Biggest Win)
# ==========================================
@st.cache_resource
def load_recommendation_artifacts():
    # 🔥 UPDATED: Pointing to your new compressed K-Means file using joblib
    with open('final_birch_recommendation_model.pkl_compressed.pkl', 'rb') as f:
        artifacts = joblib.load(f)

    # CRITICAL: Extract unique lists ONCE inside cache so it doesn't recalculate
    artifacts['unique_users'] = sorted(artifacts['df_final']['userId'].dropna().unique())
    artifacts['unique_products'] = sorted(artifacts['df_filtered']['productId'].dropna().unique())
    return artifacts

try:
    artifacts = load_recommendation_artifacts()
    df_final = artifacts['df_final']
    df_filtered = artifacts['df_filtered']
    unique_users_list = artifacts['unique_users']      # Fast cached array
    unique_products_list = artifacts['unique_products']  # Fast cached array
except FileNotFoundError:
    st.error("⚠️ 'final_birch_recommendation_model.pkl' not found. Please run your training script first to export the model.")
    st.stop()


@st.cache_resource
def build_matrices(_df):
    # Compressed Sparse Row generation
    user_matrix = csr_matrix((_df['Rating'], (_df['user_idx'], _df['product_idx'])))
    prod_matrix = user_matrix.T

    # Pre-build mappings once inside cache memory
    categories = _df['productId'].astype('category').cat.categories
    idx_to_p = dict(enumerate(categories))
    p_to_idx = {v: k for k, v in idx_to_p.items()}

    return prod_matrix, idx_to_p, p_to_idx

# Fetch matrix structures instantly from memory cache
product_user_matrix, idx_to_prod, prod_to_idx = build_matrices(df_filtered)


# ==========================================
# BACK-END CORE CORE FUNCTIONS (Untouched Logic)
# ==========================================
def get_cluster_recommendations(cluster_id, top_n):
    cluster_data = df_final[df_final['kmeans_cluster'] == cluster_id]
    product_stats = cluster_data.groupby('productId')['Rating'].agg(['count', 'mean'])
    top_products = product_stats.sort_values(by=['mean', 'count'], ascending=False)
    st.success(f"Successfully matched products for Cluster **{cluster_id}**")
    return top_products.head(top_n)


def get_user_recommendations(target_user, df, top_n_user):
    user_rows = df[df['userId'] == target_user]
    if user_rows.empty:
        return None

    user_cluster = user_rows['kmeans_cluster'].iloc[0]
    already_rated = user_rows['productId'].tolist()
    cluster_peers = df[df['kmeans_cluster'] == user_cluster]
    recommendation_pool = cluster_peers[~cluster_peers['productId'].isin(already_rated)]

    top_products = (
        recommendation_pool.groupby('productId')['Rating']
        .agg(Predicted_Rating='mean', Total_Cluster_Reviews='count')
        .reset_index()
    )

    sorted_recommendations = top_products.sort_values(
        by=['Predicted_Rating', 'Total_Cluster_Reviews'],
        ascending=[False, False]
    ).head(top_n_user).reset_index(drop=True)

    st.success(f"Successfully matched user **{selected_user}** to **Cluster {user_cluster}**")

    return sorted_recommendations


def get_cosine_product_recommendations(target_product_id, top_n=5):
    if target_product_id not in prod_to_idx:
        return f"⚠️ Product '{target_product_id}' not found."

    target_idx = prod_to_idx[target_product_id]
    target_vector = product_user_matrix[target_idx]
    similarity_scores = cosine_similarity(target_vector, product_user_matrix).ravel()

    similar_indices = np.argsort(similarity_scores)[::-1]

    prod_recommendations = []
    for idx in similar_indices:
        if idx == target_idx:
            continue

        prod_id = idx_to_prod[idx]
        score = similarity_scores[idx]

        if score <= 0.0:
            continue

        prod_recommendations.append({
            'productId': prod_id,
            'Cosine_Similarity': round(float(score), 4)
        })

        if len(prod_recommendations) >= top_n:
            break

    st.success(f"Successfully matched similar products related to the selected product **{target_product_id}**")

    return pd.DataFrame(prod_recommendations)


# ==========================================
# FRONT-END USER INTERFACE (UI) LAYOUT
# ==========================================

# SIDEBAR: Context & System Metrics
st.sidebar.title("Model Configurations")
st.sidebar.markdown("""
**Model Engine:** K-Means
* **Target Segments:** 11 Behavioral Clusters
* **Dataset Scale:** ~7.8 Million Rows
""")

total_users = len(unique_users_list)
total_products = len(unique_products_list)
st.sidebar.metric(label="Total Profiled Users", value=f"{total_users:,}")
st.sidebar.metric(label="Total Active Products", value=f"{total_products:,}")


# MAIN DISPLAY TITLE

st.html(
    """
    <div style="
        background-color: #463B68;      /* Change this Hex for your desired banner color */
        padding: 24px;
        extra-padding: 10px;
        border-radius: 12px;
        border-left: 8px solidrgb(60, 20, 168); /* Adds a clean vertical accent line on the left */
        margin-bottom: 25px;
    ">
        <h1 style="
            color: #FFFFFF !important;  /* Force title text to be white */
            margin: 0;
            font-size: 2.5rem;
            font-family: sans-serif;
        ">
            🛍️ E-Commerce Recommendation System
        </h1>
        <p style="
            color: #94A3B8 !important;  /* Soft grey color for subheadline */
            margin: 8px 0 0 0;
            font-size: 1.1rem;
        ">
            An end-to-end recommendation engine delivering real-time, personalized insights across users, clusters, and individual products.
        </p>
    </div>"""
)

st.divider()


# ------------------------------------------
# MODULE 1: USER PERSOANLIZED RECOMMENDATIONS
# ------------------------------------------

st.header("👤 User Recommendations")
st.write("Predicts what an individual customer will love based on the trends of their mapped peer cluster.")

col1, col2 = st.columns([2, 1])
with col1:
    # ⚡ FIX: Uses the small unique list instead of .tolist() on 7.8M records!
    selected_user = st.selectbox(
        "Select or Type a Target User ID:",
        options=unique_users_list,
        key="user_select_mod1"
    )
with col2:
    num_recommendations = st.slider("Number of recommendations:", min_value=3, max_value=10, value=5, key="slider_mod1")

if st.button("✨ Generate User Recommendations", type="primary", key="btn_mod1"):
    with st.spinner("Analyzing user historical profiles..."):
        user_recomm = get_user_recommendations(target_user=selected_user, df=df_final, top_n_user=num_recommendations)

        if user_recomm is not None and not user_recomm.empty:
            user_recomm.columns = ['Product ID', 'Predicted Affinity Rating', 'Cluster Demand Vol']
            st.dataframe(user_recomm.style.format({'Predicted Affinity Rating': '{:.2f} ⭐'}), use_container_width=True, hide_index=True)
            st.caption(f"*Rationale: Items favored highly by peers within this target profile, unpurchased by user {selected_user}.*")
        else:
            st.warning("No recommendations available for this user.")
st.divider()


# ------------------------------------------
# MODULE 2: CLUSTER BROAD TRENDS
# ------------------------------------------
st.header("🛍️ Cluster Recommendations")
st.write("Inspects and displays popular inventory items prioritized across an entire target cluster.")

col3, col4 = st.columns([2, 1])
with col3:
    available_clusters = sorted(df_final['kmeans_cluster'].dropna().unique())
    selected_cluster = st.selectbox("Select Target Cluster Segments:", options=available_clusters, key="cluster_select_mod2")
with col4:
    selected_top_n = st.slider("Number of Recommendations:", min_value=3, max_value=10, value=5, key="slider_mod2")

if st.button("✨ Generate Products for Selected Cluster", type="primary", key="btn_mod2"):
    with st.spinner("Calculating cluster distribution metrics..."):
        recomm = get_cluster_recommendations(cluster_id=selected_cluster, top_n=selected_top_n)
        if not recomm.empty:
            recomm.index.name = 'Product ID'
            st.dataframe(recomm.rename(columns={'count': 'Total Group Reviews', 'mean': 'Average Quality Score'}), use_container_width=True)
        else:
            st.warning("No data records found for this cluster.")
st.divider()


# ------------------------------------------
# MODULE 3: ITEM-ITEM COSINE SIMILARITY
# ------------------------------------------
st.header("🔄 Product-to-Product Recommendations")
st.write("Uses high-speed on-demand matrix calculations to find similar products related to selected product.")

col5, col6 = st.columns([2, 1])
with col5:
    selected_product = st.selectbox("Select base Catalog Product ID:", options=unique_products_list, key="prod_select_mod3")
with col6:
    selected_top_product = st.slider("Max items to display:", min_value=3, max_value=10, value=5, key="slider_mod3")

if st.button("✨ Generate Similar Products", type="primary", key="btn_mod3"):
    with st.spinner("Executing spatial cosine distance vectors..."):
        get_prod_recommendations = get_cosine_product_recommendations(target_product_id=selected_product, top_n=selected_top_product)

        if isinstance(get_prod_recommendations, pd.DataFrame) and not get_prod_recommendations.empty:
            st.dataframe(get_prod_recommendations.rename(columns={'productId': 'Alternative Product ID', 'Cosine_Similarity': 'Similarity'}), use_container_width=True, hide_index=True)
        else:
            st.warning("No similar products available for selectec product")

st.write("<br><br><br><br>", unsafe_allow_html=True)
