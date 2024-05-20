import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from flask import Flask, request, jsonify
import firebase_admin 
from firebase_admin import credentials, firestore 

# Data Loading
post_data = pd.read_csv("post_data.csv")
interaction_data = pd.read_csv("interaction_data.csv")

# Assign Default Ratings (you can adjust the default rating value here)
interaction_data['rating'] = 3

# Load DataFrame into Dataset
reader = Reader(rating_scale=(1, 5))
interaction_data_surprise = Dataset.load_from_df(interaction_data[['uid', 'post_id', 'rating']], reader)

# Firebase Initialization (replace with your credentials)
cred = credentials.Certificate("firebase-credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()  

app = Flask(__name__)

# Hybrid Recommender Function with Refinement
def hybrid_recommender(user_profile_id, post_data, interaction_data):
    # Check if user is an existing user (in interaction_data)
    if user_profile_id in interaction_data['uid'].unique():
        app.logger.info(f'User {user_profile_id} found in interaction data.')
        return recommendations_for_existing_user(user_profile_id, post_data, interaction_data)

    # If not an existing user, check in Firestore for new users
    else:
        app.logger.info(f'User {user_profile_id} not found in interaction data, checking Firestore...')
        user_doc_ref = db.collection('Users').document(user_profile_id)
        user_doc = user_doc_ref.get()

        if user_doc.exists:
            app.logger.info(f'User {user_profile_id} found in Firestore.')
            return recommendations_for_new_user(user_profile_id, post_data)  # Pass only post_data
        else:
            app.logger.error(f'User {user_profile_id} not found in either dataset.')
            return None  # User ID not found anywhere

           

def recommendations_for_existing_user(user_profile_id, post_data, interaction_data):
    # Collaborative Filtering for Existing Users 
    algo = SVD()  
    trainset = interaction_data_surprise.build_full_trainset()
    algo.fit(trainset)

    # Get interacted posts and their features
    user_post_ids = interaction_data[interaction_data['uid'] == user_profile_id]['post_id'].tolist()
    interacted_post_features = post_data[post_data['post_id'].isin(user_post_ids)][['spi_type', 'location']]

    # Remove interacted posts
    remaining_posts = post_data[~post_data['post_id'].isin(user_post_ids)]

    def calculate_similarity(post, interacted_features):
        similarity = 0
        if post['spi_type'] in interacted_features['spi_type'].tolist():
            similarity += 3
        if post['location'] in interacted_features['location'].tolist():
            similarity += 3
        return similarity

    # Calculate similarity to interacted posts
    remaining_posts = remaining_posts.copy()
    remaining_posts['similarity_score'] = remaining_posts.apply(
        lambda x: calculate_similarity(x, interacted_post_features), axis=1)

    # Sort by similarity and get top N (Adjust this number as needed)
    recommendations = remaining_posts.sort_values(by=['similarity_score'], ascending=False)['post_id'].head(8).tolist()
    return recommendations

def recommendations_for_new_user(user_profile_id, post_data):
    # Fetch user profile data from Firestore 
    user_doc_ref = db.collection('Users').document(user_profile_id)
    user_doc = user_doc_ref.get()

    if user_doc.exists:
        user_data = user_doc.to_dict()
        user_spi_type = user_doc.get('spi_type')
        user_location = user_doc.get('location')
        
        # Ensure consistent data types for filtering
        post_data['spi_type'] = post_data['spi_type'].astype(str)
        post_data['location'] = post_data['location'].astype(str)

        # ... (Rest of your content-based filtering logic from CSV dataset)
        recommendations = post_data[
            (post_data['spi_type'] == user_spi_type) &
            (post_data['location'] == user_location)
        ]['post_id'].tolist() 

        return recommendations

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations_api():
    uid = request.json.get('uid')
    app.logger.info(f'Received request for recommendations for user: {uid}')

    try:
        recommendations = hybrid_recommender(uid, post_data, interaction_data)
        
        if recommendations is None:  
            app.logger.error(f'User not found: {uid}')
            return jsonify({'error': 'User not found'}), 404

        # Fetch details for the recommended posts
        recommended_posts = post_data[post_data['post_id'].isin(recommendations)] 
        post_details = recommended_posts[['post_id', 
                                          'seller',
                                          'spi_type', 
                                          'location', 
                                          'quantity', 
                                          'price' ,  
                                          'industry' ]].to_dict(orient='records') 
        
        app.logger.info(f'Recommendations for user {uid}: {post_details}')
        return jsonify({'recommendations': post_details})
        
    except Exception as e: 
        app.logger.error(f'Error during recommendation generation: {e}')
        return jsonify({'error': 'Internal server error'}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
    
   
