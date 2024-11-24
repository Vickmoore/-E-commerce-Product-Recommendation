import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class RecommendationSystem:
    def __init__(self, data_path):
        # Load the data
        self.data = pd.read_csv(data_path)
        
        # Preprocess the data
        self.preprocess_data()
        
        # Train the models
        self.train_models()
        
    def preprocess_data(self):
        # Preprocessing steps as defined in your earlier code (e.g., converting types, handling missing values)
        self.data['PurchaseDate'] = pd.to_datetime(self.data['PurchaseDate'])
        self.data.fillna(0, inplace=True)
        self.data['Gender'] = self.data['Gender'].map({'Male': 1, 'Female': 0})
        self.data['ProductCategory'] = self.data['ProductCategory'].astype('category').cat.codes
        self.data['AverageSpendPerUser'] = self.data.groupby('UserID')['PurchaseAmount'].transform('mean')
        self.data['ProductPopularity'] = self.data.groupby('ProductID')['PurchaseAmount'].transform('sum')
        self.data['DaysSinceLastPurchase'] = (self.data['PurchaseDate'].max() - self.data['PurchaseDate']).dt.days
        self.data['UserTotalPurchases'] = self.data.groupby('UserID')['PurchaseAmount'].transform('sum')
        self.data['PurchaseFrequency'] = self.data.groupby('UserID').size()

    def train_models(self):
        # Train collaborative filtering model using SVD
        user_item_matrix = self.data.pivot_table(index='UserID', columns='ProductID', values='Rating').fillna(0)
        self.svd = TruncatedSVD(n_components=10, random_state=42)
        self.matrix_svd = self.svd.fit_transform(user_item_matrix)

        # Train content-based model using cosine similarity on ProductCategory and PurchaseAmount
        product_features = self.data[['ProductID', 'ProductCategory', 'PurchaseAmount']].drop_duplicates()
        scaler = MinMaxScaler()
        product_features[['PurchaseAmount']] = scaler.fit_transform(product_features[['PurchaseAmount']])
        self.similarity_matrix = cosine_similarity(product_features[['ProductCategory', 'PurchaseAmount']], product_features[['ProductCategory', 'PurchaseAmount']])
        self.product_features = product_features

    def recommend_products(self, user_id, n_recommendations=5):
        # Collaborative filtering recommendation
        user_index = self.matrix_svd[user_id]
        recommendations = np.argsort(-user_index)[:n_recommendations]
        return recommendations

    def content_based_recommendations(self, product_id, n_recommendations=5):
        product_index = self.product_features[self.product_features['ProductID'] == product_id].index[0]
        similarity_scores = self.similarity_matrix[product_index]
        similar_indices = np.argsort(-similarity_scores)[:n_recommendations]
        return self.product_features.iloc[similar_indices][['ProductID', 'ProductCategory', 'PurchaseAmount']]
    
    def hybrid_recommendations(self, user_id, product_id, n_recommendations=5):
        collaborative_recommendations = self.recommend_products(user_id, n_recommendations)
        content_recommendations = self.content_based_recommendations(product_id, n_recommendations)
        combined_recommendations = list(set(collaborative_recommendations) | set(content_recommendations['ProductID']))
        return combined_recommendations[:n_recommendations]
