from flask import Flask, request, jsonify
from recommendation_system import RecommendationSystem

app = Flask(__name__)

# Initialize the recommendation system
recommendation_system = RecommendationSystem('ecommerce_data.csv')

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    product_id = int(request.args.get('product_id'))
    n_recommendations = int(request.args.get('n_recommendations', 5))
    
    # Get hybrid recommendations for the given user and product
    recommendations = recommendation_system.hybrid_recommendations(user_id, product_id, n_recommendations)
    
    return jsonify({'user_id': user_id, 'product_id': product_id, 'recommendations': recommendations})

if __name__ == "__main__":
    app.run(debug=True)
