Documentation

- How the Recommendation System Works:
  Collaborative Filtering: This method uses user behavior (such as ratings) to recommend products similar to those that the user has already liked or purchased.
  Content-Based Filtering: This method recommends products similar to those the user has liked based on the content attributes of the product (e.g., product category, price).
  Hybrid Model: This model combines both methods to provide a better, more personalized recommendation.

- Steps to Retrain the Model with New Data:
  Step 1: Add new data to the dataset.
  Step 2: Run the RecommendationSystem script again with the updated data to retrain the model.
  Step 3: Restart the Flask API to serve the newly trained model.

- Instructions for Integrating the API with the Website:
  You can make HTTP requests to the API from your website's backend (e.g., using JavaScript or Python).
  javascript
  fetch('http://127.0.0.1:5000/recommend?user_id=1&product_id=101&n_recommendations=5')
  .then(response => response.json())
  .then(data => console.log(data.recommendations));

This will fetch the recommendations and display them to the user.

- Post-Delivery Support:
  Troubleshooting: Ensure that the Flask server is running, the correct PDF data file is provided, and all dependencies (like Flask and scikit-learn) are installed.
