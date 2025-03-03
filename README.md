# Model-Based Collaborative Filtering: Matrix Factorization

This repository demonstrates the **Model-Based Collaborative Filtering** approach for movie recommendations using **Matrix Factorization** (SVD - Singular Value Decomposition) from the **MovieLens dataset**. The recommendation system predicts ratings for movies that a user might like based on their past ratings and the ratings of similar users.

### Steps followed in the process:

1. **Dataset Preparation**:
   - We load and merge two datasets: one with movie details and one with ratings from users.
   - We then filter movies with fewer than 1000 ratings, as we focus on more popular movies.

2. **User-Item Matrix Creation**:
   - A user-item matrix is created where each row represents a user and each column represents a movie. The cell values represent the ratings given by users to movies.

3. **Modeling**:
   - We apply **Matrix Factorization** using **SVD (Singular Value Decomposition)** to decompose the user-item matrix into three matrices. This helps in predicting the ratings for unrated movies.
   - We evaluate the model’s performance using **RMSE (Root Mean Squared Error)**.

4. **Model Tuning**:
   - We perform **Grid Search** to find the best hyperparameters, such as the number of epochs and the learning rate, that improve the model's performance.

5. **Final Model**:
   - We use the best hyperparameters obtained from Grid Search to train the final model and make movie recommendations based on the learned latent features.

## 1. **Overview**

Matrix factorization techniques like **SVD (Singular Value Decomposition)** are widely used in collaborative filtering. This model-based approach is based on creating a matrix factorization model that predicts a user’s preferences for a set of items based on patterns in the ratings matrix.

In this implementation, we use the **Surprise library** to train an SVD model and make predictions for user-movie pairs that have not yet been rated.

## 2. **Dataset**

The **MovieLens dataset** used in this repository includes movie metadata and user ratings. The data files used are:

- **movie.csv**: Contains the `movieId` and `title` for each movie.
- **rating.csv**: Contains the `userId`, `movieId`, and the `rating` given by users.

The data is merged and preprocessed to focus on popular movies (with more than 1000 ratings), and the user-item matrix is created for training the model.

