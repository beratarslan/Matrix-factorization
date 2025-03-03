#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

#############################
# Preparing The Dataset
#############################

# Load the movie and rating datasets
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

# Merge the datasets on movieId
df = movie.merge(rating, how="left", on="movieId")
df.head()

# Create a sample dataset with specific movie IDs
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

# Get the shape of the sample dataframe
sample_df.shape

# Create a user-item matrix (user_movie_df) using pivot_table
user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

# Initialize the Reader object to specify the rating scale
reader = Reader(rating_scale=(1, 5))

# Load the data from the sample dataframe
data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader)

##############################
# Modeling
##############################

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=.25)

# Initialize the SVD (Singular Value Decomposition) model
svd_model = SVD()

# Fit the model to the training set
svd_model.fit(trainset)

# Make predictions on the test set
predictions = svd_model.test(testset)

# Calculate the RMSE (Root Mean Squared Error) of the predictions
accuracy.rmse(predictions)

# Example prediction for user 1.0 and movieId 541
svd_model.predict(uid=1.0, iid=541, verbose=True)

# Example prediction for user 1.0 and movieId 356
svd_model.predict(uid=1.0, iid=356, verbose=True)

# View the sample dataframe for userId 1
sample_df[sample_df["userId"] == 1]

##############################
# Model Tuning
##############################

# Define the parameter grid for tuning the SVD model
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

# Perform Grid Search with cross-validation
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

# Fit the grid search to the data
gs.fit(data)

# Get the best RMSE score from the grid search
gs.best_score['rmse']

# Get the best parameters based on RMSE
gs.best_params['rmse']

##############################
# Final Model and Prediction
##############################

# Check the attributes of the final SVD model
dir(svd_model)
svd_model.n_epochs

# Initialize the final SVD model with the best parameters from the grid search
svd_model = SVD(**gs.best_params['rmse'])

# Build the full training set
data = data.build_full_trainset()

# Fit the final model to the full training set
svd_model.fit(data)

# Make a final prediction for user 1.0 and movieId 541
svd_model.predict(uid=1.0, iid=541, verbose=True)
