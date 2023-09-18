import lenskit
from lenskit.datasets import MovieLens
import pandas as pd
from lenskit import batch, topn
from lenskit import crossfold as xf
from lenskit import batch, crossfold
from lenskit import topn
from lenskit.algorithms import Recommender, bias, basic
from lenskit.metrics.predict import rmse, user_metric


if __name__ == "__main__":
    #import the data using pandas, LensKit didn't wroked
    data_path = '/Users/alex_fr/Code/Group3_RecommenderSystems/DataSet'
    ratings_path = f'{data_path}/rating.csv'
    movies_path = f'{data_path}/movie.csv'

    ratings = pd.read_csv(ratings_path)

    # rename the columns to match the lenskit requirements
    ratings = ratings.rename(columns={'movieId': 'item', 'userId': 'user'})
    print(ratings.head())


    # **
    # Our MODELS
    # **
    basic_recommender = Recommender.adapt(basic.Popular())  
    bias_recommender  = Recommender.adapt(bias.Basic())

    # Split the data into training and testing sets using cross val we use 5 folds 
    splits = crossfold.partition_users(ratings, 5, crossfold.SampleFrac(0.2))

    for i, (train, test) in enumerate(splits):
        train_ratings = train.copy()
        test_ratings = test.copy()

        #fit the model at your choice from above
        bias_recommender.fit(train_ratings)

        # Generate recommendations for each user in the test set
        users = test_ratings['user'].unique()
        recommendations = batch.recommend(bias_recommender, users, 10)

        # Evaluate the recommendations using RMSE 
        test_pred = pd.merge(test_ratings[['user', 'item', 'rating']], recommendations, on=['user', 'item'])
        test_rmse = rmse(test_pred['rating'], test_pred['score'])

        print(f'Split {i + 1} RMSE: {test_rmse:.2f}')