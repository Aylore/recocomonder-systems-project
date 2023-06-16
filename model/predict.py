import pandas as pd
import pickle
# import argparse
import tensorflow as tf

# parser = argparse.ArgumentParser(description='get recommendations')

# parser.add_argument('--userId', type=int, 
# 					help='the user id you want to recommend for', required=True)
# args = parser.parse_args()
# # userId =args.userId
# # print(args)
def top_10_recommendations(userId):

    # Loading all the datasets needed:
    movies_df_mod = pd.read_csv('model/data/movies_mod.csv')
    ratings_df_removed = pd.read_csv('model/data/ratings_df_last_liked_movie_removed.csv')

    
    # Gathering all the movies in the dataset:
    not_watched = list(movies_df_mod.movieId)
    
    # Selecting all movies that have not been seen by the user:
    ratings_df_removed = ratings_df_removed[ratings_df_removed.userId == userId]
    
    if len(ratings_df_removed) ==  0:  ## First check for valid users/users with enough information 
        return print('User {} does not have enough information. 1'.format(userId))
    
    ratings_df_removed = ratings_df_removed.merge(movies_df_mod, how= 'left', on= 'movieId').dropna()
    
    if len(ratings_df_removed) ==  0:  ## Second check
        return print('User {} does not have enough information. 2'.format(userId))
    
    watched = list(ratings_df_removed.movieId)
    del ratings_df_removed  ## I find that not all variables are actually cleared in definitions; this is to ensure it removed from RAM
    
    # Finding the movies the user has not watched:
    for movie in watched:
        if movie in not_watched:
            not_watched.remove(movie)
            
    # Loading in users' like and disliked genres:
    total_user_like_df = pd.read_csv('model/data/total_user_like_df.csv')
    total_user_dislike_df = pd.read_csv('model/data/total_user_dislike_df.csv') 

    
    # Selecting from total_user_like_df and total_user_dislike_df to isolate only the userId input:
    total_user_like_df = total_user_like_df[total_user_like_df.userId == userId]
    
    if len(total_user_like_df) ==  0:  ## Third check
        return print('User {} does not have enough information. 3'.format(userId))
    
    total_user_dislike_df = total_user_dislike_df[total_user_dislike_df.userId == userId]
    if len(total_user_dislike_df) ==  0:  ## Fourth check
        return print('User {} does not have enough information. 4'.format(userId))
            
    # Changing the columns names to differentiate between the columns of total_user_like_df and total_user_dislike_df:

    like_columns = list(total_user_like_df.columns)
    like_columns_modified = []

    for column in like_columns:
        if column == 'userId':
            like_columns_modified.append('userId')
        else:
            modify_column = 'user_like_' + column
            like_columns_modified.append(modify_column)

    total_user_like_df.columns = like_columns_modified
    
    dislike_columns = list(total_user_dislike_df.columns)
    dislike_columns_modified = []

    for column in dislike_columns:
        if column == 'userId':
            dislike_columns_modified.append('userId')
        else:
            modify_column = 'user_dislike_' + column
            dislike_columns_modified.append(modify_column)

    total_user_dislike_df.columns = dislike_columns_modified

    # Loading in tags:
    movie_tags_df = pd.read_csv('model/data/final/movie_tags_df.csv')
    like_dislike_tags = (pd.read_csv('model/data/final/like_dislike_tags.csv')).astype('int64')
    
    # Selecting the movies that have not been seen from movie_tags_df and merging movies_df_mod and movie_tags_df:
    template_df = pd.DataFrame({'movieId': not_watched}, index= list(range(len(not_watched)))) ## Creating a template DF for merging
    template_df = template_df.merge(movies_df_mod, how= 'left', on= 'movieId').dropna()
    template_df = template_df.merge(movie_tags_df, how= 'left', on= 'movieId').dropna()
    del movie_tags_df
    
    # Selecting the user's tags:
    like_dislike_tags = like_dislike_tags[like_dislike_tags.userId == userId]
    if len(like_dislike_tags) ==  0:  ## Fifth check
        return print('User {} does not have enough information. 5'.format(userId))

    # Adding a userId column to the template DF so that merging is possible with total_user_like_df, total_user_dislike_df, and like_dislike_tags
    template_df['userId'] = userId
    template_df = template_df.merge(total_user_like_df, how= 'left', on= 'userId').dropna()
    del total_user_like_df
    template_df = template_df.merge(total_user_dislike_df, how= 'left', on= 'userId').dropna()
    del total_user_dislike_df
    template_df = template_df.merge(like_dislike_tags, how= 'left', on= 'userId').dropna()
    del like_dislike_tags
    
    like_columns_modified.remove('userId')
    dislike_columns_modified.remove('userId')
    like_columns.remove('userId')

    # Generating the columns for the tag inputs for random forest:
    rf_columns = []
    for x in range(20): 
        rf_columns.append('LIKE_' + str(x))
        rf_columns.append('DISLIKE_' + str(x))
    for x in range(5):
        rf_columns.append('TAG_' + str(x))
        
    # Selecting out the inputs from the template DF by column names:
    genres_like_input = template_df.loc[:, like_columns_modified]
    genres_dislike_input = template_df.loc[:, dislike_columns_modified]
    genres_movie_input = template_df.loc[:, like_columns]
    
    tags_input = template_df.loc[:, rf_columns]
    
    # Saving the movieId list:
    movieId_list = list(template_df.movieId)
    
    del template_df
    
    # Loading in all models
    genres_model = tf.keras.models.load_model('model/data/models/genres_model.h5', compile=True)
    tags_model = pickle.load(open('model/data/tags_model.sav', 'rb'))
    combine_model = pickle.load(open('model/data/combine_model.sav', 'rb'))
    
    # Predicting with the genres model and tags model:
    genres_model_predictions = (genres_model.predict(x= [genres_like_input, genres_dislike_input, genres_movie_input])) * 5 ## Rescaling up; predicts a scaled and bound (sigmoid, 0-1) values
    tags_model_predictions = tags_model.predict(tags_input)
    
    # Extracting and changing the Keras predictions into a 1-D format (list):
    genres_model_predictions_list = []

    for prediction in genres_model_predictions:
        genres_model_predictions_list.append(prediction[0])
    
    # Using the predictions from the two models as the inputs for the combine_model:
    combine_input = pd.DataFrame({'genres_model': genres_model_predictions_list, 
                                  'tag_model': tags_model_predictions}, 
                                 index= list(range(len(genres_model_predictions))))
    
    combine_model_predictions = combine_model.predict(combine_input)
    
    # Rounding the predictions that are out of bounds:
    combine_model_predictions_rounded = []

    for prediction in combine_model_predictions:
        rounded = prediction
        if rounded > 5:
            rounded = 5
        elif rounded < 0.5:
            rounded = 0.5

        combine_model_predictions_rounded.append(rounded)
    
    # Adding all predictions into one DF:
    predictions_df = pd.DataFrame({'movieId': movieId_list,
                                   'genres_model': genres_model_predictions_list, 
                                  'tag_model': tags_model_predictions,
                                  'combine_predictions': combine_model_predictions_rounded}, 
                                 index= list(range(len(movieId_list))))
    
    # Sorting by combine_model_predictions_rounded and selecting the first 10 highest predicted ratings:
    best_movies_df = predictions_df.sort_values(by= ['combine_predictions'], ascending=False).iloc[:10, :]
    
    # Finding adding the movie titles and information to highest 10:
    best_movies_df = best_movies_df.merge(movies_df_mod, how= 'left', on= 'movieId').dropna()
    del movies_df_mod
    
    return predictions_df, best_movies_df
    
    

top_10_recommendations(5)