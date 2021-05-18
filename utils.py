import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import os, sys
from tqdm import tqdm 

# sklearn libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Multiply
from keras.optimizers import Adam
from keras.regularizers import l2

pd.options.mode.chained_assignment = None  # default='warn'



def get_playlists_df(number_of_files):
    
    start = 0
    end = 1000
    list_of_df = []
    
    for i in range(number_of_files):
        
        path = 'drive/MyDrive/data/mpd.slice.' + str(start) + "-" + str(end-1) + '.json'
        json_file = json.load(open(path, 'r'))
        
        playlists_df = pd.DataFrame.from_dict(json_file['playlists'], orient='columns')
        list_of_df.append(playlists_df)
        
        start = end
        end = end + 1000
    
    concat_playlists_df = pd.concat(list_of_df).reset_index(drop=True)
        
        
    return concat_playlists_df


def get_all_songs_df(playlists_df):
    all_songs_array = []
    for index, row in playlists_df.iterrows():
        for track in row['tracks']:
            all_songs_array.append([track['track_uri'], 
                                      track['track_name'], 
                                      track['artist_uri'], 
                                      track['artist_name'],  
                                      track['album_uri'], 
                                      track['album_name'],
                                      row['pid']])

    all_songs_df = pd.DataFrame(all_songs_array, columns=['track_uri', 
                                                          'track_name', 
                                                          'artist_uri', 
                                                          'artist_name', 
                                                          'album_uri', 
                                                          'album_name', 
                                                          'pid'])
    return all_songs_df




def get_negative_samples(training_df, all_unique_songs, number_of_neg_sample=20, inp=["artist_uri"]):

  """
  number_of_neg_sample : number of negative samples will be added for each playlist,
                         or assign 'same' to add negative samples as much as number of positive samples for each playlist.
  """

  all_neg_samples_list = []
  all_pids = training_df['pid'].unique()

  inp_dict = {i: training_df[i].unique() for i in inp}

  for playlist_id in tqdm(all_pids, position=0, leave=True):

    # tracks in corresponding playlist
    tracks_in_playlist = training_df[training_df.pid == playlist_id].track_uri.values
    inp_in_playlist = {i: training_df[training_df.pid == playlist_id][i].values for i in inp}

    # take the difference between all unique songs and songs in the playlist to get possible neg samples 
    possible_neg_samples =  np.array(list( set(all_unique_songs) - set(tracks_in_playlist) ))

    possible_neg_samples_inp = {i: np.array(list( set(inp_dict[i]) - set(inp_in_playlist[i]) )) for i in inp}

    # get indices of n neg random samples
    random_neg_sample_indices = np.random.randint(0, len(possible_neg_samples), size=(number_of_neg_sample,))

    random_neg_sample_indices_inp = {i: np.random.randint(0, len(possible_neg_samples_inp[i]), size=(number_of_neg_sample,)) for i in inp}

    # get n neg random samples
    neg_samples_for_a_playlist = possible_neg_samples[random_neg_sample_indices]

    neg_samples_for_a_playlist_inp = {i: possible_neg_samples_inp[i][random_neg_sample_indices_inp[i]] for i in inp}

    for x, a_track in enumerate(neg_samples_for_a_playlist):
      all_neg_samples_list.append([playlist_id, a_track] + [neg_samples_for_a_playlist_inp[i][x] for i in inp])

  all_neg_samples_df = pd.DataFrame(data = all_neg_samples_list, columns=['pid', 'track_uri'] + inp)
  all_neg_samples_df['interaction'] = 0

  return all_neg_samples_df




def get_test_samples(training_df, number_of_test_sample):

  """
  number_of_neg_sample : number of test samples will be selected for each playlist

  """

  all_test_samples_indices = []
  all_pids = training_df['pid'].unique()

  for playlist_id in tqdm(all_pids, position=0, leave=True):

    # indices of tracks in corresponding playlist
    track_indices = training_df[(training_df.pid == playlist_id) & (training_df.interaction == 1)].index.values

    # randomly select n track
    random_indices = np.random.randint(0, len(track_indices), size=(number_of_test_sample,))
    test_samples_ind_for_a_playlist = track_indices[random_indices]

    for test_sample_ind in test_samples_ind_for_a_playlist:
      all_test_samples_indices.append(test_sample_ind)

  return all_test_samples_indices


def get_negative_samples_test(training_df, all_unique_songs, number_of_neg_sample=99, inp=["artist_uri"]):

  """
  number_of_neg_sample : number of negative samples will be added for each playlist,
                        or assign 'same' to add negative samples as much as number of positive samples for each playlist.
  """

  all_neg_samples_list = []
  all_pids = training_df['playlist_id'].unique()

  inp_dict = {i: training_df[i].unique() for i in inp}

  for p_id in tqdm(all_pids, position=0, leave=True):

    # tracks in corresponding playlist
    tracks_in_playlist = training_df[training_df.playlist_id == p_id].track_id.values
    inp_in_playlist = {i: training_df[training_df.pid == p_id][i].values for i in inp}

    # take the difference between all unique songs and songs in the playlist to get possible neg samples 
    possible_neg_samples =  np.array(list( set(all_unique_songs) - set(tracks_in_playlist) ))

    possible_neg_samples_inp = {i: np.array(list( set(inp_dict[i]) - set(inp_in_playlist[i]) )) for i in inp}

    # get indices of n neg random samples
    random_neg_sample_indices = np.random.randint(0, len(possible_neg_samples), size=(number_of_neg_sample,))

    random_neg_sample_indices_inp = {i: np.random.randint(0, len(possible_neg_samples_inp[i]), size=(number_of_neg_sample,)) for i in inp}
    # get n neg random samples
    neg_samples_for_a_playlist = possible_neg_samples[random_neg_sample_indices]

    neg_samples_for_a_playlist_inp = {i: possible_neg_samples_inp[i][random_neg_sample_indices_inp[i]] for i in inp}

    for x, a_track in enumerate(neg_samples_for_a_playlist):
      all_neg_samples_list.append([p_id, a_track] + [neg_samples_for_a_playlist_inp[i][x] for i in inp])

  all_neg_samples_df = pd.DataFrame(data = all_neg_samples_list, columns=['playlist_id', 'track_id'] + inp)
  all_neg_samples_df['interaction'] = 0
  return all_neg_samples_df


def print_top_k_acc(click_ranks, k):
  acc = np.sum(click_ranks < k) / len(click_ranks)
  acc = round(acc, 4)
  print("Top-{} accuracy: {}".format(k, acc))
