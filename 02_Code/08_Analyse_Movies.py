#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:21:40 2019

@author: Manita
"""
import glob
import ntpath
import os
# Read movie scripts

movie_path = '/Users/Manita/OneDrive - NOVAIMS/Text Mining/Poject/scripts/scraping/texts/*.txt'
file_list = glob.glob(movie_path)

# Save them in a list of lists
movie_scripts = []
movie_names = []

def get_movie_name(file):
    name = ntpath.split(file)[1]
    name = os.path.splitext(ntpath.split(name)[1])[0]
    return name

for file in file_list:
    movie_names.append(get_movie_name(file))
    with open(file, 'r') as f:
        movie_scripts.append(f.read())
    
movie_dict = dict(zip(movie_names, movie_scripts))
    
# Iterate
ofensive_levels = []
toxic_model = pickle.load(open(path + filename, 'rb'))
for key, movie in tqdm(movie_dict.items()):
    #print("Processing '{}' ".format(key))
    movie_sents = pd.DataFrame({'sentence':nltk.sent_tokenize(movie)})
    movie_sents['char_count'] = movie_sents['sentence'].apply(len)
    movie_sents['word_count'] = movie_sents['sentence'].apply(lambda x: len(x.split()))
    movie_sents['word_density'] = movie_sents['char_count'] / (movie_sents['word_count']+1)
    movie_sents['punctuation_count'] = movie_sents['sentence'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
    movie_sents['title_word_count'] = movie_sents['sentence'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    movie_sents['upper_case_word_count'] = movie_sents['sentence'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    # Clean
    cleaned_movie = [clean_text(sent) for sent in movie_sents['sentence']]
    # Vectorize
    matrix = tfidf_vec.transform(cleaned_movie)
    # Combine
    input_ = hstack((matrix, movie_sents[numeric_labels].values))
    
    #scale
    input_=scaler.transform(input_)
      
    #if using feature selection apply it:
    input_=input_[:,idx]
    
    # predict
    predictions = toxic_model.predict(input_)
    #print(predictions)
    # Return Dataframe with toxicity
    # appolo 13 is blank -> returns error
    '''
    result = pd.DataFrame({'Comment':movie_sents['sentence'],
                           'Classification':['Toxic' if prediction == 1 else 'Clean' for prediction in predictions]})
    ofensive_levels.append(result.Classification.value_counts()[1]/len(result.Classification))
    '''
    toxic_count= (predictions == 1).sum()/len(predictions)
    try:
        ofensive_levels.append(toxic_count)
    except Exception:
        pass
    

ofensive_dict = dict(zip(movie_names, ofensive_levels))
ofensive_dict.pop('Apollo_13', None) # remove Appolo 13
ofensive_dict.pop('Scary_Movie_2', None)
# remove french movies
french = ['Jeux_Interdits','Le_Diable_par_la_Queue','Ni_vu_ni_connu','Les_Tontons_Flingueurs','Un_Singe_en_Hiver']
for f in french:
    ofensive_dict.pop(f, None)
# convert to list to order
ofensive_list = [ (v,k) for k,v in ofensive_dict.items() ]
ofensive_list.sort(reverse=True)

for toxic, key in ofensive_list:
    print("'{}' Offensiveness: {:2.2%}".format(key,toxic))

df = pd.DataFrame.from_dict(ofensive_dict,orient='index', columns = ['Offensiveness'])
df.sort_values(inplace=True, by = ['Offensiveness'], ascending = False)  

# Plot all movies
'''
plt.figure(figsize = (13,8)) 
plt.bar(df.index, df.Offensiveness)  
#plt.xticks(rotation=80)
plt.tick_params(
    axis='x',
    bottom=False,
    labelbottom=False)
plt.show() 
'''

# Plot Top and Tail offensive movies
import matplotlib.ticker as mtick

def plot_top_bottom(df, top_n):
    top = df.head(top_n)
    bottom = df.tail(top_n)
    tmp =  pd.concat([top,bottom], axis=0, sort = False)
    
    movies = tmp.index
    label = np.round(tmp.Offensiveness.values,decimals=2)
    plt.figure(figsize = (35,15)) 
    plt.bar(tmp.index, tmp.Offensiveness, color = 'grey')  
    plt.xticks(rotation=80)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for a,b in zip(movies, label):
        plt.text(a, b+0.01, str(("{0:.0%}".format(b)).lstrip('0')),
                 fontweight='bold',
                 fontsize = 15,
                 horizontalalignment='center')
    plt.title("Movies' scripts toxicity comparison", fontweight = 'bold', fontsize = 25, loc = 'right')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False) 
    plt.ylim(top=1)
    plt.xticks(fontsize=18, fontweight = 'bold')
    plt.yticks(fontsize=14, fontweight = 'bold')
    plt.show()     

top_n = 10  
plot_top_bottom(df, top_n)    
    
    
    
    
    
    