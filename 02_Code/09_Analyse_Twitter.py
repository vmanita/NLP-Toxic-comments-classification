# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:10:06 2019

@author: rodri
"""

from twython import TwythonStreamer
import re
import pandas as pd
from nltk.tokenize import TweetTokenizer
import string

credentials = {}

credentials['APP_KEY'] = 'uIw8cpw2ne99pDO0ZLSIzbSqr'
credentials['APP_SECRET'] = 'aDOlkqkDNpRnTtT4k3yX33OY5BkNrEhbOHwumvMJDoPTbtXjvD'
credentials['OAUTH_TOKEN'] = '1947775687-GUWXzU7lQPrmKH1jXl8N02NNItikoo2wPcNPy0t'
credentials['OAUTH_TOKEN_SECRET'] = 'kDaKnUlz6DZKCyoq1pWB8CNWAcjZnW6WymOBPMbVfKI3U'

def process_tweet(tweet):  
    d = {}
    d['text'] = tweet['text']
    d['user'] = tweet['user']['screen_name']
    d['user_loc'] = tweet['user']['location']
    
    return d

keywords= ['trump','obama']
twitter_dict = {'Keyword':[], 'User':[],'Tweet':[],'Location':[]}

for key in tqdm(keywords):
    number_tweets = 5 
    keyword = key
    language = 'en'
    locations = None
    users = None
    
 
    class MyStreamer(TwythonStreamer):
        
        def on_success(self, data, i=[0]):
            
            
            tweet_data = process_tweet(data)
            
            if re.search("RT", tweet_data['text']) == None:
                
                i[0]+=1
                print("Fetching tweets...")
                twitter_dict['Keyword'].append(keyword)
                twitter_dict['User'].append(tweet_data['user'])
                twitter_dict['Tweet'].append(tweet_data['text'])
                twitter_dict['Location'].append(tweet_data['user_loc'])
                if i[0] == number_tweets:
                    self.disconnect()
                
        def on_error(self, status_code, data):
            
            print(status_code, data)
            self.disconnect()
    
    stream = MyStreamer(credentials['APP_KEY'], credentials['APP_SECRET'],
                        credentials['OAUTH_TOKEN'], credentials['OAUTH_TOKEN_SECRET'])
       
    stream.statuses.filter(track = keyword, language = language,
                               locations = locations, follow = users)
    
twitter_df = pd.DataFrame(twitter_dict)
tknzr = TweetTokenizer(strip_handles=True, reduce_len = True)
twitter_df['char_count'] = twitter_df['Tweet'].apply(len)
twitter_df['word_count'] = twitter_df['Tweet'].apply(lambda x: len(x.split()))
twitter_df['word_density'] = twitter_df['char_count'] / (twitter_df['word_count']+1)
twitter_df['punctuation_count'] = twitter_df['Tweet'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
twitter_df['title_word_count'] = twitter_df['Tweet'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
twitter_df['upper_case_word_count'] = twitter_df['Tweet'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
tweets = twitter_df['Tweet']
twitter_df['processed_tweet'] = [clean_text(tweet, tok = tknzr) for tweet in tweets]
twitter_df['processed_tweet'] = [''.join(tweet) for tweet in twitter_df['processed_tweet']]
twitter_df['processed_tweet'].replace('', np.nan, inplace=True)
twitter_df.dropna(subset=['processed_tweet'], inplace=True)


# predict
ofensive_levels = []
for word in keywords:
    tmp = twitter_df.loc[twitter_df['Keyword'].str.contains(word)]
    # Vectorize
    matrix = tfidf_vec.transform(tmp['processed_tweet'])
    # Combine
    input_ = hstack((matrix, tmp[numeric_labels].values))
    
    #scale
    input_=scaler.transform(input_)
      
    #if using feature selection apply it:
    input_=input_[:,idx]
    
    predictions = model.predict(input_)
    toxic_count= (predictions == 1).sum()/len(predictions)
    try:
        ofensive_levels.append(toxic_count)
    except Exception:
        pass


    
ofensive_dict = dict(zip(keywords, ofensive_levels))

ofensive_list = [ (v,k) for k,v in ofensive_dict.items() ]
ofensive_list.sort(reverse=True)
df = pd.DataFrame({'Keyword':list(ofensive_dict.keys()),
                  'Toxicity_Level':list(ofensive_dict.values())})

for toxic, key in ofensive_list:
    print("'{}' Offensiveness: {:2.2%}".format(key,toxic))
    
def plot_results(df):
    plt.figure(figsize = (10,8)) 
    p = sns.barplot(x = df['Keyword'], y = df['Toxicity_Level'])
    p.set(ylim=(0, 1))
    keys = df.index
    toxicities = np.round(df.Toxicity_Level.values,decimals=2)
    for key, toxicity in zip(keys, toxicities):
        p.text(key, toxicity+0.01, 
               str(("{0:.0%}".format(toxicity)).lstrip('0')),
                     fontweight='bold',
                     fontsize = 10,
                     horizontalalignment='center')
    plt.show()

plot_results(df)
# export dataframe to CSV
# twitter_df.to_csv("twitter_df.csv")

# export results to TXT
"""f = open("results.txt","w")
f.write(str(ofensive_dict))
f.close()"""

