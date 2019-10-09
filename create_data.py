"""
Author:   Kyle Sadler
Date:     October 7th, 2019
Purpose:  Pre-process amazon_reviews.txt and create word ambiguity training data
Language: Python3

"""


import datetime
import sys
import pprint
import os
import random


def preprocess(line):
    """ processes a line of raw text and 
    returns a list of individual words 
    removes stop words
    """

    characters_to_remove = [":", ";", ")", "\"", ",", "(", "]", "!", "?", "[",
    "{", "}", "$", "&", "'", "--", "*", " - ", "%", ".", "^", "#", "+"]	

    # stopwords from https://www.ranks.nl/stopwords
    stopwords = ['a','about','above','after','again','against','all','am','an',
    'and','any','are','arent','as','at','be','because','been','before',
    'being','below','between','both','but','by','cant','cannot','could',
    'couldnt','did','didnt','do','does','doesnt','doing','dont','down',
    'during','each','few','for','from','further','had','hadnt','has',
    'hasnt','have','havent','having','he','hed','hell','hes','her','here',
    'heres','hers','herself','him','himself','his','how','hows','i','id',
    'ill','im','ive','if','in','into','is','isnt','it','its','its','itself',
    'lets','me','more','most','mustnt','my','myself','no','nor','not','of',
    'off','on','once','only','or','other',"ought","our", "ours", 'ourselves',
    'out','over','own','same','shant','she','shed','shell','shes','should',
    'shouldnt','so','some','such','than','that','thats','the','their','theirs',
    'them','themselves','then','there','theres','these','they','theyd','theyll',
    'theyre','theyve','this','those','through','to','too','under','until','up',
    'very','was','wasnt','we','wed','well','were','weve','were','werent','what',
    'whats','when','whens','where','wheres','which','while','who','whos','whom',
    'why','whys','with','wont','would','wouldnt','you','youd','youll','youre',
    'youve','your','yours','yourself','yourselves']

    parsed = line.lower()

    for a in characters_to_remove:
        parsed = parsed.replace(a, "")

    parsed = parsed.split()

    # take out stopwords
    parsed = [x for x in parsed if(x not in stopwords)] 

    return parsed

def write_data(s, c, w, train_test):
    """	
    write the given data to the correct file
    @param s the string who's context is to be saved
    @param c the context around s
    @param w the size of the context windows (ex. window=2 -> +-2 context window)
    @param train_test type of file to write to
    @output the context is appended to the data file

    """
    with open(os.path.join("data", s+"_"+train_test+str(w)+".txt"), "a+") as f:
        f.write(" ".join(c[0][max(len(c[0])-w,0):] + c[1][:min(w, len(c[1])-1)]) + "\n")



def create_data(strings, windows):
    """	
    create a file of training data using the context 
    around each instance of either input string
    @param strings list of words to create data for
    @param windows list of sizes of context windows (ex. window=2 -> +-2 context window)
    @output file of training data string1_string2.txt

    """
    
    contexts = {} # dictionary of lists of space-separated words in context window
    for s in strings:
        contexts[s] = []

    largest_window = max(windows)

    with open('../amazon_reviews.txt') as file:
        for line in file:

            words = preprocess(line)

            for i in range(len(words)):
                if(words[i] in strings):
                    first_window = words[max(0,i-largest_window):i]
                    second_window = words[min(len(words)-1,i+1):min(len(words)-1,i+1+largest_window)]
                    
                    assert(len(first_window) <= largest_window)
                    assert(len(second_window) <= largest_window)
                    
                    contexts[words[i]].append([first_window, second_window])

    if(not os.path.isdir("./data")):
        os.mkdir("data")

    for s in strings:
        context_list = contexts[s]
        random.seed(1)
        random.shuffle(context_list)
        split = int(len(context_list)*.8)

        
        # training data
        for c in context_list[:split]:
            for w in windows:
                write_data(s, c, w, "train")


        # testing data
        for c in context_list[split:]:
            for w in windows:
                write_data(s, c, w, "test")


start = datetime.datetime.now()

strings = ["night", "seat", "kitchen", "cough", "car", "bike", "manufacturer", "big", "small", "huge", "heavy"]
windows = [5,10,20]
create_data(strings, windows)

end = datetime.datetime.now()

print("datafiles created in " +str(end - start)+" seconds\n")
