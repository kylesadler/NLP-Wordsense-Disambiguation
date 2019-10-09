"""
Author:   Kyle Sadler
Date:     October 7th, 2019
Purpose:  Create Naive Bayes Classifiers to disambiguate pseudowords created with create_data.py 
Language: Python3

"""

import os
import datetime

SCALING_FACTOR = 10e3

def train_nbc(string, window):
    """	
    compute frequency of context words of the given string with given window 
    @param string word of context word frequency to calculate
    @param window integer window size
    @output freq dictionary of word occurrences
    @output totalwords total number of context words
    @output occurrences total number of data samples

    """
    
    freq = {} # dictionary of words mapping to num occurrences
    totalwords = 0 # total number of context words for string
    occurrences = 0

    # go though training datafiles and calculate word frequencies
    with open(os.path.join("data", string+"_train"+window+".txt"), "r") as datafile:
        for line in datafile:
            occurrences += 1
            words = line.split()

            for w in words:
                totalwords += 1
                if(w in freq):
                    freq[w] += 1
                else:
                    freq[w] = 1

    return freq, totalwords, occurrences


def test_nbc(groundtruth, groundtruth_idx, window, nbc):
    """	
    test naive bayes classifier on given testing set using pre-computed frequencies
    @param groundtruth groundtruth string
    @param groundtruth groundtruth_idx string index
    @param window integer window size
    @param nbc list of dictionaries containing frequencies, occurrences, total examples
    @output predictions list of number of predictions [correct, incorrect]

    """
    predictions = [0,0]
    # go though training datafiles and calculate word frequencies
    with open(os.path.join("data", groundtruth+"_test"+window+".txt"), "r") as datafile:
        for line in datafile:

            words = line.split()
            
            # compute arg_max{ P(C_k) * P(x_1 | C_k)*P(x_2 | C_k)*...*P(x_n | C_k) }
            num_training_examples = nbc[0]["occurrences"]+nbc[1]["occurrences"]
            class_prob=[0,0]

            for i in range(2):
                class_prob[i] = nbc[i]["occurrences"]/num_training_examples
                for w in words:
                    if(w in nbc[i]["freq"]):
                        class_prob[i] *= nbc[i]["freq"][w] * SCALING_FACTOR / nbc[i]['totalwords']
                    else:
                        class_prob[i] *= .1*SCALING_FACTOR / nbc[i]['totalwords']
                
            if(class_prob[groundtruth_idx] >= class_prob[(groundtruth_idx+1)%2]):
                predictions[0] +=1
            else:
                predictions[1] +=1

    return predictions


def naive_bayes_classifier(string1, string2, window):
    """	
    train a naive bayes classifier to disambiguate string1 and string2 
    @param string1 first string to disambiguate
    @param string2 second string to disambiguate
    @window integer window size
    @output results in results dir

    """

    window = str(window)
    outfile = open(os.path.join("results", "nbc"+window+"_"+string1+"_"+string2+".txt"), "w")

    # train NB classifier
    start = datetime.datetime.now() 
    freq1, totalwords1, occurrences1 = train_nbc(string1, window)
    freq2, totalwords2, occurrences2 = train_nbc(string2, window)
    string1_data = {"freq":freq1, "totalwords":totalwords1, "occurrences":occurrences1}
    string2_data = {"freq":freq2, "totalwords":totalwords2, "occurrences":occurrences2}
    nbc = [string1_data, string2_data]
    time_elapsed = str(datetime.datetime.now()-start)
    
    print('naive bayes classifier :: '+string1+" vs. "+string2+" :: window +-"+window)
    print("trained in "+time_elapsed+" seconds")
    outfile.write("training time: "+time_elapsed+' seconds\n')
    
    # test NB classifier   
    start = datetime.datetime.now()
    predictions1 = test_nbc(string1, 0, window, nbc)
    predictions2 = test_nbc(string2, 1, window, nbc)
    time_elapsed = str(datetime.datetime.now()-start)

    print("tested in "+time_elapsed+" seconds\n")
    outfile.write("testing time: "+time_elapsed+'\n\n')
    outfile.write("confusion matrix:\n")
    confusion_matrix = ("{0: <16}\tpredicted {1: <8}\tpredicted {2: <16}\n"+
                        "actual {1: <16}\t{3: <16}\t{4: <16}\n"+
                        "actual {2: <16}\t{5: <16}\t{6: <16}\n\n").format(" ", 
                        string1, string2, str(predictions1[0]), str(predictions1[1]), 
                        str(predictions2[1]), str(predictions2[0]))
    
    print(confusion_matrix)
    outfile.write(confusion_matrix)
    

if(not os.path.isdir("results")):
    os.mkdir("results")

for window in [5,10,20]:
    start = datetime.datetime.now()
    naive_bayes_classifier('night', 'seat', window)
    naive_bayes_classifier('kitchen', 'cough', window)
    naive_bayes_classifier('car', 'bike', window)
    naive_bayes_classifier('manufacturer', 'bike', window)
    naive_bayes_classifier('big', 'small', window)
    naive_bayes_classifier('huge', 'heavy', window)
    time_elapsed = str(datetime.datetime.now()-start)
    print("\ntotal training and testing: "+time_elapsed+" seconds")
