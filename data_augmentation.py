from slot_aligner import SlotAligner
from nltk.tokenize import word_tokenize
from data_processing import tokenize_mr


def insert_dot(utt):
    """
        Makes sure that the reference sentence ends with a dot, this is important for the splitting part
    """
    if utt[-1] == ",":
        utt = utt[:-1]+"."
    elif utt[-1] == "!":
        utt = utt[:-1]+"."
    elif utt[-1] == " ":
        utt = utt[:-1]+"."
    elif utt[-1] in "0123456789abcdefghijklmnopqrstuvwxyzéABCDEFGHIJKLMNOPQRSTUVWXYZ'":
        utt += "."
    
    return utt

def define_boundaries(splits):
    """
        Defines the boundaries of each of the sentences based on a split
    """

    sent_boundaries = []
    for idx in range(0, len(splits)-1):
        sent_boundaries.append([splits[idx], splits[idx+1]])

    return sent_boundaries

def find_all(string,substring):
    """
        Finds all occurences of a substring within a string
    """
    length = len(substring)
    c=0
    indexes = []
    while c < len(string):
        if string[c:c+length] == substring:
            indexes.append(c)
        c=c+1
    return indexes

def split_utterance(input_mrs, ref):
    aligner = SlotAligner()

    #Make sure that the sentence ends with a dot
    ref = insert_dot(ref)

    # Get boundaries of all sentences inside the utterance
    splits = find_all(ref, ".")

    # If only one sentence - there is nothing to split - just return the utterance
    if len(splits) == 1:
        return (input_mrs, ref)

    else: 
        splits = [0] + splits

        mrs = tokenize_mr(input_mrs)

        # We have as many boundaries as sentences within the utterance
        sent_boundaries = define_boundaries(splits)

        # Get the slots that each one of the sentences realizes
        realizations = [[] for x in xrange(0,len(sent_boundaries))]

        slot_realization_positions = None

        for mr in mrs.keys():
            pass

    # # If there is only one sentence, there is nothing to split
    # if pos_split == -1:
    #     return [(input_mrs, ref)]

    # first_sentence_slots = []
    # second_sentence_slots = []

    # augmented_data = []




