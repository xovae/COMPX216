import re
import math
import random

def tokenise(filename):
    with open(filename, 'r') as f:
        return [i for i in re.split(r'(\d|\W)', f.read().replace('_', ' ').lower()) if i and i != ' ' and i != '\n']

def build_unigram(sequence):
    # Return a unigram model.
    inner_dict = {} 
    for word in sequence:
        if word in inner_dict:
            inner_dict[word] += 1
        else:
            inner_dict[word] = 1
    outer_dict = {(): inner_dict}
    
    print(outer_dict)
 
def build_bigram(sequence):
    # Return a bigram model.
    outer_dict = {}
    for i in range(len(sequence) - 1):
        if sequence[i] in outer_dict:
            if sequence[i + 1] in outer_dict[sequence[i]]:
                outer_dict[sequence[i]] += 1
            else:
                outer_dict[sequence[i]] = 1
        else:
            outer_dict[(sequence[i])] = {}
            

def build_n_gram(sequence, n):
    # Return an n-gram model.
    outer_dict = {}
    for i in range(len(sequence) - n):
        context_list = []
        for i2 in range(n):
            context_list.append(sequence[i + i2])
        context = tuple(context_list)
        if context in outer_dict:
            if sequence[i + n] in outer_dict[context]:
                outer_dict[context] += 1
            else:
                outer_dict[context] = 1
        else:
            outer_dict[context] = {}
    

def query_n_gram(model, sequence):
    # Task 2
    # Return a prediction as a dictionary.
    # Replace the line below with your code.
    raise NotImplementedError

def blended_probabilities(preds, factor=0.8):
    blended_probs = {}
    mult = factor
    comp = 1 - factor
    for pred in preds[:-1]:
        if pred:
            weight_sum = sum(pred.values())
            for k, v in pred.items():
                if k in blended_probs:
                    blended_probs[k] += v * mult / weight_sum
                else:
                    blended_probs[k] = v * mult / weight_sum
            mult = comp * factor
            comp -= mult
    pred = preds[-1]
    mult += comp
    weight_sum = sum(pred.values())
    for k, v in pred.items():
        if k in blended_probs:
            blended_probs[k] += v * mult / weight_sum
        else:
            blended_probs[k] = v * mult / weight_sum
    weight_sum = sum(blended_probs.values())
    return {k: v / weight_sum for k, v in blended_probs.items()}

def sample(sequence, models):
    # Task 3
    # Return a token sampled from blended predictions.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_ramp_up(sequence, models):
    # Task 4.1
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

def log_likelihood_blended(sequence, models):
    # Task 4.2
    # Return a log likelihood value of the sequence based on the models.
    # Replace the line below with your code.
    raise NotImplementedError

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    '''
    model = build_unigram(sequence[:20])
    print(model)
    '''

    # Task 1.2 test code
    '''
    model = build_bigram(sequence[:20])
    print(model)
    '''

    # Task 1.3 test code
    '''
    model = build_n_gram(sequence[:20], 5)
    print(model)
    '''

    # Task 2 test code
    '''
    print(query_n_gram(model, tuple(sequence[:4])))
    '''

    # Task 3 test code
    '''
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()
    '''

    # Task 4.1 test code
    '''
    print(log_likelihood_ramp_up(sequence[:20], models))
    '''

    # Task 4.2 test code
    '''
    print(log_likelihood_blended(sequence[:20], models))
    '''
