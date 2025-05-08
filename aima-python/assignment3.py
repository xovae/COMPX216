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
    return outer_dict

def build_bigram(sequence):
    # Return a bigram model.
    outer_dict = {}
    for i in range(len(sequence) - 1):
        #If the context is not already in the dictionary, add it
        if sequence[i] not in outer_dict:
            outer_dict[(sequence[i])] = {}
        
        #Check if the following token is in the dictionary
        following_token = sequence[i + 1]
        if following_token in outer_dict[sequence[i]]:
            outer_dict[sequence[i]][following_token] += 1
        else:
            outer_dict[sequence[i]][following_token] = 1

    return outer_dict

def build_n_gram(sequence, n):
    # Return an n-gram model.
    outer_dict = {}
    for i in range(len(sequence) - (n-1)):
        context_list = []
        
        #Get the current context
        for i2 in range(n-1):
            context_list.append(sequence[i + i2])
        context = tuple(context_list)
        
        #If the context is not already in the dictionary, add it
        if context not in outer_dict:
            outer_dict[context] = {}
            
        #Check if the following token is in the dictionary 
        following_token = sequence[i + (n-1)]
        
        if following_token in outer_dict[context]:
            outer_dict[context][following_token] += 1
        else:
            outer_dict[context][following_token] = 1
    return outer_dict
    

def query_n_gram(model, sequence):
    # Return a prediction as a dictionary.
    if sequence in model:
        return model[sequence]
    else:
        return None

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
    # Return a token sampled from blended predictions.
    preds = []
    for model in models:
        #Get the context length (n-1) of the model
        model_length = len(list(model.keys())[0])
        #Check if the sequence is of sufficient length (length is equal to or greater than the model's n-1 value)
        if len(sequence) >= model_length:
            pred = query_n_gram(model, tuple(sequence[-model_length:]))
            if pred is not None:
                preds.append(pred)
                
    probs = blended_probabilities(preds)
     
    return random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]

def log_likelihood_ramp_up(sequence, models):
    # Return a log likelihood value of the sequence based on the models.
    log_likelihood_sum = 0.0
    
    for i in range(len(sequence)):
        #Get the n-gram model
        if i < len(models):
            model = models[-(i+1)]
        else:
            model = models[0]
            
        #Get the context length (n-1) of the model
        model_length = len(list(model.keys())[0])

        #Get the context and token
        context = tuple(sequence[i - model_length:i])
        token = sequence[i]

        #Get the dictionary entry for the corresponding context
        context_values = query_n_gram(model, context)
        
        #Check if the context does not exist in the dictionary for the given model, or the token does not exist for the given context 
        if not context_values or token not in context_values:
            return -math.inf
        
        #Calculate the probability of the given token occurring given the current context
        prob_sum = sum(context_values.values())
        prob = context_values[token] / prob_sum
        
        #Add the probability to the log sum
        log_likelihood_sum += math.log(prob)
    
    return log_likelihood_sum
        
def log_likelihood_blended(sequence, models):
    # Return a log likelihood value of the sequence based on the models.
    blended_log_likelihood_sum = 0.0
    
    for i in range(len(sequence)):
        
        preds = []
        
        #Get the n-gram model
        if i < len(models):
            model = models[-(i+1)]
        else:
            model = models[0]
            
        #Get the context length (n-1) of the model
        model_length = len(list(model.keys())[0])

        #Get the context and token
        context = tuple(sequence[i - model_length:i])
        token = sequence[i]
        
        # Get the probability of the current token given the current context for each applicable model
        for model in models:
            #Get the context length (n-1) of the model
            model_length = len(list(model.keys())[0])
    
            #Check if the context is of sufficient length for the model
            if len(context) >= model_length:
                pred = query_n_gram(model, context[-model_length:])
                #If the context does exist in the current model
                if pred is not None:
                    preds.append(pred)

        #Get the blended probability
        probs = blended_probabilities(preds)
        
        #Check if the token does not exist in blended probabilities
        if token not in probs:
            return -math.inf
        
        #Calculate the probability of the given token occurring given the current context from each model
        prob = probs[token]
        
        #Add the probability to the log sum
        blended_log_likelihood_sum += math.log(prob)
    
    return blended_log_likelihood_sum

if __name__ == '__main__':

    sequence = tokenise('assignment3corpus.txt')

    # Task 1.1 test code
    model = build_unigram(sequence[:20])
    print(model)

    # # Task 1.2 test code
    model = build_bigram(sequence[:20])
    print(model)

    # Task 1.3 test code
    model = build_n_gram(sequence[:20], 5)
    print(model)

    # Task 2 test code
    print(query_n_gram(model, tuple(sequence[:4])))

    # Task 3 test code
    models = [build_n_gram(sequence, i) for i in range(10, 0, -1)]
    head = []
    for _ in range(100):
        tail = sample(head, models)
        print(tail, end=' ')
        head.append(tail)
    print()

    # Task 4.1 test code
    print(log_likelihood_ramp_up(sequence[:20], models))

    # Task 4.2 test code
    print(log_likelihood_blended(sequence[:20], models))