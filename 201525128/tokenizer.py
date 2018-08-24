import operator
import sys
from collections import OrderedDict
from operator import itemgetter
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

print(len(sys.argv))
if len(sys.argv) == 2:
    file = sys.argv[1]

else:
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

apostrophe = ["he's", "i'm", "she's", "there's", "they're", "i'll", "he'll", "she'll", "they'll", "we'll", "you'll",
              "you're", "we're", "it's", "can't", "don't", "wouldn't", "won't", "shouldn't", "mustn't", "couldn't",
              "i've", "they've", "we've"]


def tokenize(file):
    final = []
    lCount = 0
    with open(file, 'r') as f:
        for line in f:
            to_insert = []
            lCount += 1
            if lCount > 150:
                break
            print(lCount)
            split = line.split()
            i = 0
            while i < len(split):
                if i + 1 != len(split) and (split[i] + "'" + split[i + 1]) in apostrophe:
                    to_insert.append(split[i] + "'" + split[i + 1])
                    i += 2
                elif split[i] == 'http' or split[i] == 'https':

                    ins = split[i] + split[i + 1]
                    to_insert.append(ins)
                    i += 2
                else:
                    to_insert.append(split[i])
                    i += 1
            final = final + to_insert

    # print(len(final))
    return final


def make_freq_dict(final):
    freq_dict = {}

    for i in range(len(final)):
        key = final[i]
        if key not in freq_dict.keys():
            freq_dict[key] = 1
        else:
            freq_dict[key] += 1

    sorted_x = sorted(freq_dict.items(), key=operator.itemgetter(1))

    zipf_dict = {}
    for i in range(len(sorted_x)):
        word = list(sorted_x[i])
        word[1] = float(int(word[1]))
        zipf_dict[word[0]] = word[1]

    # print(zipf_dict)
    # print('\n')

    d = OrderedDict(sorted(zipf_dict.items(), key=itemgetter(1)))
    d = list(d.items())
    # print(len(d))
    # print('\n')
    return d


def make_graph(d, level):
    word = []
    value = []
    for i in range(len(d)):
        pair = d[i]
        word.append(pair[0])
        value.append(pair[1])

    # print(word)
    # print(value)

    word = word[::-1]
    value = value[::-1]
    # for pr in range(len(value)):
    #     if value[pr] is not 0:
    #         value[pr] = math.log(value[pr])
    x = [i for i in range(len(word))]  # [math.log(i + 0.5) for i in range(len(word))]
    x = np.array(x)
    if level == 1:
        plt.plot(x, value, label='Unigram MLE')
    elif level == 2:
        plt.plot(x, value, label='Bigram MLE')
    elif level == 3:
        plt.plot(x, value, label='Trigram MLE')
    # plt.xticks(x, word)
    plt.legend(loc='upper right')
    plt.show()


def three_in_one():
    word = []
    value = []
    for i in range(len(unifreq)):
        pair = unifreq[i]
        word.append(pair[0])
        value.append(pair[1])

    # print(word)
    # print(value)

    word = word[::-1]
    value = value[::-1]
    # for pr in range(len(value)):
    #     if value[pr] is not 0:
    #         value[pr] = math.log(value[pr])
    x = [i for i in range(len(word))]  # [math.log(i + 0.5) for i in range(len(word))]
    x = np.array(x)
    plt.plot(x, value, label='Unigram')

    word = []
    value = []
    for i in range(len(bifreq)):
        pair = bifreq[i]
        word.append(pair[0])
        value.append(pair[1])

    # print(word)
    # print(value)

    word = word[::-1]
    value = value[::-1]
    # for pr in range(len(value)):
    #     if value[pr] is not 0:
    #         value[pr] = math.log(value[pr])
    x = [i for i in range(len(word))]  # [math.log(i + 0.5) for i in range(len(word))]
    x = np.array(x)
    plt.plot(x, value, label='Bigram')

    word = []
    value = []
    for i in range(len(trifreq)):
        pair = trifreq[i]
        word.append(pair[0])
        value.append(pair[1])

    # print(word)
    # print(value)

    word = word[::-1]
    value = value[::-1]
    # for pr in range(len(value)):
    #     if value[pr] is not 0:
    #         value[pr] = math.log(value[pr])
    x = [i for i in range(len(word))]  # [math.log(i + 0.5) for i in range(len(word))]
    x = np.array(x)
    plt.plot(x, value, label='Trigram')
    plt.legend(loc='upper right')
    plt.show()


def make_bigrams(final):
    bigrams = []
    for i in range(len(final) - 1):
        bigram = str(final[i] + ' ' + final[i + 1])
        bigrams.append(bigram)

    return bigrams


def make_trigrams(final):
    trigrams = []
    for i in range(len(final) - 2):
        trigram = str(final[i] + ' ' + final[i + 1] + ' ' + final[i + 2])
        trigrams.append(trigram)

    return trigrams


def make_bigram_freq(bigrams):
    freq_dict = {}
    for i in bigrams:
        if i not in freq_dict.keys():
            freq_dict[i] = 1
        else:
            freq_dict[i] += 1
    sorted_x = sorted(freq_dict.items(), key=operator.itemgetter(1))

    zipf_dict = {}
    for i in range(len(sorted_x)):
        word = list(sorted_x[i])
        zipf_dict[word[0]] = word[1]

    d = OrderedDict(sorted(zipf_dict.items(), key=itemgetter(1)))
    d = list(d.items())
    return d


def make_trigram_freq(trigrams):
    freq_dict = {}
    for i in trigrams:
        if i not in freq_dict.keys():
            freq_dict[i] = 1
        else:
            freq_dict[i] += 1
    sorted_x = sorted(freq_dict.items(), key=operator.itemgetter(1))

    zipf_dict = {}
    for i in range(len(sorted_x)):
        word = list(sorted_x[i])
        zipf_dict[word[0]] = word[1]

    d = OrderedDict(sorted(zipf_dict.items(), key=itemgetter(1)))
    d = list(d.items())
    return d


def unigram_probabilities(unigrams, total_tokens):
    unigram_probs = {}
    # print(unigrams)
    for i in range(len(unigrams)):
        pair = unigrams[i]
        # print(pair)
        word = pair[0]
        val = pair[1]
        prob = float(val) / total_tokens
        unigram_probs[word] = prob

    return unigram_probs


def bigram_probabilities(bigrams, total_tokens, unifreq):
    bigram_probs = {}
    # print(unigrams)
    for i in range(len(bigrams)):
        pair = bigrams[i]
        # print(pair)
        word = pair[0]
        second = word.split(' ')[1]
        val = pair[1]
        for j in range(len(unifreq)):
            term = unifreq[j][0]
            if term == second:
                total_tokens = float(unifreq[j][1])
        prob = float(val) / total_tokens
        bigram_probs[word] = prob

    return bigram_probs


def trigram_probabilities(trigrams, total_tokens, bifreq):
    trigram_probs = {}
    # print(unigrams)
    for i in range(len(trigrams)):
        pair = trigrams[i]
        # print(pair)
        word = pair[0]
        second = ' '.join(word.split(' ')[:2])
        val = pair[1]
        for j in range(len(bifreq)):
            term = bifreq[j][0]
            if term == second:
                total_tokens = float(bifreq[j][1])
        prob = float(val) / total_tokens
        trigram_probs[word] = prob

    return trigram_probs


def laplace_smoothing(frequency, N, V, type):
    laplace = {}
    if type == 1:
        for i in range(len(frequency)):
            pair = frequency[i]
            word = pair[0]
            val = pair[1]
            laplace[word] = float(val + 1) / (N + V)
    elif type == 2:
        for i in range(len(frequency)):
            pair = frequency[i]
            # print(pair)
            word = pair[0]
            val = pair[1]
            first = word.split()[0]
            firstval = 0
            for j in range(len(N)):
                wordtemp = N[j][0]
                if wordtemp == first:
                    firstval = float(N[j][1])

            laplace[word] = float(val + 1) / (firstval + V)
    elif type == 3:
        for i in range(len(frequency)):
            pair = frequency[i]
            word = pair[0]
            val = pair[1]
            first = ' '.join(word.split()[:2])
            firstval = 0
            for j in range(len(N)):
                wordtemp = N[j][0]
                if wordtemp == first:
                    firstval = float(N[j][1])
            laplace[word] = float(val + 1) / (firstval + V)
    return laplace


def laplace_graph(laplace, level):
    vals = []
    # print(laplace)
    for i in laplace.keys():
        laplace[i] *= len(final)
        vals.append(laplace[i])
    x = [i for i in range(len(laplace))]
    sorted_x = sorted(laplace.items(), key=operator.itemgetter(1))
    y = []
    for i in sorted_x:
        y.append(i[1])
    y = y[::-1]
    if level == 1:
        plt.plot(x, y, label='Laplace-Unigram')
    elif level == 2:
        plt.plot(x, y, label='Laplace-Bigram')
    elif level == 3:
        plt.plot(x, y, label='Laplace-Trigram')
    plt.legend(loc='upper right')
    plt.show()


def prob_mle(word, level):
    # print('In prob_mle with:', word, level)
    if level == 3:
        for i in trigram_probs.keys():
            if i == word:
                return trigram_probs[i]
    elif level == 2:
        for i in bigram_probs.keys():
            if i == word:
                return bigram_probs[i]
    elif level == 1:
        for i in unigram_probs.keys():
            if i == word:
                return unigram_probs[i]


def lambd(term, level):
    # print('In lambda with:', term, level)
    total = 0
    deno = 0
    if level is 3:
        for i in range(len(trifreq)):
            pair = trifreq[i]
            word = pair[0].split()[2]
            deno += pair[1]
            if word == term:
                total += 1
    elif level is 2:
        for i in range(len(bifreq)):
            pair = bifreq[i]
            deno += pair[1]
            word = pair[0].split()[1]
            if word == term:
                total += 1
    total = total / float(total + deno)
    return 1 - total


def witten_bell(trifreq, level):
    prob = 0
    # print('In witten-bell with:', trifreq, level)
    if level is 3:
        prob = lambd(trifreq.split()[2], level) * prob_mle(trifreq, level) + \
               (1 - lambd(trifreq.split()[2], level)) * witten_bell(' '.join(trifreq.split()[1:]), level - 1)
    if level is 2:
        prob = lambd(trifreq.split()[1], level) * prob_mle(trifreq, level) + \
               (1 - lambd(trifreq.split()[1], level)) * prob_mle(trifreq.split()[1], level - 1)
    return prob


delta = 0.75


def get_follower_count(word1, bifreq):
    count = 0
    for i in range(len(bifreq)):
        pair = bifreq[i]
        first = pair[0].split(' ')[0]
        if first == word1:
            count += 1

    return count


def get_followee_count(word2, bifreq):
    count = 0
    for i in range(len(bifreq)):
        pair = bifreq[i]
        second = pair[0].split(' ')[1]
        if second == word2:
            count += 1
    return count


def get_unique_bigrams(bifreq):
    return len(bifreq)


def kn_low(word2, bifreq):
    return float(get_followee_count(word2, bifreq)) / get_unique_bigrams(bifreq)


def kneser_ney(word1, word2, unifreqtemp, bifreqtemp, level):
    prob = 0
    if level == 1:

        pass

    elif level == 3:
        prob1 = 0
        prob2 = 0
        prob3 = 0
        """ Prob1 """
        val = 0
        for i in range(len(trifreq)):
            pair = trifreq[i]
            word = pair[0]
            if word == word1:
                val = pair[1]
                break
        prob1 += max(val - delta, 0)

        bi = ' '.join(word1.split(' ')[:2])
        for i in range(len(bifreq)):
            term = bifreq[i][0]
            if term == bi:
                prob1 /= float(bifreq[i][1])
                break

        """ Prob2 """
        w1 = word1.split(' ')[0]
        w2 = word1.split(' ')[1]
        w3 = word1.split(' ')[2]

        nume = 0
        for i in range(len(trifreq)):
            pair = trifreq[i]
            first = pair[0].split(' ')[0]
            second = pair[0].split(' ')[1]
            if first == w1 and second == w2:
                nume += 1

        deno = 0
        term = w1 + ' ' + w2
        for i in range(len(bifreq)):
            word = bifreq[i][0]
            if word == term:
                deno += float(bifreq[i][1])
                break
        nume = nume / float(deno)
        nume *= delta
        prob2 = nume

        """ Prob3 """
        nume = 0
        deno = 0
        for i in range(len(bifreq)):
            pair = bifreq[i]
            if pair[0].split(' ')[0] == w2:
                nume += 1
        for i in range(len(trifreq)):
            pair = trifreq[i]
            if pair[0].split(' ')[1] == w2:
                deno += 1
        nume /= float(deno)
        prob3 = nume

        nume = 0
        deno = 0
        for i in range(len(bifreq)):
            pair = bifreq[i]
            if pair[0].split(' ')[1] == w3:
                nume += 1
        deno = len(bifreq)
        nume /= float(deno)
        prob3 *= nume
        prob3 *= delta

        nume = 0
        deno = 0

        for i in range(len(trifreq)):
            pair = trifreq[i]
            if pair[0].split(' ')[1] == w2:
                deno += 1
        for i in range(len(trifreq)):
            pair = trifreq[i]
            # print(pair)
            if pair[0].split(' ')[1] == w2 and pair[0].split(' ')[2] == w3:
                nume += 1
        nume = max(nume - delta, 0)
        nume /= float(deno)

        prob3 += nume

        prob2 *= prob3

        prob1 += prob2

        prob = prob1

    elif level == 2:
        unicount = 0
        for i in range(len(bifreqtemp)):
            pair = bifreqtemp[i]
            first = pair[0].split(' ')[0]
            second = pair[0].split(' ')[1]
            # print(first,second)
            if first == word1 and second == word2:
                # print('mil gaya')
                for j in range(len(unifreqtemp)):
                    unipair = unifreqtemp[j]
                    unifirst = unipair[0]
                    # print(unipair)
                    if unifirst == word1:
                        # print('uni mil gaya',unifirst)
                        unicount = float(unipair[1])
                        # print('unicount=',unicount)
                        prob += max(0, float(pair[1]) - delta) / float(unipair[1])

        prob += (delta / unicount) * get_follower_count(word1, bifreqtemp) * kn_low(word2, bifreqtemp)

    return prob


def nine_in_one(num):
    final = tokenize(sys.argv[num])  # Final == list of all tokens
    unifreq = make_freq_dict(final)  # Freq_dict of all unigrams (List of tuples)
    unigrams = deepcopy(final)  # List of all unigrams
    bigrams = make_bigrams(final)  # List of all bigrams
    bifreq = make_bigram_freq(bigrams)  # Freq_dict of all bigrams (List of tuples)
    trigrams = make_trigrams(final)  # List of all trigrams
    trifreq = make_trigram_freq(trigrams)  # Freq_dict of all trigrams (List of tuples)
    unigram_probs = unigram_probabilities(unifreq, len(unigrams))  # MLE Probabilities of unigrams
    bigram_probs = bigram_probabilities(bifreq, len(bigrams), unifreq)  # MLE Probabilities of bigrams
    trigram_probs = trigram_probabilities(trifreq, len(trigrams), bifreq)  # MLE Probabilities of trigrams
    x = [i for i in range(len(unigram_probs))]
    y = []
    for i in unigram_probs:
        y.append(unigram_probs[i])
    # print(y)
    y = sorted(y, reverse=True)
    plt.plot(x, y, label='File' + str(num) + 'Uni')

    x = [i for i in range(len(bigram_probs))]
    y = []
    for i in bigram_probs:
        y.append(bigram_probs[i])
    # print(y)
    y = sorted(y, reverse=True)
    plt.plot(x, y, label='File' + str(num) + 'Bi')

    x = [i for i in range(len(trigram_probs))]
    y = []
    for i in trigram_probs:
        y.append(trigram_probs[i])
    # print(y)
    y = sorted(y, reverse=True)
    plt.plot(x, y, label='File' + str(num) + 'Tri')

    plt.legend(loc='upper right')
    # plt.show()


def unigram_for_all():
    laplace = laplace_smoothing(unifreq, len(final), len(unifreq), 1)
    vals = []
    # print(laplace)
    for i in laplace.keys():
        laplace[i] *= 1  # len(final)
        vals.append(laplace[i])
    x = [i for i in range(len(laplace))]
    sorted_x = sorted(laplace.items(), key=operator.itemgetter(1))
    y = []
    for i in sorted_x:
        y.append(i[1])
    y = y[::-1]
    plt.plot(x, y, label='Laplace')

    kn_prob = {}
    kn_vals = []
    for i in range(len(unifreq)):
        pair = unifreq[i]
        term = pair[0]
        val = pair[1]
        kn_prob[term] = (val - delta) / float(len(unigrams))
        kn_vals.append(kn_prob[term])
    x = [i for i in range(len(kn_prob))]
    kn_vals = sorted(kn_vals, reverse=True)
    # print(kn_vals)
    plt.plot(x, kn_vals, label='Kneser-Ney')

    wt_val = []
    for i in unigram_probs:
        wt_val.append(unigram_probs[i])
    x = [i for i in range(len(unigram_probs))]
    wt_val = sorted(wt_val, reverse=True)
    plt.plot(x, wt_val, label='Witten-Bell')

    plt.legend(loc='upper right')
    plt.title('Unigrams smoothing comparison for all three smoothing techniques')
    plt.show()


def get_next_word(prev, bi_kn_prob):
    max = -1
    ret = ''
    for i in bi_kn_prob:
        sp = i.split(' ')
        if sp[0] == prev:
            if bi_kn_prob[i] > max:
                max = bi_kn_prob[i]
                ret = sp[1]
    del bi_kn_prob[prev + ' ' + ret]
    return ret, bi_kn_prob


def laplace_multiple_V(laplace, V):

    vals = []
    # print(laplace)
    for i in laplace.keys():
        laplace[i] *= len(final)
        vals.append(laplace[i])
    x = [i for i in range(len(laplace))]
    sorted_x = sorted(laplace.items(), key=operator.itemgetter(1))
    y = []
    for i in sorted_x:
        y.append(i[1])
    y = y[::-1]
    plt.plot(x, y, label='Laplace' + ':' + str(V))


final = tokenize(sys.argv[1])  # Final == list of all tokens

unigrams = deepcopy(final)  # List of all unigrams
unifreq = make_freq_dict(final)  # Freq_dict of all unigrams (List of tuples)
# make_graph(unifreq, 1)
bigrams = make_bigrams(final)  # List of all bigrams
bifreq = make_bigram_freq(bigrams)  # Freq_dict of all bigrams (List of tuples)
# make_graph(bifreq, 2)
trigrams = make_trigrams(final)  # List of all trigrams
trifreq = make_trigram_freq(trigrams)  # Freq_dict of all trigrams (List of tuples)
# make_graph(trifreq, 3)

unigram_probs = unigram_probabilities(unifreq, len(unigrams))  # MLE Probabilities of unigrams
x = [i for i in range(len(unigram_probs))]
y = []
for i in unigram_probs:
    y.append(unigram_probs[i])
y = sorted(y, reverse=True)
plt.plot(x, y, label='Unigram Prob')
plt.legend(loc='upper right')
plt.show()
# print(unigram_probs)
bigram_probs = bigram_probabilities(bifreq, len(bigrams), unifreq)  # MLE Probabilities of bigrams
x = [i for i in range(len(bigram_probs))]
y = []
for i in bigram_probs:
    y.append(bigram_probs[i])
y = sorted(y, reverse=True)
plt.plot(x, y, label='Bigram Prob')
plt.legend(loc='upper right')
plt.show()

# print(bigram_probs)
trigram_probs = trigram_probabilities(trifreq, len(trigrams), bifreq)  # MLE Probabilities of trigrams
x = [i for i in range(len(trigram_probs))]
y = []
for i in trigram_probs:
    y.append(trigram_probs[i])
y = sorted(y, reverse=True)
plt.plot(x, y, label='Trigram Prob')
plt.legend(loc='upper right')
plt.show()

# print(trigram_probs)

"""LAPLACE SMOOTHING"""

# uni_laplace = laplace_smoothing(unifreq, len(final), len(unifreq), 1)  # Dictionary={N-gram:Prob}
# laplace_graph(uni_laplace, 1)
# bi_laplace = laplace_smoothing(bifreq, unifreq, len(unifreq), 2)
# laplace_graph(bi_laplace, 2)
# tri_laplace = laplace_smoothing(trifreq, bifreq, len(bifreq), 3)
# laplace_graph(tri_laplace, 3)

"""WITTEN BELL"""

# wb_vals = []
# for i in range(len(trifreq)):
#     # print(i)
#     pair = trifreq[i][0]
#     val = witten_bell(pair, 3)
#     wb_vals.append(val)
#     # print(pair, val)
#
# wb_vals = sorted(wb_vals)
# wb_vals = wb_vals[::-1]
# # print(wb_vals)
# x = [i for i in range(len(wb_vals))]
# plt.plot(x, wb_vals, label='Witten Bell Trigram')
# plt.legend(loc='upper right')
# plt.show()
#
# wb_vals = []
# for i in range(len(bifreq)):
#     # print(i)
#     pair = bifreq[i][0]
#     val = witten_bell(pair, 2)
#     wb_vals.append(val)
#     # print(pair, val)
#
# wb_vals = sorted(wb_vals)
# wb_vals = wb_vals[::-1]
# # print(wb_vals)
# x = [i for i in range(len(wb_vals))]
# plt.plot(x,wb_vals, label='Witten Bell Bigram')
# plt.legend(loc='upper right')
# plt.show()


""" KNESER NEY"""

# bi_kn_prob = {}
# bi_kn_vals = []
# for i in range(len(bifreq)):
#     pair = bifreq[i]
#     # print(pair)
#     w1 = pair[0].split(' ')[0]
#     w2 = pair[0].split(' ')[1]
#     # print(w1, w2)
#     bi_kn_prob[pair[0]] = kneser_ney(w1, w2, unifreq, bifreq, 2)
#     bi_kn_vals.append(bi_kn_prob[pair[0]])
#
# # print(bi_kn_prob)
# x = [i for i in range(len(bi_kn_prob))]
# kn_vals = sorted(bi_kn_vals, reverse=True)
# plt.plot(x, kn_vals,label='Kneser Ney Bigram')
# plt.legend(loc='upper right')
# plt.show()
# # print(kn_prob)
#
# tri_kn_prob = {}
# tri_kn_vals = []
# for i in range(len(trifreq)):
#     pair = trifreq[i]
#     term = pair[0]
#     tri_kn_prob[term] = kneser_ney(term, 0, 0, 0, 3)
#     tri_kn_vals.append(tri_kn_prob[term])
# x = [i for i in range(len(tri_kn_prob))]
# tri_kn_vals = sorted(tri_kn_vals, reverse=True)
# # print(tri_kn_vals)
# plt.plot(x, tri_kn_vals, label='Kneser Ney Trigram')
# plt.legend(loc='upper right')
# plt.show()
# uni_kn_prob = {}
# uni_kn_vals = []
# for i in range(len(unifreq)):
#     pair = unifreq[i]
#     term = pair[0]
#     val = pair[1]
#     uni_kn_prob[term] = (val - delta) / float(len(unigrams))
#     uni_kn_vals.append(uni_kn_prob[term])
# x = [i for i in range(len(uni_kn_prob))]
# uni_kn_vals = sorted(uni_kn_vals, reverse=True)
# # print(kn_vals)
# plt.plot(x, uni_kn_vals, label='Kneser Ney Unigram')
# plt.legend(loc='upper right')
# plt.show()

"""MISCELLANEOUS"""
# three_in_one()  # Graph comparing unifreq, bifreq. trifreq
for i in range(3):
    nine_in_one(i + 1)  # Zipf curve of all three sources in one graph
plt.show()

# unigram_for_all()  # Unigram comparison for all three smoothings

# uni_laplace = laplace_smoothing(unifreq, len(final), 200, 1)  # Dictionary={N-gram:Prob}
# laplace_multiple_V(uni_laplace, 200)
# uni_laplace = laplace_smoothing(unifreq, len(final), 2000, 1)  # Dictionary={N-gram:Prob}
# laplace_multiple_V(uni_laplace, 2000)
# uni_laplace = laplace_smoothing(unifreq, len(final), len(unifreq), 1)  # Dictionary={N-gram:Prob}
# laplace_multiple_V(uni_laplace, len(unifreq))
# uni_laplace = laplace_smoothing(unifreq, len(final), 10*len(unifreq), 1)  # Dictionary={N-gram:Prob}
# laplace_multiple_V(uni_laplace, 10*len(unifreq))
# plt.legend(loc='upper right')
# plt.title('Laplace Smoothing for 200, 2000, VocabSize, 10*VocabSize')
# plt.show()

"""WORD PREDICTION"""

# sentence = 'the'
# start = 'the'
# done = {}
# done[start] = []
# flag = 0
#
# bi_kn_prob_copy = deepcopy(bi_kn_prob)
#
# for i in range(15):
#     while flag is 0:
#         word, bi_kn_prob_copy = get_next_word(start, bi_kn_prob_copy)
#         if word in done[start]:
#             flag = 0
#         else:
#             flag = 1
#     done[start].append(word)
#     sentence += ' ' + word
#     start = word
#     flag = 0
#     done[start] = []
# print(sentence)
