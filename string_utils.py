from functools import reduce
from itertools import takewhile, product, starmap
import numpy as np
from random import choice

def union(Ss):
    return reduce(set.union, Ss)

def language_concatenation(A, B, concat=None):
    if concat is None:
        concat = lambda u, v: u + v
    return set(starmap(concat,
                       product(A, B)))

leftEdge = '⋊'
rightEdge = '⋉'
edgeSymbols = {leftEdge, rightEdge}



def tupleToDottedString(pair): 
    return '.'.join(pair)

def dottedStringToTuple(s): 
    return tuple(s.split('.'))

t2ds = tupleToDottedString
ds2t = dottedStringToTuple

def ds_l(s):
    #let 
    #  l = len(s)              ; length of a dotted string s as a string
    #  n = len(  ds2t(s)  )    ; # of symbols in s
    #  d = l - n               ; # of dots in s
    #  
    #  ∀s, d = n-1 ∴ l = n + (n - 1) = 2n - 1
    #  ∴ n = (l+1)/2
    return (len(s)+1)/2

def padInputSequenceWithBoundaries(inputSeq):
    temp = list(dottedStringToTuple(inputSeq))
    temp = tuple([leftEdge] + temp + [rightEdge])
    return tupleToDottedString(temp)

def trimBoundariesFromSequence(seq):
    temp = list(dottedStringToTuple(seq))
    if temp[0] == leftEdge:
        temp = temp[1:]
    if temp[-1] == rightEdge:
        temp = temp[:-1]
    return tupleToDottedString(tuple(temp))


def dsToInventory(s):
    s_t = ds2t(s)
    symbols = set(s_t)
    return symbols

def lexiconToInventory(DSs):
    inventories = list(map(dsToInventory, DSs))
    return union(inventories)



def subInDS(dottedString, to_replace, replacement):
    '''
    Replace each instance of symbol 'to_replace' 
    with 'replacement' symbol in 'dottedString'.
    '''
    old_symbol_tuple = dottedStringToTuple( dottedString )

    replacer = lambda symb: symb if symb != to_replace else replacement
    new_symbol_tuple = tuple( map(replacer, old_symbol_tuple) )

    dottedSymbols = tupleToDottedString( new_symbol_tuple ) 
    return dottedSymbols


def dsToKfactors(k, ds):
    seq = ds2t(ds)
    l = len(seq)
    if k > l:
        return tuple()
    kFactor_start_indices = takewhile(lambda pair: pair[0] <= l-k, enumerate(seq))
    kFactors = tuple(seq[index[0]:index[0]+k] for index in kFactor_start_indices)
    return set(map(t2ds, kFactors))

def dsTo2factors(ds):
    return dsToKfactors(2, ds)
def dsTo3factors(ds):
    return dsToKfactors(3, ds)

def lexiconToKfactors(DSs, k):
    myDsToKfactors = lambda ds: dsToKfactors(k, ds)
    return union(map(set, map(myDsToKfactors, DSs)))

def lexiconTo2factors(DSs):
    return union(map(set, map(dsTo2factors, DSs)))
def lexiconTo3factors(DSs):
    return union(map(set, map(dsTo3factors, DSs)))


def compareKfactors(DSs_A, DSs_B, k):
    A = lexiconToKfactors(DSs_A, k)
    B = lexiconToKfactors(DSs_B, k)
    return {"A == B":A == B, "A - B": A - B, "B - A": B - A}

def sameKfactors(DSs_A, DSs_B, k):
    return compareKfactors(DSs_A, DSs_B, k)["A == B"]

def hasIllicitKfactors(W, illicit_k_factors):
    if type(W) == str:      
        # gather the k-factors into an immutable data structure
        illicit_kfs = tuple(illicit_k_factors)
        # get the set of k-factor lengths (values of k) among the illicit_kfs
        illicit_factor_lengths = set([len(ds2t(kf)) for kf in illicit_kfs])
        # map each k to the set of k-factors of dotted string ds
        kFactorSets = {kf_l:dsToKfactors(kf_l, W) for kf_l in illicit_factor_lengths}
        illegal_kfactors_discovered = tuple(ikf for ikf in illicit_kfs if ikf in kFactorSets[len(ds2t(ikf))])
        if illegal_kfactors_discovered == tuple():
            return False
        return illegal_kfactors_discovered
    else:
        myFunc = lambda w: hasIllicitKfactors(w, illicit_k_factors)
        results = tuple(map(myFunc, W))
        if not any(results):
            return False
        return set(t2ds(each) for each in results if each != False)


    
def sigmaK(sigma, k):
    return product(sigma, repeat=k)



def dsToKfactorSequence(k, ds):
    seq = ds2t(ds)
    l = len(seq)
    if k > l:
        return tuple()
    kFactor_start_indices = takewhile(lambda pair: pair[0] <= l-k, enumerate(seq))
    kFactors = tuple(seq[index[0]:index[0]+k] for index in kFactor_start_indices)
    return tuple(map(t2ds, kFactors))

def threeFactorSequenceToDS(threeFactors):
    wLE = ds2t(threeFactors[0])[0]
    wRE = ds2t(threeFactors[-1])[-1]
    w_NE = '.'.join([ds2t(eachTriphone)[1] for eachTriphone in threeFactors])
    return '.'.join([wLE, w_NE, wRE])



def randomString(sigma, l, hasLeftEdge=True):
    s_t = tuple([choice(list(sigma)) for each in range(l)])
    s = t2ds(s_t)
    if hasLeftEdge:
        return leftEdge + '.' + s
    return s

# def randomPrefix(l, alphabet):
#     return randomString(alphabet, l, hasLeftEdge=True)

def randomPrefixFromTriphones(triphones, l, hasLeftEdge=True):
    def foo(triphonesSoFar, max_length):
        s = threeFactorSequenceToDS(triphonesSoFar)
        s_t = ds2t(s)
        l = len(s_t)
        if l == max_length:
            return s

        rightmost_symbol = s_t[-1]
        triphonesBeginningWithRMS = {t for t in triphones if ds2t(t)[0] == rightmost_symbol}
        if l + 2 == max_length:
            wordFinalTriphones = list({t for t in triphonesBeginningWithRMS if ds2t(t)[2] == rightEdge})
            triphonesToChooseFrom = wordFinalTriphones
        else:
            wordMedialTriphones = list({t for t in triphonesBeginningWithRMS if ds2t(t)[2] != rightEdge})
            triphonesToChooseFrom = wordMedialTriphones
        nextTriphone = choice(triphonesToChooseFrom)
        triphonesSoFar.append(nextTriphone)
        return foo(triphonesSoFar, max_length)
    if hasLeftEdge:
        wordInitialTriphones = list({t for t in triphones if ds2t(t)[0] == leftEdge})
        return foo([choice(wordInitialTriphones)], max_length = l)
    else:
        raise Exception("Currently unsupported.")
#         return foo([choice(wordInitialTriphones)



def replaceXj(s, j, x):
    s_t = ds2t(s)
    s_l = list(s_t)
    s_l[j] = x
    s_t = tuple(s_l)
    return t2ds(s_t)

def removeXj(s, j):
    return replaceXj(s, j, '_')

def removeXi(x0k):
    l = len(ds2t(x0k))
    return removeXj(x0k, l-2)



def getPrefixes(s):
    if type(s) == str:
        sAsTup = ds2t(s)
    elif type(s) == tuple:
        sAsTup = s
    else:
        raise Exception('s must be a string or a tuple.')
    prefsAsTuples = set(sAsTup[0:i] for i in range(1, len(sAsTup)+1))
    return set(map(t2ds, prefsAsTuples))

def getProperPrefixes(s):
    Ps = getPrefixes(s)
    return {p for p in Ps if p[-1] != rightEdge}

def isProperPrefix(word, prefix):
    PPs = getProperPrefixes(word)
    return prefix in PPs

def hasAsPrefix(word, prefix):
    if type(prefix) == str:
        prefix_t = ds2t(prefix)
    elif type(prefix) == tuple:
        prefix_t = prefix
    else:
        raise Exception('prefix should be a dotted string or a tuple.')
    if type(word) == str:
        word_t = ds2t(word)
    elif type(word) == tuple:
        word_t = word
    else:
        raise Exception('word should be a dotted string or a tuple.')
    
    l = len(prefix_t)
    return word_t[0:l] == prefix_t

def wordsWithPrefix(p, Ws):
    return {w for w in Ws if hasAsPrefix(w, p)}



def d_s(x, y):
    '''
    Hamming distance between symbol x and symbol y.
    '''
    return x != y

def d_h(u, v):
    '''
    Hamming distance between strings u and v.
    '''
    u_t = ds2t(u)
    v_t = ds2t(v)
    if len(u_t) != len(v_t):
        return np.infty
    return sum(tuple(starmap(d_s, zip(u_t,v_t))))

def hamming_neighbors(s, W):
    '''
    Returns the strings of W that are exactly Hamming distance 1 from s.
    '''
    return h_sphere(1, s, W)

def h_sphere(k, s, W, exclude_s = False):
    '''
    Returns the strings of W that are exactly Hamming distance k from s.
    '''
    sphere = {v for v in W if d_h(s,v) == k}
    if exclude_s:
        return sphere - {s}
    return sphere

def h_neighborhood(k, s, W, exclude_s = False):
    '''
    Returns all strings of W whose Hamming distance from s is <= k.
    '''
    N = {v for v in W if d_h(s,v) <= k}
    if exclude_s:
        return N - {s}
    return N

def getSpheres(s, W):
    '''
    Returns a mapping from [0,len(s)-1] to the corresponding 
    Hamming spheres of s in W.
    '''
    D = range(len(ds2t(s)))
    spheres = {d:h_sphere(d, s, W) for d in D}
    return spheres

def neighborhood_measures(k, s, W, M, exclude_s = False):
    '''
    Applies a measure M (dictionary) to each of the k-neighbors
    of s in W.
    '''
    N = h_neighborhood(k, s, W, exclude_s)
    Ms = {v:M[v] for v in N}
    return Ms



def are_k_cousins(prefix, wordform, k, prefixes, exactlyK = True):
    if exactlyK:
        k_cousins = h_sphere(k, prefix, prefixes)
    else:
        k_cousins = h_neighborhood(k, prefix, prefixes)
    prefixesOfw = getPrefixes(wordform)
    return any(p in k_cousins for p in prefixesOfw)

def get_k_cousins(prefix, k, Ws, prefixes, exactlyK = True):
    if exactlyK:
        k_cousins = h_sphere(k, prefix, prefixes)
    else:
        k_cousins = h_neighborhood(k, prefix, prefixes)
    return {w for w in Ws if any(p in k_cousins for p in getPrefixes(w))}

def count_k_cousins(prefix, k, Ws, prefixes, exactlyK = True):
    if exactlyK:
        k_cousins = h_sphere(k, prefix, prefixes)
    else:
        k_cousins = h_neighborhood(k, prefix, prefixes)
    return len({w for w in Ws if any(p in k_cousins for p in getPrefixes(w))})




def wordformsOfLength(l, Ws, includingEdges = False):
    #Ws assumed to have word edges
    if includingEdges:
        return {w for w in Ws if len(ds2t(w)) == l}
    return {w for w in Ws if (len(ds2t(w)) + 2) == l}

def wordformsAtLeastLlong(l, Ws, includingEdges = False):
    #Ws assumed to have word edges
#     maxL = len( ds2t(sorted(list(Ws), key=len, reverse=True)[0]) )
    if includingEdges:
#         maxL = max(wordlengthsInclEdges)
        return {w for w in Ws if len(ds2t(w)) >= l}
#         return union([wordformsOfLength(eachl, Ws, includingEdges) for eachl in range(l, maxL+1)])
    if not includingEdges:
#         maxL = max(wordlengthsNotIncludingEdges)
#         maxL = maxL - 2
        return {w for w in Ws if (len(ds2t(w)) - 2) >= l}
#         return union([wordformsOfLength(eachl, Ws, includingEdges) for eachl in range(l, maxL+1)])
    
def getWordformsWithx(x, Ws):
    return {w for w in Ws if x in ds2t(w)}

def wordsWhereXiIs(x, i, Ws):
#     wordsWithX = xToWs[x]
    wordsWithX = getWordformsWithx(x, Ws)
    ws = set(map(ds2t, wordsWithX))
    return {t2ds(w) for w in ws if i <= len(w) - 1 and w[i] == x}