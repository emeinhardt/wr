"""
This module documents some Python functions (and the associated mathematical
concepts) for
 - functional, non-/less-stateful manipulation of dictionaries.
 - picking out subsets of the domain of an 'enumerated' partial function
   represented as a Python dictionary.
 - combining and modifying partial functions via 'priority union' and
   'stretched composition'.
     - Priority union and stretched composition define a left near-ring over
       the set of all possible partial functions from a finite set X to itself,
       where
         - Priority union defines a non-commutative group with the empty
           function ('{}') as additive identity.
         - Stretched composition defines a monoid with the identity function
           id_X as multiplicative identity.
 - comparing how different two partial functions are from each other (their
   edit distance), picking out the set of all partial functions in a list
   that have exactly some edit distance or are within some edit distance of a
   reference partial function.

The application of these functions/mathematical concepts is
 - implementation of a software library for defining and manipulating
   phonological feature matrices, phonological rules and finite state
   automata / transducers.
 - understanding and defining the conditions under which two phonological maps
   'interact'.

Conventions adopted here:
 - dom(f) refers to f's *domain of definition*
 - cod(f) = image(f)

TODO:
 - Test dict_to_relation, curry, and flip more thoroughly.
 - Define inverses for dict_to_relation and curry.
 - Add documentation/examples to relevant notebooks for dict_to_relation,
   curry, and flip.
"""

__version__ = '0.1'
__author__ = 'Eric Meinhardt'

from functools import reduce, wraps
from funcy import walk_keys
from itertools import product
from json import dumps
from frozendict import FrozenDict


##################################
#  PRETTY PRINTING A DICTIONARY  #
##################################

def pprint_dict(a_dict, suppress_return=True):
    if isinstance(a_dict, FrozenDict):
        the_dict = dict(a_dict)
    else:
        the_dict = a_dict
    json_string = dumps(the_dict, sort_keys=True, indent=4,
                        ensure_ascii=False)
    print(json_string)
    if not suppress_return:
        return json_string


def pprint_dictlist(dlist, suppress_return=True):
    a_dict = dlist[0]
    if isinstance(a_dict, FrozenDict):
        my_dlist = list(map(dict, dlist))
    elif isinstance(a_dict, dict):
        my_dlist = dlist
    else:
        raise Exception("dlist must be a list of dictionaries or FrozenDicts; instead got {0}".format(dlist))
    json_string = dumps(my_dlist, sort_keys=True, indent=4,
                        ensure_ascii=False)
    print(json_string)
    if not suppress_return:
        return json_string


############################
#  PRESERVING FROZENDICTS  #
############################

def conserve_frozendicts(func):
    '''
    Decorator for functions that accept or return dictionaries; if any
    arguments of func are FrozenDicts, then the result will be casted to a
    FrozenDict
    '''
    @wraps(func)
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        args = func_args[:len(arg_names)]
        # print('0 args: {0}'.format(args))
        defaults = func.__defaults__ or ()
        args = args + defaults[len(defaults) - (func.__code__.co_argcount - len(args)):]
        # print('1 args: {0}'.format(args))
        # params = list(zip(arg_names, args))
        # args = func_args[len(arg_names):]
        # print('2 args: {0}'.format(args))
        if any([isinstance(arg, FrozenDict) for arg in args]):
            return FrozenDict(func(*func_args, **func_kwargs))
        return func(*func_args, **func_kwargs)
    return wrapper


###################################################
#  CONVERTING BETWEEN DICTIONARIES AND FUNCTIONS  #
###################################################


def dict_to_function(a_dict, use_default_return_value=False,
                             default_return_value=None):
    '''
    Returns a function that looks up its argument in a_dict.

    If use_default_return_value is True, the function will return
    default_return_value for any argument not in a_dict. If
    use_default_return_value is False, the function will throw the normal
    error.

    If the keys of a_dict are all iterables of the same type and length k,
    then this will return a function of k arguments (rather than a
    function expecting an iterable). For example, if the keys of a dict d
    were [0,'a'] and ['foo','bar'], then d's two keys would both be lists of
    length 2 and you would retrieve corresponding values from
    dict_to_function(d) by calling
        dict_to_function(d)(0, 'a')
    or
        dict_to_function(d)('foo','bar')
    *rather* than the cumbersome
        dict_to_function(d)([0, 'a']).
    '''
    ks = list(a_dict.keys())
    all_iterables = all(map(lambda k: hasattr(k, '__iter__',), ks))
    if all_iterables:
        a_k = list(ks)[0]
        a_len = len(a_k)
        a_type = type(a_k)
        same_len = all(map(lambda k: len(k) == a_len, ks))
        same_type = all(map(lambda k: type(k) == a_type, ks))
        if same_len and same_type and a_len > 1:
            def func(*args):
                if not use_default_return_value:
                    return a_dict[a_type(args)]
                else:
                    return a_dict.get(a_type(args),
                                      default=default_return_value)
            return func

    def func(arg):
        if not use_default_return_value:
            return a_dict[arg]
        else:
            return a_dict.get(arg, default=default_return_value)
    return func

def function_to_dict(a_function, domain):
    '''
    Returns a dictionary that maps from each of the keys k in domain to
    a_function(k).
    '''
    return {k:a_function(*k) for k in domain}


#############################################
#  MISC. FUNCTIONAL PROGRAMMING OPERATIONS  #
#############################################

def dict_to_relation(a_dict):
    '''
    Returns a tuple representing the *relation* defined by the function
    defined by a_dict.
    '''
    return tuple(map(lambda kv_pair: kv_pair[0] + (kv_pair[1],),
                     tuple(a_dict.items())))

@conserve_frozendicts
def curry_dict(a_dict):
    '''
    Given an (immutable = 'frozen') dictionary whose keys are all iterables of
    the same type and length, returns a 'curried' form of the dictionary.
    '''
    ks = list(a_dict.keys())
    all_iterables = all(map(lambda k: hasattr(k, '__iter__',), ks))
    if not all_iterables:
        raise Exception('Dictionary is not curryable: {0}'.format(a_dict))
    a_k = list(ks)[0]
    a_len = len(a_k)
    a_type = type(a_k)
    same_len = all(map(lambda k: len(k) == a_len, ks))
    same_type = all(map(lambda k: type(k) == a_type, ks))
    if not (same_len and same_type and a_len > 1):
        raise Exception('Dictionary is not curryable: {0}'.format(a_dict))
    
    my_L = dict_to_relation(a_dict)
    head   = lambda seq: seq[0]
    tail   = lambda seq: seq[1:]
    heads  = lambda seqs: set(map(head, seqs))
    tails  = lambda seqs: set(map(tail, seqs))
    follow = lambda h, seqs: tails({seq for seq in seqs if head(seq) == h})
    
    sing_of_sing = lambda x: len(x) == 1 and len(list(x)[0]) == 1
    
    
    def curr(Ss):
        if isinstance(a_dict, FrozenDict):
            return FrozenDict({h:curr(follow(h, Ss)) if not sing_of_sing(follow(h, Ss)) else list(follow(h, Ss))[0][0] for h in heads(Ss)})
        return {h:curr(follow(h, Ss)) if not sing_of_sing(follow(h, Ss)) else list(follow(h, Ss))[0][0] for h in heads(Ss)}
    # curr = lambda Ss: {h:curr(follow(h, Ss)) if not sing_of_sing(follow(h, Ss)) else list(follow(h, Ss))[0][0] for h in heads(Ss)}
    
    return curr(my_L)


def flip_dict(a_dict):
    '''
    Dictionary analogue of the functional programming 'flip' function:
        flip: ((A x B) ⟶ C) ⟶ ((B x A) ⟶ C)
    
    That is, given a dictionary whose keys are all sequences of uniform
    type and length, this function returns a new version of that dictionary
    where the order of the keys is reversed.
    '''
    rev = lambda l: list(reversed(l))
    return walk_keys(lambda l: tuple(rev(l)), a_dict)


################################################
#  IMMUTABLE/COMPOSABLE DICTIONARY OPERATIONS  #
################################################


@conserve_frozendicts
def change_key(a_dict, key, value):
    '''
    Returns a new copy of a_dict where key = value.
    '''
    new_dict = {k: a_dict[k] for k in a_dict}
    new_dict[key] = value
    return new_dict


@conserve_frozendicts
def dict_update(dict_to_update, dict_with_update):
    '''
    If d_b and d_a are dictionaries, d_b.update(d_a) is a statement that
    returns nothing. This is a composable dictionary update that returns a new
    dictionary that is the result of updating a copy of dict_to_update with
    dict_with_update.
    '''
    newDict = {k: dict_to_update[k] for k in dict_to_update}
    newDict.update(dict_with_update)
    return newDict


@conserve_frozendicts
def dict_updates(dict_to_update, dicts_with_updates):
    '''
    Updates dict_to_update with all of the updates in dicts_with_updates in the
    order they are presented in dicts_with_updates.
    
    E.g. if
        dicts_with_updates = [{'a':'b'},{'b':'d'}]
    and
        d = dict()
    then
        dict_updates(d, dicts_with_updates)
    is the same as
        dict_update(dict_update(d, {'a':'b'}), {'b':'d'})
    '''
    return reduce(dict_update, dicts_with_updates, dict_to_update)


@conserve_frozendicts
def remove_key(a_dict, key):
    '''
    Returns a new copy of a_dict without key.
    '''
    newDict = dict(a_dict)
    del newDict[key]
    return newDict


@conserve_frozendicts
def project_to(the_dict, desired_keys):
    """
    Projects a dictionary down to just having the desired keys.
    """
    d = {key: the_dict[key] for key in desired_keys}
    return d


@conserve_frozendicts
def project_off(the_dict, undesired_keys):
    """
    Projects away the undesired keys from a dictionary.
    """
    d = {key: the_dict[key] for key in the_dict if key not in undesired_keys}
    return d


###############################
#  SELECTING KEYS AND VALUES  #
###############################

def dom(a_dict):
    '''
    Returns the keys of a_dict as a set.
    '''
    return set(a_dict.keys())


def image(a_dict, keys=None):
    '''
    Returns the set containing all and only values that a_dict maps elements of
    keys to. If keys isn't provided, returns the values of a_dict as a set.

    If f: X ⟶ Y and S ⊆ X, the image of S under f ("f[S]") is the subset of the
    codomain that f maps elements in S to:
        image_f(S) = f[S] = {y ∈ Y | ∃ x ∈ S s.t. f(x) = y}
    '''
    if keys is None:
        keys = dom(a_dict)
    img = {a_dict[k] for k in keys}
    return img


def common_keys(dictA, dictB):
    '''
    Returns the keys where dictA and dictB are both defined.

    common_keys(f,g) = dom(f) ∩ dom(g)
    '''
    return set(dictA.keys()).intersection(dictB.keys())


def equalizer(dictA, dictB, keys_to_compare=None):
    '''
    Returns the keys where dictA and dictB have equal values.
    
    Examines keys_to_compare if provided; otherwise looks at all and only keys
    that are common to both.

    If f: X ⟶ Y and g: X ⟶ Y, then
        equalizer(f,g) = {x | f(x) = g(x)}.
    '''
    if keys_to_compare is None:
        keys_to_compare = common_keys(dictA, dictB)
    return set([k for k in keys_to_compare if dictA[k] == dictB[k]])


def differing_keys(dictA, dictB, keys_to_compare=None):
    '''
    Returns the keys where dictA and dictB have different values: the
    complement of equalizer(dictA, dictB, keys_to_compare).
    
    Examines keys_to_compare if provided; otherwise looks at all and only keys
    that are common to both.
    '''
    if keys_to_compare is None:
        keys_to_compare = common_keys(dictA, dictB)
    equalizer_keys = equalizer(dictA, dictB)
    differingKeys = set.difference(keys_to_compare, equalizer_keys)
    return differingKeys


def fiber(value, a_dict):
    '''
    Returns the keys mapping to value in a_dict.

    If f: X ⟶ Y, the fiber of y ∈ Y is the set of elements in dom(f) s.t. f
    maps them to y:
        f⁻¹(y) = {x ∈ X| f(x) = y}
    '''
    return {k for k in a_dict if a_dict[k] == value}


def preimage(values, a_dict):
    '''
    Returns all and only those keys that map to values
    in a_dict.

    If f: X ⟶ Y, the pre-image of S ⊆ Y is the set of elements in dom(f) s.t. f
    maps them to some element in S:
        f⁻¹[S] = {x ∈ X| f(x) ∈ S}
    '''
    return {k for k in a_dict if a_dict[k] in values}


@conserve_frozendicts
def fibers(a_dict):
    '''
    Returns a dictionary mapping every value to the set of keys that map to it
    in a_dict.

    Where the pre-image of S bundles all the fibers into a 'flat' set, we can
    define fibers[S] as defining a function taking each element of S to its
    corresponding fiber.
    '''
    newDict = {value: fiber(value, a_dict) for value in set(a_dict.values())}
    return newDict


def common_fibers(dict_a, dict_b):
    '''
    If f: X ⟶ Z and g: Y ⟶ Z, then the common fibers of f and g are all and 
    only
        {z ∈ Z | z ∈ image(f) ∨ z ∈ image(g)} = image(f) ∩ image(g).
    I.e.
        common_fibers(f,g)
    indexes the equivalence classes of fibers of f and g defined by the fiber
    product of f and g.
    '''
    return set.intersection( image(dict_a), image(dict_b) )


def fiber_product(dict_a, dict_b):
    '''
    If f: X ⟶ Z and g: Y ⟶ Z, then the fiber product (or pullback) of f, g
        X x_Z Y = {(x,y) | f(x) = g(y)}
    or, equivalently
        X x_Z Y = select(X x Y, λ(x, y).f(x) = g(y))
    
    This function returns the fiber product of the functions defined by
    dict_a and dict_b.
    '''
    product_of_keys = list(product(dom(dict_a), dom(dict_b)))
    matching_key_pairs = {each for each in product_of_keys if dict_a[each[0]] == dict_b[each[1]]}
    return matching_key_pairs


#############################################
#  IDENTITIES, INVERSES, AND INVERTIBILITY  #
#############################################

def get_identity(keys):
    '''
    Returns a dictionary mapping every key in keys to itself.
    '''
    newDict = {k: k for k in keys}
    return newDict


def is_surjective_over(codomain, a_dict):
    '''
    Returns true iff image(a_dict) == codomain.
    '''
    return set(a_dict.values()) == codomain


@conserve_frozendicts
def get_all_right_inverses(a_dict):
    '''
    Returns a set of dictionaries d such that
        {k:a_dict[ d[k] ] for k in keys} == get_identity(keys)
    where keys = set( a_dict.values() ).
    
    Note that this returns a set of dictionaries because in general there exist
    multiple right-inverses; only bijective functions have a unique right-
    inverse.
    '''
    myFibers = fibers(a_dict)
    obligatoryMap = {value: list(myFibers[value])[0] for value in myFibers
                     if len(myFibers[value]) == 1}
    
    valuesWithChoices = {value for value in myFibers
                         if len(myFibers[value]) > 1}
    # mapsToMakeChoicesFrom = {value: myFibers[value] for value in
    #                           valuesWithChoices}
    
    if len(valuesWithChoices) == 0:
        return obligatoryMap
    
    fiberToDicts = lambda value: [{value: key} for key in myFibers[value]]
    fibersAsDictLists = map( fiberToDicts, valuesWithChoices )
    choosableMaps = list(product(*fibersAsDictLists))
    
    update_obligatoryMap = lambda chosenMap: dict_updates(obligatoryMap,
                                                          chosenMap)
    rightInverses = list(map(update_obligatoryMap, choosableMaps))
    return rightInverses


def is_right_identity(dictB, dictA):
    '''
    Returns true iff dictA is a right-identity of dictB:
        dictB ⚬ dictA = id_cod(B)
    where id_cod(B) is the identity map on the values of dictB. 
    I.e. this function returns true iff
        {k:dictB[ dictA[k] ] for k in keys} == get_identity(keys)
    where keys = set( dictB.values() ).
    '''
    keys = set( dictB.values() )
    return {k: dictB[ dictA[k] ] for k in keys} == get_identity(keys)


def is_injective(a_dict):
    '''
    Returns true if the fiber of every element in the image of a_dict contains
    at most one element.
    '''
    return all(map(lambda fiber: len(fiber) <= 1, tuple(fibers(a_dict).values())))


@conserve_frozendicts
def get_left_inverse(a_dict):
    '''
    Returns the left inverse of a_dict if a_dict is invertible.
    '''
    assert is_injective(a_dict), "{0} isn't invertible because it isn't injective.".format(a_dict)
    
    getFiber = lambda value: list(fiber(value, a_dict))[0]
    newDict = {v: getFiber(v) for v in image(a_dict)}
    return newDict


################################################################
#  GENERATING ALL (PARTIAL/TOTAL) FUNCTIONS OVER A FINITE SET  #
################################################################

def generate_all_functions_over(keys, partial=False):
    '''
    Returns a generator that will yield all functions (dictionaries) from keys
    to keys. By default returns total functions; if partial is False, will also
    include partial functions.
    '''
    if partial:
        ks = set.union( set(keys), {None})
    else:
        ks = set(keys)
    domain = keys
    l = len(domain)
    images = product(ks, repeat=l)
    for image in images:
        noNulls = lambda kv: kv[1] is not None
        mapping = filter(noNulls, zip(domain, image))
        yield dict(mapping)


#########################################
#  PRIORITY UNION OF PARTIAL FUNCTIONS  #
#########################################

@conserve_frozendicts
def left_priority_union(dictA, dictB):
    '''
    Returns a new dictionary containing all the keys and values in dictA, plus
    whatever keys and values are in dictB that don't conflict with dictA.
    '''
    # empty dictionary is the unique dict s.t. it is both a left-identity and
    #   a right-identity for all dicts
    # dictB is a right-identity of dictA if dom(dictB) == equalizer(dictA, dictB)
    # dictB is always a right-identity of dictA if dom(dictB) ⊆ dom(dictA)
    # dictA is a left-identity of dictB if dom(dictA) == equalizer(dictA, dictB)
    
    newDict = {k: dictB[k] for k in dictB}
    newDict.update(dictA)
    return newDict


@conserve_frozendicts
def right_priority_union(dictA, dictB):
    '''
    Returns a new dictionary containing all the keys and values in dictB, plus
    whatever keys and values are in dictA that don't conflict with dictA.
    '''
    # empty dictionary is the unique dict s.t. it is both a left-identity and
    #   a right-identity for all dicts
    # dictA is a right-identity of dictB if dom(dictA) == equalizer(dictA, dictB)
    # dictA is always a right-identity of dictB if dom(dictA) ⊆ dom(dictB)
    # dictB is a left-identity of dictA if dom(dictB) == equalizer(dictA, dictB)
    
    newDict = {k: dictA[k] for k in dictA}
    newDict.update(dictB)
    return newDict


@conserve_frozendicts
def subtract_dicts(dictA, dictB):
    '''
    Returns a new dictionary containing all the key-value pairs in dictA that
    are not found in dictB.
    '''
    newDict = {k: dictA[k] for k in dictA if dictA[k] != dictB.get(k)}
    return newDict


@conserve_frozendicts
def common_mappings(dictA, dictB):
    '''
    Returns a dictionary containing only those keys in the equalizer of dictA
    and dictB.
    '''
#     newDict = {k:dictA[k] for k in equalizer(dictA, dictB)}
    return subtract_dicts(dictA, subtract_dicts(dictA, dictB))


###########################
#  STRETCHED COMPOSITION  #
###########################

def are_composable(dictB, dictA):
    '''
    Returns true iff
        values(dictA) ⊆ keys(dictB)
    i.e. if dictB and dictA are composable as
        dictB ⚬ dictA
    '''
    return set.issubset( image(dictA), dom(dictB) )


def total(keys_to_ensure_totality_over):
    '''
    Maps
        keys_to_ensure_totality_over
    to a function that maps a_dict to
        a version of a_dict that is total over the keys
        already in it plus all keys in keys_to_ensure_totality_over.
    
    Achieves this by taking the priority union of
        a_dict
    with
        get_identity(keys_to_ensure_totality_over).
    
    That is, this curried function is literally just syntactic sugar
    for
        priority_union(a_dict, get_identity(keys_to_ensure_totality_over))
    '''
    @conserve_frozendicts
    def totalize(a_dict):
        return left_priority_union(a_dict, 
                                   get_identity(keys_to_ensure_totality_over))
    return totalize


@conserve_frozendicts
def naive_compose(dictB, dictA):
    '''
    Composes B with A as B ⚬ A.
    '''
    return {k: dictB[ dictA[k] ] for k in dictA}


@conserve_frozendicts
def stretched_compose(dictB, dictA):
    '''
    Returns total(image(dictA))(dictB) ⚬ dictA.
    '''
    return naive_compose(total(image(dictA))(dictB), dictA)


###################
#  EDIT DISTANCE  #
###################


def edit_distance(dictA, dictB, keys_to_compare=None):
    '''
    Returns the number of keys that need to be added, deleted, or whose value
    needs to be changed to make dictA and dictB the same.
    
    If keys_to_compare is specified, all keys not in keys_to_compare will be
    ignored when calculating distance.
    '''
    if keys_to_compare is None:
        my_A = dictA
        my_B = dictB
    else:
        my_A = project_to(dictA, keys_to_compare)
        my_B = project_to(dictB, keys_to_compare)
        
    missing_from_A = set.difference(set(my_B.keys()), set(my_A.keys()))
    missing_from_B = set.difference(set(my_A.keys()), set(my_B.keys()))
    
    diff_keys = differing_keys(my_A, my_B)
    
    # To turn A to B, you'd need to
    #   1. add    all keys in B that are missing_from_A
    #   2. remove all keys in A that are missing_from_B
    #   3. change all keys in A in co-equalizer(A,B)
    return len(missing_from_A) + len(missing_from_B) + len(diff_keys)


def edit_circle(target_dict, dict_list, edit_dist, keys_to_compare=None):
    '''
    Returns the set of all dictionaries in dict_list that have exactly the
    specified edit distance from target_dict.
    
    If keys_to_compare is specified, all keys not in keys_to_compare will be
    ignored when calculating distance.
    '''
    if keys_to_compare is None:
        my_target = target_dict
        my_list = dict_list
    else:
        my_target = project_to(target_dict, keys_to_compare)
        my_list = list(map(lambda d: project_to(d, keys_to_compare), dict_list))
    
    in_circle = set([d for d in my_list if edit_distance(my_target, d) == edit_dist])
    return in_circle


def edit_neighborhood(target_dict, dict_list, edit_dist, keys_to_compare=None):
    '''
    Returns the set of all dictionaries in dict_list that have *at most* the
    specified edit distance from target_dict, excluding target_dict.
    
    If keys_to_compare is specified, all keys not in keys_to_compare will be
    ignored when calculating distance.
    '''
    if keys_to_compare is None:
        my_target = target_dict
        my_list = dict_list
    else:
        my_target = project_to(target_dict, keys_to_compare)
        my_list = list(map(lambda d: project_to(d, keys_to_compare), dict_list))
    
    neighbors = set([d for d in my_list if edit_distance(my_target, d) <= edit_dist and edit_distance(my_target, d) != 0])
    return neighbors
