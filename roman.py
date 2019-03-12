"""
This module implements functions for accessing, manipulating, and modifying
collections of feature matrices that represent phonological inventories, as 
well as for constructing functions and dictionaries that represent phonological
rules.

TODO:
 - Feature charts:
   1. Go over 'bakovic_chart_riggle_hayes.json' with Eric B.
   2. Include more feature charts out of the box (e.g. SPE).
   3. Figure out how licensing works for e.g. the Phonological Corpus Tools-
      provided Hayes feature matrices...
   4. Allow import of csv files.
   5. Allow export/import of inventory as json or csv.
   6. Add support for adding features (and default values).
 7. Inferring a more compact description of changes.
 8. Automatically infer (likely) natural classes from feature structure (i.e.
    without being told/hard-coding what linguistically significant natural
    classes correspond to what combinations of feature values):
     - What non-trivial subsets of an inventory can be picked out by a small
       natural class?
 9. Overhaul for consistency: camelcase vs. underscore,
    no_symbol vs. removeSymbol, argument order, ?
 - Phonetically/featurally pathological symbols:
   10. The boundary-symbol related checking in fmsToWord is a mess; perhaps 
       there is a better solution than the tangle of cases. (See next item...)
   11. Add support/extend feature matrices for more things like word boundary 
       symbols:
       - Morpheme boundary symbol (a word boundary is also a morpheme boundary)
       - Root boundaries (root boundaries are also morpheme boundaries)
       - Other structural...boundaries? (e.g. syllable structure positions)
   12. Related: figure out how you want to handle the empty string in terms of
       features - this is critical for representing insertion rules.
 13. Take advantage of the relational algebra module?
 14. Fix patternSequenceToStringSet as noted in the docstring?
"""

__version__ = '0.1'
__author__ = 'Eric Meinhardt' # Except where noted in comments

from frozendict import FrozenDict
import harpoon as pf
# import relations as rel
import json
from functools import reduce
from random import choice
from itertools import islice, starmap, repeat, product
import re

# taken from https://github.com/aimacode/aima-python, circa 04.22.2018
from search import Problem, uniform_cost_search, astar_search


##########
#  MISC  #
##########

union = lambda sets: reduce(set.union, sets)


def nth(iterable, n, default=None):
    """
    Returns the nth item or a default value
    """
    return next(islice(iterable, n, None), default)


def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def myPipe(arg, funcs):
    '''
    Pipes in terminal shells
        arg | f | g | h
    compactly compose functions and shove an argument through the composition:
        h(g(f(arg))).
    This function imitates that behavior: If handed arg and [f, g, h],
    this computes h(g(f(arg))).
    '''
    return reduce(lambda x, f: f(x), reversed(funcs), arg)


###############################
#  IMPORTING A FEATURE CHART  #
###############################

hayes = 'ipa2hayes.json'
bakovic_complete = 'bakovic_chart_riggle_hayes.json'

default_feature_chart = bakovic_complete


def get_raw_feature_chart(filename):
    '''
    Feature charts are assumed to be .json files representing a list of Python
    dictionaries where every dictionary in the list has the same set of keys;
    these keys represent the features of the feature chart, plus a key
    ('symbol') indicating the symbol of the associated feature matrix.

    Returns the list of dictionaries.

    '''
    with open(filename, 'r', encoding='utf8') as f:
        # inventory = json.load(f)
        inventory = json.loads(f.read())
    return inventory


def get_feature_chart(filename):
    '''
    Feature charts are assumed to be .json files representing a list of Python
    dictionaries where every dictionary in the list has the same set of keys;
    these keys represent the features of the feature chart, plus a key
    ('symbol') indicating the symbol of the associated feature matrix.

    Differs from get_raw_feature_chart in that it casts each feature matrix to
    an immutable, hashable version of a dictionary (a 'FrozenDict') and assumes
    each feature matrix (including the symbol field) is unique, so it returns
    the feature chart as a *set*.

    Note that when this library is imported, this function is called on the
    default feature chart (currently 'bakovic_chart_riggle_hayes.json') and the
    result assigned to the default inventory ('IPA').
    '''
    return set(map(FrozenDict, get_raw_feature_chart(filename)))


IPA = get_feature_chart(default_feature_chart)


def reload_feature_chart(filename):
    '''
    Reassigns the default inventory ('IPA') to the feature chart in filename.
    '''
    IPA = get_feature_chart(filename)
    return IPA

def export_feature_chart(filename):
    '''
    Saves the default inventory ('IPA') as a JSON file.
    '''
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        json.dump(IPA, f, ensure_ascii = False, indent = 4)

################################
#  OPERATIONS ON AN INVENTORY  #
################################


# ACESSING BASIC INFORMATION

def getSymbols(inventory=None):
    '''
    Returns the set of IPA symbols defined in the specified inventory (or
    whatever portion of the IPA that's defined in the loaded feature matrix).
    '''
    if inventory is None:
        inventory = IPA
    return set([fm['symbol'] for fm in inventory])


def getFeatures(inventory=None):
    '''
    Returns the set of features defined in the specified inventory (or whatever
    portion of the IPA that's defined in the loaded feature matrix).
    
    Note that (as intuitively expected), this does not include the 'symbol'
    field.
    '''
    if inventory is None:
        inventory = IPA
    random_fm = choice(list(inventory))
    features = set(random_fm.keys()) - {'symbol'}
    return features


def getFeatureValues(inventory=None):
    '''
    Returns the set of feature values defined in the specified inventory (or
    whatever portion of the IPA that's defined in the loaded feature matrix).
    
    (As is probably intuitively expected, this does not include IPA symbols
    themselves.)
    '''
    if inventory is None:
        inventory = IPA
    symbols = getSymbols(inventory)
    getVals = lambda d: set(d.values())
    values = union(map(getVals, inventory))
    values = set.difference(values, symbols)
    return values


def opposite(feature_value):
    '''
    Returns the opposite of feature_value:
        - The opposite of '+' is '-' and vice versa.
        - The opposite of True is False and vice versa.
        - '0' is left unchanged.
    '''
    if feature_value == '+':
        return '-'
    elif feature_value == '-':
        return '+'
    elif feature_value == '0':
        return '0'
    elif feature_value is True:
        return False
    elif feature_value is False:
        return True
    raise Exception('Feature value {0} has no defined opposite.'.format(feature_value))


# FUNCTIONS FOR DETECTING NON-UNIQUE SYMBOLS IN AN INVENTORY

def same_fms(full_fm_a, full_fm_b):
    '''
    Returns true if the feature matrices are the same except for the symbol
    field.
    '''
    a = remove_symbol(full_fm_a)
    b = remove_symbol(full_fm_b)
    return a == b


def have_distinct_symbols(full_fm_a, full_fm_b):
    return full_fm_a['symbol'] != full_fm_b['symbol']


def is_matching_fm(target_full_fm):
    return lambda other_full_fm: same_fms(target_full_fm, other_full_fm)


def matching_fms(target_full_fm, inventory=None):
    '''
    Returns all feature matrices in inventory that match target and have
    distinct symbols.
    '''
    if inventory is None:
        inventory = IPA
    
    matches_target = is_matching_fm(target_full_fm)
    return set(filter(lambda an_fm: matches_target(an_fm) and
                      have_distinct_symbols(target_full_fm, an_fm),
                      inventory))


def non_unique_symbols(inventory=None):
    '''
    Returns all sets of entries in the specified inventory (or whatever portion
    of the IPA that's defined in the loaded feature matrix) that do not have
    distinct feature matrices.
    '''
    if inventory is None:
        inventory = IPA
    
    # map from each IPA symbol to the set of (distinct) IPA symbols
    #  that have identical feature matrices in the inventory
    matching_symbols = {s_fm['symbol']: matching_fms(s_fm, inventory) for s_fm in inventory}
    non_uniques = set([s for s in matching_symbols if len(matching_symbols[s]) > 0])
    return non_uniques


# ADDING AND REMOVING SYMBOLS TO AN INVENTORY

def addToInventory(new_feature_matrix, inventory=None, forceAdd=False):
    '''
    Adds new_feature_matrix to copy of specified/default inventory.

    If forceAdd is False (as by default), this will raise an error if
    new_feature_matrix matches something that already exists in the inventory
    (not including the 'symbol' field).
    '''
    if inventory is None:
        inventory = IPA
    
    missingFeatures = {feature for feature in getFeatures(inventory)
                       if feature not in new_feature_matrix.keys()}
    assert missingFeatures == set(), "New feature matrix must be defined on all features; missing {0}".format(missingFeatures)
    
    without_symbol = remove_symbol(new_feature_matrix)
    fm_matches = set([each for each in inventory
                     if set(without_symbol.keys()) == pf.equalizer(without_symbol,
                                                                   remove_symbol(each))])
    if fm_matches != set() and not forceAdd:
        raise Exception('New feature matrix matches features of existing entry in inventory: compare {0} with {1}'.format(without_symbol, fm_matches))
    
    fm_matches = set([each for each in inventory
                     if set(new_feature_matrix.keys()) in
                     pf.equalizer(new_feature_matrix, each)])
    if fm_matches != set() and not forceAdd:
        raise Exception('New feature matrix is a duplicate of an existing entry in inventory: compare {0} with {1}'.format(new_feature_matrix, fm_matches))
    
    new_inventory = {fm for fm in inventory}
    new_inventory.add(new_feature_matrix)
    return new_inventory


def addBoundarySymbols(left_boundary_symbol='⋊', right_boundary_symbol='⋉',
                       boundaryPattern=None, inventory=None):
    '''
    Adds the default/specified left and right word boundary symbols to a copy
    of default/specified inventory and then returns the inventory.

    If you want a single boundary symbol, enter the same symbol for both
    left_boundary_symbol and right_boundary_symbol. If you *don't*, functions
    elsewhere involving boundary symbols *may not work*.
    
    The default feature matrix (boundaryPattern) for both boundaries, unless
    otherwise specified, is '0' for all features. (This assumption - and the
    lack of appealing alternatives - is the source of difficulties elsewhere.)

    There is currently no special support in this package for morpheme
    boundary symbols; since morpheme and word boundary symbols are a bit of a
    hack in some ways and especially hacky in the context of phonetic feature
    matrices, it might be worth considering adding features specifically for
    different kinds of boundary symbols.
    '''
    if inventory is None:
        inventory = IPA
    if boundaryPattern is None:
        boundaryPattern = {k: '0' for k in getFeatures(inventory)}
    
    if left_boundary_symbol not in getSymbols(inventory):
        left_fm = boundaryPattern
        left_fm['symbol'] = left_boundary_symbol
        left_fm = FrozenDict(left_fm)
        inventory = addToInventory(left_fm, inventory=inventory, forceAdd=False)
    else:
        raise Exception('Left boundary symbol already in inventory.')
    
    if right_boundary_symbol != left_boundary_symbol:
        if right_boundary_symbol not in getSymbols(inventory):
            right_fm = boundaryPattern
            right_fm['symbol'] = right_boundary_symbol
            right_fm = FrozenDict(right_fm)
            inventory = addToInventory(right_fm, inventory=inventory, forceAdd=True)
        else:
            raise Exception('Right boundary symbol already in inventory.')
    
    return inventory


def removeFromInventory(pattern, inventory=None):
    '''
    Removes all feature matrices matching pattern from copy of
    specified/default inventory.
    '''
    if inventory is None:
        inventory = IPA
    
    new_inventory = {fm for fm in inventory
                        if not symbol_matches_pattern(fm['symbol'], pattern)}
    
    return new_inventory


###########################################
#  OPERATIONS ON A SINGLE FEATURE MATRIX  #
###########################################


def lookup_symbol(IPA_symbol, no_symbol=True, inventory=None,
                  null_behavior='error'):
    '''
    Retrieves the single feature matrix associated with IPA_symbol in the
    specified inventory (or whatever portion of the IPA that's defined in the
    loaded feature matrix).
    
    Will return the feature matrix without the 'symbol' field, unless no_symbol
    is False.
    
    Will throw an error if it discovers more than 1 matching feature matrix.
    
    If IPA_symbol doesn't have a match, null_behavior determines what this
    function does:
        - If null_behavior == 'exception' then this function will
          throw an error.
        - If null_behavior has any other value, this function will
          return that value.
    '''
    if inventory is None:
        inventory = IPA
    matches = set([each for each in inventory if each['symbol'] == IPA_symbol])
    assert len(matches) < 2, "Error - multiple matches for {0}! Check inventory and feature matrix for duplicates!".format(IPA_symbol)
    if len(matches) == 1:
        if no_symbol:
            return remove_symbol( list(matches)[0] )
        else:
            return list(matches)[0]
    else:
        if null_behavior == 'error':
            raise Exception('Error: {0} has no matching feature matrix.'.format(IPA_symbol))
        else:
            return null_behavior


def remove_symbol(feature_matrix):
    '''
    Returns a copy of feature_matrix without a symbol field.
    '''
    return pf.project_off(feature_matrix, ['symbol'])


def wordToFMs(IPAwordform, inventory=None):
    '''
    Returns a sequence of feature matrices corresponding to each of the IPA
    symbols in IPAwordform.
    '''
    if inventory is None:
        inventory = IPA
    return list(map(lambda symb: lookup_symbol(symb, inventory=inventory),
                    IPAwordform))


def symbol_matches_pattern(symbol, pattern):
    '''
    Returns true iff all feature values specified in pattern are specified in
    symbol as they are in pattern.
    '''
    extension = getMatches(pattern, asSymbol=True)
    return symbol in extension


def symbols_match_pattern_seq(symbols, patterns):
    '''
    Returns ture iff the ith symbol in symbols matches the ith pattern in
    patterns for all i.
    '''
    return all(starmap(symbol_matches_pattern, zip(symbols, patterns)))


########################################
#  OPERATIONS ON TWO FEATURE MATRICES  #
########################################


def common_feature_values(fm_a, fm_b):
    '''
    Returns just the feature values of fm_a and fm_b that have the same value
    across both.
    
    This is just a wrapper for
        remove_symbol( common_mappings(fm_a, fm_b) )
    That is, where equalizer(fm_a, fm_b) will give you the *keys* (≈ the
    features) that have the same values (but not those values) and
    common_mappings(fm_a, fm_b) would give the common keys and those values,
        (1) by default it will include the 'symbol' field of feature matrices,
        if present, which is not always desirable.
        (2) None of "equalizer", "differing_keys", or "common_mappings" are
        immediately suggestive titles to most users, so it doesn't hurt to have
        a convenient function that fills a likely/expected use case and is
        built through a simple composition of other functions and some of their
        optional functionality.
    '''
    return remove_symbol( pf.common_mappings(fm_a, fm_b) )


#############################################
#  OPERATIONS ON A SET OF FEATURE MATRICES  #
#############################################





############################################################
#  OPERATIONS ON AN INVENTORY AND A SINGLE FEATURE MATRIX  #
############################################################

# SELECTING ALL SYMBOLS MATCHING A PARTICULAR SET OF FEATURE VALUES


def getMatches(pattern, asSymbol=False, inventory=None, removeSymbol=True):
    '''
    Retrieves all feature matrix entries in the specified inventory (or
    whatever portion of the IPA that's defined in the loaded feature matrix)
    that match pattern.
    
    A pattern is a dictionary/frozen dictionary specifying keys and values that
    results must match.
    
    If asSymbol is True, will return all matching entries as IPA symbols.
    If asSymbol is False, will return all matching as feature matrices.
    
    If asSymbol is False and removeSymbol is True, will remove the symbol field
    from the retrieved feature matrices.
    '''
    if inventory is None:
        inventory = IPA
    fm_matches = set([each for each in inventory
                      if set(pattern.keys()) == pf.equalizer(pattern, each)])
    if not asSymbol:
        if removeSymbol:
#             return list(map(remove_symbol, fm_matches))
            return set(map(remove_symbol, fm_matches))
        return fm_matches
    else:
        symbol_matches = set([each['symbol'] for each in fm_matches])
        return symbol_matches


def lookup_fm(full_pattern, inventory=None):
    '''
    Assuming you think full_pattern matches exactly one symbol in the specified
    inventory (or whatever portion of the IPA that's defined in the loaded
    feature matrix), this will return exactly that symbol.
    
    A pattern is a dictionary/frozen dictionary specifying keys and values that
    results must match.
    '''
    if inventory is None:
        inventory = IPA
    fm_matches = set([each for each in inventory
                      if set(full_pattern.keys()) == pf.equalizer(full_pattern, each)])
    symbol_matches = set([each['symbol'] for each in fm_matches])
    assert len(symbol_matches) == 1, 'Expected exactly 1 match but {0} matches found: {1}'.format(len(symbol_matches), symbol_matches)
    match = list(symbol_matches)[0]
    return match


# There's **no problem** iff:
#  - The boundary symbols are **not** in the inventory.
#  - The boundary symbols **are** in the inventory but not in the wordform.
#  - The boundary symbols **are** in the inventory and in the wordform, but the
#    left and right boundary symbols are **the same**.
#  - The boundary symbols **are** in the inventory, in the wordform, and
#    there's two of them, **but their feature matrices are distinguishable
#    without reference to their symbol fields**.
#  - The boundary symbols **are** in the inventory, in the wordform, there's
#    two of them, their feature matrices aren't distinguishable without
#    reference to their symbol fields, **but in this particular wordform,
#    symbol fields are available.**

# There *is* a problem iff:
#  - There are exactly one or two boundary symbols in the inventory. (These are
#    the only possibilities I consider.)
#  - The left and right boundary symbols are distinct.
#  - One or more of those symbols are in the segment sequence represented by
#    fms.
#  - The feature matrices of the symbols are the same (except for the symbol
#    field) and symbol fields (which would disambiguate them) are not present.

def fmsToWord(fms, inventory=None, left_boundary_symbol='⋊',
                                   right_boundary_symbol='⋉',
                                   disambiguate=True):
    '''
    Returns the sequence of IPA symbols that corresponds to the sequence of
    feature matrices in fms.
    
    As noted elsewhere,
        - If there are boundary symbols in fms, they must already be present
          in the relevant inventory.
        - If there is only a single boundary symbol in the inventory, I
          assume that you will pass the same symbol to both
          left_boundary_symbol and right_boundary_symbol.
    
    Also, this function assumes that every feature matrix in fms can be
    uniquely mapped to a symbol, with the limited exception of boundary
    symbols. That is, iff
        (A) disambiguate=True.
        (B) Both specified/default boundary symbols are in the relevant
            inventory.
        (C) Both specified/default boundary symbols are distinct.
        (D) The feature matrices of the symbols are the same (except for the
        symbol field).
        (E) One or both of those symbols are in the segment sequence
        represented by fms.
        (F) The symbol fields (which would disambiguate them) are not
        present in fms.
    then technically the feature matrices for boundary symbols are *ambiguous*,
    so for convenience this function will map such a feature matrix at the left
    edge of fms to the left boundary symbol and a feature matrix at the right
    edge of fms to the right boundary symbol. If the conditions above hold and
    you pass a multi-word feature matrix sequence, feature matrices not at the
    left or right edge of fms will be seen as ambiguous and this function will
    choke. ('Parse' fms first!)
    
    Iff
        (2) Neither specified/default boundary symbol is in the relevant
            inventory.
    or
        (3) One or both of the specified/default boundary symbols *are* in the 
            relevant inventory, but neither is in fms.
    or
        (4) The specified/default boundary symbols aren't distinct. (I.e. there
        is really only one boundary symbol.)
    or
        (5) The specified/default boundary symbols are distinct and have
            distinct feature matrices in the relevant inventory (not including
            the symbol field).
    or
        (6) The specified/default boundary symbols are distinct, *don't* have
            distinct feature matrices in the relevant inventory, but the
            feature matrices in fms that correspond to boundary symbols have
            their symbol field included.
    then there is no ambiguity problem.
    '''
    if inventory is None:
        inventory = IPA

    # First priority is checking for whether nothing special needs to be done
    #  (= cases 2-5 above), something slightly special needs to be done (case
    #  6), or something very special needs to be done (iff case 1 = all of A-E
    #  are true).

    # Check for whether nothing special need be done (cases 2-5 above)...

    # Check for whether inventory has specified boundary symbols (case 2).
    left_boundary_fm = lookup_symbol(left_boundary_symbol,
                                     inventory=inventory, null_behavior=None)
    if left_boundary_fm is None:
        inv_has_left_boundary = False
    else:
        inv_has_left_boundary = True
    right_boundary_fm = lookup_symbol(right_boundary_symbol,
                                      inventory=inventory, null_behavior=None)
    if right_boundary_fm is None:
        inv_has_right_boundary = False
    else:
        inv_has_right_boundary = True
    # Case 2 is true iff not inv_has_both_boundary_symbols:
    inv_has_both_boundary_symbols = inv_has_left_boundary and inv_has_right_boundary
    inv_has_either_boundary_symbol = inv_has_left_boundary or inv_has_right_boundary
    case2 = not inv_has_either_boundary_symbol

    # Check for whether neither specified/default boundary symbol is in fms
    #   (case 3).
    fms_no_symbols = list(map(remove_symbol, fms))
    if inv_has_both_boundary_symbols:
        word_has_left_boundary_fm = left_boundary_fm in fms_no_symbols
        word_has_right_boundary_fm = right_boundary_fm in fms_no_symbols
        word_has_boundary_fms = word_has_left_boundary_fm or word_has_right_boundary_fm
        if not word_has_boundary_fms:
            case3 = True
        else:
            case3 = False
    else:
        case3 = False

    # Check for whether the specified/default boundary symbols aren't distinct.
    #   (I.e. there is really only one boundary symbol = case 4.)
    # We only need to check case 4 if case 2 is false and case 3 is false.
    boundary_symbols_are_distinct = left_boundary_symbol != right_boundary_symbol
    if (not case2) and (not case3):
        case4 = not boundary_symbols_are_distinct
    else:
        case4 = False

    # Check for whether the specified/default boundary symbols are distinct and
    #   have distinct feature matrices in the relevant inventory (not including
    #   the symbol field).
    # We only need to check case 5 if cases 2, 3, and 4 are false.
    feature_matrices_are_distinct = left_boundary_fm != right_boundary_fm
    if (not case2) and (not case3) and (not case4):
        case5 = feature_matrices_are_distinct
    else:
        case5 = False

    # Check whether the specified/default boundary symbols are distinct,
    #   *don't* have distinct feature matrices in the relevant inventory, but
    #   the feature matrices in fms that correspond to boundary symbols have
    #   their symbol field included.
    # We only need to check case 6 if cases 2, 3, 4, and 5 are false.
    if (not case2) and (not case3) and (not case4) and (not case5):
        matches_left_boundary_fm = lambda fmat: remove_symbol(fmat) == left_boundary_fm
        matches_right_boundary_fm = lambda fmat: remove_symbol(fmat) == right_boundary_fm
        matches_boundary_fm = lambda fmat: matches_left_boundary_fm(fmat) or matches_right_boundary_fm(fmat)
        fms_in_word_matching_boundaries = [fmat for fmat in fms if matches_boundary_fm(fmat)]
        has_symbol_field = lambda fmat: 'symbol' in fmat
        boundary_fms_in_word_have_symbols = all(map(has_symbol_field, fms_in_word_matching_boundaries))
        case6 = boundary_fms_in_word_have_symbols
    else:
        case6 = False

    # for i, case in zip(range(2,7), [case2, case3, case4, case5, case6]):
    #     if case:
    #         print('Case {0} true.'.format(i))

    # If any of cases 2-6 are true, then the feature matrices in fms are
    #  unambiguous and lookup_fm can be applied to each of them and be expected
    #  to behave as expected (i.e. return the unique IPA symbol associated with
    #  each feature matrix in fms).
    if case2 or case3 or case4 or case5 or case6:
        return ''.join(list(map(lambda fmat: lookup_fm(fmat, inventory=inventory),
                            fms)))

    # Check whether we're supposed to try to disambiguate fms.
    condA = disambiguate

    # Check whether the inventory has both specified/default boundary symbols.
    if condA:
        condB = inv_has_both_boundary_symbols
    else:
        condB = False

    # Check whether the specified/default boundary symbols are distinct.
    if condA and condB:
        condC = boundary_symbols_are_distinct
    else:
        condC = False

    # Check whether the specified/default boundary symbols have feature
    #   matrices that aren't distinct (except for the symbol field).
    if condA and condB and condC:
        condD = not feature_matrices_are_distinct
    else:
        condD = False

    # Check whether one or both of the boundary symbols are present in fms.
    if condA and condB and condC and condD:
        word_has_left_boundary_fm = left_boundary_fm in fms_no_symbols
        word_has_right_boundary_fm = right_boundary_fm in fms_no_symbols
        if word_has_left_boundary_fm or word_has_right_boundary_fm:
            condE = True
        else:
            condE = False
    else:
        condE = False

    # Check whether one or both of the specified/default boundary symbols are
    #   present in fms (SOMEwhere) *without* a defined 'symbol' field:
    matches_left_boundary_fm = lambda fmat: remove_symbol(fmat) == left_boundary_fm
    matches_right_boundary_fm = lambda fmat: remove_symbol(fmat) == right_boundary_fm
    matches_boundary_fm = lambda fmat: matches_left_boundary_fm(fmat) or matches_right_boundary_fm(fmat)
    has_symbol_field = lambda fmat: 'symbol' in fmat
    if condA and condB and condC and condD and condE:
        fms_in_word_matching_boundaries = [fmat for fmat in fms if matches_boundary_fm(fmat)]
        boundary_fms_in_word_have_symbols = all(map(has_symbol_field,
                                                    fms_in_word_matching_boundaries))
        condF = not boundary_fms_in_word_have_symbols
    else:
        condF = False

    # Check whether all ambiguous boundary symbol feature matrices are at the
    #   left or right edge of fms.
    if condA and condB and condC and condD and condE and condF:
        # Get indices of all feature matrices in fms matching the left and
        #   right boundary symbols and not containing a 'symbol' field.
        enum_fms = [(i, each) for i, each in enumerate(fms)]
        boundary_fms = [(i, each) for i, each in enum_fms
                                                 if matches_boundary_fm(each)]
        ambiguous_boundary_fms = [(i, each) for i, each in boundary_fms
                                                if not has_symbol_field(each)]

        # If the identified indices are anywhere other than the left and right
        #   edge of fms, then the boundary symbol fms are not disambiguatable.
        left_edge_index = 0
        right_edge_index = len(fms) - 1
        edge_indices = {left_edge_index, right_edge_index}
        ambiguous_boundary_fm_indices = {i for i, each in ambiguous_boundary_fms}
        condG = all([i in edge_indices for i in ambiguous_boundary_fm_indices])
    else:
        condG = False

    if condA and condB and condC and condD and condE and condF and condG:
        # Tear off the left and right edge fms, manually specify them as the
        #   passed boundary symbols, calculate the symbols for the rest of the
        #   word as usual, join the results together and return the results.
        if left_edge_index in ambiguous_boundary_fm_indices:
            left_pad = left_boundary_symbol
            rest_of_word = fms[1:]
        else:
            left_pad = ''
            rest_of_word = fms

        if right_edge_index in ambiguous_boundary_fm_indices:
            right_pad = right_boundary_symbol
            rest_of_word = rest_of_word[:-1]
        else:
            right_pad = ''
            # rest_of_word = rest_of_word

        rest_of_word_as_symbols = ''.join(list(map(lambda fmat: lookup_fm(fmat,
                                                                          inventory=inventory),
                                               rest_of_word)))

        return left_pad + rest_of_word_as_symbols + right_pad
    else:
        raise Exception("Unresolvably ambiguous boundary symbols in fms. Check this function's docstring.")


###########################################################
#  TESTING WHETHER A SET OF SYMBOLS FORM A NATURAL CLASS  #
###########################################################

def form_natural_class(symbolset, inventory=None):
    '''
    Returns true iff the given symbols form a natural class in the
    specified/default inventory.
    
    Symbolset can be a collection of IPA characters or a collection of feature
    matrices.
    '''
    if inventory is None:
        inventory = IPA
    if any(map(lambda s: s in getSymbols(inventory), symbolset)): #we were handed a collection of IPA characters...
        assert all(map(lambda s: s in getSymbols(inventory), symbolset)), '{0} contains non-IPA symbols.'.format(symbolset)
        fms = set(map(lambda s: lookup_symbol(s, inventory=inventory), symbolset))
    else:
        fms = symbolset
    common_fvs = reduce(common_feature_values, fms)
    fms_picked_out = set(getMatches(common_fvs, inventory=inventory))
    if fms_picked_out == fms:
        return True
    else:
        return False


#################################################################
#  FINDING THE MINIMAL SET OF FEATURES EQUIVALENT TO A PATTERN  #
#################################################################


def equal_extension(pattern_A, pattern_B, inventory=None):
    '''
    Returns true iff pattern_A and patternB have the same matches in the
    inventory (either the specified one or the default).
    '''
    if inventory is None:
        inventory = IPA
    matches_A = getMatches(pattern_A, True, inventory=inventory)
    matches_B = getMatches(pattern_B, True, inventory=inventory)
    return matches_A == matches_B
 

class MinimumEncoding(Problem):
    """
    Given a set of feature-value pairs p and a set S of possible segments, find
    a set of feature-value pairs p* as small as possible that still picks out
    the same subset of S as the original set of feature-value pairs p.
    
    As elsewhere, if no inventory is specified, the default one (the entire 
    IPA) is set as the inventory.
    
    If no background_pattern is offered, S is the inventory. If a
    background_pattern is offered, S is all elements of the inventory that 
    match background_pattern.

    Any patterns contained in excluded_patterns are ineligible for being
    returned as solutions.
    """
    
    def __init__(self, initial, goal_pattern_to_minimize,
                 background_pattern=None, inventory=None,
                 excluded_patterns=None):
#         Problem.__init__(self, initial)
        self.initial = initial
#         self.goal = goal
        if inventory is None:
            inventory = IPA
        if background_pattern is None:
            fms_to_check = inventory
        else:
            fms_to_check = getMatches(background_pattern, False,
                                      inventory=inventory)
        if excluded_patterns is None:
            excluded_patterns = set()
        self.inventory = inventory
        self.fms_to_check = fms_to_check
        self.goal_pattern = goal_pattern_to_minimize
        self.goal_keys = set(goal_pattern_to_minimize.keys())
        self.goal_extension = getMatches(goal_pattern_to_minimize, False,
                                         fms_to_check)
        self.excluded_states = excluded_patterns

    def actions(self, state):
#         choose_keys = lambda s: pf.differing_keys(s, self.goal_pattern, keys_to_compare = self.goal_keys)
#         keys_to_chose = choose_keys(state)
        make_action = lambda key: lambda s: pf.change_key(s, key,
                                                      self.goal_pattern[key])
        keys_to_chose = pf.differing_keys(state, self.goal_pattern,
                                          keys_to_compare=self.goal_keys)
        possible_actions = {make_action(key) for key in keys_to_chose}
        return possible_actions

    def result(self, state, action):
        return action(state)

    def goal_test(self, state):
        if (getMatches(state, False, self.fms_to_check) == self.goal_extension 
            and state not in self.excluded_states):
                return True
        return False

    def h(self, node):
        # 'distance from goal' = cost to minimize
        state = node.state
#         return -1.0 * self.value(state)
        matches = getMatches(state, False, self.fms_to_check)
        num_missing = len([each for each in self.goal_extension if each not in matches])
        num_extra = len([each for each in matches if each not in self.goal_extension])
        return (num_missing + num_extra)
    
    def value(self, state):
        # 'distance from goal' = cost to minimize
        matches = getMatches(state, False, self.fms_to_check)
        num_missing = len([each for each in self.goal_extension if each not in matches])
        num_extra = len([each for each in matches if each not in self.goal_extension])
        return 1.0 * (num_missing + num_extra)


# def minimize_pattern(pattern, background_pattern=None, inventory=None,
#                      excluded_patterns=None):
#     '''
#     Given an input pattern p to minimize, returns the smallest pattern p* that
#     is no bigger than the input (in general it will be smaller) that picks out
#     the same symbols as the input pattern in the context of background_pattern
#     (if specified) and the specified/default inventory.
    
#     If there are multiple patterns that are all equally best, this returns the
#     first one found.
    
#     If excluded_patterns is specified, then this will exclude all listed
#     patterns from consideration to be returned.
#     '''
#     if inventory is None:
#         inventory = IPA
#     if excluded_patterns is None:
#         excluded_patterns = set()
#     if pattern is None or pattern == dict():
#         return pattern
#     if len( getMatches(pattern, inventory) ) == 0:
#         raise Exception('Pattern does not pick out any elements in (background_pattern of) inventory!')
#     return uniform_cost_search(MinimumEncoding(FrozenDict({}), pattern,
#                                                background_pattern=background_pattern,
#                                                inventory=inventory,
#                                                excluded_patterns=excluded_patterns)).state


def get_minimal_patterns(pattern, background_pattern=None, inventory=None,
                         excluded_patterns=None):
    '''
    Given an input pattern to minimize, returns an iterator that yields each
    pattern no bigger than the input (in general it will be smaller) that picks
    out the same symbols as the input pattern in the context of
    background_pattern (if specified) and the specified/default inventory.
    
    If excluded_patterns is specified, then this will exclude all listed
    patterns from consideration to be returned.
    '''
    if inventory is None:
        inventory = IPA
    if excluded_patterns is None:
        excluded_patterns = set()
    
    while minimize_pattern(pattern, background_pattern, inventory,
                           excluded_patterns) is not None:
        next_minimal_pattern = minimize_pattern(pattern, background_pattern, 
                                                inventory, excluded_patterns)
#         print('Excluded patterns before: {0}'.format(excluded_patterns))
        excluded_patterns.add(next_minimal_pattern)
#         print('Excluded patterns after: {0}'.format(excluded_patterns)
        yield next_minimal_pattern


def minimize_pattern(pattern, background_pattern=None, inventory=None,
                     excluded_patterns=None):
    '''
    Given an input pattern to minimize, returns the smallest pattern that is no
    bigger than the input (in general it will be smaller) that picks out the
    same symbols as the input pattern in the context of background_pattern (if
    specified) and the specified/default inventory.
    
    If there are multiple patterns that are all equally best, this returns the
    first one found.
    
    If excluded_patterns is specified, then this will exclude all listed
    patterns from consideration to be returned.
    '''
    if inventory is None:
        inventory = IPA
    if excluded_patterns is None:
        excluded_patterns = set()
    if pattern is None or pattern == dict():
        return pattern
    if background_pattern is not None:
        fms_to_check = getMatches(background_pattern, inventory=inventory)
    else:
        fms_to_check = inventory
    if len( getMatches(pattern, inventory = fms_to_check) ) == 0:
        raise Exception('Pattern does not pick out any elements in (background_pattern of) inventory!')
    if len(fms_to_check) == 1:
        raise Exception('Only one element - every feature value picks out one thing or no thing.')
        #This should perhaps be a warning rather than an exception...
    
#     return uniform_cost_search(MinimumEncoding(FrozenDict({}), pattern, background_pattern = background_pattern, inventory = inventory, excluded_patterns = excluded_patterns)).state
    return astar_search(MinimumEncoding(FrozenDict({}), pattern,
                                        background_pattern=background_pattern,
                                        inventory=inventory,
                                        excluded_patterns=excluded_patterns)).state


########################################################
#  DETERMINING THE EFFECTS OF CHANGING FEATURE VALUES  #
########################################################


def literalChange(pattern, changes):
    '''
    Takes a pattern (dict/frozen dict specifying keys and values) and makes
    exactly the specified changes to that pattern.
    
    If changes is also a pattern, the input pattern is coerced to match. If
    changes is a function, then it is applied to the input pattern.
    '''
    if callable(changes):
        return changes(pattern)
    
    if isinstance(changes, dict) or isinstance(changes, FrozenDict):
        newDict = {k: pattern[k] for k in pattern}
        newDict.update(changes)

        if isinstance(pattern, FrozenDict):
            newDict = FrozenDict(newDict)
        return newDict
    
    raise Exception('Changes must be a function or a dict/frozen dict: got {0} of type {1}'.format(changes, type(changes)))


def literalFeatureFlipper(feature):
    '''
    Returns a function that takes a pattern and returns a version of it where
    feature is flipped to its opposite:
        '+'   ⟶ '-'
        '-'   ⟶ '+'
        '0'   ⟶ '0'
        True  ⟶ False
        False ⟶ True
    '''
    return lambda pattern: pf.change_key(pattern, feature,
                                     opposite(pattern[feature]))


def image_of_pattern(change, pattern_to_apply_to=None, inventory=None):
    '''
    Applies literal change to each symbol matching pattern_to_apply_to (if any
    is specified, otherwise applies to each symbol/feature matrix in specified 
    or default inventory). Returns feature matrices that result.
    '''
    if inventory is None:
        inventory = IPA
    if pattern_to_apply_to is None:
        symbols_to_apply_to = set(map(remove_symbol, inventory))
    else:
        symbols_to_apply_to = getMatches(pattern_to_apply_to, asSymbol=False,
                                         inventory=inventory, removeSymbol=True)
    results = {s_in: literalChange(s_in, change) for s_in in symbols_to_apply_to}
    image = set(results.values())
    return image


def test_IO_relation(change, predicate, pattern_to_apply_to=None,
                     inventory=None):
    '''
    If pattern_to_apply_to is specified, applies literal change to all symbols
    matching pattern_to_apply_to; if pattern_to_apply_to is not specified,
    applies change to all elements of the inventory (if specified, otherwise it
    defaults to the entire inventory).
    
    The function then returns whether it is always the case that the specified
    predicate holds between each symbol changed and the result of applying the
    change to that symbol.
    '''
    if inventory is None:
        inventory = IPA
    if pattern_to_apply_to is None:
        symbols_to_apply_to = set(map(remove_symbol, inventory))
    else:
        symbols_to_apply_to = getMatches(pattern_to_apply_to, asSymbol=False,
                                         inventory=inventory, removeSymbol=True)
    results = {s_in: literalChange(s_in, change) for s_in in symbols_to_apply_to}
    predicate_tests = {s_in: predicate(s_in, results[s_in]) for s_in in symbols_to_apply_to}
    if not all(set(predicate_tests.values())):
        return False
    return True


def change_defines_function(change, pattern_in=None, pattern_out=None,
                            inventory=None):
    '''
    Returns true iff literal change is a function from pattern_in (or the
    specified/default inventory) to pattern_out (or the specified/default
    inventory), i.e. iff applying the literal change to each symbol matching
    pattern_in (or the specified/default inventory) results in a symbol
    matching pattern_out (or a symbol in the specified/default inventory, if no
    pattern_out is provided).
    '''
    if inventory is None:
        inventory = IPA
    if pattern_in is None:
        fms_in = set(map(remove_symbol, inventory))
    else:
        fms_in = getMatches(pattern_in, asSymbol=False, inventory=inventory,
                            removeSymbol=True)
    if pattern_out is None:
        fms_out = set(map(remove_symbol, inventory))
    else:
        fms_out = getMatches(pattern_out, asSymbol=False, inventory=inventory,
                             removeSymbol=True)
    results = {fm_in: literalChange(fm_in, change) for fm_in in fms_in}
    result_in_fms_out = lambda fm: fm in fms_out
    closed = list(map(result_in_fms_out, list(results.values())))
    return all(closed)


#########################################################################
#  CONSTRUCTING RULE-LIKE FUNCTIONS ON (SEQUENCES OF) FEATURE MATRICES  #
#########################################################################


def padWordformWithBoundarySymbols(wordform, left_boundary_symbol='⋊',
                                   right_boundary_symbol='⋉',
                                   inventory=None):
    '''
    Returns a copy of wordform (sequence of symbols or feature matrices from
    the default/specified inventory) padded with the default/specified boundary
    symbols.

    Note that the inventory in question must already contain boundary symbols,
    and that whatever your boundary symbols are, the boundary symbols passed
    here must match.

    Note also that if you don't have distinct left and right boundary symbols
    that you should pass the same symbol as the value for both the left and
    right boundary symbol arguments. By "should", I mean that if you *don't*,
    *things involving boundary symbols will not work*.
    '''
    if inventory is None:
        inventory = IPA
    
    if wordform[0] in getSymbols(inventory):
        # wordform is a symbol sequence
        if (left_boundary_symbol not in getSymbols(inventory)) or (right_boundary_symbol not in getSymbols(inventory)):
            raise Exception('Add boundary symbols to inventory first.')
    elif getMatches(wordform[0], inventory=inventory) != set():
        # wordform is a feature matrix sequence
        if 'symbol' in set(wordform[0].keys()):
            left_fm = lookup_symbol(left_boundary_symbol,
                                    no_symbol=False,
                                    inventory=inventory)
            right_fm = lookup_symbol(right_boundary_symbol,
                                     no_symbol=False,
                                     inventory=inventory)
        else:
            left_fm = lookup_symbol(left_boundary_symbol,
                                    no_symbol=True,
                                    inventory=inventory)
            right_fm = lookup_symbol(right_boundary_symbol,
                                     no_symbol=True,
                                     inventory=inventory)
    else:
        raise Exception('Wordform must either consist of recognized IPA symbols or feature matrices in the inventory: instead received {0}'.format(wordform))    
    
    if wordform[0] in getSymbols(inventory):
        return left_boundary_symbol + wordform + right_boundary_symbol
    elif getMatches(wordform[0], inventory=inventory) != set():
        return [left_fm] + wordform + [right_fm]
    else:
        raise Exception('Wordform must either consist of recognized IPA symbols or feature matrices in the inventory: instead received {0}'.format(wordform))


def concatenateLanguages(lang_a, lang_b):
    '''
    The concatenation of two formal languages L_a, L_b is the result of
    concatenating every string in L_a with every string in L_b:
        L_a * L_b = starmap(*, product(L_a, L_b))
    where * is string concatenation.
    '''
    concat = lambda a, b: a + b
    return set(starmap(concat, product(lang_a, lang_b)))


def concatenatePatterns(pattern_a, pattern_b, background_pattern=None,
                        inventory=None):
    '''
    The concatenation of two formal languages L_a, L_b is the result of
    concatenating every string in L_a with every string in L_b:
        L_a * L_b = starmap(*, product(L_a, L_b))
    where * is string concatenation.
    
    This function performs language (stringset) concatenation for the case of
    two string (symbol) sets defined by two patterns. The extension of each
    pattern (as elsewhere) is determined by the specified/default inventory and
    the background_pattern (if any).
    
    Use with reduce to concatenate more than two patterns:
        reduce(concatenatePatterns, [{'strid':'+'}, {'round':'+'}, {'c.g.':'+'}])
        reduce(lambda pat_a, pat_b: concatenatePatterns(pat_a, pat_b, inventory = ame), 
                                                        [{'strid':'+'}, {'round':'+'}, 
                                                         {'c.g.':'+'}])
    '''
    if inventory is None:
        inventory = IPA
    if background_pattern is None:
        fms_to_check = inventory
    else:
        fms_to_check = getMatches(background_pattern, asSymbol=False,
                                  removeSymbol=False, inventory=inventory)
    # this permits using concatenatePatterns with reduce
    if isinstance(pattern_a, set):
        symbols_a = pattern_a
    else:
        symbols_a = getMatches(pattern_a, asSymbol=True,
                               inventory=fms_to_check)
    
    symbols_b = getMatches(pattern_b, asSymbol=True,
                           inventory=fms_to_check)
    return concatenateLanguages(symbols_a, symbols_b)


def patternSequenceToStringSet(patterns, background_pattern=None,
                               inventory=None):
    '''
    Given a finite sequence (= tuple or list or iterator) of patterns
    representing a set of possible string sequences, returns the set of actual
    strings matching this sequence, where symbols matching the patterns must be
    found in the specified/default inventory or (if background_pattern is
    specified) in the subset of the specified/default inventory satisfying
    background_pattern.
    
    Use with caution.
    
    FIXME TODO: make this return an iterator so it can (in principle) generate
    samples from infinite stringsets / from an iterator of patterns...
    '''
    my_concat = lambda pat_a, pat_b: concatenatePatterns(pat_a, pat_b,
                                                         background_pattern=background_pattern,
                                                         inventory=inventory)
    return reduce(my_concat, patterns)


def patternToRegex(pattern, background_pattern=None, inventory=None):
    '''
    Given a pattern, finds all matching symbols in the specified/default
    inventory (or the subset of that matching background_pattern, if specified),
    compiles them into a string and then wraps them in '[' and ']' for use in
    constructing regular expressions from patterns.
    
    E.g. patternToRegex({'s.g.':'+'}) will return "[hɦ]"
         patternToRegex({'syll':'+'}, inventory = ame) will return 
         "[ʉœɒwɵoʏyøɶuɞɔɥ]"
    
    Such strings can then be used with the standard Python re library.
    '''
    if inventory is None:
        inventory = IPA
    if background_pattern is None:
        fms_to_check = inventory
    else:
        fms_to_check = getMatches(background_pattern, asSymbol=False,
                                  removeSymbol=False, inventory=inventory)
    symbols_matching_pattern = getMatches(pattern, asSymbol=True,
                                          inventory=fms_to_check)
    symbol_string = ''.join(symbols_matching_pattern)
    match_string  = '[' + symbol_string + ']'
    return match_string


def patternSeqToRegex(patternRegex, inventory=None):
    '''
    patternRegex must be a sequence (i.e. tuple or list or iterator) where each 
    element is either a pattern or one of the regular expression operators 
    '|', '*', '+', '?','(', ')', or '(?:'.
    For example,
        [{'s.g.':'+'}, '+',
         '(?:', {'round':'+', 'syll':'+'}, '|', {'voice':'-', 'cont':'+'}, ')','?']
    will return the string (for use in Python regular expressions)
        '[hɦ]+(?:[ʉœɒɵoʏyøɶɞuɔ]|[ʃçfxhsʂχθħɸ])?'
    which will match 1 or more instances of any IPA symbol with +s.g. followed
    by either exactly zero or exactly one instance of either a round vowel or a
    voiceless continuant.
    '''
    if inventory is None:
        inventory = IPA
    operators = set(['|','*','+','?','(',')','(?:'])

    # Ensure that patternRegex is well-formed
    assert all(map(lambda term: isinstance(term, dict) or isinstance(term, FrozenDict) or term in operators,
                   patternRegex)), "Every element of patternRegex must either be a pattern or an operator: {0}".format(patternRegex)

    # Ensure that every pattern is a FrozenDict
    patternRegex = list(map(lambda item: FrozenDict(item)
                            if isinstance(item, dict) or isinstance(item, FrozenDict) else item,
                            patternRegex))

    # Remap each pattern to a regular expression over the appropriate inventory
    remap = lambda eachItem: patternToRegex(eachItem, inventory=inventory) \
                             if eachItem not in operators else eachItem
    # remap = lambda eachItem: eachItem if eachItem in operators else patternToRegex(eachItem, inventory = inventory)
    regexStrings = list(map(remap, patternRegex))
    return ''.join(regexStrings)


def regexRule(patternRegex, replacementFunction, inventory=None):
    '''
    patternRegex must be a sequence (i.e. tuple or list or iterator) where each
    element is either a pattern or one of the regular expression operators
    '|', '*', '+', '?','(', ')', or '(?:'.
    For example,
        ['(', {'son':'-'}, ')', '(', '⋉', ')']
    will result in the string (for use in Python regular expressions)
        '([βqtʈdçɟɦɢʒfʦɖkɡvt̪cʁθʔʐhsʃʕzbxʂʧɣħʝʣd̪χðʤpɸ])(⋉)'
    which will match any substring containing a sonorant followed by the word-
    boundary symbol; it will also put the sonorant in one capture group in the
    match object, and the word-boundary symbol in another capture group.

    replacementFunction must be a function on regular expression match objects
    that returns the string to replace any given matching substring with.
    
    This function will return a function that replaces any substring matching
    patternRegex (in the default or specified inventory) according to
    replacementFunction.
    '''
    if inventory is None:
        inventory = IPA
    patternToMatch = patternSeqToRegex(patternRegex, inventory)
    return lambda substring: re.sub(patternToMatch, replacementFunction,
                                    substring)


def starFreeRule(target, change, left_env, right_env, inventory, toDict):
    '''
    \FIXME
    '''
    if inventory is None:
        inventory = IPA
    
#     target_fms = getMatches(target, asSymbol = False, inventory = inventory)
    assert change_defines_function(change,
                                   pattern_in=target,
                                   inventory=inventory), "Applying change to at least one element doesn't result in a unique symbol in inventory."
    
    LHS = left_env[:]
    LHS.extend([target])
    LHS.extend(right_env[:])
    print('LHS: {0}'.format(LHS)) #FIXME
    
    if not toDict:
        def rule(substring):
            if len(substring) != sum(map(len, LHS)):
                return substring
            if not symbols_match_pattern_seq(substring, LHS):
                return substring
            
            left_env_substring = substring[0:len(left_env)]
            
            target_symbol = substring[len(left_env)]
            changed_target = literalChange(lookup_symbol(target_symbol,
                                                         inventory=inventory),
                                           change)
            changed_target_symbol = lookup_fm(changed_target,
                                              inventory=inventory)
            
            right_env_substring = substring[len(left_env) + 1:]
            
            return left_env_substring + changed_target_symbol + right_env_substring
        return rule
    else:
        my_concat = lambda pat_a, pat_b: concatenatePatterns(pat_a, pat_b, inventory=inventory)
        left_env_strings = reduce(my_concat, left_env, set())
        print('Left env strings:') #FIXME
        print(left_env_strings)
        target_strings = getMatches(target, asSymbol=True, inventory=inventory)
        print('Target strings:') #FIXME
        print(target_strings)
        right_env_strings = reduce(my_concat, right_env, set())
        print('Right env strings:') #FIXME
        print(right_env_strings)
        makeLHS = lambda l, t, r: l + t + r
        # FIXME t should be replaced according to change...
        makeRHS = lambda l, t, r: l + lookup_fm(lookup_symbol(t, inventory=inventory), inventory=inventory) + r
        explicit_mapping = {makeLHS(l, t, r):makeRHS(l, t, r) for l, t, r in product(left_env_strings, target_strings, right_env_strings)}
#         explicit_mapping = dict()
#         for l,t,r in product(left_env_strings, target_strings, right_env_strings):
#             lhs = makeLHS(l, t, r)
#             rhs = makeRHS(l, t, r)
#             explicit_mapping[lhs] = rhs
        return explicit_mapping
