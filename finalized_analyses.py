# all analyses here were run in "Analysis - Lower Lambdas - Buckeye.ipynb"

# I. UNIGRAM POSTERIOR ANALYSES

## A 
#unigram posterior; all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 1, direction = 'uni', pc = 0.01, scale_factor = 0.015625, include_main_post = True,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_w_pc0dot01_sf0dot015625_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate

## B
my_analysis = duration_analysis(avg_post   = False, order = 1, direction = 'uni', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = True,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_w_pc0dot01_sf0dot015625__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate




## BIGRAM PRIORS in BOTH directions

## C
#unigram posterior; all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = True, 
                                include_unigram_lm = True,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = True, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_bwd1_ + h_w_given_fwd1_ + h_w_given_w_pc0dot01_sf0dot015625_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## D
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = True, include_unigram_lm_as_rank = True,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = True,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_bwd1__rank + h_w_given_fwd1__rank + h_w_given_w_pc0dot01_sf0dot015625__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate

# II. UNIGRAM POSTERIOR ANALYSES WITHOUT CLASS FEATURES - TENTATIVE

## A'
#unigram posterior; all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 1, direction = 'uni', pc = 0.01, scale_factor = 0.015625, include_main_post = True,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_w_pc0dot01_sf0dot015625_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + speaker_name + isAdj + isAdv + isV + isN + speech_rate

## B'
my_analysis = duration_analysis(avg_post   = False, order = 1, direction = 'uni', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = True,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_w_pc0dot01_sf0dot015625__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + speaker_name + isAdj + isAdv + isV + isN + speech_rate




## BIGRAM PRIORS in BOTH directions
## C'
#unigram posterior; all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = True, 
                                include_unigram_lm = True,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = True, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_bwd1_ + h_w_given_fwd1_ + h_w_given_w_pc0dot01_sf0dot015625_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## D'
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = True, include_unigram_lm_as_rank = True,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = True,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_bwd1__rank + h_w_given_fwd1__rank + h_w_given_w_pc0dot01_sf0dot015625__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + speaker_name + isAdj + isAdv + isV + isN + speech_rate


# III. UNIGRAM POSTERIOR ANALYSES BUT
# no unigram posterior
# backwards bigram posterior only
#
# no unigram prior
# both bigram priors
# = drop A'' and B''


## C''
#all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = True,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = True, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625_ + h_w_given_bwd1_ + h_w_given_fwd1_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## D''
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = True,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = True,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625__rank + h_w_given_bwd1__rank + h_w_given_fwd1__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## ADD UNIGRAM PRIOR ON TOP OF C'' and D''

## E''
#all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = True,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = False, 
                                include_unigram_lm = True,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = True, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625_ + h_w_given_bwd1_ + h_w_given_fwd1_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## F''
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = False, #new features start here
                                include_classBag_features = True,
                                include_main_post_as_rank    = True,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = True,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = True,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625__rank + h_w_given_bwd1__rank + h_w_given_fwd1__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + voiced_oral_stop_or_affricate + sonorant_consonant + voiceless_stop_or_affricate + vowel + voiced_fricative + voiceless_fricative + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## SAME AS E'' and F'' BUT SWAP OUT BROAD NATURAL/SONORITY CLASSES FOR FINE-GRAINED ONES

## G''
#all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = True,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = False, 
                                include_unigram_lm = True,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = True, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = True, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625_ + h_w_given_bwd1_ + h_w_given_fwd1_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + ɹ + k + ɪ + f + d + u + w + b + g + h + aɪ + v + ʌ + i + ʊ + ð + m + s + ɚ + dʒ + p + tʃ + oʊ + n + æ + z + eɪ + j + ɔɪ + θ + t + ɛ + ŋ + aʊ + ɑ + l + ʒ + ʃ + speaker_name + isAdj + isAdv + isV + isN + speech_rate


# H''
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = True, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = True,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = True,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = True,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625__rank + h_w_given_bwd1__rank + h_w_given_fwd1__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + ɹ + k + ɪ + f + d + u + w + b + g + h + aɪ + v + ʌ + i + ʊ + ð + m + s + ɚ + dʒ + p + tʃ + oʊ + n + æ + z + eɪ + j + ɔɪ + θ + t + ɛ + ŋ + aʊ + ɑ + l + ʒ + ʃ + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## SAME AS G'' and H'' but with FORWARD BIGRAM POSTERIOR INSTEAD OF BWD BIGRAM POSTERIOR

## I''
#all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'fwd', pc = 0.01, scale_factor = 0.015625, include_main_post = True,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = False, 
                                include_unigram_lm = True,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = True, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = True, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_w_fwd1_pc0dot01_sf0dot015625_ + h_w_given_fwd1_ + h_w_given_bwd1_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + ɹ + k + ɪ + f + d + u + w + b + g + h + aɪ + v + ʌ + i + ʊ + ð + m + s + ɚ + dʒ + p + tʃ + oʊ + n + æ + z + eɪ + j + ɔɪ + θ + t + ɛ + ŋ + aʊ + ɑ + l + ʒ + ʃ + speaker_name + isAdj + isAdv + isV + isN + speech_rate


# J''
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'fwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = True, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = True,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = True,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = True,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_w_fwd1_pc0dot01_sf0dot015625__rank + h_w_given_fwd1__rank + h_w_given_bwd1__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + ɹ + k + ɪ + f + d + u + w + b + g + h + aɪ + v + ʌ + i + ʊ + ð + m + s + ɚ + dʒ + p + tʃ + oʊ + n + æ + z + eɪ + j + ɔɪ + θ + t + ɛ + ŋ + aʊ + ɑ + l + ʒ + ʃ + speaker_name + isAdj + isAdv + isV + isN + speech_rate


## G'' AND H'' BUT INCLUDING UNIGRAM POSTERIOR

#K''
#all continuous variables
# 
# 
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = True,
                                avg_lm     = False, include_lm = True, 
                                include_unigram_post   = True, 
                                include_unigram_lm = True,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = True, 
                                include_NS     = True, include_lWND = True, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = True, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = False,
                                include_main_lm_as_rank      = False, 
                                include_unigram_post_as_rank = False, include_unigram_lm_as_rank = False,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = False,
                                include_NS_as_rank = False, include_lWND_as_rank = False,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = False)
# my_analysis.keys()
# Formula:
# log_duration ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625_ + h_w_given_bwd1_ + h_w_given_fwd1_ + h_w_given_w_pc0dot01_sf0dot015625_ + h_w_ + neighborhood_size + weighted_neighborhood_density + syll_seg_cat + ɹ + k + ɪ + f + d + u + w + b + g + h + aɪ + v + ʌ + i + ʊ + ð + m + s + ɚ + dʒ + p + tʃ + oʊ + n + æ + z + eɪ + j + ɔɪ + θ + t + ɛ + ŋ + aʊ + ɑ + l + ʒ + ʃ + speaker_name + isAdj + isAdv + isV + isN + speech_rate


# L''
# 
my_analysis = duration_analysis(avg_post   = False, order = 2, direction = 'bwd', pc = 0.01, scale_factor = 0.015625, include_main_post = False,
                                avg_lm     = False, include_lm = False, 
                                include_unigram_post   = False, 
                                include_unigram_lm = False,
                                include_other_post_dir = False, 
                                include_other_lm_dir   = False, 
                                include_NS     = False, include_lWND = False, 
                                include_NS_cat = False, 
                                include_phones_length     = False,
                                include_phones_length_cat = False,
                                include_syll_length       = False,
                                include_syll_length_cat   = False,
                                include_seg_syll_length   = False,
                                include_seg_syll_length_cat = True,
                                include_phones_syll_interaction = False, #added after the fact
                                include_speaker_id = True, include_POS = True, include_speech_rate = True,
                                include_segBag_features = True, #new features start here
                                include_classBag_features = False,
                                include_main_post_as_rank    = True,
                                include_main_lm_as_rank      = True, 
                                include_unigram_post_as_rank = True, include_unigram_lm_as_rank = True,
                                include_other_post_dir_as_rank = False, include_other_lm_dir_as_rank = True,
                                include_NS_as_rank = True, include_lWND_as_rank = True,
                                include_phones_length_as_rank = False, include_syll_length_as_rank = False,
                                include_speech_rate_as_rank = False,
                                include_log_duration_as_rank = True)
# my_analysis.keys()
# Formula:
# log_duration_rank ~ h_w_given_w_bwd1_pc0dot01_sf0dot015625__rank + h_w_given_bwd1__rank + h_w_given_fwd1__rank + h_w_given_w_pc0dot01_sf0dot015625__rank + h_w__rank + neighborhood_size_rank + weighted_neighborhood_density_rank + syll_seg_cat + ɹ + k + ɪ + f + d + u + w + b + g + h + aɪ + v + ʌ + i + ʊ + ð + m + s + ɚ + dʒ + p + tʃ + oʊ + n + æ + z + eɪ + j + ɔɪ + θ + t + ɛ + ŋ + aʊ + ɑ + l + ʒ + ʃ + speaker_name + isAdj + isAdv + isV + isN + speech_rate


# "PRIMARY ANALYSES" = G'' and H'' and K'' and L''
