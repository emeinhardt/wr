#alignment related commands and paths

#WORKFLOW: run each shell command, then open up the relevant notebook and finish it, as described there

#GD_AmE_destressed_aligned_w_LTR_Buckeye/AmE-diphones-IPA-annotated-columns.csv
#LTR_Buckeye_aligned_w_GD_AmE_destressed/LTR_buckeye.tsv

papermill "Gating Data - Transcription Lexicon Alignment Maker.ipynb" "AmE-diphones - LTR_Buckeye alignment.ipynb" -p g GD_AmE_destressed_aligned_w_LTR_Buckeye/AmE-diphones-IPA-annotated-columns.csv -p l LTR_Buckeye_aligned_w_GD_AmE_destressed/LTR_buckeye.tsv



#GD_AmE_destressed_aligned_w_LTR_CMU_destressed/AmE-diphones-IPA-annotated-columns.csv
#LTR_CMU_destressed_aligned_w_GD_AmE_destressed/LTR_CMU_destressed.tsv

papermill "Gating Data - Transcription Lexicon Alignment Maker.ipynb" "AmE-diphones - LTR_CMU_destressed alignment.ipynb" -p g GD_AmE_destressed_aligned_w_LTR_CMU_destressed/AmE-diphones-IPA-annotated-columns.csv -p l LTR_CMU_destressed_aligned_w_GD_AmE_destressed/LTR_CMU_destressed.tsv


#GD_AmE_destressed_aligned_w_LTR_newdic_destressed/AmE-diphones-IPA-annotated-columns.csv
#LTR_newdic_destressed_aligned_w_GD_AmE_destressed/LTR_newdic_destressed.tsv

papermill "Gating Data - Transcription Lexicon Alignment Maker.ipynb" "AmE-diphones - LTR_newdic_destressed alignment.ipynb" -p g GD_AmE_destressed_aligned_w_LTR_newdic_destressed/AmE-diphones-IPA-annotated-columns.csv -p l LTR_newdic_destressed_aligned_w_GD_AmE_destressed/LTR_newdic_destressed.tsv