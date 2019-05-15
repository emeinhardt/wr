#alignment related commands and paths

#WORKFLOW: run each shell command, then open up the relevant notebook and finish it, as described there

# GD-AmE/AmE-diphones-IPA-annotated-columns.csv
# LTR_Buckeye/LTR_buckeye.tsv

papermill "Gating Data - Transcription Lexicon Alignment Maker.ipynb" "AmE-diphones - LTR_Buckeye alignment.ipynb" -p g GD-AmE/AmE-diphones-IPA-annotated-columns.csv -p l LTR_Buckeye/LTR_buckeye.tsv



# GD-AmE/AmE-diphones-IPA-annotated-columns.csv
# LTR_CMU_destressed/LTR_CMU_destressed.tsv

papermill "Gating Data - Transcription Lexicon Alignment Maker.ipynb" "AmE-diphones - LTR_CMU_destressed alignment.ipynb" -p g GD-AmE/AmE-diphones-IPA-annotated-columns.csv -p l LTR_CMU_destressed/LTR_CMU_destressed.tsv


# GD-AmE/AmE-diphones-IPA-annotated-columns.csv
# LTR_newdic_destressed/LTR_newdic_destressed.tsv

papermill "Gating Data - Transcription Lexicon Alignment Maker.ipynb" "AmE-diphones - LTR_newdic_destressed alignment.ipynb" -p g GD-AmE/AmE-diphones-IPA-annotated-columns.csv -p l LTR_newdic_destressed/LTR_newdic_destressed.tsv