#!/bin/bash


for MODEL in 'p-cons' 't-cons' 'k-cons' 'b-cons' 'd-cons' 'g-cons' 'S-cons' 'f-cons' 's-cons' 'v-cons' 'Z-cons' 'm-cons' 'n-cons' 'l-cons' 'R-cons' 'z-cons'; do

	echo "---------------------- $MODEL ----------------------"
	
	echo "p-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/p_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "t-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/t_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "k-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/k_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "b-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/b_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "d-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/d_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "g-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/g_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "S-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/S_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "f-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/f_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "s-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/s_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "v-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/v_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "Z-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/Z_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "m-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/m_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "n-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/n_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "l-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/l_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "R-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/R_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth
	echo "z-cons"
	python test_phonemes.py ./ptest_mfcc6/$MODEL/ ./reference/z_filenames.csv ./checkpoints/pretrained_models/$MODEL.pth

done
