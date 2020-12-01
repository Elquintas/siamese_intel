#!/bin/bash

for PATIENT in 'ANC150' 'AVG063' 'BEH066' 'BIV111' 'BOF124' 'BOM094' 'BOM123' 'BOM148' 'BOP097' 'BOT138' 'CAA116' 'CAA143' 'CAF128' 'CAM101' 'CNH136' 'COC005' 'COC137' 'COM100' 'DEG119' 'DES144' 'FAS009' 'FOD065' 'FOD132' 'GAC073' 'GAC115' 'GAV008' 'GOF131' 'GUM099' 'GUY016' 'HEB114' 'JAC104' 'JAR011' 'JEJ077' 'JOP013' 'JOP070' 'JUM122' 'KUN026' 'LAA110' 'LAA133' 'LAJ002' 'LAM113' 'LAM118' 'LEA103' 'LEH135' 'LEM015' 'LEY012' 'LEY067' 'MAC126' 'MAF006' 'MAX068' 'MER069' 'MIH091' 'MOA112' 'MOS096' 'NAC105' 'NIG145' 'NIR008' 'NOF004' 'OBJ087' 'OLD003' 'ORE062' 'PEJ149' 'PEM074' 'PIP072' 'PLF078' 'PRA142' 'PRG014' 'QUD071' 'REA127' 'README' 'RIA120' 'ROA141' 'ROD003' 'ROD121' 'ROJ146' 'RUM139' 'SAL108' 'SER007' 'SOJ125' 'SOM147' 'THN010' 'THS140' 'TTP001' 'TTP004' 'TTP005' 'TTP043' 'TTP044' 'WAA107' 'ZAJ130'; do


	echo "****************************** $PATIENT ***********************************"
	echo "p-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/p-cons/ ./reference/p_filenames.csv ./checkpoints/pretrained_models/p-cons.pth
	echo "t-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/t-cons/ ./reference/t_filenames.csv ./checkpoints/pretrained_models/t-cons.pth
	echo "k-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/k-cons/ ./reference/k_filenames.csv ./checkpoints/pretrained_models/k-cons.pth
	echo "b-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/b-cons/ ./reference/b_filenames.csv ./checkpoints/pretrained_models/b-cons.pth
	echo "d-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/d-cons/ ./reference/d_filenames.csv ./checkpoints/pretrained_models/d-cons.pth
	echo "g-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/g-cons/ ./reference/g_filenames.csv ./checkpoints/pretrained_models/g-cons.pth
	echo "S-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/S-cons/ ./reference/S_filenames.csv ./checkpoints/pretrained_models/S-cons.pth
	echo "f-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/f-cons/ ./reference/f_filenames.csv ./checkpoints/pretrained_models/f-cons.pth
	echo "s-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/s-cons/ ./reference/s_filenames.csv ./checkpoints/pretrained_models/s-cons.pth
	echo "v-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/v-cons/ ./reference/v_filenames.csv ./checkpoints/pretrained_models/v-cons.pth
	echo "Z-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/Z-cons/ ./reference/Z_filenames.csv ./checkpoints/pretrained_models/Z-cons.pth
	echo "m-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/m-cons/ ./reference/m_filenames.csv ./checkpoints/pretrained_models/m-cons.pth
	echo "n-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/n-cons/ ./reference/n_filenames.csv ./checkpoints/pretrained_models/n-cons.pth
	echo "l-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/l-cons/ ./reference/l_filenames.csv ./checkpoints/pretrained_models/l-cons.pth
	echo "R-cons"
	python test_phonemes.py ./pats_full_mfccs/$PATIENT/R-cons/ ./reference/R_filenames.csv ./checkpoints/pretrained_models/R-cons.pth
	echo "z-cons"
        python test_phonemes.py ./pats_full_mfccs/$PATIENT/z-cons/ ./reference/z_filenames.csv ./checkpoints/pretrained_models/z-cons.pth

done
