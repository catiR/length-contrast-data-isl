import os, json
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# make subsets of words for convenience
def make_sets(db,shorts,longs):
	
	def _wspec(wd,l1,l2):
		if (wd in l1) and (wd in l2):
			return(wd,wd)
		elif wd in l1:
			return(f'{wd} [L1]',wd)
		elif wd in l2:
			return(f'{wd} [L2]',wd)
		else:
			return ('','')

	def _ksrt(k):
		if ' ' in k:
			return((k[0],1/len(k)))
		else:
			return (k.replace(':',''),k[-1] )
		
	words = set([(t['word'],t['speaker_lang']) for t in db])
	l1 = [w for w,l in words if l == 'L1']
	l2 = [w for w,l in words if l == 'L2']
	words = set([w for w,l in words])

	wdict = defaultdict(list)
	for w in words:
		if 'agg' in w:
			wdict['AG:'].append(_wspec(w,l1,l2))
		elif 'all' in w:
			wdict['AL:'].append(_wspec(w,l1,l2))
		elif 'egg' in w:
			wdict['EG:'].append(_wspec(w,l1,l2))
		elif 'eki' in w:
			wdict['E:G'].append(_wspec(w,l1,l2))
		elif 'aki' in w:
			wdict['A:G'].append(_wspec(w,l1,l2))
		elif 'ala' in w:
			wdict['A:L'].append(_wspec(w,l1,l2))
		elif w in shorts:
			wdict['OTHER - SHORT'].append(_wspec(w,l1,l2))
		elif w in longs:
			wdict['OTHER - LONG'].append(_wspec(w,l1,l2))
		else:
			print(f'something should not have happened: {w}')

			
	sets = [(k, sorted(wdict[k])) for k in sorted(list(wdict.keys()),key = _ksrt)]
	
	return sets
			

# compile data for a token record
def get_tk_data(tk,shorts,longs):

	# merge intervals
	# from list of phones
	# to word part
	def _merge_intervals(plist):
		if not plist:
			return np.nan
		tot_start, tot_end = plist[0]['start'],plist[-1]['end']
		tot_dur = tot_end-tot_start
		return tot_dur

	tkdat = {}
	tkdat['word'] = tk['word']
	tkdat['speaker_lang'] = tk['speaker_lang']
	tkdat['n_pre_phone'] = len(tk['gold_annotation']['prevowel'])
	tkdat['n_post_phone'] = len(tk['gold_annotation']['postvowel'])
	
	if tk['word'] in longs:
		tkdat['vlen'] = 1
	else:
		assert tk['word'] in shorts
		tkdat['vlen'] = 0
		
	for s in ['gold','mfa']:
		tkdat[f'{s}_pre_dur'] = _merge_intervals(tk[f'{s}_annotation']['prevowel'])
		tkdat[f'{s}_v_dur'] = _merge_intervals(tk[f'{s}_annotation']['vowel'])
		tkdat[f'{s}_post_dur'] = _merge_intervals(tk[f'{s}_annotation']['postvowel'])
		tkdat[f'{s}_word_dur'] = tk[f'{s}_annotation']['target_word_end'] -\
		  tk[f'{s}_annotation']['target_word_start']

	return tkdat


# code short vowels 0, long 1
def prep_dat(d):
	df = d.copy()
	for s in ['gold','mfa']:
		df[f'{s}_ratio'] = df[f'{s}_v_dur'] / (df[f'{s}_v_dur']+df[f'{s}_post_dur'])
		df[f'{s}_pre_dur'] = df[f'{s}_pre_dur'].fillna(0) # set absent onsets dur zero
	df = df.convert_dtypes()
	return df


def setup(annot_json):

	longs = set(['aki', 'ala', 'baki', 'bera', 'betri', 'blaki', 'breki',
				'brosir', 'dala', 'dreki', 'dvala', 'fala', 'fara', 'færa',
				'færi', 'gala', 'hausinn', 'jónas', 'katrín', 'kisa', 'koma',
				'leki', 'leyfa', 'maki', 'muna', 'nema', 'raki', 'sama',
				'speki', 'svala', 'sækja', 'sömu', 'taki', 'tala', 'tvisvar',
				'vala', 'veki', 'vinur', 'ása', 'þaki'])
	
	shorts = set(['aggi', 'baggi', 'balla', 'beggi', 'eggi', 'farðu', 'fossinn',
				'færði', 'galla', 'hausnum', 'herra', 'jónsson', 'kaggi', 'kalla',
				'lalla', 'leggi', 'leyfðu', 'maggi', 'malla', 'mamma', 'missa',
				'mömmu', 'nærri', 'palla', 'raggi', 'skeggi', 'snemma', 'sunna',
				'tommi', 'veggi','vinnur', 'ásta'])

	with open(annot_json, 'r') as handle:
		db = json.load(handle)

	sets = make_sets(db,shorts,longs)

	db = [get_tk_data(tk,shorts,longs) for tk in db]
	dat = pd.DataFrame.from_records(db)
	dat = prep_dat(dat)

	return sets,dat



def vgraph(dat1,l1,src1,lab1,dat2,l2,src2,lab2):

	def _gprep(df,l,s):

		# color by length + speaker group
		ccs = { "lAll" : (0.0, 0.749, 1.0),
				"lL1" : (0.122, 0.467, 0.706),
				"lL2" : (0.282, 0.82, 0.8),
				"sAll" :(0.89, 0.467, 0.761),
				"sL1" : (0.863, 0.078, 0.235),
				"sL2" : (0.859, 0.439, 0.576),
				"xAll" : (0.988, 0.69, 0.004),
				"xL1" : (0.984, 0.49, 0.027),
				"xL2" : (0.969, 0.835, 0.376)}

		vdurs = np.array(df[f'{s}_v_dur'])*1000
		cdurs = np.array(df[f'{s}_post_dur'])*1000
		rto = np.mean(df[f'{s}_ratio'])

		if sum(df['vlen']) == 0:
			vl = 's'
		elif sum(df['vlen']) == df.shape[0]:
			vl = 'l'
		else:
			vl = 'x'

		cc = ccs[f'{vl}{l}']

		return vdurs, cdurs, rto, cc

	
	vd1,cd1,ra1,cl1 = _gprep(dat1,l1,src1)
	lab1 += f'\n Ratio: {ra1:.3f}'
	if src1 == 'gold':
		mk1 = '^'
	else:
		mk1 = '<'


	fig, ax = plt.subplots(figsize=(9,7))
	ax.set_xlim(0.0,350)
	ax.set_ylim(0.0,350)

	ax.scatter(vd1,cd1,marker = mk1, label = lab1,
				   c = [cl1 + (.7,)], edgecolors = [cl1] )

	if lab2:
		vd2,cd2,ra2,cl2 = _gprep(dat2,l2,src2)
		lab2 += f'\n Ratio: {ra2:.3f}'
		if src2 == 'gold':
			mk2 = 'v'
		else:
			mk2 = '>'
		ax.scatter(vd2,cd2, marker = mk2, label = lab2,
					c = [cl2 + (.05,)], edgecolors = [cl2] )


	ax.set_title("Stressed vowel & following consonant(s) duration" )
	ax.set_xlabel("Vowel duration (ms)")
	ax.set_ylabel("Consonant duration (ms)")
	#fig.legend(loc=8,ncols=2)
	fig.legend(loc=7)
	
	ax.axline((0,0),slope=1,color="darkgray")
	
	fig.tight_layout()
	#fig.subplots_adjust(bottom=0.15)
	fig.subplots_adjust(right=0.75)

	#plt.xticks(ticks=[50,100,150,200,250,300],labels=[])
	#plt.yticks(ticks=[100,200,300],labels=[])

	return fig

