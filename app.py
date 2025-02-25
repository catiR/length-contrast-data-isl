import gradio as gr
import vowel_length as vln




annotation_json = 'Data/Length_in_spoken_icelandic.json'

menus, vdata = vln.setup(annotation_json)


grouplist = [g for g,ws in menus]
worddict = {g:ws for g,ws in menus}




def get_group_words(group):
	if group == '[NONE]':
		choices = ['[NONE]']
	else:
		choices = [ '[ALL]' ] + [n for n,v in worddict[group]]
	return gr.Dropdown(choices = choices, value = choices[0], interactive=True)

def check_word_langs(word,cur_lang):
	if ' [L' not in word:
		return gr.Radio(value=cur_lang,interactive=True)
	elif ' [L1]' in word:
		return gr.Radio(value='L1',interactive=False)
	else:
		return gr.Radio(value='L2',interactive=False)



def subset_words_spks(g,w,l,s,wsets,db):
	if w == '[ALL]':
		swords = [v for n,v in wsets[g]]
		labl = g
	else:
		labl = w.split(' ')[0]
		swords = [labl]
		
	if l == 'All':
		slang = ['L1', 'L2']
		labl += f'\n L1+L2, '
	else:
		slang = [l]
		labl += f'\n {l}, '

	labl += f'{s}'
	
	db1 = db.copy()
	db1 = db1.loc[ (db1['speaker_lang'].isin(slang)) & (db1['word'].isin(swords)) ]
	db1.reset_index()

	if s.lower() == 'mfa':
		src = 'mfa'
	else:
		assert s[:3].lower() == 'ann'
		src = 'gold'
		
	return db1, src, labl
	


def plott(g1,w1,l1,s1,g2,w2,l2,s2):

	dat1,src1,lab1 = subset_words_spks(g1,w1,l1,s1,worddict,vdata)

	if '[NONE]' in [g2, w2]:
		dat2, l2, src2, lab2 = None, None, None, None
	else:
		dat2,src2,lab2 = subset_words_spks(g2,w2,l2,s2,worddict,vdata)

	fig = vln.vgraph(dat1,l1,src1,lab1,dat2,l2,src2,lab2)

	return fig



bl = gr.Blocks()#theme=gr.themes.Glass())

with bl:
	
	with gr.Tabs():
		
		with gr.TabItem("Vowel quantity"):

			with gr.Row():
				with gr.Column():
					gr.Markdown(
					""" 
					#### Select data (1)
					"""
						)
					gmenu1 = gr.Dropdown(choices=grouplist,label="Group", value='AL:')
					wmenu1 = gr.Dropdown(label="Word", choices=['[ALL]'] + [n for n,v in worddict['AL:']])
					lmenu1 = gr.Radio(["L1", "L2","All"],label="Speaker group",value="L1")
					smenu1 = gr.Dropdown(["Annotated", "MFA"],label="Source",value="Annotated")

					gmenu1.change(get_group_words,inputs=[gmenu1],outputs = [wmenu1])
					wmenu1.input(check_word_langs,inputs=[wmenu1,lmenu1],outputs = [lmenu1])
					

				with gr.Column():
					gr.Markdown(
					""" 
					#### Select data (2)
					"""
						)
					gmenu2 = gr.Dropdown(choices=['[NONE]'] + grouplist,label="Group", value='A:L')
					wmenu2 = gr.Dropdown(label="Word", choices=['[ALL]'] + [n for n,v in worddict['A:L']])
					lmenu2 = gr.Radio(choices=["L1", "L2","All"],label="Speaker group",value="L1")
					smenu2 = gr.Dropdown(["Annotated", "MFA"],label="Source",value="Annotated")

					gmenu2.change(get_group_words,inputs=[gmenu2],outputs = [wmenu2])
					wmenu2.input(check_word_langs,inputs=[wmenu2,lmenu2],outputs = [lmenu2])

	
			btn = gr.Button(value="Update Plot")
			plo = gr.Plot(value=plott('AL:','[ALL]',"L1","Annotated",'A:L','[ALL]',"L1","Annotated"))
			btn.click(plott, [gmenu1,wmenu1,lmenu1,smenu1,gmenu2,wmenu2,lmenu2,smenu2], plo)




			gr.Markdown(
			""" 
			# Long and short Icelandic vowels 
			Check the About tab for more info about the project.
			"""
				)
					

		with gr.TabItem("About"):
			gr.Markdown(
			"""
			## Assessed and Annotated Vowel Lengths in Spoken Icelandic Sentences\
			for L1 and L2 Speakers: A Resource for Pronunciation Training
			"""
				 )

			
			gr.Markdown(
			"""
			### About

			This annotated data and its demo application accompany the paper 
			*Assessed and Annotated Vowel Lengths in Spoken Icelandic Sentences\
			for L1 and L2 Speakers: A Resource for Pronunciation Training*, \
			Caitlin Laura Richter, Kolbrún Friðriksdóttir, Kormákur Logi Bergsson, \
			Erik Anders Maher, Ragnheiður María Benediktsdóttir, Jon Gudnason - NoDaLiDa/Baltic-HLT 2025, Tallinn, Estonia.
			

			"""
				 )
			
			gr.Markdown(
				""" 
				## Demo: Viewing the data
				Use the menus to choose words, speaker group, and data source. 
				Words are split into related groups and either the whole group or a single word can be selected. 
				Available speaker groups are native Icelandic speakers (L1), second-language speakers (L2), or all. 
				Data source options are gold (human) annotations or automated Montreal Forced Aligner (MFA).

				The display is a scatter plot of vowel and consonant durations,
				supplemented with density plots for each dimension separately.

				The general expectation is that, all else being equal, syllables with long stressed vowels 
				followed by short consonants have a higher vowel:(vowel+consonant) duration ratio, 
				while syllables with short stressed vowels followed by long consonants have a lower ratio. 

				Many other factors also affect relative durations in any particular recorded token, 
				and these factors have considerable - not necessarily balanced - variation throughout this dataset. 
				This demo is provided to begin exploring the data and suggest hypotheses for follow-up. 
				See Pind 1999, 'Speech segment durations and quantity in Icelandic' 
				(J. Acoustical Society of America, 106(2)) for a review of the acoustics of Icelandic vowel duration. 
			"""
				)


			gr.Markdown(
				""" 
			## Accessing the data

			Annotations can be downloaded as 
			[json](https://github.com/catiR/length-contrast-data-isl/blob/main/Data/Length_in_spoken_icelandic.json) 
			or [tsv](https://github.com/catiR/length-contrast-data-isl/blob/main/Data/Length_in_spoken_icelandic.tsv) files. 
			See [the paper](https://github.com/catiR/length-contrast-data-isl/blob/main/Data/133_Annotated_Vowel_Lengths.pdf) 
			for complete information. 
			"""
				 )

			gr.Markdown(
				""" 
			Audio is available from [Clarin](https://repository.clarin.is/repository/xmlui/) (Samrómur). 
			The 'collection' field plus recording filename in the annotations metadata 
			specify the original audio file, including which Samrómur collection it is found in. 
			"""
				 )

			gr.Markdown(
				""" 
Annotation records are in the following scheme: 

```
[ { recording: source-file-id.wav,
    collection: samromur-collection,
    speaker_lang: L1/L2,
    word: target-word,
    word_context: { 
        normalised: normalised-carrier-sentence-text,
        before: sentence-context-preceding-token,
        after: sentence-context-following-token 
    },
    gold_annotation: {
        target_word_start: seconds,
        target_word_end: seconds,
        prevowel: [ { 
                phone: ipa-character,
                start: seconds,
                end: seconds, 
                }, 
                { phone2 ... } ,
        ], 
        vowel: [ { 
                phone: ipa-character,
                start: seconds,
                end: seconds, 
                }, 
        ], 
        postvowel: [ { 
                phone: ipa-character,
                start: seconds,
                end: seconds, 
                }, 
        ]
    },
    mfa_annotation : {
     ... as for gold ...
    }
 },
]
```	
		
				"""
				 )
			



			gr.Markdown(
			"""
			### Contact caitlinr@ru.is about bugs, feedback, or collaboration!

			"""
				 )




if __name__ == "__main__":
	bl.launch()

	
