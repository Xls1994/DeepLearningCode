from itertools import izip
cuePath ='Cue/Ftraincue.context'
wordPath='Word/Ftrainword.context'
mergePath='MergeFtrain.context'

cue =open(cuePath,'r')
word =open(wordPath,'r')
merge =open(mergePath,'w')
for line1,line2 in izip(word,cue):
    line1=line1.strip()
    line2 =line2.strip()
    if line1!='':
        line =line1+' '+line2
        merge.write(line+'\n')

cue.close()
word.close()
merge.close