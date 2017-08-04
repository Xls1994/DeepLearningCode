# -*- encoding: utf-8 -*-
'''
autuor: yangyl
@site: 
@software: PyCharm Community Edition
@file: process_MEDLINE.py
@time: 2017/8/4 16:32

'''
def extract_abstract(file_path,output_path):
    writer =open(output_path,'w')
    doc =[]
    corpus =[]
    with open(file_path,'r')as f:
        for line in f:
            line =line.strip()
            if line !="":
                if line.find("PMID-")!=-1:
                    pmid =line.split(' ')[1]
                    doc.append(pmid)
                elif line.find("TI")!=-1:
                    title =line[line.index("-")+2:]
                    doc.append(title)
                elif line.find("AB")!=-1:
                    abstract =line[line.index("-")+2:]
                    doc.append(abstract)
            else:
                corpus.append(doc)
                doc =[]
    for c in corpus:
        for d in c:
            writer.write(d+"\n")
        writer.write("\n")



if __name__ == '__main__':
    file_path=''
    pass