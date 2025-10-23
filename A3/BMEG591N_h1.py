import numpy as np

def align(cigar, ref):
    '''
    purpose: aligns a sequence to a ref based on a cigar string
    inputs: 
        cigar-string of numbers and letters indicating matches and mismatches (insertions/deletions)
        ref-string of ref sequence (ACGT combination)
        pos-indicates number of ref string at which to start alignment, defaults to 0
    outputs: array with 2 rows that contains the ref string in the first row and the aligned seq in the second row
    '''
    top=[]
    bottom=[]
    for i in range(len(cigar)//2):
        num = cigar[i]
        letter = cigar[i+1]
        # if matching, add all entries from ref up to num
        if letter == 'M':
            entry=list(ref[i*2:i*2+num])
            top.extend(entry)
            bottom.extend(entry)
        elif letter == 'D':
            entry=list(ref[i*2:i*2+num])
            top.extend(entry)
            bottom.extend(['']*num)
        elif letter == 'I':
            # not given the identity of the seq to be aligned thus not sure what to put for the mystery base...
            entry=list('X'*num)
            bottom.extend(entry)
            top.extend(['']*num)
            pass
    result = np.array([top,bottom])
    return result