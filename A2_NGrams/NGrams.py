import re
import math
import operator
import matplotlib.pyplot as plt


acronyms = ["I.O.U.","R.I.P.","R.I.P","M.D.","N.B.","P.O.","U.S","U.K.","U.S.","U.S.A.","P.S.","A.M.","P.M.","a.m.","p.m."]

beg_periods = ["mr.","Mr.","mrs.","Mrs.","dr.","Dr.","st.","St."]

all_tokens = []

def white_space(line):
	tokens = line.split()
	return(tokens)

def full_stop(tokens):
	for i,token in enumerate(tokens) :
		if token not in beg_periods and token not in acronyms:
			p = re.compile("[A-Za-z]+[.]+([A-Za-z])*")
			m = p.match(token)
			if m :
				fi = token.find(".")
				ri = token.rfind(".")
				first = token[:fi]
				last = token[ri+1:]
				tokens.insert(i+1,token[fi:ri+1])
				if last is not "":
					tokens.insert(i+2,last)
				tokens[i] = first;
	return tokens

def question_mark(tokens):
	for i,token in enumerate(tokens) :
		p = re.compile("([^?]*)([?]+)([^?]*)")
		m = p.match(token)
		if m :
			first = p.match(token).group(1)
			mark = p.match(token).group(2)
			last = p.match(token).group(3)
			if first is not "":
				tokens[i] = first
				tokens.insert(i+1,mark)
				if last is not "":
					tokens.insert(i+2,last)
			elif last is not "":
				tokens[i] = mark
				tokens.insert(i+1,last)
	return tokens

def exclaimation_mark(tokens):
	for i,token in enumerate(tokens) :
		p = re.compile("([^!]*)([!]+)([^!]*)")
		m = p.match(token)
		if m :
			first = p.match(token).group(1)
			mark = p.match(token).group(2)
			last = p.match(token).group(3)
			if first is not "":
				tokens[i] = first
				tokens.insert(i+1,mark)
				if last is not "":
					tokens.insert(i+2,last)
			elif last is not "":
				tokens[i] = mark
				tokens.insert(i+1,last)
	return tokens

def commas(tokens):
	for i,token in enumerate(tokens) :
		p = re.compile("([^,]*)([,]+)([^,]*)")
		m = p.match(token)
		if m :
			first = p.match(token).group(1)
			comma = p.match(token).group(2)
			last = p.match(token).group(3)
			
			if first is not "":
				tokens[i] = first
				tokens.insert(i+1,comma)
				if last is not "":
					tokens.insert(i+2,last)
				
			elif last is not "":
				tokens[i] = comma
				tokens.insert(i+1,last)
			
	return tokens

def key_emoticons(tokens):
	for i,token in enumerate(tokens) :
		p = re.compile("(.*)([:,;][-]*[D,d,P,p,),(]+)(.*)")
		m = p.match(token)
		if m :
			first = p.match(token).group(1)
			emoticon = p.match(token).group(2)
			last = p.match(token).group(3)
			if first is not "":
				tokens[i] = first
				tokens.insert(i+1,emoticon)
				if last is not "":
					tokens.insert(i+2,last)	
				u = re.compile("(.+)([:,;][-]*[D,d,P,p,),(]+)")		
				ka = u.match(tokens[i])
				if ka :
					tokens = key_emoticons(tokens)	
			elif last is not "":
				tokens[i] = emoticon
				tokens.insert(i+1,last)
		else :
			p = re.compile("([^(]*)([(]+)([^(]*)")
			m = p.match(token)
			if m :
				first = p.match(token).group(1)
				paren = p.match(token).group(2)
				last = p.match(token).group(3)
				if first is not "":
					tokens[i] = first
					tokens.insert(i+1,paren)
					if last is not "":
						tokens.insert(i+2,last)	
				elif last is not "":
					tokens[i] = paren
					tokens.insert(i+1,last)
			else :
				p = re.compile("([^)]*)([)]+)([^)]*)")
				m = p.match(token)
				if m :
					first = p.match(token).group(1)
					paren = p.match(token).group(2)
					last = p.match(token).group(3)
					if first is not "":
						tokens[i] = first
						tokens.insert(i+1,paren)
						if last is not "":
							tokens.insert(i+2,last)	
					elif last is not "":
						tokens[i] = paren
						tokens.insert(i+1,last)

	return tokens

def double_quotes(tokens):
	for i,token in enumerate(tokens) :
		p = re.compile("([^\"]*)([\"]+)([^\"]*)")
		m = p.match(token)
		if m :
			first = p.match(token).group(1)
			double_quote = p.match(token).group(2)
			last = p.match(token).group(3)
			if first is not "":
				tokens[i] = first
				tokens.insert(i+1,double_quote)
				if last is not "":
					tokens.insert(i+2,last)	
			elif last is not "":
				tokens[i] = double_quote
				tokens.insert(i+1,last)

	return tokens

def person(tokens):
	for i,token in enumerate(tokens) :
		p = re.compile("([@i].+)(:)")
		m = p.match(token)
		if m :
			first = p.match(token).group(1)
			last = p.match(token).group(2)
			tokens[i] = first
			tokens.insert(i +1 , last)

	return tokens

def single_quotes(tokens):
	for i,token in enumerate(tokens) :
		p = re.compile("([\w]+)([\']+)([\w]+)")
		m = p.match(token)
		if m :
			first = p.match(token).group(1)
			single_quote = p.match(token).group(2)
			last = p.match(token).group(3)
			if first is not "":
				tokens[i] = first
				tokens.insert(i+1,single_quote)
				if last is not "":
					tokens.insert(i+2,last)	
			elif last is not "":
				tokens[i] = single_quote
				tokens.insert(i+1,last)
		else :
			p = re.compile("([\w]*)([\']+)([\w]+)([\']*)")
			m = p.match(token)
			if m :
				first = p.match(token).group(1)
				single_quote = p.match(token).group(2)
				last = p.match(token).group(3)
				last_quote = p.match(token).group(4)
				if first is not "":
					tokens[i] = first
					tokens.insert(i+1,single_quote)
					if last is not "":
						tokens.insert(i+2,last)	
						if last_quote is not "":
							tokens.insert(i+3,last_quote)
				elif last is not "":
					tokens[i] = single_quote
					tokens.insert(i+1,last)
					if last_quote is not "":
						tokens.insert(i+2,last_quote)
			else :
				p = re.compile("([\w]+)([\']+)([\w]*)")
				m = p.match(token)
				if m :
					first = p.match(token).group(1)
					single_quote = p.match(token).group(2)
					last = p.match(token).group(3)
					if first is not "":
						tokens[i] = first
						tokens.insert(i+1,single_quote)
						if last is not "":
							tokens.insert(i+2,last)	
					elif last is not "":
						tokens[i] = single_quote
						tokens.insert(i+1,last)

	for i,token in enumerate(tokens) :
		if i <= len(tokens)-2:
			
			if token == "'" and tokens[i+1] in ["s","d","ll","ve","t","re"]:
				
				tokens[i+1] = "'"+tokens[i+1]
				tokens.pop(i)
	return tokens

def tags(tokens):
	for i,token in enumerate(tokens):
		p = re.compile("[.?!]+")
		m = p.match(token)
		if m:
			if tokens[i+1] is not "</s>" :
				tokens.insert(i+1,"</s>")
				tokens.insert(i+2,"<s>")
	return tokens

def emoticon(tokens):
	for i,token in enumerate(tokens):
		
		if token not in ["<s>","</s>"] :
			p = re.compile("""
					    (?:
					      [<>]?
					      [:;=8]                     # eyes
					      [\-o\*\']?                 # optional nose
					      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
					      |
					      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
					      [\-o\*\']?                 # optional nose
					      [:;=8]                     # eyes
					      [<>]?
					    )""")
			try :
				m = p.match(token)
			except TypeError :
				pass
			else :
				if m:
					first = p.match(token).group(1)
					special_char = p.match(token).group(2)
					last = p.match(token).group(3)
					op = []
					count = 1;
					for j in special_char :
						op.append(j)
					if first is not "":
						# print(i)
						# print(token)
						tokens[i] = first
						for j in op :
							tokens.insert(i+count,j)
							count+=1
						if last is not "":
							tokens.insert(i+count,last)
					elif last is not "":
						count =1;
						for j in op :
							tokens.insert(i+count,j)
							count+=1
						tokens.insert(i+count,last)
						tokens.pop(i)
					elif len(special_char) > 1:
						count =1
						for j in op :
							tokens.insert(i+count,j)
							count+=1
						tokens.pop(i)
				
	return tokens

def prin(di):
	seen = []
	for key,value in di.items():
		if value not in seen :
			seen.append(value)
	print(seen)
	seen.sort()
	seen.reverse()
	print(seen)
	for i in seen:
		print(i)

## PostTOKENIZER

#####################################

f = open('output.txt','r')
lines = f.read().lower()
f.close()
lines = lines.split('\n')

unigramcount = {}
bigramcount = {}
trigramcount = {}
fourgramcount = {}
fivegramcount = {}
sixgramcount = {}
onegram ={}
twogram = {}
threegram = {}
fourgram = {}
fivegram = {}
sixgram = {}
nextwordunigram = {}
nextwordbigram = {}
nextwordtrigram = {}
nextwordfourgram = {}
nextwordfivegram = {}
nextwordsixgram = {}

for line in lines:
  #print(line)
  line = line.split(' ')
  ll = len(line)

  for idx in xrange(len(line)):

    if (idx == ll-2 and idx>=0):
      if onegram.has_key(line[idx]):
        onegram[line[idx]] += 1
      else:
        onegram[line[idx]] = 1

    if (idx == ll-3 and idx>=0):
      if twogram.has_key(line[idx]+' '+line[idx+1]):
        twogram[line[idx]+' '+line[idx+1]] += 1
      else:
        twogram[line[idx]+' '+line[idx+1]] = 1

    if (idx == ll-4 and idx>=0):
      if threegram.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]):
        threegram[line[idx]+' '+line[idx+1]+' '+line[idx+2]] += 1
      else:
        threegram[line[idx]+' '+line[idx+1]+' '+line[idx+2]] = 1

    if (idx == ll-5 and idx>=0):
      if fourgram.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]):
        fourgram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]] += 1
      else:
        fourgram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]] = 1

    if (idx == ll-6 and idx>=0):
      if fivegram.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]):
        fivegram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]] += 1
      else:
        fivegram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]] = 1

    if (idx == ll-7 and idx>=0):
      if sixgram.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]+' '+line[idx+5]):
        sixgram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]+' '+line[idx+5]] += 1
      else:
        sixgram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]+' '+line[idx+5]] = 1

    if(idx <= ll-2 and idx >=0):
      if unigramcount.has_key(line[idx]):
        unigramcount[line[idx]] += 1
      else:
        unigramcount[line[idx]] = 1

    if (idx <= ll-3 and idx>=0):
      if bigramcount.has_key(line[idx]+' '+line[idx+1]):
        bigramcount[line[idx]+' '+line[idx+1]] += 1
      else:
        bigramcount[line[idx]+' '+line[idx+1]] = 1

    if (idx <= ll-4 and idx>=0):
      if trigramcount.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]):
        trigramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]] += 1
      else:
        trigramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]] = 1

    if (idx <= ll-5 and idx>=0):
      if fourgramcount.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]):
        fourgramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]] += 1
      else:
        fourgramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]] = 1

    if (idx <= ll-6 and idx>=0):
      if fivegramcount.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]):
        fivegramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]] += 1
      else:
        fivegramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]] = 1

    if (idx == ll-7 and idx>=0):
      if sixgramcount.has_key(line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]+' '+line[idx+5]):
        sixgramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]+' '+line[idx+5]] += 1
      else:
        sixgramcount[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]+' '+line[idx+5]] = 1

for line in lines:
  #print(line)
  line = line.split(' ')
  ll = len(line)

  for idx in xrange(len(line)):
    if(idx <= ll-2 and idx >=0):
      if not nextwordunigram.has_key(line[idx]):
        nextwordunigram[line[idx]] = line[idx+1]
      try:
        if unigramcount[line[idx]] > unigramcount[nextwordunigram[line[idx]]]:
          nextwordunigram[line[idx]] = line[idx+1]
      except:
        pass

    if (idx <= ll-3 and idx>=0):
      nextwordbigram[line[idx]+' '+line[idx+1]] = line[idx+2]

    if (idx <= ll-4 and idx>=0):
        nextwordtrigram[line[idx]+' '+line[idx+1]+' '+line[idx+2]] = line[idx+3]

    if (idx <= ll-5 and idx>=0):
        nextwordfourgram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]] = line[idx+4]

    if (idx <= ll-6 and idx>=0):
        nextwordfivegram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]] = line[idx+5]

    if (idx == ll-7 and idx>=0):
        nextwordsixgram[line[idx]+' '+line[idx+1]+' '+line[idx+2]+' '+line[idx+3]+' '+line[idx+4]+' '+line[idx+5]] = line[idx+6]

uniprob = 0.0
biprob = 0.0
triprob = 0.0
fourprob = 0.0
fiveprob = 0.0
sixprob = 0.0

for token in onegram:
  uniprob += (onegram[token]/unigramcount[token])

for token in twogram:
  biprob += (twogram[token]/bigramcount[token])

for token in threegram:
  triprob += (threegram[token]/trigramcount[token])

for token in fourgram:
  fourprob += (fourgram[token]/fourgramcount[token])

for token in fivegram:
  fiveprob += (fivegram[token]/fivegramcount[token])

for token in sixgram:
  sixprob += (sixgram[token]/sixgramcount[token])

try:
  uniprob /= len(onegram) 
  biprob /= len(twogram)
  triprob /= len(threegram)
  fourprob /= len(fourgram)
  fiveprob /= len(fivegram)
  sixprob /= len(sixgram)
except:
  print("Horrible error!")

print("Part A: The probabilities of tokens being sentence ending tokens are as follows:")
print("Unigram: "+str(uniprob))
print("Bigrams: "+str(biprob))
print("Trigrams: "+str(triprob))
print("Fourgrams: "+str(fourprob))
print("Fivegrams: "+str(fiveprob))
print("Sixgrams: "+str(sixprob))
print("Hence, the probability order is: Sixgrams > Fivegrams > Fourgrams > Trigrams >> Bigrams >> Unigrams")

print('Part B:')
y = list(reversed(sorted(unigramcount.values())))
x= []
logy= []
for idx in xrange(len(unigramcount)):
  x.append(idx+1)
  logy.append(math.log10(y[idx]))

f = plt.figure(0)
plt.plot(x,y)
plt.ylabel('Frequency of unigrams')
plt.xlabel('Rank')
#plt.show()

#plt.plot(x,logy)
#plt.ylabel('Log Frequency')
#plt.xlabel('Rank')
#plt.show()

print('Part C:')
i = 1
for ngram in [onegram,twogram,threegram,fourgram,fivegram,sixgram]:
  plt.figure(i)
  y = list(reversed(sorted(ngram.values())))
  x = [idx for idx in xrange(len(ngram))]
  plt.plot(x,y)
  plt.ylabel('Frequency of '+str(i)+' gram model for P(X|<\s>)')
  plt.xlabel('Rank')
  axes=plt.gca()
  axes.set_ylim([0,max(y)])
  i+=1
  #plt.show()

plt.show()

noinputs = 1
while(noinputs>=0):
  print('Part D:')
  inp = raw_input('Enter the value of \'n\' in the n-gram model to generate sentences. Enter \'-1\' to exit.\n')
  noinputs = int(inp) - 1
  idx =0
  
  line = []
  line.append('<s>')
  if noinputs>0 and noinputs<7:
    inputs = raw_input('Enter '+str(noinputs)+' words to start the sentence\n')
    inputs = inputs.strip('\n').split(' ')
    for sinput in inputs:
      line.append(sinput)

  while(noinputs>=0 and line[-1]!='<\s>'):
    try:
      if noinputs == 0:
        while line[-1]!='<\s>':
          line.append(nextwordunigram[line[-1]])

      if noinputs == 1:
        while line[-1]!='<\s>':
          line.append(nextwordbigram[line[-2]+' '+line[-1]])
  
      if noinputs == 2:
        while line[-1]!='<\s>':
          line.append(nextwordtrigram[line[-3]+' '+line[-2]+' '+line[-1]])

      if noinputs == 3:
        while line[-1]!='<\s>':
          line.append(nextwordfourgram[line[-4]+' '+line[-3]+' '+line[-2]+' '+line[-1]])
  
      if noinputs == 4:
        while line[-1]!='<\s>':
          line.append(nextwordfivegram[line[-5]+' '+line[-4]+' '+line[-3]+' '+line[-2]+' '+line[-1]])
  
      if noinputs == 5:
        while line[-1]!='<\s>':
          line.append(nextwordsixgram[line[-6]+' '+line[-5]+' '+line[-4]+' '+line[-3]+' '+line[-2]+' '+line[-1]])
    
    except:
      noinputs -= 1 
      print('Sorry! '+str(noinputs+2)+' gram not found! Using Katz Backoff to revert to '+str(noinputs+1)+' gram model!')
    #print(noinputs)
    #print(line)

  if(noinputs>=0):
    print(line)
  
  else:
    print("Sorry! Even unigrams were missing! Katz Backoff failed!")

if __name__ == "__main__":
	with open("tweets.en.txt") as f:
		for line in f:
			line = line.strip()
			tokens = white_space(line)
			tokens = full_stop(tokens)
			tokens = question_mark(tokens)
			tokens = exclaimation_mark(tokens)
			tokens = tags(tokens)
			tokens = key_emoticons(tokens)
			tokens = commas(tokens)
			tokens = double_quotes(tokens)
			tokens = person(tokens)
			tokens = single_quotes(tokens)
			tokens = emoticon(tokens)
			all_tokens += tokens
			with open("output.txt",'a') as fa:
				fa.write(str(tokens) + "\n")
