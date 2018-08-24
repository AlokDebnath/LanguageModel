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

if __name__ == "__main__":
	with open("./Gutenberg/Zane Grey___The Light of Western Stars.txt") as f:
		for line in f:
			line = "<s> " + line.strip() + " </s>"
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
