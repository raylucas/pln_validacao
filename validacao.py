# coding: utf-8
import nltk

# nltk.download()


#remove acentos
from unicodedata import normalize
def removeacentos(txt):
    return normalize('NFKD', txt).encode('ASCII','ignore').decode('ASCII')

if __name__ == '__main__':
    from doctest import testmod
    testmod()


#remove pontuação
def removepontuacao(txt):
    if ',' in txt:
        txt = txt.replace(',','')
    if '.' in txt:
        txt = txt.replace('.','')
    if '!' in txt:
        txt = txt.replace('!','')
    if '?' in txt:
        txt = txt.replace('?','')
    if '"' in txt:
        txt = txt.replace('"','')
    if ':' in txt:
        txt = txt.replace(':', '')
    if ';' in txt:
        txt = txt.replace(';','')
    if '(' in txt:
        txt = txt.replace('(','')
    if ')' in txt:
        txt = txt.replace(')','')
    if '|' in txt:
        txt = txt.replace('|','')
    return txt

arquivo = open('/home/abner/Dropbox/Alaska/TCC/lista-treinamento-oficial.txt','r')
basetrei = arquivo.readlines()
#print(basetrein)
basefrasetreinamento = []
baseclassetreinamento = []
basetreinamento = []

for i in basetrei:
    frase = i.split("', '")[0].replace("('","")

    frase = removeacentos(frase)
    frase = removepontuacao(frase)

    basefrasetreinamento.append(frase)

    classe = i.split("', '")[-1].replace("'),","").replace("\n","").replace("')","").replace(" ","").replace("  ","")
    baseclassetreinamento.append(classe)

basetreinamento = zip(basefrasetreinamento, baseclassetreinamento)



arquivoteste = open('/home/abner/Dropbox/Alaska/TCC/lista-testes-oficial.txt','r')
basetes = arquivoteste.readlines()
#print(basetrein)
basefraseteste = []
baseclasseteste = []
baseteste = []

for i in basetes:
    frase = i.split("', '")[0].replace("('","")

    frase = removeacentos(frase)
    frase = removepontuacao(frase)

    basefraseteste.append(frase)

    classe = i.split("', '")[-1].replace("'),","").replace("\n","").replace("')","").replace(" ","").replace("  ","")
    baseclasseteste.append(classe)

baseteste = zip(basefraseteste, baseclasseteste)

#print (basefraseteste)

#print(baseteste)



# stopword
stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
stopwordsadd = open('/home/abner/Dropbox/Alaska/TCC/stopwords','r')
stopwords = stopwordsadd.readlines()
for i in stopwords:
    stopwordsnltk.append(i.replace('\n', ''))
#print(stopwordsnltk)

'''
stopwordsaddverbos = open('/home/abner/Dropbox/Alaska/TCC/stopwords-verbos','r')
stopwordsverbos = stopwordsaddverbos.readlines()
for i in stopwordsverbos:
    stopwordsnltk.append(i.replace('\n', ''))
print(stopwordsnltk)
'''


def removestopword(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwordsnltk]
        frases.append((semstop, emocao))
    return frases


#print(removestopword(basetreinamento))

# stemming

def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming


frasescomstemmingtreinamento = aplicastemmer(basetreinamento)
frasescomstemmingteste = aplicastemmer(baseteste)


# print(frasescomstemming)

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras


palavrastreinamento = buscapalavras(frasescomstemmingtreinamento)
palavrasteste = buscapalavras(frasescomstemmingteste)
#print(palavrastreinamento)

def buscafrequencia(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras


frequenciatreinamento = buscafrequencia(palavrastreinamento)
frequenciateste = buscafrequencia(palavrasteste)


#print(frequenciatreinamento.most_common())
#print(len(frequenciatreinamento.most_common()))

def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq


palavrasunicastreinamento = buscapalavrasunicas(frequenciatreinamento)
palavrasunicasteste = buscapalavrasunicas(frequenciateste)

#print (palavrasunicastreinamento)

#print(palavrasunicastreinamento)

def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicastreinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas


caracteristicasfrase = extratorpalavras(['am', 'nov', 'dia'])
#print(caracteristicasfrase)


basecompletatreinamento = nltk.classify.apply_features(extratorpalavras, frasescomstemmingtreinamento)
basecompletateste = nltk.classify.apply_features(extratorpalavras, frasescomstemmingteste)
# print(basecompleta[15])

#print (basecompletatreinamento)

# constroi a tabela de probabilidade
classificador = nltk.NaiveBayesClassifier.train(basecompletatreinamento)
#print(classificador.labels())
#print(classificador.show_most_informative_features(20))

print("accuracy")
print (nltk.classify.accuracy(classificador, basecompletateste))

erros = []
for(frase, classe) in basecompletateste:
    #print(frase)
    #print(classe)
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))

for (classe, resultado, frase) in erros:
    print (classe, resultado, frase)

from nltk.metrics import ConfusionMatrix

esperado = []
previsto = []

for(frase, classe) in basecompletateste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

#esperado = 'alegria alegria alegria alegria medo medo supresa supresa'.split()
#previsto = 'alegria alegria medo supresa medo medo medo supresa'.split()

matriz = ConfusionMatrix(esperado, previsto)
print (matriz)


# 1. Cenário
# 2. Número de Classes - 16%
# 3. ZeroRules - 21.05%
