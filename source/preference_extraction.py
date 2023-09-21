from Levenshtein import distance as levdistance 
import csv
import re

test_list = [    
    "I'm looking for world food",
    "I want a restaurant that serves world food",
    "I want a restaurant serving Swedish food",
    "I'm looking for a restaurant in the center",
    "I would like a cheap restaurant in the west part of town",
    "I'm looking for a moderately priced restaurant in the west part of town",
    "I'm looking for a restaurant in any area that serves Tuscan food",
    "Can I have an expensive restaurant",
    "I'm looking for an expensive restaurant and it should serve international food",
    "I need a Cuban restaurant that is moderately priced",
    "I'm looking for a moderately priced restaurant with Catalan food",
    "What is a cheap restaurant in the south part of town",
    "What about Chinese food",
    "I wanna find a cheap restaurant",
    "I'm looking for Persian food please",
    "Find a Cuban restaurant in the center"
]

def fetchKeywords(filename):
    file = open(filename)
    file = csv.DictReader(file)
    keyword_names = file.fieldnames[1:4]
    keyword_dict = {key: set() for key in keyword_names}

    for row in file:
        for keyword in keyword_names:
            if(row[keyword]) != "":
                keyword_dict[keyword].add(row[keyword])
    return keyword_dict

def patternMatch(data, keyword_dict):
    data = data.lower()
    temp = None
    result = []
    if temp := re.findall("(\w+) food", data):
        result.append(("food", temp[0]))
    if temp := re.findall("in the (\w+)", data):
        result.append(("area", temp[0]))
    if temp := re.findall("(\w+) priced", data):
       result.append(("pricerange", temp[0]))
    if temp := re.findall("(\w+) restaurant", data):
       for key, values in keyword_dict.items():
           for value in values:
               if levdistance(temp[0], value) <= 2:
                result.append((key, value))
    return result
    
