
from cmath import isnan
import os
from sys import stderr
from flask import Flask, request, make_response, redirect, url_for, jsonify
from flask import render_template, session
from runmodel import get_outputs
import pandas as pd
import zipfile

app = Flask(__name__, template_folder='templates')

# Generated by os.urandom(16)
app.secret_key = b'\xcdt\x8dn\xe1\xbdW\x9d[}yJ\xfc\xa3~/'

responses = []

@app.route('/', methods=['GET'])
def home():
    html = render_template('main.html', length=0, to_generate="recipe")
    response = make_response(html)
    return response

def get_title_recipe(s):
  tokens = s.split(" ")
  for i in range(len(tokens)):
    if tokens[i][0].islower():
      ttokens = tokens[:i-1]
      rtokens = tokens[i-1:]
      return ' '.join(ttokens), ' '.join(rtokens)

@app.route('/generate_recipe2', methods=['GET'])
def generate_recipe2():
    ingredients = request.args.get("ingredients")
    title = request.args.get("title")
    if ingredients != "" and ingredients is not None and title != "" and title is not None:
        input = title + " | " + ingredients
        recipes = get_outputs(input, "recipe generation: ", ["ti-r-50k-12.sav"])
        results = pd.DataFrame()
        results["recipes"] = recipes
        results["titles"] = title
        results["ingredients"] = ingredients
        results["models"] = ["StepChefT5"]
        length = len(results)
    else:
        ingredients = ""
        title = ""
        results = "error"
        length = -1
    
    html = render_template('main.html',
                           results=results,
                           ingredients=ingredients,
                           title=title,
                           length=length,
                           to_generate="recipe2")
    response = make_response(html)
    return response

@app.route('/generate_recipe', methods=['GET'])
def generate_recipe():
    ingredients = request.args.get("ingredients")
    if ingredients is not None and ingredients != "":
        input = ingredients
        titles = get_outputs(input, "recipe title generation: ", ["i-t-50k-5.sav"])
        recipes = get_outputs(titles[0], "recipe generation: ", ["ti-r-50k-12.sav"])

        baseline_recipe = get_outputs(input, "recipe and title generation: ", ["i-tr-50k-10.sav"])
        btitle, brecipe = get_title_recipe(baseline_recipe[0])
        titles.append(btitle)
        recipes.append(brecipe)

        results = pd.DataFrame()
        results["recipes"] = recipes
        results["titles"] = titles
        results["ingredients"] = ingredients
        results["models"] = ["StepChefT5", "Baseline"]
        length = len(results)
    else:
        ingredients = ""
        results = "error"
        length = -1
    
    html = render_template('main.html',
                           results=results,
                           ingredients=ingredients,
                           length=length,
                           to_generate="recipe")
    response = make_response(html)
    return response