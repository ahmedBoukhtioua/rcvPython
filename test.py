import os
import numpy as np
import cv2
from PIL import Image
import pytesseract

import io
import re
import pandas as pd
import spacy
import datetime
import string
from collections import Counter

from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

for file in os.listdir(r"C:\Users\hp\PycharmProjects\untitled/set10"):

    if file.endswith(".jpg"):
        file_path = r"C:\Users\hp\PycharmProjects\untitled/set10/" + str(file)
        img = cv2.imread(file_path)
        ratio = img.shape[0] / 500.0
        original_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
        (__, cnts, __) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (240, 0, 159), 3)
        H, W = img.shape[:2]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 100 and (0.7 < w / h < 1.3) and (W / 4 < x + w // 2 < W * 3 / 4) and (
                    H / 4 < y + h // 2 < H * 3 / 4):
                break
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        scanned_file_name = r"C:\Users\hp\PycharmProjects\untitled/set10/" + str(file[:-4]) + "-Scanned.png"
        cv2.imwrite(scanned_file_name, dst)
        file_text = pytesseract.image_to_string(Image.open(scanned_file_name))
        textCVTest = r"C:\Users\hp\PycharmProjects\untitled/set10/" + str(file[:-4]) + "-Scanned.txt"

        with open(textCVTest, "a") as f:
            f.write(file_text + "\n")


def extract_mobile_number(text):
    pattern = re.compile(
        r'(\+\d{3}\s\d{8})|(\+\d{3}\s\d{2}.\d{3}.\d{3})|(\+\d{3}\s\d{2}\s\d{3}\s\d{3})|(\d{2}\s\d{3}\s\d{3})|(\d{2}.\d{3}.\d{3})|(\d{2}-\d{3}-\d{3})|(\(\+\d{3}\)\s\d{8})')
    phone = pattern.findall(text)
    if phone:
        number = ''.join(phone[0])
        return number


# function that extract email from CvTmp
def extract_email(text):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].lower()
        except IndexError:
            return None


# return common letters between two strings
def CountCommonChar(s1, s2):
    s3 = s1.lower()
    s4 = s2.lower()
    common_letters = Counter(s3) & Counter(s4)
    return sum(common_letters.values())


# function that extract name from CvTmp
def extract_name(resume_text):
    PersonsMatches = []
    ListSub = []
    Persons = []
    nlp_text = nlp(resume_text)

    pattern = [{'LIKE_EMAIL': False, 'LIKE_NUM': False, 'LIKE_URL': False, "IS_ALPHA": True},
               {'LIKE_EMAIL': False, 'LIKE_NUM': False, 'LIKE_URL': False, "IS_ALPHA": True},
               {'LIKE_EMAIL': False, 'LIKE_NUM': False, 'LIKE_URL': False, "IS_ALPHA": True, 'OP': '?'}]
    matcher.add('NAME', None, pattern)
    matches = matcher(nlp_text)
    if extract_email(resume_text) is None:
        return "Non trouve"
    else:
        for match in matches:
            PersonsMatches.append(nlp_text[match[1]:match[2]])

        for Person in PersonsMatches:
            nlp_person = nlp(Person.text)

            if (nlp_person[0].text.lower() in extract_email(resume_text).lower() and nlp_person[
                -1].text.lower() in extract_email(resume_text).lower()):
                Persons.append(Person.text)
                ListSub.append(CountCommonChar(Person.text, extract_email(resume_text)))
        if len(ListSub) == 0:
            return "Non trouve"
        else:
            maxMatch = max(ListSub)
            max_index = ListSub.index(maxMatch)
            return Persons[max_index]


# function that extract skills from CvTmp
def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]

    # reading the csv file
    data = pd.read_csv('skills.csv')

    # extract values
    skills = list(data.columns.values)
    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize().lower() for i in set([i.lower() for i in skillset])]


# function that extract languages from CvTmp
def extract_langues(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    # reading the csv file
    data = pd.read_csv('langue.csv').apply(lambda x: x.astype(str).str.lower())
    # extract values
    skills = list(data.columns.str.lower().values)
    skillset = []
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize().lower() for i in set([i.lower() for i in skillset])]


# function that extract age from CvTmp
def extract_age(text):
    match = re.findall(r'\d{4}', text)
    age = []
    for m in match:
        date = datetime.datetime.strptime(m, '%Y').strftime("%Y")

        if 1960 < int(date) < 2000:
            now = datetime.datetime.now()
            age.append((int(now.year) - int(date)))

    if len(age) == 0:
        return None
    else:
        return max(age)


# function that extract years of experience from CvTmp
def extract_Year_of_experience(text):
    match = re.findall(r'\d{4}', text)
    experience = []
    for m in match:
        date = datetime.datetime.strptime(m, '%Y').strftime("%Y")
        now = datetime.datetime.now()
        if extract_age(text) is not None:

            if ((int(now.year) - extract_age(text)) + 1) < int(date) < int(now.year):
                experience.append(int(date))
        else:
            if 2000 < int(date) < int(now.year):
                experience.append(int(date))
    if len(experience) == 0:
        return None
    else:
        return max(experience) - min(experience)


import json

Cv = {"Nom & prenom": extract_name(file_text), "Email": extract_email(file_text),
      "Phone Number": extract_mobile_number(file_text), "Skills": extract_skills(file_text),
      "Langues": extract_langues(file_text), "age": extract_age(file_text),
      "Years of experience": extract_Year_of_experience(file_text)}

with open(" data.json", 'w', encoding='latin-1') as outfile:
    json.dump(Cv, outfile, ensure_ascii=False)
