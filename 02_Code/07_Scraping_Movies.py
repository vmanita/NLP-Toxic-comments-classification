#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:18:41 2019

@author: Manita
"""
'''
This script does the automatic web scapping of all movies in IMSDB.
It takes a really long time. We ran it one time and saved the scripts in a folder
'''

# SCRAPPING
from bs4 import BeautifulSoup
import requests
import urllib.request
from selenium import webdriver
import codecs

def get_all_movies():
    link_all_scripts = 'http://www.imsdb.com/all%20scripts/'
    response_all_scripts = requests.get(link_all_scripts)
    soup = BeautifulSoup(response_all_scripts.text, 'html.parser')

    # we want the third table -> has movie names
    find_tables = soup.findAll('td', valign='top')
    all_movies = find_tables[2].findAll('a')

    movies = [(movie_info.string, \
              movie_info["href"], \
              re.split("[,.]",movie_info.string)[0].replace(' ', '_'))
              for movie_info in all_movies]
    return movies

def check_movie_info(movies):
    for movie in movies:
        if movie[1][0:15] !='/Movie Scripts/':
            return 'One of the movie link does not start with /Movie Scripts/.'
    return 'All movie URLs have a correct format.'

movies = get_all_movies()
#------------------------------------------------------------------------------
folder = '/Users/Manita/OneDrive - NOVAIMS/Text Mining/Poject/scripts/scraping/'
#------------------------------------------------------------------------------

def handle_movie (movie, browser):
    # Unpack tuple
    title, link_to_movie_page, movie_title = movie

    # Interrogate the page with all the movie information (ratings, writer,
    # genre, link to script)
    full_html_link = u'http://www.imsdb.com' + link_to_movie_page
    response_script = requests.get(full_html_link)
    soup = BeautifulSoup(response_script.text, 'html.parser')

    # Get all relevant information (genre, writer, script) from page
    list_links = soup.findAll('table', "script-details")[0].findAll('a')
    genre = []
    writer = []
    script = ''
    for link in list_links:
        href = link['href']
        if href[0:7]== "/writer":
            writer.append(link.get_text())
        if href[0:7]== "/genre/":
            genre.append(link.get_text())
        if href[0:9]== "/scripts/":
            script = href

    # If the link to the script points to a PDF, skip this movie, but log
    # the information in `movies_pdf_script.csv`
    if script == '' or script[-5:] != '.html':
        path_to_directory = folder
        pdf_logging_filename = path_to_directory + 'movies_pdf_script.csv'
        with open(pdf_logging_filename, 'a') as f:
            new_row = title + '\n'
            f.write(new_row)

    # If the link to the script points to an html page, write the corresponding
    # text to a file and include the movie in a csv file, with meta-information
    else:

        # Parse the webpage which contains the script text
        full_script_url =  u'http://www.imsdb.com' + script
        browser.get(full_script_url)
        page_text = browser.page_source
        soup = BeautifulSoup(page_text, 'html.parser')

        # If the scraping does not go as planned (unexpected structure),
        # log the file name in an error file
        if len(soup.findAll('td', "scrtext"))!=1:
            error_file_name = folder + 'scraping_error.csv'
            with open(error_file_name, 'a') as error_file:
                new_row = title + '\n'
                error_file.write( new_row )

        # Normal scraping:
        else:
            # Write the script text to a file
            path_to_directory = folder+'texts/'
            filename = path_to_directory + movie_title + '.txt'
            text = soup.findAll('td', "scrtext")[0].get_text()
            with codecs.open(filename, "w",
                    encoding='ascii', errors='ignore') as f:
                f.write(text)

            # Add the meta-information to a CSV file
            path_to_directory = folder
            success_filename = path_to_directory + 'successful_files.csv'
            new_row = title + ';' + str(genre) + ';' + str(writer) + ';' \
                    + movie_title + ';' + filename + '\n'
            with open(success_filename, 'a') as f:
                f.write(new_row)
                
                

CHROME_DRIVER_PATH = "/Users/Manita/OneDrive - NOVAIMS/Text Mining/Poject/scripts/scraping/chromedriver_2"
browser = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH)

# Download all movie scripts
# ERROR IN MOVIE 814 -> start from oblivion movies[1168:][0]
for i,movie in enumerate(movies[1168:]):
        handle_movie(movie, browser)
        print("----------------------")
        print(movie)
        print("----------------------")
        print("\n")















