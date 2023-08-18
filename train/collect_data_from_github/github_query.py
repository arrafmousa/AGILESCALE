#!/usr/bin/env python3.6.2
# -*- coding: utf-8 -*-
import base64
import os

# run script from command line via python3 keyword_search_github_repositories.py

import click
import time

from github import Github
from github.GithubException import RateLimitExceededException
from tqdm import tqdm


def search_github(auth: Github, keyword: list) -> list:
    """Search the GitHub API for repositories using an input keyword.
    Args:
        auth: A Github authenticate object.
        keyword: A keyword string.
    Returns:
        A nested list of GitHub repositories returned for a keyword. Each result list contains the repository name,
        url, and description.
    """

    print('Searching GitHub using keyword: {}'.format(keyword))

    # set-up query
    query = keyword
    results = auth.search_code("".join(query))

    # print results
    print(f'Found {results.totalCount} repo(s)')

    results_list = []
    for repo in tqdm(range(0, results.totalCount)):
        try:
            results_list.append([results[repo].name, results[repo].url])
            time.sleep(2)
        except RateLimitExceededException:
            time.sleep(60)
            results_list.append([results[repo].name, results[repo].url, results[repo].description])
        try:
            for content in [results[repo]]:
                # check if it's a Python file
                if content.path.endswith(".py"):
                    # save the file
                    filename = os.path.join("pubchempy-files", f"{results[repo].name.replace('/', '-')}")
                    with open(filename, 'w') as f:
                        f.write(results[repo].decoded_content)
                    with open("combined.txt", "ab") as comb:
                        comb.write(results[repo].decoded_content)
        except Exception as e:
            print("Error:", e)

    return results_list


def pull_github(keywords: str, filename: str) -> None:
    # initialize and authenticate GitHub API
    auth = Github(
        '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')  # TODO: add your token here

    # search a list of keywords
    search_list = [keyword.strip() for keyword in keywords.split(',')]

    # search repositories on GitHub
    github_results = dict()
    for key in search_list:
        github_results[key] = []
        github_results[key] += search_github(auth, key)
        if len(search_list) > 1: time.sleep(120)


@click.command()
# @click.option('--token', prompt='Please enter your GitHub Access Token')
@click.option('--keywords', prompt='Please enter the keywords separated by a comma')
@click.option('--filename', prompt='Please provide the file path')
def main(keywords: str, filename: str) -> None:
    pull_github(keywords, filename)


if __name__ == '__main__':
    main()
