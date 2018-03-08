#!/usr/bin/python
#
# Copyright 2012 Google Inc. All Rights Reserved.

"""Download more than 100 files product of a VT Intelligence search.

VirusTotal Intelligence allows you to download up to the top100 files that
match a given search term. The 100 file limit is a server-side limitation
when creating the file packages. In order to overcome this limitation this
small script has been developed, it will paginate over a given Intelligence
search and download the matching files individually.
"""

__author__ = 'emartinez@virustotal.com (Emiliano Martinez)'


import json
import logging
import optparse
import os
import queue
import re
import socket
import sys
import threading
import time
import urllib
from urllib.request import urlopen, Request


API_KEY = 'NOTHING - You can put your key here.'
INTELLIGENCE_SEARCH_URL = ('https://www.virustotal.com/intelligence/search/'
                           'programmatic/')
INTELLIGENCE_DOWNLOAD_URL = ('https://www.virustotal.com/intelligence/download/'
                             '?hash=%s&apikey=%s')

NUM_CONCURRENT_DOWNLOADS = 5

LOCAL_STORE = 'intelligencefiles'

socket.setdefaulttimeout(60)

LOGGING_LEVEL = logging.INFO  # Modify if you just want to focus on errors
logging.basicConfig(level=LOGGING_LEVEL,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout)


class Error(Exception):
  """Base-class for exceptions in this module."""


class InvalidQueryError(Error):
  """Search query is not valid."""


def create_download_folder(query=None):
  """Creates a folder to store the downloaded files.

  The Intelligence query issues is stored in a separate txt file inside the
  directory created, this will allow the user to remember the query he performed
  at a later time.

  Args:
    query: the Intelligence search query, as a string, that is issued in order
      to save the corresponding files to the directory being created.

  Returns:
    String with the path of the created folder.
  """
  folder_name = time.strftime('%Y%m%dT%H%M%S')
  if not os.path.exists(LOCAL_STORE):
    os.mkdir(LOCAL_STORE)
  folder_path = os.path.join(LOCAL_STORE, folder_name)
  if not os.path.exists(folder_path):
    os.mkdir(folder_path)
  if query:
    query_path = os.path.join(folder_path, 'intelligence-query.txt')
    with open(query_path, 'wb') as query_file:
      query_file.write(query)
  return folder_path


def get_matching_files(search, page=None):
  """Get a page of files matching a given Intelligence search.

  Args:
    search: a VirusTotal Intelligence search phrase. More about Intelligence
      searches at: https://www.virustotal.com/intelligence/help/
    page: a token indicating the page of file results that should be retrieved.

  Returns:
    Tuple with a token to retrieve the next page of results and a list of sha256
    hashes of files matching the given search conditions.

  Raises:
    InvalidQueryError: if the Intelligence query performed was not valid.
  """
  response = None
  page = page or 'undefined'
  attempts = 0
  parameters = {'query': search, 'apikey': API_KEY, 'page': page}
  data = urllib.urlencode(parameters)
  request = Request(INTELLIGENCE_SEARCH_URL, data)
  while attempts < 10:
    try:
      response = urlopen(request).read()
      break
    except Exception:
      attempts += 1
      time.sleep(1)
  if not response:
    return (None, None)

  try:
    response_dict = json.loads(response)
  except ValueError:
    return (None, None)

  if not response_dict.get('result'):
    raise InvalidQueryError(response_dict.get('error'))

  next_page = response_dict.get('next_page')
  hashes = response_dict.get('hashes', [])
  return (next_page, hashes)


def download_file(file_hash, destination_file=None):
  """Downloads the file with the given hash from Intelligence.

  Args:
    file_hash: either the md5, sha1 or sha256 hash of a file in VirusTotal.
    destination_file: full path where the given file should be stored.

  Returns:
    True if the download was successful, False if not.
  """
  destination_file = destination_file or file_hash
  download_url = INTELLIGENCE_DOWNLOAD_URL % (file_hash, API_KEY)
  attempts = 0
  while attempts < 3:
    try:
      urllib.urlretrieve(download_url, destination_file)
      return True
    except Exception:
      attempts += 1
  return False


def main():
  """Download the top-n results of a given Intelligence search."""
  usage = 'usage: %prog [options] <intelligence_query/local_file_with_hashes>'
  parser = optparse.OptionParser(
      usage=usage,
      description='Allows you to download the top-n files returned by a given'
      'VirusTotal Intelligence search. Example: '
      'python %prog -n 10 type:"peexe" positives:5+')
  parser.add_option(
      '-n', '--numfiles', dest='numfiles', default=100,
      help='number of files to download')
  parser.add_option(
      '-k', '--apikey', dest='API_KEY',
      help='VT Public API Key')
  (options, args) = parser.parse_args()
  if not args:
    parser.error('No search query provided')

  end_process = False
  search = ' '.join(args)
  search = search.strip().strip('\'')
  numfiles = int(options.numfiles)

  if os.path.exists(search):
    with open(search, 'rb') as file_with_hashes:
      content = file_with_hashes.read()
      requested_hashes = re.findall(
          '([0-9a-fA-F]{64}|[0-9a-fA-F]{40}|[0-9a-fA-F]{32})', content)
      search = ','.join(set(requested_hashes))

  logging.info('Starting VirusTotal Intelligence downloader')
  logging.info('* VirusTotal Intelligence search: %s', search)
  logging.info('* Number of files to download: %s', numfiles)

  work = queue.Queue()  # Queues files to download
  end_process = False

  def worker():
    while not end_process:
      try:
        sha256, folder = work.get(True, 3)
      except Queue.Empty:
        continue
      destination_file = os.path.join(folder, sha256)
      logging.info('Downloading file %s', sha256)
      success = download_file(sha256, destination_file=destination_file)
      if success:
        logging.info('%s download was successful', sha256)
      else:
        logging.info('%s download failed', sha256)
      work.task_done()

  threads = []
  for unused_index in range(NUM_CONCURRENT_DOWNLOADS):
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    threads.append(thread)

  logging.info('Creating folder to store the requested files')
  folder = create_download_folder(search)

  queued = 0
  wait = False
  next_page = None
  while not end_process:
    try:
      logging.info('Retrieving page of file hashes to download')
      try:
        next_page, hashes = get_matching_files(search, page=next_page)
      except InvalidQueryError as e:
        logging.info('The search query provided is invalid... %s', e)
        return
      if hashes:
        logging.info(
            'Retrieved %s matching files in current page, queueing them',
            len(hashes))
        for file_hash in hashes:
          work.put([file_hash, folder])
          queued += 1
          if queued >= numfiles:
            logging.info('Queued requested number of files')
            wait = True
            break
      if not next_page or not hashes:
        logging.info('No more matching files')
        wait = True
      if wait:
        logging.info('Waiting for queued downloads to finish')
        while work.qsize() > 0:
          time.sleep(5)
        end_process = True
        for thread in threads:
          if thread.is_alive():
            thread.join()
        logging.info('The downloaded files have been saved in %s', folder)
    except KeyboardInterrupt:
      end_process = True
      logging.info('Stopping the downloader, initiated downloads must finish')
      for thread in threads:
        if thread.is_alive():
          thread.join()


if __name__ == '__main__':
  main()