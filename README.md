# malgazer

A Python malware analysis library.  Mostly for machine learning purposes.

# Documentation

More info coming soon, along with my dissertation, which will go much deeper into what this is.
For now, this page is all of the documentation for this project.

## Training Logs

You can find logs from different training session in the [training folder](training). 

# Training Data

You can access all the training data I used at:

https://keithjjones-my.sharepoint.com/:f:/p/keith/EqyQqJCh0o9BuKnI2RuVIhYBp-njSmQCT86Wuf9WRhTm4w?e=g4WCnT

# Bugs and Issues

Please file any bugs or issues using the GitHub issues facility above.

# Branches

The "master" branch is for users.  The bleeding edge, and often broken, branch of "develop" is for new features.

# Timeline

This source code supports my dissertation.  The code is not production ready until that time.
Be aware that this code will change often as I add more functionality.  There will be frequent breaking changes.

# Docker

To run the Docker portion of this project, you will need a trained classifier that will predict classifications
with the "predict_sample" function, such as the library.ml.ML class.  Dill Pickle this classifier
(or use train_classifier.py and the resulting saved classifier output from this script)
and place it in samples/ml.dill.

Next, copy ".env.template" to ".env" and fill in any information for your instance.

Next, you can stand up this project with the following command
after you have installed Docker on your system:

```
docker-compose up
```

You can rebuild all of the docker images at any time with the following command:

```
docker-compose build --no-cache
```

This was developed using Docker on a Mac.  Other operating systems have not been tested (yet).

You can start a local registry with:

```
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

## Web

After bringing it up in Docker, you can access the web portion of this project at https://localhost.
Information about the API is on the "API" page of the website.

## API

After bringing it up in Docker, you can access the API portion of this project at https://localhost/api

## Portainer

After bringing it up in Docker, you can access portainer at https://localhost/portainer

## Logs

Logs can be found in docker/logs in a directory for each node in the docker stack.

# Installation

To use this module outside Docker, you will need the requirement.  The following command installs the requirements:

```
pip install -r requirements.txt
```

## python-magic

If you are running Windows or macOS, please make sure the dependencies for 
python-magic are installed.  More information can be found 
at https://github.com/ahupp/python-magic.

# License
This application(s) is/are covered by the Creative Commons BY-SA license.

- https://creativecommons.org/licenses/by-sa/4.0/
- https://creativecommons.org/licenses/by-sa/4.0/legalcode

# Resources

- Magic File
  - https://github.com/ahupp/python-magic
- PE File Structures
  - https://github.com/erocarrera/pefile
- ELF File Structures
  - https://github.com/eliben/pyelftools
  - https://github.com/jacob-baines/elfparser
    - http://www.elfparser.com/
- Mach-O File Structures
  - https://bitbucket.org/ronaldoussoren/macholib
    - http://pythonhosted.org/macholib/
  - https://github.com/rodionovd/machobot
