# malgazer

A Python malware analysis library.  Mostly for machine learning purposes.

# Documentation

More info coming soon, along with my dissertation, which will go much deeper into what this is.
For now, this page is all of the documentation for this project.

# Bugs and Issues

Please file any bugs or issues using the GitHub issues facility above.

# Timeline

This source code supports my dissertation.  The code is not production ready until that time.
Be aware that this code will change often as I add more functionality.  There will be frequent breaking changes.

# Docker

To run the Docker portion of this project, you will need a trained classifier that will predict classifications
with the "predict_sample" function, such as the library.ml.ML class.  Dill Pickle this classifier and place it in
samples/ml.dill.

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

## Web

After bringing it up in Docker, you can access the web portion of this project at http://localhost:8080.
Information about the API is on the "API" page of the website.

## API

After bringing it up in Docker, you can access the API portion of this project at http://localhost:8888

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