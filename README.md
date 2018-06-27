# malgazer

A Python malware analysis library.  Mostly for machine learning purposes.

# Documentation

More info coming soon.  This is it for now.

# Bugs and Issues

Please file any bugs or issues using the GitHub issues facility above.

# Timeline

This source code supports my dissertation.  The code is not production ready until that time.
Be aware that this code will change often as I add more functionality.  There will be frequent breaking changes.

# Docker

You can stand up this project with the following command, after you have installed Docker on your system:

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

## API

After bringing it up in Docker, you can access the API portion of this project at http://localhost:8888

# Installation

To use this module, you will need the requirement.  The following command installs the requirements:

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