"""
=================================== LICENSE ==================================
Copyright (c) 2021, Consortium Board ROXANNE
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

Neither the name of the ROXANNE nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY CONSORTIUM BOARD ROXANNE ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL CONSORTIUM BOARD TENCOMPETENCE BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
==============================================================================

 Helpers to deal with files """
from os.path import isfile
from falcon import HTTP_400
from ..exceptions import ValidationError


CHUNK_SIZE = 4096


class FileExistsError(ValidationError):
    """ Error saving file, file already exists """
    code = "FileExistsError"
    response_code = HTTP_400


def save_file(file, path):
    """
    Saves file into given path

    :param file: WTForms file
    :param path: Path for the file
    :raises FileExistsError: Can't create file, because file with given name already exists
    :returns: path to the file
    """
    if isfile(path):
        raise FileExistsError("File already exists!", "filename")
    with open(path, "wb") as writefile:
        while True:
            buffer = file.read(CHUNK_SIZE)
            if not buffer:
                break
            writefile.write(buffer)
    return path


def stream_file(filepath):
    """
    Stream file by path

    :param filepath: path to the file to be streamed
    """
    with open(filepath, "rb") as read_file:
        while True:
            buffer = read_file.read(CHUNK_SIZE)
            if not buffer:
                break
            yield buffer
