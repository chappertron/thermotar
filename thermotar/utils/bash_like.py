### funcs that act like tail and split bash commands
import os
import io
import re


def my_tail(file, N=10):
    """Grab the last N lines of file and save as a string IO object????
        Using Method 2 from: https://www.geeksforgeeks.org/python-reading-last-n-lines-of-a-file/

    Might be slow??? But doesn't need whole file to be loaded in to memory, just the desired lines!

    """

    # take buffer size of 8192 bytes
    buffsize = (
        io.DEFAULT_BUFFER_SIZE
    )  # 8192 # how much data is stored from a file stream at a time# can tell the buffer size
    fsize = os.stat(file).st_size  # calculate the size of the file in bytes

    iter = 0

    with open(file) as stream:
        if buffsize > fsize:
            # tweak the buffer size if too big
            buffsize = fsize - 1

        # list for the last N lines
        fetched_lines = []
        output = io.StringIO()

        # start at eof
        stream.seek(fsize)

        # while loop for fetching these lines:
        while not (
            len(fetched_lines) >= N or stream.tell() == 0
        ):  # greater than used to ensure that no partial lines are captured eg if N lines requested is part way through the buffer
            # iterate until the number of lines desired is found.
            # essentially looks through 8 kb (default on my system at least) of data from eof at a time until at least the N desired lines are found or BOF is found

            # counts the number of iterations
            iter += 1

            # move cursor to the last nth line of the file
            stream.seek(fsize - buffsize * iter)

            # storing each line in list to end of fil
            fetched_lines.extend(
                stream.readlines()
            )  # the very first line could be partial, by selecting only [1:] this ensures it isn't kept

            # write directly to StringIO object

            # end program when eof or > requested lines
            print(iter)
            if iter == 1:
                print("".join(fetched_lines))
                print("Lines = ", len(fetched_lines))

        output.writelines(fetched_lines[-N:])
        output.getvalue()

        output.seek(0)  # reset position of output to beginning so it can be read later

        # print('methods are the same',(contents == feteched_contents))
        # print(contents)
    return output


def _tail(file, N=20):
    """From https://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-similar-to-tail TODO understand and merge with above!!!"""

    total_lines_wanted = N

    with open(file, "rb") as f:  # open file as binary
        BLOCK_SIZE = 1024  # io.DEFAULT_BUFFER_SIZE #1024
        f.seek(0, 2)  # get last line from the end
        block_end_byte = f.tell()
        lines_to_go = total_lines_wanted
        block_number = -1
        blocks = []
        while lines_to_go > 0 and block_end_byte > 0:
            if (
                block_end_byte - BLOCK_SIZE > 0
            ):  # if previous block ended on one larger than the blocksize, see -*n iters from end
                f.seek(block_number * BLOCK_SIZE, 2)
                blocks.append(f.read(BLOCK_SIZE))
            else:
                f.seek(0, 0)
                blocks.append(f.read(block_end_byte))
            lines_found = blocks[-1].count(
                b"\n"
            )  # grab blocks found so far and chuck into a list #reverse tje order of the block too # number of lines total # covert the \n into binary before searching # does this check for broken lines?
            lines_to_go -= lines_found
            block_end_byte -= BLOCK_SIZE
            block_number -= 1
        all_read_text = b"".join(
            reversed(blocks)
        )  # is reversed, because last block is found first
    desired_bytes = b"\n".join(
        all_read_text.splitlines()[-total_lines_wanted:]
    )  # string it all together with just the lines wanted
    desired_bytes.decode(
        "UTF-8"
    )  ## perhaps can use what ever encoding is used initially or by default??

    output = io.TextIOWrapper(
        io.BytesIO(desired_bytes), encoding="utf-8"
    )  # io.StringIO(desired_text)
    # output.writelines(desired_text)
    # output.seek(0)

    return output  # desired_text


def tail(file, N=20):
    """
    Returns the last `window` lines of file `f` as a list.
    f - a byte file-like object
    """

    with open(file, "rb") as f:
        window = N

        if window == 0:
            return []
        BUFSIZ = 1024
        f.seek(0, 2)
        curr_bytes = f.tell()
        size = window + 1
        block = -1
        data = []
        while size > 0 and curr_bytes > 0:
            if curr_bytes - BUFSIZ > 0:
                # Seek back one whole BUFSIZ
                f.seek(block * BUFSIZ, 2)
                # read BUFFER
                data.insert(0, f.read(BUFSIZ))
            else:
                # file too small, start from begining
                f.seek(0, 0)
                # only read what was not read
                data.insert(0, f.read(curr_bytes))
            linesFound = data[0].count(b"\n")
            size -= linesFound
            curr_bytes -= BUFSIZ
            block -= 1

    desired_bytes = b"\n".join(b"".join(data).splitlines()[-window:])
    output = io.TextIOWrapper(io.BytesIO(desired_bytes), encoding="utf-8")

    return output  #''.join(data).splitlines()[-window:]


def glob_to_re_group(glob_path: str):
    """Convert a glob expression to an equivalent re that captures what the unknowns are in the original glob"""
    # create groups for the matched terms
    map_re_glob = {
        r"\?\*": r"(.+)",
        r"\*": r"(.*)",
    }  # converts * to a group with 0 or more, converts '?*' to a group with one or more

    re_glob = glob_path

    for glob_form, re_form in map_re_glob.items():
        re_glob = re.sub(glob_form, re_form, re_glob)

    return re_glob


if __name__ == "__main__":
    filename = "./test_files/profile2d.dat"  #'../../test_files/rep1/aver.xvg'

    N = 26896  # 147

    tail_out = tail(filename, N=N)

    # checking the Nth line from the bottom with the naieve approach

    with open(filename) as stream:
        line_N = stream.readlines()[-N]

    print("basic approach: ", line_N)

    with tail_out as stream:
        print(stream.readline())
        print(stream.readline())
        print(stream.readline())


# csplit command I managed to get work csplit -f prof3D_ ../profile3d.dat /^[0-9][0-9]*\ [0-9][0-9]*\ [0-9][0-9]*$/ '{*}'
# execept for 3d cuz has an e+ in csplit -f prof3D_ ../profile3d.dat /^[0-9][0-9]*\ [0-9][0-9]*\ [0-9][0-9e.+]*$/ '{*}'
