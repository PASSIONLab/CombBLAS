import os
import re
import shutil

# Configuration
TARGET_EXTENSIONS = ('.c', '.cpp', '.h', '.hpp')
PLACEHOLDER = 'std::placeholders::_1'
INCLUDE_STATEMENT = '#include <functional>'
BACKUP_SUFFIX = '.bak'
LOG_FILE = 'bind2nd_replacement.log'

def backup_file(file_path):
    backup_path = file_path + BACKUP_SUFFIX
    #shutil.copyfile(file_path, backup_path)
    print(f"Backup created: {backup_path}")

def add_include(content):
    include_pattern = re.compile(r'#include\s*<functional>')
    if include_pattern.search(content):
        return content  # Include already present

    # Find the last include statement to insert after
    includes = list(re.finditer(r'#include\s*[<"].*[>"]', content))
    if includes:
        last_include = includes[-1]
        insert_pos = last_include.end()
        new_content = content[:insert_pos] + '\n' + INCLUDE_STATEMENT + content[insert_pos:]
        return new_content
    else:
        # If no includes are present, add at the top
        return INCLUDE_STATEMENT + '\n' + content

def replace_bind2nd(content):
    """
    Replaces bind2nd with std::bind and inserts std::placeholders::_1.

    Example:
    std::bind2nd(std::greater<int>(), 3)
    becomes
    std::bind(std::greater<int>(), std::placeholders::_1, 3)
    """
    pass
    # # Regex to find bind2nd usage
    # # This pattern looks for bind2nd(function, value)
    # pattern = re.compile(r'bind2nd\s*\(\s*(.*)\s*,\s*(.*)\s*\)')

    # def replacer(match):
    #     func = match.group(1).strip()
    #     value = match.group(2).strip()
    #     # Replace with std::bind(func, std::placeholders::_1, value)
    #     return f'bind({func}, {PLACEHOLDER}, {value})'

    # new_content, num_subs = pattern.subn(replacer, content)
    # return new_content, num_subs
    




def traverse_and_process(root_dir):
    bind2ndcnt = 0
    
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(TARGET_EXTENSIONS):
                    file_path = os.path.join(subdir, file)
                    try:
                        newcontent = ""
                        with open(file_path, 'r', encoding='utf-8') as file:
                            while 1:
                                line = file.readline()
                                if not line:
                                    break
                                substring = "bind2nd"
                                start_index = line.find(substring)
                                if start_index != -1:
                                    # print(f"Substring found at index {start_index}")
                                    endidx = start_index + len(substring)
                                    left = 0
                                    right = 0
                                    coma = 0
                                    comaloc = 0
                                    endidxstart = -1
                                    while endidx < len(line):
                                        if line[endidx] == '(':
                                            if left == 0: # left parenthesis is found first time
                                                endidxstart = endidx
                                            left += 1
                                        elif line[endidx] == ')':
                                            right += 1
                                            if coma == 1 and left == right:
                                                bind2ndcnt += 1
                                                # print("bind2nd found")
                                                break
                                        elif line[endidx] == ',' and left > right:
                                            coma += 1
                                            comaloc = endidx
                                        endidx += 1
                                    print(line[start_index:endidx+1], line[endidxstart+1:comaloc], line[comaloc+1:endidx])
                                    print("old line: ", line)
                                    line = line[0:start_index] + "bind(" + line[endidxstart+1:comaloc] + ", " + PLACEHOLDER + ", " + line[comaloc+1:endidx] + ")" + line[endidx+1:]
                                    print("new line: ", line)
                                substring = "bind1st"
                                start_index = line.find(substring)
                                if start_index != -1:
                                    print(f"Substring found at index {start_index}")
                                    endidx = start_index + len(substring)
                                    left = 0
                                    right = 0
                                    coma = 0
                                    comaloc = 0
                                    endidxstart = -1
                                    while endidx < len(line):
                                        if line[endidx] == '(':
                                            if left == 0: # left parenthesis is found first time
                                                endidxstart = endidx
                                            left += 1
                                        elif line[endidx] == ')':
                                            right += 1
                                            if coma == 1 and left == right:
                                                bind2ndcnt += 1
                                                # print("bind2nd found")
                                                break
                                        elif line[endidx] == ',' and left > right:
                                            coma += 1
                                            comaloc = endidx
                                        endidx += 1
                                    print(line[start_index:endidx+1], line[endidxstart+1:comaloc], line[comaloc+1:endidx])
                                    print("old line: ", line)
                                    line = line[0:start_index] + "bind(" + line[endidxstart+1:comaloc] + ", " + line[comaloc+1:endidx] + ", " + PLACEHOLDER + " )" + line[endidx+1:]
                                    print("new line: ", line)
                                newcontent += line
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(newcontent)
                    except Exception as e:
                        log.write(f"Error processing {file_path}: {e}\n")
                        print(f"Error processing {file_path}: {e}")
    print(f"\nProcessing complete. See '{LOG_FILE}' for details.")
    print("bind2nd count: ", bind2ndcnt)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replace bind2nd with std::bind in C/C++ files.")
    parser.add_argument('directory', help="Path to the target directory.")
    args = parser.parse_args()

    target_directory = args.directory

    if not os.path.isdir(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
        exit(1)

    traverse_and_process(target_directory)

