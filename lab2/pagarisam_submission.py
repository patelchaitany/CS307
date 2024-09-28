import heapq
import random
from collections import defaultdict
import spacy
import string
import re
from spacy.cli import download
import nltk
from nltk.tokenize import sent_tokenize

# download("en_core_web_sm")
"""
State Representation:
    index of statment in input document.
    index of statment in targate document.
    move which we perform to reach that state
"""


def read_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


def extract_sentences(text):
    # nltk.download("punkt_tab")
    text = text.lower()
    nlp = spacy.load("en_core_web_sm")
    sentences = nlp(text)
    sentences = [sent.text for sent in sentences.sents]
    # sentences = re.split(r"(?<=[.!?])(?<!\.\.)", text)
    # sentences = sent_tokenize(text)
    return sentences


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans("\n", " ", string.punctuation))


# Main function
def process_text_file(file_path):
    # Step 1: Read the file
    content = read_file(file_path)

    # Step 2: Extract sentences
    sentences = extract_sentences(content)

    # Step 3: Remove punctuation from each sentence
    print(sentences)
    cleaned_sentences = [remove_punctuation(sentence) for sentence in sentences]

    return cleaned_sentences


document1 = list()

document2 = list()


class Node:
    def __init__(self, state, parent=None, g=0, h=0, w1=1, w2=1):
        self.state = state
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost estimate to goal
        self.f = w1 * g + w2 * h  # Total cost

    def __lt__(self, other):
        return self.f < other.f


def get_dffrance(index_doc1=None, index_doc2=None):
    """
    Input:
        index_doc1 : index of sentences in document one (source document)
        index_doc2 : index of sentences in document two (target document)

    Output:
        Number of characters that are present in sentence2 but not in sentence1.
        Also, add the difference between the lengths of sentence1 and sentence2
        only if len(sentence1) > len(sentence2).

    Method:
        Count characters that are present in sentence2 and not present in sentence1.
        Also add the difference in sentence lengths if len(sentence1) > len(sentence2).
        Do this for the whole length of document 1 and 2, starting from index_doc1 and index_doc2.
    """
    sentence1 = document1[index_doc1] if index_doc1 is not None else ""
    sentence2 = document2[index_doc2] if index_doc2 is not None else ""

    # Convert sentences into character lists
    chars1 = list(sentence1)
    chars2 = list(sentence2)

    char_count = defaultdict(int)
    char_count1 = defaultdict(int)

    for char in chars1:
        char_count1[char] += 1

    # Decrement count for characters in sentence2
    difference = 0
    if len(chars1) > len(chars2):
        difference += len(chars1) - len(chars2)

    for char in chars2:
        if char_count1[char] == 0:
            char_count[char] += 1
            difference += 1
        else:
            char_count1[char] -= 1

    return difference


def distance(state, goal_state):
    index_doc1, index_doc2, m = list(state)
    len_doc1, len_doc2, _ = list(goal_state)

    distance = 0

    while index_doc1 <= len_doc1 or index_doc2 <= len_doc2:
        if index_doc1 <= len_doc1 and index_doc2 <= len_doc2:
            distance = distance + get_dffrance(index_doc1, index_doc2)

        elif index_doc1 <= len_doc1 and index_doc2 > len_doc2:
            distance = distance + get_dffrance(index_doc1, None)

        elif index_doc1 > len_doc1 and index_doc2 <= len_doc2:
            distance = distance + get_dffrance(None, index_doc2)

        index_doc2 = index_doc2 + 1
        index_doc1 = index_doc1 + 1

    return distance


def char_level_edit_distance(sentence1, sentence2):
    # Convert sentences to strings of characters
    str1 = sentence1
    str2 = sentence2

    # Get the lengths of both strings
    n = len(str1)
    m = len(str2)

    # Initialize the matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Fill the matrix
    for i in range(1, n + 1):
        dp[i][0] = i  # Deletion cost
    for j in range(1, m + 1):
        dp[0][j] = j  # Insertion cost

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost if the characters are the same
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + 1,  # Substitution
                )

    # The edit distance is the value in the bottom-right corner of the matrix
    return dp[n][m]


def Edit_distance(state, goal_state):
    index_doc1 = list(state)[0]
    index_doc2 = list(state)[1]
    move = list(state)[2]
    ans = 0

    if move == 0:
        sentence1 = document1[index_doc1 - 1]
        sentence2 = document2[index_doc2 - 1]
        ans = char_level_edit_distance(sentence1, sentence2)
    elif move == 1:
        sentence2 = document2[index_doc2 - 1]
        ans = len(sentence2)
    elif move == 2:
        sentence1 = document2[index_doc1 - 1]
        ans = len(sentence1)

    return ans


def get_successors(node: Node):
    """
    Moves (a,b,c):
        a = index in the input document
        b = index in the targate document
        c = is the move which required to perform like(insertion,deletion,skip)

    C :
        0 : aligment is performed.
        1 : insertion happend (on the first document happend).
        2 : deletion happend (on the first document happend).
    """
    moves = [(1, 1, 0), (0, 1, 1), (1, 0, 2)]
    parent_node = node
    curr_state = list(node.state)
    successor = list()
    for move in moves:
        mv = list(move)
        new_state = (curr_state[0] + mv[0], curr_state[1] + mv[1], mv[2])
        new_node = Node(new_state, node)
        successor.append(new_node)

    return successor


def a_star(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited = set()
    nodes_explored = 0

    while open_list:
        _, node = heapq.heappop(open_list)
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))

        nodes_explored += 1

        if (
            list(node.state)[0] == list(goal_node.state)[0] + 1
            and list(node.state)[1] == list(goal_node.state)[1] + 1
        ):
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("Total nodes explored", nodes_explored)

            return path

        for successor in get_successors(node):
            if (
                list(successor.state)[0] <= list(goal_node.state)[0] + 1
                and list(successor.state)[1] <= list(goal_node.state)[1] + 1
            ):
                successor.g = node.g + Edit_distance(successor.state, goal_node.state)
                successor.h = distance(successor.state, goal_node.state)
                successor.f = successor.g + successor.h
                heapq.heappush(open_list, (successor.f, successor))
            print(node.state)
            print(
                f"Cost : {successor.state} {successor.f} {successor.h} {successor.g} -> {successor.parent.state if successor.parent is not None else None}"
            )
            print()

    print("Total nodes explored", nodes_explored)
    return None


def aligment_doc(states, start_state, goal_state):
    new_doc = []
    for i in states:
        if list(i)[0] == list(start_state)[0] and list(i)[1] == list(start_state)[1]:
            continue
        elif list(i)[-1] == 0:
            new_doc.append(document1[list(i)[0] - 1])
        elif list(i)[-1] == 1:
            new_doc.append(document2[list(i)[1] - 1])
        elif list(i)[-1] == 2:
            continue

        if list(i)[0] == list(goal_state)[0] and list(i)[1] == list(goal_state)[1]:
            print("goal state is reached")
    return new_doc


def calculate(sentence):
    words = sentence.split()
    return len(words)


if __name__ == "__main__":
    document1 = process_text_file("doc1.txt")
    document2 = process_text_file("doc2.txt")
    start_state = (0, 0, 0)
    end_state = (len(document1) - 1, len(document2) - 1, 0)
    print(end_state)
    ans = a_star(start_state, end_state)
    ans.reverse()
    print(ans)
    doc3 = aligment_doc(ans, start_state, end_state)
    print(doc3)
    print(f"len {len(document1)} : {document1}")
    print()
    print(f"len {len(document2)} : {document2}")
    print()

    total_words = 0

    for i in range(0, len(document1)):
        total_words = total_words + calculate(document1[i])

    for i in range(0, len(document2)):
        if char_level_edit_distance(doc3[i], document2[i]) >= 0:
            print(
                f"sentances in both document are \n : {doc3[i]}\n doc2 : {document2[i]}"
            )
            print(
                f"Edit_distance this two document is : {char_level_edit_distance(doc3[i], document2[i]) }\n"
            )

    print(total_words)
