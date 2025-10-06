import re
import heapq

# ---------- Text Preprocessing ----------
def Normalizetext(s):
    arr = re.split(r'[.?!]', s)
    result = []
    for a in arr:
        a = a.strip()
        if not a:
            continue
        a = re.sub(r'[^\w\s]', '', a)
        result.append(a.lower())
    return result

# ----------   LEVENSHTEIN DISTANCE (word-level) ----------
def levenstein(s1, s2):
    tokens1, tokens2 = s1.split(), s2.split()
    n, m = len(tokens1), len(tokens2)
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]

    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])

    return dp[n][m]

# ---------- A* State ----------
class State:
    def __init__(self, i, j, g, f, alignment):
        self.i = i
        self.j = j
        self.g = g
        self.f = f
        self.alignment = alignment

    def __lt__(self, other):
        return self.f < other.f

# ---------- A* Search for Alignment ----------
def heuristic(doc1, doc2, i, j):
    remaining = min(len(doc1) - i, len(doc2) - j)
    return remaining  

def align(doc1, doc2):
    open_list = []
    heapq.heappush(open_list, State(0, 0, 0, 0, []))
    visited = set()

    while open_list:
        curr = heapq.heappop(open_list)

        # Goal state
        if curr.i == len(doc1) and curr.j == len(doc2):
            return curr.alignment

        key = (curr.i, curr.j)
        if key in visited:
            continue
        visited.add(key)

        # Align two sentences
        if curr.i < len(doc1) and curr.j < len(doc2):
            s1, s2 = doc1[curr.i], doc2[curr.j]
            cost = levenstein(s1, s2)
            new_alignment = curr.alignment + [(s1, s2, cost)]
            g = curr.g + cost
            h = heuristic(doc1, doc2, curr.i + 1, curr.j + 1)
            heapq.heappush(open_list, State(curr.i + 1, curr.j + 1, g, g + h, new_alignment))

        # Skip doc1
        if curr.i < len(doc1):
            new_alignment = curr.alignment + [(doc1[curr.i], None, len(doc1[curr.i].split()))]
            g = curr.g + len(doc1[curr.i].split())
            h = heuristic(doc1, doc2, curr.i + 1, curr.j)
            heapq.heappush(open_list, State(curr.i + 1, curr.j, g, g + h, new_alignment))

        # Skip doc2
        if curr.j < len(doc2):
            new_alignment = curr.alignment + [(None, doc2[curr.j], len(doc2[curr.j].split()))]
            g = curr.g + len(doc2[curr.j].split())
            h = heuristic(doc1, doc2, curr.i, curr.j + 1)
            heapq.heappush(open_list, State(curr.i, curr.j + 1, g, g + h, new_alignment))


    return []

# ---------- Detect Plagiarism ----------
#DETECT THE PLAGARISM BASED ON THE SIMILARITY IN WORDS

def detect_plagiarism(doc1, doc2, threshold=4):
    alignment = align(doc1, doc2)
    plagiarized = []
    for s1, s2, cost in alignment:
        if s1 and s2:  # aligned pair
            if cost <= threshold:
                plagiarized.append((s1, s2, cost))
    return alignment, plagiarized

# ---------- Main ----------
if __name__ == "__main__":

    text1 = """Artificial intelligence is transforming industries across the world. 
    It enables machines to learn from data and perform tasks that usually require human intelligence. 
    Applications include healthcare, finance, education, and transportation. 
    However, ethical concerns and job displacement remain significant challenges. 
    Developing responsible AI systems is crucial for the future."""

    text2 = """Artificial intelligence is changing industries globally. 
    It allows computers to learn from data and carry out tasks requiring human intelligence. 
    Some applications are in healthcare, banking, education, and travel. 
    But ethical issues and unemployment are still big concerns. 
    Building safe and responsible AI for the future is very important. 
    Additionally, AI can also improve environmental sustainability."""
    doc1 = Normalizetext(text1)
    doc2 = Normalizetext(text2)

    alignment, plagiarized = detect_plagiarism(doc1, doc2, threshold=4)

    print("Alignment Result:")
    for s1, s2, cost in alignment:
        if s1 and s2:
            print(f'Align: "{s1}" <-> "{s2}" | cost={cost}')
        elif s1:
            print(f'Skip Doc1: "{s1}"')
        elif s2:
            print(f'Skip Doc2: "{s2}"')
    print()
    print("\nPotential Plagiarism (edit distance <= 4):")
    for s1, s2, cost in plagiarized:
        print(f'"{s1}" <-> "{s2}" | cost={cost}')

plag_percent = (len(plagiarized) / len(doc1)) * 100
print()

print(f"The percantage of plag is : {plag_percent}%")
