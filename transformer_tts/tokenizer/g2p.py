import datetime as dt
import math
import re

# import optparse
# Option
# parser = optparse.OptionParser()
# parser.add_option(
#     "-v",
#     action="store_true",
#     dest="verbose",
#     default="False",
#     help="This option prints the detail information of g2p process.",
# )

# options, args = parser.parse_args()

ALL_TOKENS = {
    "aa",
    "c0",
    "cc",
    "ch",
    "ee",
    "h0",
    "ii",
    "k0",
    "kf",
    "kh",
    "kk",
    "ks",
    "lb",
    "lh",
    "lk",
    "ll",
    "lm",
    "lp",
    "ls",
    "lt",
    "mf",
    "mm",
    "nc",
    "nf",
    "ng",
    "nh",
    "nn",
    "oh",
    "oo",
    "p0",
    "pf",
    "ph",
    "pp",
    "ps",
    "qq",
    "rr",
    "s0",
    "ss",
    "t0",
    "tf",
    "th",
    "tt",
    "uu",
    "vv",
    "wa",
    "we",
    "wi",
    "wo",
    "wq",
    "wv",
    "xi",
    "xx",
    "ya",
    "ye",
    "yo",
    "yq",
    "yu",
    "yv",
    "._",
    ",_",
    "?_",
    "~_",
    "!_",
    "…_",
}
CON = [
    "K0",  # ㄱ
    "KK",  # ㄲ
    "ks",  # ㄳ
    "nn",  # ㄴ
    "nc",  # ㄵ
    "nh",  # ㄶ
    "t0",  # ㄷ
    "tt",  # ㄸ
    "rr",  # ㄹ
    "lk",  # ㄺ
    "lm",  # ㄻ
    "lb",  # ㄼ
    "ls",  # ㄽ
    "lt",  # ㄾ
    "lp",  # ㄿ
    "lh",  # ㅀ
    "mm",  # ㅁ
    "p0",  # ㅂ
    "pp",  # ㅃ
    "ps",  # ㅄ
    "s0",  # ㅅ
    "ss",  # ㅆ
    "oh",  # ㅇ
    "c0",  # ㅈ
    "cc",  # ㅉ
    "ch",  # ㅊ
    "kh",  # ㅋ
    "th",  # ㅌ
    "ph",  # ㅍ
    "h0",  # ㅎ
]
ONS = [
    "k0",  # ㄱ
    "kk",  # ㄲ
    "nn",  # ㄴ
    "t0",  # ㄷ
    "tt",  # ㄸ
    "rr",  # ㄹ
    "mm",  # ㅁ
    "p0",  # ㅂ
    "pp",  # ㅃ
    "s0",  # ㅅ
    "ss",  # ㅆ
    "oh",  # ㅇ?
    "c0",  # ㅈ
    "cc",  # ㅉ
    "ch",  # ㅊ
    "kh",  # ㅋ
    "th",  # ㅌ
    "ph",  # ㅍ
    "h0",  # ㅎ
]
NUC = [
    "aa",  # ㅏ
    "qq",  # ㅐ
    "ya",  # ㅑ
    "yq",  # ㅒ
    "vv",  # ㅓ
    "ee",  # ㅔ
    "yv",  # ㅕ
    "ye",  # ㅖ
    "oo",  # ㅗ
    "wa",  # ㅘ
    "wq",  # ㅙ
    "wo",  # ㅚ
    "yo",  # ㅛ
    "uu",  # ㅜ
    "wv",  # ㅝ
    "we",  # ㅞ
    "wi",  # ㅟ
    "yu",  # ㅠ
    "xx",  # ㅡ
    "xi",  # ㅢ
    "ii",  # ㅣ
]
COD = [
    "",
    "kf",  # ㄱ
    "kk",  # ㄲ
    "ks",  # ㄳ
    "nf",  # ㄴ
    "nc",  # ㄵ
    "nh",  # ㄶ
    "tf",  # ㄷ
    "ll",  # ㄹ
    "lk",  # ㄺ
    "lm",  # ㄻ
    "lb",  # ㄼ
    "ls",  # ㄽ
    "lt",  # ㄾ
    "lp",  # ㄿ
    "lh",  # ㅀ
    "mf",  # ㅁ
    "pf",  # ㅂ
    "ps",  # ㅄ
    "s0",  # ㅅ
    "ss",  # ㅆ
    "oh",  # ㅇ?
    "c0",  # ㅈ
    "ch",  # ㅊ
    "kh",  # ㅋ
    "th",  # ㅌ
    "ph",  # ㅍ
    "h0",  # ㅎ
]
SPE = ["._", ",_", "?_", "~_", "!_", "…_"]
num_phoneme = len(set(ONS + NUC + COD + SPE))


def writefile(body, fname):
    out = open(fname, "w")
    for line in body:
        out.write("{}\n".format(line))
    out.close()


def readRules(rule_book):
    f = open(rule_book, "r", encoding="utf-8")

    rule_in = []
    rule_out = []

    while True:
        line = f.readline()
        line = re.sub(r"\n", "", line)

        if line != "":
            if line[0] != "#":
                IOlist = line.split("\t")
                rule_in.append(IOlist[0].replace(",", "/"))
                if IOlist[1]:
                    # rule_out.append(IOlist[1])
                    rule_out.append(IOlist[1].replace(",", "/"))
                else:  # If output is empty (i.e. deletion rule)
                    rule_out.append("")
        if not line:
            break
    f.close()

    return rule_in, rule_out


def isHangul(charint):
    def isin(x: int, minimum: int, maximum: int) -> bool:
        return x >= minimum and x <= maximum

    # hangul
    h_a = 44032
    h_b = 55203
    # consonant
    c_a = 12593
    c_b = 12622
    # vowel
    v_a = 12623
    v_b = 12643

    if isin(charint, h_a, h_b):
        return 0
    if isin(charint, c_a, c_b):
        return 1
    if isin(charint, v_a, v_b):
        return 2

    return -1


def checkCharType(var_list):
    #  0: hangul
    #  1: consonant
    #  2: vowel
    #  3: whitespace
    #  4: spcecial
    # -1: non-hangul
    checked = []
    for i in var_list:
        ishangul = isHangul(i)

        if ishangul > -1:  # Hangul character
            checked.append(ishangul)
        elif i == 32:  # whitespace
            checked.append(3)
        elif i in [33, 44, 46, 63, 126, 8230]:  # special character
            checked.append(4)
        else:  # Non-hangul character
            checked.append(-1)
    return checked


def graph2phone(graphs):
    # Encode graphemes as utf8
    try:
        graphs = graphs.decode("utf8")
    except AttributeError:
        pass

    integers = [ord(i) for i in graphs]

    # Romanization (according to Korean Spontaneous Speech corpus; 성인자유발화코퍼스)
    phones = ""

    # Pronunciation
    chartype = checkCharType(integers)
    for idx, integer in enumerate(integers):
        if chartype[idx] == 0:  # not space characters
            base = 44032
            df = int(integer) - base
            iONS = int(math.floor(df / 588)) + 1
            iNUC = int(math.floor((df % 588) / 28)) + 1
            iCOD = int((df % 588) % 28) + 1

            s1 = "-" + ONS[iONS - 1]  # onset
            s2 = NUC[iNUC - 1]  # nucleus

            if COD[iCOD - 1]:  # coda
                s3 = COD[iCOD - 1]
            else:
                s3 = ""
            phones += s1 + s2 + s3

        elif chartype[idx] == 1:  # consonant
            i = -(12593 - integer)
            phones += "-" + CON[i]

        elif chartype[idx] == 2:  # vowel
            i = -(12623 - integer)
            phones += "-" + NUC[i]

        elif chartype[idx] == 3:  # space character
            tmp = "#"
            phones += tmp

        elif chartype[idx] == 4:  # special character
            phones += graphs[idx] + "_"

        phones = re.sub(r"-(oh)", "-", phones)
        tmp = ""

    # 초성 이응 삭제

    phones = re.sub(r"^oh", "", phones)
    phones = re.sub(r"-(oh)", "", phones)

    # 받침 이응 'ng'으로 처리 (Velar nasal in coda position)
    phones = re.sub(r"oh-", "ng-", phones)
    phones = re.sub(r"oh([# ]|$)", "ng", phones)

    # Remove all characters except Hangul and syllable delimiter (hyphen; '-')
    phones = re.sub(r"(\W+)\-", "\\1", phones)
    phones = re.sub(r"[^a-zA-Z0-9_.,?~!…]+$", "", phones)
    phones = re.sub(r"^\-", "", phones)

    return phones


def phone2prono(phones, rule_in, rule_out):
    # Apply g2p rules
    for pattern, replacement in zip(rule_in, rule_out):
        # print pattern
        phones = re.sub(pattern, replacement, phones)
        prono = phones
    return prono


def addPhoneBoundary(phones):
    # Add a comma (,) after every second alphabets to mark phone boundaries
    ipos = 0
    newphones = ""
    while ipos + 2 <= len(phones):
        if phones[ipos] == "-":
            newphones = newphones + phones[ipos]
            ipos += 1
        elif phones[ipos] == " ":
            ipos += 1
        elif phones[ipos] == "#":
            newphones = newphones + phones[ipos]
            ipos += 1

        newphones += phones[ipos : ipos + 2] + "/"
        ipos += 2

    return newphones


def addSpace(phones):
    ipos = 0
    newphones = ""
    while ipos < len(phones):
        if ipos == 0:
            newphones = newphones + phones[ipos] + phones[ipos + 1]
        else:
            newphones = newphones + " " + phones[ipos] + phones[ipos + 1]
        ipos += 2

    return newphones


def graph2prono(graphs, rule_in, rule_out):
    romanized = graph2phone(graphs)
    romanized_bd = addPhoneBoundary(romanized)
    prono = phone2prono(romanized_bd, rule_in, rule_out)

    prono = re.sub(r"/", " ", prono)
    prono = re.sub(r" $", "", prono)
    prono = re.sub(r"#", "-", prono)
    prono = re.sub(r"-+", "-", prono)

    prono_prev = prono
    identical = False
    loop_cnt = 1

    while not identical:
        prono_new = phone2prono(re.sub(r" ", "/", prono_prev + "/"), rule_in, rule_out)
        prono_new = re.sub(r"/", " ", prono_new)
        prono_new = re.sub(r" $", "", prono_new)

        if re.sub(r"-", "", prono_prev) == re.sub(r"-", "", prono_new):
            identical = True
            prono_new = re.sub(r"-", "", prono_new)

        else:
            loop_cnt += 1
            prono_prev = prono_new

    return prono_new


def testG2P(rulebook, testset):
    [testin, testout] = readRules(testset)
    cnt = 0
    body = []
    for idx in range(len(testin)):
        print("Test item #: " + str(idx + 1) + "/" + str(len(testin)))
        item_in = testin[idx]
        item_out = testout[idx]
        ans = graph2phone(item_out)
        ans = re.sub(r"-", "", ans)
        ans = addSpace(ans)

        [rule_in, rule_out] = readRules(rulebook)
        pred = graph2prono(item_in, rule_in, rule_out)

        if pred != ans:
            print(
                "G2P ERROR:  [result] "
                + pred
                + "\t\t\t[ans] "
                + item_in
                + " ["
                + item_out
                + "] "
                + ans
            )
            cnt += 1
        else:
            body.append(
                "[result] "
                + pred
                + "\t\t\t[ans] "
                + item_in
                + " ["
                + item_out
                + "] "
                + ans
            )

    print("Total error item #: " + str(cnt))
    writefile(body, "good.txt")


def runKoG2P(graph, rulebook):
    [rule_in, rule_out] = readRules(rulebook)
    prono = graph2prono(graph, rule_in, rule_out)

    print(prono)


def runTest(rulebook, testset):
    print("[ G2P Performance Test ]")
    beg = dt.datetime.now()

    testG2P(rulebook, testset)

    end = dt.datetime.now()
    print("Total time: ")
    print(end - beg)


# # Usage:
# if __name__ == "__main__":

#     if args[0] == "test":  # G2P Performance Test
#         runTest("rulebook.txt", "testset.txt")

#     else:
#         graph = args[0]
#         runKoG2P(graph, "rulebook.txt")
