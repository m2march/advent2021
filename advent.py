import numpy as np
import pandas as pd
import itertools
import collections 
import scipy.signal

def input_filename(num, is_test):
    return '{}{}.input'.format(num, '.test' if is_test else '')

def read_input_1(file):
    with open(file) as f:
        return [float(x) for x in f]

def puzzle_1_1(inputs):
    return np.sum(np.diff(np.array(inputs)) > 0)

def puzzle_1_2(inputs):
    s = pd.Series(inputs)
    c = s.rolling(3).mean().dropna()
    return puzzle_1_1(c)

def read_input_2(file):
    with open(file) as f:
        return [l.split(' ') for l in f]

def puzzle_2_1(inputs):
    dir_vectors = {
        'forward': np.array((0, 1)),
        'up': np.array((-1, 0)),
        'down': np.array((1, 0))
    }
    start = np.array((0, 0))
    for d, x in inputs:
        start += dir_vectors[d] * int(x)
    return np.prod(start)

def puzzle_2_2(inputs):
    dir_vectors = {
        'forward': np.array((0, 1)),
        'up': np.array((-1, 0)),
        'down': np.array((1, 0))
    }
    start = np.array([0, 0])
    depth = 0
    for d, x in inputs:
        start += dir_vectors[d] * int(x)
        depth += start[0] * int(x) * dir_vectors[d][1]
    return depth * start[1]

def read_input_3(file):
    with open(file) as f:
        return np.array([list(l.strip()) for l in f.readlines()], dtype=int)
    

def digits_to_dec(array):
    return int(''.join([str(x) for x in array.astype(int)]), base=2)


def puzzle_3_1(inputs):
    gamma = (np.sum(inputs, axis=0) / inputs.shape[0]) > 0.5
    epsilon = 1 - gamma

    return digits_to_dec(gamma) * digits_to_dec(epsilon)


def puzzle_3_2(inputs):
    def most_common_bit(inputs, most_common=True):
        most_common_bit = (np.sum(inputs, axis=0) / inputs.shape[0]) >= 0.5
        if not most_common:
            most_common_bit = 1 - most_common_bit
        return most_common_bit

    def extract_rating(inputs, most_common=True):
        current_bit = 0
        search = inputs.copy()
        while search.shape[0] > 1 and current_bit < search.shape[-1]:
            mcb = most_common_bit(search, most_common)
            search = search[search[:,current_bit] == mcb[current_bit], :]
            current_bit += 1
        return digits_to_dec(search[0, :])

    o2 = extract_rating(inputs)
    co2 = extract_rating(inputs, False)
    return o2 * co2


def read_input_4(file):
    with open(file) as f:
        numbers = [int(x) for x in f.readline().split(',')]
        f.readline()
        boards = [[]]
        for l in f:
            if len(l.strip()) == 0:
                boards.append([])
            else:
                try:
                    l = l.strip().replace('  ', ' ')
                    boards[-1].append([int(x) for x in l.split(' ')])
                except Exception as e:
                    print(l)
                    raise e
        boards = np.array(boards)
        return numbers, boards

def winning_board_indexes(boards):
    for axis in [1, 2]:
        if np.any(boards.sum(axis=axis) == 0):
             w = np.where(boards.sum(axis=axis) == 0)
             if w[0].shape != (0,):
                return w[0]
    return None

def puzzle_4_1(numbers, boards):
    assert(np.all(boards >= 0))
    boards = boards + 1
    for n in numbers:
        boards[boards == (n+1)] = 0
        wb = boards[winning_board_indexess(boards)[0],:,:]
        wb[wb != 0] -= 1
        print(wb, n)
        s = wb.sum()
        return s * n

def puzzle_4_2(numbers, boards):
    boards = boards + 1
    for n in numbers:
        boards[boards == (n+1)] = 0
        index = winning_board_indexes(boards)
        if index is not None:
            if boards.shape[0] == 1:
                print('Wb:', boards)
                wb = boards[0, :, :]
                wb[wb != 0] -= 1
                print(wb, n)
                s = wb.sum()
                return s * n
            full_index = np.ones(boards.shape[0], dtype=bool)
            full_index[index] = False
            boards = boards[full_index, :, :]


def read_input_5_1(file):
    def parse_line(l):
        parts = l.split('->')
        a, b = [[int(x.strip()) for x in p.split(',')]
                for p in parts]
        xs, ys = zip(*(a, b))
        return ((min(xs), min(ys)), (max(xs), max(ys)))

    with open(file) as f:
        return [parse_line(l) for l in f]

def puzzle_5_1(inputs):
    inputs = [i for i in inputs
              if i[0][0] == i[1][0] or i[0][1] == i[1][1]]

    m = np.zeros((max([i[j][0] for i in inputs for j in [0, 1]])+1,
                  max([i[j][0] for i in inputs for j in [0, 1]])+1))

    for i in inputs:
        m[i[0][1]:i[1][1]+1, i[0][0]:i[1][0]+1] += 1

    return np.sum(m > 1)

def read_input_5_2(file):
    with open(file) as f:
        return [[
                [int(x.strip()) for x in p.split(',')]
                for p in l.split('->')]
                for l in f
               ]

def puzzle_5_2(inputs):
    all_points = [y for x in inputs for y in x]
    m = np.zeros([max(x)+1 for x in zip(*all_points)], dtype=int)

    for i in inputs:
        dx = i[1][0] - i[0][0]
        dy = i[1][1] - i[0][1]
        assert(dx == 0 or dy == 0 or abs(dx) == abs(dy))
        dx = dx // abs(dx) if dx != 0 else dx
        dy = dy // abs(dy) if dy != 0 else dy
        assert(dx != 0 or dy != 0)
        s = i[0]
        while s != i[1]:
            m[s[1], s[0]] += 1
            s[0] += dx
            s[1] += dy 
        m[s[1], s[0]] += 1

    print(m)
    return np.sum(m > 1)


def read_input_6(file):
    with open(file) as f:
        return [int(x) for x in f.readline().split(',')]

def puzzle_6_1(inputs, days=80):
    lanterns = np.array(inputs, dtype=int)
    for d in range(days):
        lanterns -= 1
        n_l_count = np.sum(lanterns < 0)
        lanterns[lanterns < 0] = 6
        lanterns = np.concatenate([lanterns, 
                                   np.ones(n_l_count, dtype=int) * 8])
    return lanterns.shape[0]

def puzzle_6_2(inputs, days=256):
    lantern_groups = np.zeros(9, dtype=int)
    for i in inputs:
        lantern_groups[i] += 1

    for d in range(days):
        nl = np.zeros(9, dtype=int)
        nl[0:8] = lantern_groups[1:9]
        nl[8] = lantern_groups[0]
        nl[6] += lantern_groups[0]
        lantern_groups = nl
    return np.sum(lantern_groups)



def read_input_7(file):
    with open(file) as f:
        return [int(x) for x in f.readline().split(',')]

def puzzle_7_1(inputs):
    m = np.median(inputs)
    d = np.abs(np.array(inputs) - m)
    return int(d.sum())

def puzzle_7_2(inputs):
    m = np.mean(inputs)
    ma = np.array([[np.floor(m), np.ceil(m)]])
    ad = np.abs(np.array(inputs) - ma.T)
    ds = ((ad + 1) * ad) / 2
    s = ds.sum(axis=1)
    return int(min(s))


def read_input_8(file):
    with open(file) as f:
        hints, digits = zip(*[l.split('|') for l in f])

    hints = [[x.strip() for x in h.strip().split(' ')] for h in hints]
    digits = [[x.strip() for x in d.strip().split(' ')] for d in digits]
    return hints, digits

segments = {
    0: 'abcefg',
    1: 'cf',
    2: 'acdeg',
    3: 'acdfg',
    4: 'bcdf',
    5: 'abdfg',
    6: 'abdefg',
    7: 'acf',
    8: 'abcdefg',
    9: 'abcdfg'
}


def puzzle_8_1(inputs):
    hints, digits = inputs
    segment_count = {
        k : len(v)
        for k, v in segments.items()
    }
    segment_counts = list(segment_count.values())
    grouped = itertools.groupby(sorted(segment_counts))
    grouped_counts = [(k, len(list(v))) for k, v in grouped]
    easy_digits = [(k, v) for k, v in grouped_counts if v == 1]
    easy_lengths = [k for k, v in easy_digits]

    return len([
        d
        for ds in digits
        for d in ds
        if len(d) in easy_lengths
    ])


def puzzle_8_2(test):
    hints, digits = read_input_8(input_filename(8, test))

    segments_by_length = {
        k: list(v)
        for k, v in itertools.groupby(
            sorted(segments.values(), key=lambda x: len(x)),
            lambda x: len(x))
    }

    def seg2vec(ss):
        v = np.zeros((7,), dtype=bool)
        v[[ord(x) - ord('a') for x in ss]] = True
        return v

    numbers = []
    for hint_set, digit_set in zip(hints, digits):
        w2s = np.ones((7, 7), dtype=bool) # wires x segments
        h2n = np.zeros((10, 10), dtype=bool)

        for hi, h in enumerate(hint_set):
            for n, s in segments.items():
                if len(s) == len(h):
                    h2n[hi, n] = True
        
        unique_hints = np.where((np.sum(h2n, axis=1) == 1))[0]
        for hi in unique_hints:
            h = hint_set[hi]
            for s in segments_by_length[len(h)]:
                hv = seg2vec(h)
                sv = seg2vec(s)
                w2s[hv, :] &= sv
                w2s[:, sv] &= np.expand_dims(hv, 1)

        non_unique_hints = np.where((np.sum(h2n, axis=1) > 1))[0]
        choices = [
            (hi, np.where(h2n[hi] == 1)[0])
            for hi in non_unique_hints
        ]

        c_choices = []
        last_choice = 0
        while len(c_choices) < non_unique_hints.shape[0]:
            if len(c_choices) == 0:
                nc_h2n = h2n.copy()
                nc_w2s = w2s.copy()
            else:
                nc_h2n = c_choices[-1][1].copy()
                nc_w2s = c_choices[-1][-1].copy()

            cont = True
            for next_choice_i in range(last_choice,
                                       len(choices[len(c_choices)][1])):

                # Eval
                next_choice = choices[len(c_choices)][1][next_choice_i]
                h = hint_set[non_unique_hints[len(c_choices)]]
                hv = seg2vec(h)
                s = segments[next_choice]
                sv = seg2vec(s)

                if (nc_w2s[hv, :][:, sv].sum() >= len(h)):
                    nc_w2s[hv, :] &= sv
                    nc_w2s[:, sv] &= np.expand_dims(hv, 1)
                    nc_h2n[non_unique_hints[len(c_choices)]] = 0
                    nc_h2n[non_unique_hints[len(c_choices)]][next_choice] = 1
                    c_choices.append((next_choice_i, nc_h2n, nc_w2s))
                    cont = False
                    last_choice = 0
                    break

            if cont:
                last_choice = c_choices[-1][0] + 1
                c_choices = c_choices[:-1]

        h2n = c_choices[-1][1]
        mapping = {
            ''.join(sorted(h)): np.where(h2n[i,:] == 1)[0][0]
            for i, h in enumerate(hint_set)
        }
            
        number = [mapping[''.join(sorted(d))] for d in digit_set]
        number = int(''.join([str(x) for x in number]))
        numbers.append(number)
    print(numbers)
    return sum(numbers)



def read_input_9(test):
    with open(input_filename(9, test)) as f:
        return np.array([
            [int(x) for x in l.strip()]
            for l in f
        ], dtype=int)

def puzzle_9_1(inputs):
    #ms = np.ones(np.array(inputs.shape) + 2) * 10000
    #ms[1:-1, 1:-1] = inputs
    #print(ms)
    #print(ms[1:, :] < ms[:-1, :])
    #maxes = (
    #    (ms[1:, 1:-1] < ms[:-1, 1:-1]) & # top
    #    (ms[:-1, 1:-1] < ms[1:, 1:-1]) & # bottom
    #    (ms[1:-1, 1:] < ms[1:-1, :-1]) & # left
    #    (ms[1:-1, :-1] < ms[1:-1, 1:])   # right
    #)
    #print(maxes)
    nn = [np.array([-1, 0]), np.array([1, 0]),
          np.array([0, -1]), np.array([0, 1])]
    max_pos = []
    for r in range(inputs.shape[0]):
        for c in range(inputs.shape[1]):
            p = np.array([r, c])
            is_max = []
            for n in nn:
                if (np.all((p + n) >= np.array([0, 0])) and 
                    np.all((p + n) < np.array(inputs.shape))):
                    is_max.append(inputs[r, c] < inputs[r + n[0], c + n[1]])

            if all(is_max):
                max_pos.append((r, c))
    return sum([inputs[r, c] + 1 for r, c in max_pos])

def puzzle_9_2(inputs):
    colors = np.zeros(inputs.shape)
    cc = 1
    nn = [np.array([-1, 0]), np.array([0, -1])]
    for r in range(inputs.shape[0]):
        for c in range(inputs.shape[1]):
            if inputs[r, c] == 9:
                continue
            possble_colors = [
                colors[r + n[0], c + n[1]]
                for n in nn
                if ((r + n[0] >= 0) and (c + n[1] >= 0) and 
                    (colors[r + n[0], c + n[1]] != 0) and
                    (inputs[r + n[0], c + n[1]] != 0)
                   )
            ]
            if len(possble_colors) == 0:
                colors[r, c] = cc
                cc += 1
            else:
                nc = min(possble_colors)
                colors[r, c] = nc
                for p in possble_colors:
                    colors[colors == p] = nc

    u_colors = np.unique(colors)
    sizes = [np.sum(colors == c)
             for c in u_colors
             if c != 0
            ]
    return np.prod(sorted(sizes)[-3:])


def read_input_10(test):
    with open(input_filename(10, test)) as f:
        return [l.strip() for l in f.readlines()]

opening_characters = list('([{<')
matching_opening = { 
    ')': '(',
    ']': '[',
    '}': '{',
    '>': '<'
}
scores = {
    ')': 3,
    ']': 57,
    '}': 1197,
    '>': 25137
}

def parse(line):
    stack = []
    for x in line:
        if x in opening_characters:
            stack.append(x)
        else:
            if stack[-1] != matching_opening[x]:
                return x
            else:
                stack = stack[:-1]
    return stack

def puzzle_10_1(test):
    inputs = read_input_10(test)

    corrupt_chars = [parse(l) for l in inputs]
    return sum([scores[x] for x in corrupt_chars if not isinstance(x, list)])

def puzzle_10_2(test):
    inputs = read_input_10(test)
    parsing = [parse(l) for l in inputs]
    incomplete = [p for p in parsing if isinstance(p, list)]

    def incomplete_score(s):
        char_scores = {
            '(': 1,
            '[': 2,
            '{': 3,
            '<': 4
        }
        score = 0
        for x in reversed(s):
            score = score * 5
            score += char_scores[x]
        return score

    scores = [incomplete_score(p) for p in incomplete]
    return sorted(scores)[len(scores) // 2]


def read_input_11(file):
    with open(file) as f:
        return np.array([
            [int(x) for x in l.strip()]
            for l in f
        ], dtype=int)

def flashing_step(m):
    flash_count = 0
    m += 1
    flashing_full = np.zeros(m.shape, dtype=bool)
    while np.sum(m > 9) != 0:
        flashing = m > 9
        increase = scipy.signal.convolve2d(flashing, 
                                           np.ones((3, 3), dtype=int), 
                                           'same')
        m += increase
        m[flashing] = 0
        flashing_full |= flashing
        flash_count += np.sum(flashing)
    m[flashing_full] = 0
    return m, flash_count

def puzzle_11_1(test):
    inputs = read_input_11(input_filename(11, test))

    steps = 100
    full_flash_count = 0
    m = inputs.copy()

    for s in range(steps):
        if s < 10 or (s % 10) == 0:
            print(s)
            print(m)
        m, flash_count = flashing_step(m)
        full_flash_count += flash_count

    return full_flash_count

def puzzle_11_2(test):
    inputs = read_input_11(input_filename(11, test))

    m = inputs.copy()
    flash_count = None
    s = 0
    while flash_count != m.size:
        m, flash_count = flashing_step(m)
        s += 1
    return s


def is_small_cave(name):
    return name.lower() == name

def read_input_12(file):
    with open(file) as f:
        links = [l.strip().split('-') for l in f]

    cave_names = set([n for l in links for n in l])
    caves = {n: set() for n in cave_names}
    for a, b in links:
        caves[a].add(b)
        caves[b].add(a)
    return caves

def calculate_paths(caves, revisit_cave=[]):
    def can_revisit(fp):
        return not is_small_cave(fp[-1]) or (
            fp.count(fp[-1]) < (2 if fp[-1] not in revisit_cave else 3)
        )

    unfinished_paths = [['start']]
    finished_paths = []
    while len(unfinished_paths) != 0:
        forward_paths = [
            up + [nc]
            for up in unfinished_paths
            for nc in caves[up[-1]]
        ]
        finished_paths.extend([fp for fp in forward_paths if fp[-1] == 'end'])
        unfinished_paths = [
            fp
            for fp in forward_paths 
            if (fp[-1] != 'end' and can_revisit(fp))
        ]
    return finished_paths

def puzzle_12_1(test):
    caves = read_input_12(input_filename(12, test))
    print(caves)
    return len(calculate_paths(caves, []))

def puzzle_12_2(test):
    caves = read_input_12(input_filename(12, test))
    print(caves)
    return len(set(
        ['-'.join(p)
         for nc in caves
            if is_small_cave(nc) and nc not in ['start', 'end']
         for p in calculate_paths(caves, [nc])
        ]
    ))


def read_input_13(file):
    with open(file) as f:
        l = f.readline()
        dot_locs = []
        while len(l.strip()) != 0:
            dot_locs.append([int(x) for x in l.strip().split(',')])
            l = f.readline()

        instructions = []
        for l in f:
            i = l.split(' ')
            axis, line = i[-1].split('=')
            instructions.append([i, axis, int(line)])

        m = np.zeros((max([y for x,y in dot_locs]) + 1, 
                      max([x for x,y in dot_locs]) + 1), dtype=int)
        for x, y in dot_locs:
            m[y, x] = 1

        return dot_locs, instructions, m

def fold(m, axis, line):
    if axis == 'x':
        m = m.T

    assert(m[line, :].sum() == 0)

    tf = m[:line, :].copy()
    bf = m[line+1:, :].copy()

    if tf.shape[0] > bf.shape[0]:
        tf[tf.shape[0]-bf.shape[0]:tf.shape[0], :] += np.flip(bf, 0)
        ff = tf
    else:
        bf[:tf.shape[0], :] += np.flip(tf, 0)
        ff = bf

    if axis == 'x':
        return ff.T

    return ff

def puzzle_13_1(test):
    dot_locs, instructions, m = read_input_13(input_filename(13, test))
    print(m)
    f = fold(m, *instructions[0][-2:])
    print(m.shape, instructions[0], f.shape)
    print(f)
    return (f > 0).sum()

def puzzle_13_2(test):
    dot_locs, instructions, m = read_input_13(input_filename(13, test))
    for _, axis, line in instructions:
        m = fold(m, axis, line)
    pm = np.full(m.shape, '.')
    pm[m > 0] = '#'
    print('\n'.join(' '.join(r) for r in iter(np.flip(np.flip(pm, 1), 0))))


def read_input_14(file):
    with open(file) as f:
        pattern = f.readline().strip()
        f.readline()
        convertions = [
            [x.strip() for x in l.split('->')]
            for l in f
        ]
        return pattern, convertions

def puzzle_14_1(test, steps=10):
    pattern, convertions = read_input_14(input_filename(14, test))
    convertions = dict(convertions)
    n_pattern = ""
    for s in range(steps):
        i = 0
        for i in range(len(pattern)):
            n = pattern[i:i+2]
            if n in convertions:
                n_pattern += n[0] + convertions[n]
            else:
                n_pattern += n[0]
        pattern = n_pattern
        n_pattern = ""
    cc = sorted([len(list(v)) for k, v in itertools.groupby(sorted(pattern))])
    return cc[-1] - cc[0]

def puzzle_14_2(test, steps=40):
    pattern, convertions = read_input_14(input_filename(14, test))
    letters = list(set([l for a, b in convertions for l in list(a) + [b]]))
    letters_to_index = dict((k, i) for i, k in enumerate(letters))
    convertions_d = dict(convertions)
    convertions_to_index = dict((k, i) for i, (k, v) in enumerate(convertions))

    m = np.zeros((len(letters), len(convertions), steps+1))
    for si in range(steps+1):
        for ci in range(len(convertions)):
            if si == 0:
                m[letters_to_index[convertions[ci][0][0]], ci, 0] += 1
            else:
                c, r = convertions[ci]
                left = ''.join([c[0], r])
                right = ''.join([r, c[1]])
                if left in convertions_d:
                    m[:, ci, si] += m[:, convertions_to_index[left], si-1]
                if right in convertions_d:
                    m[:, ci, si] += m[:, convertions_to_index[right], si-1]


    r = np.zeros((len(letters),))
    for i in range(len(pattern)):
        if pattern[i:i+2] in convertions_d:
            r += m[:, convertions_to_index[pattern[i:i+2]], steps]
        else:
            r[letters_to_index[pattern[i]]] += 1
    s = sorted(r)
    return s[-1] - s[0]
